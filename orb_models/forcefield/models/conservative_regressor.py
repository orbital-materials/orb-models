from collections.abc import Mapping
from typing import Any, Literal, cast

import torch

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.dataset.property_definitions import PROPERTIES, PropertyDefinition
from orb_models.common.models import base
from orb_models.common.models.gns import MoleculeGNS
from orb_models.common.models.graph_regressor import _validate_heads_and_loss_weights
from orb_models.common.models.load import load_regressor_state_dict
from orb_models.common.models.nn_util import ScalarNormalizer
from orb_models.common.models.segment_ops import split_prediction
from orb_models.forcefield.models.coulomb_module import CoulombModule
from orb_models.forcefield.models.forcefield_heads import (
    ChargeConditionedEnergyHead,
    ConfidenceHead,
    EnergyHead,
    ForcefieldHead,
)
from orb_models.forcefield.models.forcefield_utils import (
    compute_gradient_forces_and_stress,
    torch_full_3x3_to_voigt_6_stress,
)
from orb_models.forcefield.models.loss import forces_loss_function, stress_loss_function
from orb_models.forcefield.models.pair_repulsion import ZBLBasis


class ConservativeForcefieldRegressor(base.RegressorModelMixin[AtomGraphs]):
    """A specialized regressor that handles conservative (and optionally direct) predictions.

    This class is used to train a model that produces both conservative predictions of
    forces/stress via gradients of its energy with respect to positions/cell.

    Args:
        heads: A mapping of head names to heads.
        model: A pretrained model to use for transfer learning/finetuning.
        loss_weights: The weight of the energy loss in the total loss.
            Additionally, the conservative model must also have two keys:
                - "grad_forces"
                - "grad_stress"
            which weight the gradient based losses of forces/stress respectively.
        coulomb_module: Optional CoulombModule for long-range electrostatics.
            When present, a latent_charges head must also be in heads.
        **kwargs: Additional kwargs, used for backwards compatibility of deprecated arguments.
    """

    _deprecated_kwargs = [
        "model_requires_grad",
        "cutoff_layers",
        "ensure_grad_loss_weights",
    ]

    def __init__(
        self,
        heads: Mapping[str, ForcefieldHead | ConfidenceHead],
        model: MoleculeGNS,
        loss_weights: dict[str, float] | None = None,
        online_normalisation: bool = True,
        level_of_theory: str | None = None,
        forces_loss_type: Literal["mae", "mse", "huber_0.01", "condhuber_0.01"] = "condhuber_0.01",
        pair_repulsion: bool = False,
        has_stress: bool = True,
        coulomb_module: CoulombModule | None = None,
        **kwargs,
    ):
        super().__init__()
        for kwarg in kwargs:
            if kwarg not in self._deprecated_kwargs:
                raise ValueError(
                    f"Unknown kwargs: {kwarg}, expected only backward compatible kwargs "
                    f"from {self._deprecated_kwargs}"
                )
        if "energy" not in heads:
            raise ValueError("Missing required energy head.")

        loss_weights = loss_weights or {}
        loss_weights = {k: v for k, v in loss_weights.items() if v is not None}
        nongrad_loss_weights = {
            k: v
            for k, v in loss_weights.items()
            if k not in ["grad_forces", "grad_stress", "rotational_grad"]
        }
        _validate_heads_and_loss_weights(heads, nongrad_loss_weights)

        self.loss_weights = loss_weights
        self.forces_loss_type = forces_loss_type

        self.model = model
        self.heads = torch.nn.ModuleDict(heads)
        self.grad_forces_normalizer = ScalarNormalizer(online=online_normalisation)
        self.grad_stress_normalizer = ScalarNormalizer(online=online_normalisation)

        self.pair_repulsion = pair_repulsion
        if self.pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(p=6, compute_gradients=False, node_aggregation="sum")

        self.coulomb_module = coulomb_module
        if self.coulomb_module is not None:
            assert "latent_charges" in self.heads, (
                "CoulombModule requires a 'latent_charges' head in heads"
            )

        # Target names
        self.energy_name = heads["energy"].target.fullname
        self.grad_prefix = "grad"

        self.forces_name = f"forces-{level_of_theory}" if level_of_theory else "forces"
        self.forces_target = PROPERTIES[self.forces_name]
        self.grad_forces_name = f"{self.grad_prefix}_{self.forces_name}"

        # Stress is optional since only periodic systems have it
        self.has_stress = has_stress
        if self.has_stress:
            self.stress_name: str | None = (
                f"stress-{level_of_theory}" if level_of_theory else "stress"
            )
            self.stress_target: PropertyDefinition | None = PROPERTIES[self.stress_name]
            self.grad_stress_name: str | None = f"{self.grad_prefix}_{self.stress_name}"
        else:
            self.stress_name = None
            self.stress_target = None
            self.grad_stress_name = None
        assert self.has_stress == (self.grad_stress_name is not None), (
            "grad_stress_name must be set if has_stress is True"
        )

        self.grad_rotation_name = "rotational_grad"

        self.extra_properties = []
        for name in heads.keys() - {"energy", "latent_charges", "latent_spins"}:
            if heads[name] is not None:
                self.extra_properties.append(heads[name].target.fullname)

    def enable_stress(self) -> None:
        """Enable stress computation. No-op if already enabled."""
        if self.has_stress:
            return
        self.has_stress = True
        self.stress_name = "stress"
        self.stress_target = PROPERTIES["stress"]
        self.grad_stress_name = f"{self.grad_prefix}_{self.stress_name}"

    def prepare_for_inference(self) -> None:
        """Enable stress for inference — always available via autograd."""
        self.enable_stress()

    def disable_stress(self) -> None:
        """Disable stress computation."""
        self.has_stress = False

    @property
    def properties(self):
        """List of names of predicted properties."""
        props = [
            self.energy_name,
            "free_energy",
            self.grad_forces_name,
            self.grad_rotation_name,
        ]
        if self.has_stress:
            assert self.grad_stress_name is not None, (
                "grad_stress_name must be set if has_stress is True"
            )
            props.append(self.grad_stress_name)
        props.extend(self.extra_properties)
        return props

    def forward(self, batch: AtomGraphs) -> dict[str, torch.Tensor]:
        """Forward pass computing both direct and conservative predictions."""
        vectors, stress_displacement, generator = batch.compute_differentiable_edge_vectors()
        assert stress_displacement is not None
        assert generator is not None
        batch.system_features["stress_displacement"] = stress_displacement
        batch.system_features["generator"] = generator
        batch.edge_features["vectors"] = vectors

        # Get base model features
        out = self.model(batch)
        node_features = out["node_features"]

        # Predict per-atom charges/spins BEFORE energy head so they can
        # be used as conditioning features in ChargeConditionedEnergyHead and CoulombModule.
        latent_charges = None
        if "latent_charges" in self.heads:
            latent_charges = self.heads["latent_charges"](node_features, batch)

        latent_spins = None
        if "latent_spins" in self.heads:
            latent_spins = self.heads["latent_spins"](node_features, batch)

        energy_head = self.heads[self.energy_name]
        energy_head = cast(ForcefieldHead, energy_head)
        if isinstance(energy_head, ChargeConditionedEnergyHead):
            interaction_energy = energy_head(
                node_features,
                batch,
                per_atom_charges=latent_charges,
                per_atom_spins=latent_spins,
            )
        else:
            assert latent_spins is None, "Latent spins are predicted but not used."
            interaction_energy = energy_head(node_features, batch)
        if self.pair_repulsion:
            interaction_energy += self.pair_repulsion_fn(batch)["energy"]

        coulomb_explicit_forces = None
        coulomb_explicit_virial = None
        if self.coulomb_module is not None:
            assert latent_charges is not None, "CoulombModule requires a LatentChargeHead"
            coulomb_energy, coulomb_explicit_forces, coulomb_explicit_virial = self.coulomb_module(
                latent_charges, batch
            )
            interaction_energy += coulomb_energy

        out[self.energy_name] = interaction_energy

        forces, stress, rotational_grad = compute_gradient_forces_and_stress(
            energy=interaction_energy,
            positions=batch.node_features["positions"],
            displacement=batch.system_features["stress_displacement"],
            cell=batch.system_features["cell"],
            training=self.training,
            compute_stress=self.has_stress,
            generator=batch.system_features["generator"],
        )

        # Add explicit/spatial Coulomb force/stress corrections (see CoulombModule docstring).
        if self.coulomb_module is not None:
            assert coulomb_explicit_forces is not None, "Explicit/spatial forces are not computed"
            assert coulomb_explicit_virial is not None, "Explicit/spatial virial is not computed"
            forces = forces + coulomb_explicit_forces
            if self.has_stress:
                assert stress is not None, "has_stress is True but stress is None"
                cell_3d = batch.system_features["cell"].view(-1, 3, 3)
                volume = torch.linalg.det(cell_3d).abs()
                coulomb_stress_3x3 = -coulomb_explicit_virial / volume.view(-1, 1, 1)
                coulomb_stress_3x3 = torch.where(
                    torch.abs(coulomb_stress_3x3) < 1e10,
                    coulomb_stress_3x3,
                    torch.zeros_like(coulomb_stress_3x3),
                )
                stress = stress + torch_full_3x3_to_voigt_6_stress(coulomb_stress_3x3)

        out[self.grad_forces_name] = forces  # eV / A
        if self.has_stress:
            out[self.grad_stress_name] = stress  # eV / A^3

        out[self.grad_rotation_name] = rotational_grad
        for name in self.extra_properties:
            out[name] = self.heads[name](node_features, batch)

        return out

    def predict(
        self,
        batch: AtomGraphs,
        split: bool = False,
        fp64_energy: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Predict energy, forces, and stress.

        Args:
            batch: Input batch.
            split: If True, split predictions per graph.
            fp64_energy: If True (default), return absolute energy in fp64;
                required to preserve kJ/mol resolution since reference
                energies can be as high as ~1e4-1e5 eV. If False, returns
                energy in the input dtype.
        """
        preds = self(batch)

        out = {}
        energy_head = cast(EnergyHead, self.heads[self.energy_name])
        out[self.energy_name] = energy_head.absolute_energy(
            preds[self.energy_name], batch, fp64=fp64_energy
        )
        out[self.grad_forces_name] = preds[self.grad_forces_name]
        if self.has_stress:
            assert self.grad_stress_name is not None, (
                "grad_stress_name must be set if has_stress is True"
            )
            out[self.grad_stress_name] = preds[self.grad_stress_name]
        out[self.grad_rotation_name] = preds[self.grad_rotation_name]
        for name in self.extra_properties:
            head = self.heads[name]
            if isinstance(head, ForcefieldHead):
                out[name] = preds[name]
            elif isinstance(head, ConfidenceHead):
                out[name] = torch.softmax(preds[name], dim=-1)
            else:
                raise ValueError(f"Expected ForcefieldHead or ConfidenceHead, got {type(head)}.")

        if split:
            for name, pred in out.items():
                out[name] = split_prediction(pred, batch.n_node)

        return out

    def loss(self, batch: AtomGraphs) -> base.ModelOutput:
        """Compute loss including both direct and conservative terms."""
        out = self(batch)

        energy_pred = out[self.energy_name]
        raw_grad_forces_pred = out[self.grad_forces_name]

        # metrics
        metrics: dict = {}

        total_loss = torch.tensor(
            0.0,
            device=batch.positions.device,
            dtype=batch.positions.dtype,
        )

        # Energy
        energy_head = self.heads[self.energy_name]
        energy_head = cast(EnergyHead, energy_head)
        loss_out = energy_head.loss(energy_pred, batch)
        loss = self.loss_weights[self.energy_name] * loss_out.loss
        total_loss += loss
        metrics.update(loss_out.log)
        metrics[f"{self.energy_name}_loss"] = loss

        # Conservative forces
        loss_out = forces_loss_function(
            raw_pred=raw_grad_forces_pred,
            raw_target=batch.node_targets[self.forces_name],
            raw_gold_target=batch.node_targets[self.forces_name],
            name=self.forces_name,
            normalizer=self.grad_forces_normalizer,
            n_node=batch.n_node,
            fix_atoms=batch.fix_atoms,
            loss_type=self.forces_loss_type,
            training=self.training,
        )
        loss = self.loss_weights[self.grad_forces_name] * loss_out.loss
        total_loss += loss
        metrics.update({f"{self.grad_prefix}-{k}": v for k, v in loss_out.log.items()})
        metrics[f"{self.grad_forces_name}_loss"] = loss

        # Conservative stress (optional)
        if self.has_stress and self.grad_stress_name in out:
            assert self.stress_name is not None, "stress_name must be set if has_stress is True"
            assert self.grad_stress_name is not None, (
                "grad_stress_name must be set if has_stress is True"
            )
            raw_grad_stress_pred = out[self.grad_stress_name]
            loss_out = stress_loss_function(
                raw_pred=raw_grad_stress_pred,
                raw_target=batch.system_targets[self.stress_name],
                raw_gold_target=batch.system_targets[self.stress_name],
                name=self.stress_name,
                normalizer=self.grad_stress_normalizer,
                loss_type=energy_head.loss_type,
            )
            loss = self.loss_weights[self.grad_stress_name] * loss_out.loss
            loss_out.log[f"{self.grad_stress_name}_loss"] = loss
            total_loss += loss
            metrics.update({f"{self.grad_prefix}-{k}": v for k, v in loss_out.log.items()})

        # Direct forces / stress predictions
        for grad_name in [self.grad_forces_name] + (
            [self.grad_stress_name] if self.has_stress and self.grad_stress_name in out else []
        ):
            assert grad_name is not None
            direct_name = grad_name.replace(self.grad_prefix + "_", "")
            if direct_name in self.extra_properties:
                direct_head = cast(ForcefieldHead, self.heads[direct_name])
                direct_pred = out[direct_name]
                loss_out = direct_head.loss(direct_pred, batch)
                loss = self.loss_weights[direct_name] * loss_out.loss
                total_loss += loss
                metrics.update(loss_out.log)
                metrics[f"{direct_name}_loss"] = loss

        # Equigrad
        if self.grad_rotation_name in self.loss_weights:
            rotational_grad_rms = torch.linalg.norm(
                out[self.grad_rotation_name],
                dim=(1, 2),
            ).mean()
            loss = self.loss_weights[self.grad_rotation_name] * rotational_grad_rms
            total_loss += loss
            metrics["equigrad_loss"] = loss
            metrics[f"{self.grad_rotation_name}_rms"] = rotational_grad_rms

        # Confidence
        if "confidence" in self.heads:
            confidence_head = self.heads["confidence"]
            confidence_head = cast(ConfidenceHead, confidence_head)
            raw_forces_target = batch.node_targets[self.forces_name]
            forces_error = torch.abs(raw_grad_forces_pred - raw_forces_target).mean(dim=-1)
            confidence_logits = out["confidence"]
            loss_out = confidence_head.loss(confidence_logits, forces_error, batch)
            loss = self.loss_weights["confidence"] * loss_out.loss
            total_loss += loss
            metrics.update(loss_out.log)
            metrics["confidence_loss"] = loss

        metrics["loss"] = total_loss
        return base.ModelOutput(loss=total_loss, log=metrics)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
        skip_artifact_reference_energy: bool = False,
    ):
        """Load state dict for ConservativeGraphRegressor."""
        load_regressor_state_dict(
            self,
            state_dict,
            strict=strict,
            assign=assign,
            skip_artifact_reference_energy=skip_artifact_reference_energy,
        )

    def compile(self, *args, **kwargs):
        """Override the default Module.compile method to compile only the GNS model."""
        self.model.compile(*args, **kwargs)

    def is_compiled(self):
        """Check if the model is compiled."""
        return self._compiled_call_impl or self.model._compiled_call_impl
