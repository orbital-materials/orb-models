from collections.abc import Mapping
from typing import Any, Literal, cast

import torch

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
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
from orb_models.forcefield.models.forcefield_utils import compute_gradient_forces_and_stress
from orb_models.forcefield.models.loss import forces_loss_function, stress_loss_function
from orb_models.forcefield.models.pair_repulsion import ZBLBasis


class ConservativeForcefieldRegressor(base.RegressorModelMixin[AtomGraphs]):
    """A specialized regressor that handles conservative (and optionally direct) predictions.

    This class is used to train a model that produces both conservative predictions of
    forces/stress via gradients of its energy with respect to positions/cell.

    Args:
        heads: A mapping of head names to heads.
        model: A pretrained model to use for transfer learning/finetuning.
        loss_weights: The weight of each loss term. Expected keys:
                - "energy", "forces", "stress", and optionally "rotational_grad".
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
        heads = {k: v for k, v in heads.items() if v is not None}
        if "energy" not in heads:
            raise ValueError("Missing required energy head.")

        loss_weights = loss_weights or {}
        loss_weights = {k: v for k, v in loss_weights.items() if v is not None}
        # BC: rename old grad-prefixed loss weight keys
        _bc = {"grad_forces": "forces", "grad_stress": "stress"}
        loss_weights = {_bc.get(k, k): v for k, v in loss_weights.items()}
        nongrad_loss_weights = {
            k: v
            for k, v in loss_weights.items()
            if k not in ["forces", "stress", "rotational_grad"]
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

        self.has_stress = has_stress

        collisions = {"forces", "stress"} & heads.keys()
        assert not collisions, (
            f"Heads {collisions} collide with gradient-based prediction keys in predict()."
        )

        self.extra_properties = []
        for name in heads.keys() - {"energy", "latent_charges", "latent_spins"}:
            if heads[name] is not None:
                self.extra_properties.append(heads[name].target.fullname)

    def enable_stress(self) -> None:
        """Enable stress computation. No-op if already enabled."""
        self.has_stress = True

    def prepare_for_inference(self) -> None:
        """Enable stress for inference — always available via autograd."""
        self.enable_stress()

    def disable_stress(self) -> None:
        """Disable stress computation."""
        self.has_stress = False

    @property
    def properties(self) -> list[str]:
        """Canonical names of properties available from predict()."""
        props = ["energy", "free_energy", "forces"]
        if self.has_stress:
            props.append("stress")
        props.extend(self.extra_properties)
        return props

    @property
    def autograd_derivative_keys(self) -> frozenset[str]:
        """Forward() output keys computed via autograd on the energy.

        Pipeline frameworks use this to know which outputs they should compute via their own
        autograd pass rather than reading from the model directly.
        """
        keys: set[str] = {"forces"}
        if self.has_stress:
            keys.add("stress")
        return frozenset(keys)

    @property
    def analytic_derivative_keys(self) -> frozenset[str]:
        """Forward() output keys for spatial derivatives that cannot be computed via autograd.

        Returns the dict keys of analytically computed derivatives (e.g. from Coulomb PME)
        that are *not* obtained via autograd on the energy, and should be added to the gradient-based forces/stress.
        """
        if self.coulomb_module is not None:
            keys = {"analytic_forces"}
            if self.has_stress:
                keys.add("analytic_stress")
            return frozenset(keys)
        return frozenset()

    def forward(
        self,
        batch: AtomGraphs,
        *,
        compute_forces: bool = True,
        compute_stress: bool = True,
        fp64_energy: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute absolute energy and conservative forces/stress.

        Returns a dict with total quantities and their components:

            Totals (consumer-facing):
                "energy"              — absolute energy (B,); fp64 when fp64_energy=True
                "forces"              — total forces (N, 3), autograd + explicit
                "stress"              — total stress (B, 6) in Voigt notation

            Components (used by loss and pipeline frameworks):
                "interaction_energy"  — energy without reference (B,)
                "rotational_grad"     — equivariance regularisation gradient (B, 3, 3)
                "analytic_forces"     — non-autograd forces, e.g. Coulomb (N, 3)
                "analytic_stress"     — non-autograd stress, e.g. Coulomb (B, 6)
        """
        compute_stress = compute_stress and self.has_stress

        if compute_forces or compute_stress:
            vectors, stress_displacement, generator = batch.compute_differentiable_edge_vectors()
            assert stress_displacement is not None
            assert generator is not None
            batch.system_features["stress_displacement"] = stress_displacement
            batch.system_features["generator"] = generator
            batch.edge_features["vectors"] = vectors

        out = self.model(batch)
        node_features = out["node_features"]

        # Latent charges (and spins) are fed into the energy head
        latent_charges = None
        if "latent_charges" in self.heads:
            latent_charges = self.heads["latent_charges"](node_features, batch)

        latent_spins = None
        if "latent_spins" in self.heads:
            latent_spins = self.heads["latent_spins"](node_features, batch)

        energy_head = self.heads["energy"]
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

        if self.coulomb_module is not None:
            assert latent_charges is not None, "CoulombModule requires a LatentChargeHead"
            coulomb_energy, explicit_forces, explicit_stress = self.coulomb_module(
                latent_charges, batch, compute_forces=True, compute_stress=self.has_stress
            )
            interaction_energy = interaction_energy + coulomb_energy
            out["analytic_forces"] = explicit_forces
            if explicit_stress is not None:
                out["analytic_stress"] = explicit_stress

        out["interaction_energy"] = interaction_energy
        out["energy"] = cast(EnergyHead, self.heads["energy"]).absolute_energy(
            interaction_energy, batch, fp64_energy
        )

        if compute_forces or compute_stress:
            forces, stress, rotational_grad = compute_gradient_forces_and_stress(
                energy=interaction_energy,
                positions=batch.node_features["positions"],
                displacement=batch.system_features["stress_displacement"],
                cell=batch.system_features["cell"],
                training=self.training,
                compute_stress=compute_stress,
                generator=batch.system_features["generator"],
            )

            if "analytic_forces" in out:
                forces = forces + out["analytic_forces"]
            if "analytic_stress" in out:
                stress = stress + out["analytic_stress"]

            if compute_forces:
                out["forces"] = forces  # eV / A
            if compute_stress:
                out["stress"] = stress  # eV / A^3
            out["rotational_grad"] = rotational_grad

        for name in self.extra_properties:
            out[name] = self.heads[name](node_features, batch)

        return out

    def predict(
        self,
        batch: AtomGraphs,
        split: bool = False,
        *,
        compute_forces: bool = True,
        compute_stress: bool = True,
        fp64_energy: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Consumer-facing inference: absolute energy, forces, and stress.

        Args:
            batch: Input atomic graph batch.
            split: If True, split per-system predictions into a list.
            compute_forces: Include forces in the output.
            compute_stress: Include stress in the output.
            fp64_energy: Upcast energy to fp64 before adding reference;
                required to preserve kJ/mol resolution since reference
                energies can be as high as ~1e4-1e5 eV.

        If both compute_forces and compute_stress are False, the autograd computation
        is skipped and the energy retains its computation graph.

        Returns:
            "energy"  — absolute energy (B,)
            "forces"  — total forces (N, 3)
            "stress"  — total stress in Voigt notation (B, 6)
        """
        # self() not self.forward() to respect torch.compile
        preds = self(
            batch,
            compute_forces=compute_forces,
            compute_stress=compute_stress,
            fp64_energy=fp64_energy,
        )

        out: dict[str, torch.Tensor] = {"energy": preds["energy"]}
        if compute_forces and "forces" in preds:
            out["forces"] = preds["forces"]
        if compute_stress and "stress" in preds:
            out["stress"] = preds["stress"]

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

        energy_pred = out["interaction_energy"]
        forces_pred = out["forces"]

        metrics: dict = {}
        total_loss = torch.tensor(0.0, device=batch.positions.device, dtype=batch.positions.dtype)

        # Energy
        energy_head = cast(EnergyHead, self.heads["energy"])
        loss_out = energy_head.loss(energy_pred, batch)
        loss = self.loss_weights["energy"] * loss_out.loss
        total_loss += loss
        metrics.update(loss_out.log)
        metrics["energy_loss"] = loss

        # Conservative forces
        loss_out = forces_loss_function(
            raw_pred=forces_pred,
            raw_target=batch.node_targets["forces"],
            raw_gold_target=batch.node_targets["forces"],
            name="forces",
            normalizer=self.grad_forces_normalizer,
            n_node=batch.n_node,
            fix_atoms=batch.fix_atoms,
            loss_type=self.forces_loss_type,
            training=self.training,
        )
        loss = self.loss_weights["forces"] * loss_out.loss
        total_loss += loss
        metrics.update(loss_out.log)
        metrics["forces_loss"] = loss

        # Conservative stress (optional)
        if self.has_stress and "stress" in out:
            loss_out = stress_loss_function(
                raw_pred=out["stress"],
                raw_target=batch.system_targets["stress"],
                raw_gold_target=batch.system_targets["stress"],
                name="stress",
                normalizer=self.grad_stress_normalizer,
                loss_type=energy_head.loss_type,
            )
            loss = self.loss_weights["stress"] * loss_out.loss
            total_loss += loss
            metrics.update(loss_out.log)
            metrics["stress_loss"] = loss

        # Equigrad
        if "rotational_grad" in self.loss_weights:
            rotational_grad_rms = torch.linalg.norm(out["rotational_grad"], dim=(1, 2)).mean()
            loss = self.loss_weights["rotational_grad"] * rotational_grad_rms
            total_loss += loss
            metrics["equigrad_loss"] = loss
            metrics["rotational_grad_rms"] = rotational_grad_rms

        # Confidence
        if "confidence" in self.heads:
            confidence_head = cast(ConfidenceHead, self.heads["confidence"])
            raw_forces_target = batch.node_targets["forces"]
            forces_error = torch.abs(forces_pred - raw_forces_target).mean(dim=-1)
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

    def is_compiled(self):
        """Check if the model is compiled."""
        return self._compiled_call_impl or self.model._compiled_call_impl
