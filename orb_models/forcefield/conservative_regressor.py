from typing import Any, Mapping, Optional, Dict, Literal
import torch
from torch import nn

from orb_models.forcefield import base
from orb_models.forcefield.gns import MoleculeGNS
from orb_models.forcefield.forcefield_utils import (
    split_prediction,
    validate_regressor_inputs,
)
from orb_models.forcefield.forcefield_utils import compute_gradient_forces_and_stress
from orb_models.forcefield.load import load_forcefield_state_dict
from orb_models.forcefield.pair_repulsion import ZBLBasis
from orb_models.forcefield.nn_util import ScalarNormalizer
from orb_models.forcefield.property_definitions import PROPERTIES
from orb_models.forcefield.loss import forces_loss_function, stress_loss_function
from orb_models.forcefield.atomic_system import SystemConfig


class ConservativeForcefieldRegressor(nn.Module):
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
        distill_direct_heads: Whether to distill the direct heads into the conservative heads.
        ensure_grad_loss_weights: Whether to ensure that the grad_forces and grad_stress keys are
            present in the loss_weights. Should only be used during training.
        **kwargs: Additional kwargs, used for backwards compatibility of deprecated arguments.
    """

    _deprecated_kwargs = ["model_requires_grad", "cutoff_layers"]

    def __init__(
        self,
        heads: Mapping[str, torch.nn.Module],
        model: MoleculeGNS,
        loss_weights: Optional[Dict[str, float]] = None,
        distill_direct_heads: bool = False,
        ensure_grad_loss_weights: bool = True,
        online_normalisation: bool = True,
        level_of_theory: Optional[str] = None,
        forces_loss_type: Literal[
            "mae", "mse", "huber_0.01", "condhuber_0.01"
        ] = "condhuber_0.01",
        pair_repulsion: bool = False,
        system_config: Optional[SystemConfig] = None,
        **kwargs,
    ):
        super().__init__()
        for kwarg in kwargs:
            if kwarg not in self._deprecated_kwargs:
                raise ValueError(
                    f"Unknown kwargs: {kwarg}, expected only backward compatible kwargs "
                    f"from {self._deprecated_kwargs}"
                )
        if "energy" not in heads.keys():
            raise ValueError("Missing required energy head.")

        loss_weights = loss_weights or {}
        loss_weights = {k: v for k, v in loss_weights.items() if v is not None}
        validate_regressor_inputs(
            heads, loss_weights, ensure_grad_loss_weights=ensure_grad_loss_weights
        )
        self.loss_weights = loss_weights
        self.distill_direct_heads = distill_direct_heads
        self.forces_loss_type = forces_loss_type

        self.model = model
        self.heads = torch.nn.ModuleDict(heads)
        self.grad_forces_normalizer = ScalarNormalizer(online=online_normalisation)
        self.grad_stress_normalizer = ScalarNormalizer(online=online_normalisation)

        self.pair_repulsion = pair_repulsion
        if self.pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(p=6, compute_gradients=False)

        # Target names
        self.energy_name = heads["energy"].target.fullname  # type: ignore
        self.grad_prefix = "grad"

        self.forces_name = f"forces-{level_of_theory}" if level_of_theory else "forces"
        self.forces_target = PROPERTIES[self.forces_name]
        self.grad_forces_name = f"{self.grad_prefix}_{self.forces_name}"

        self.stress_name = f"stress-{level_of_theory}" if level_of_theory else "stress"
        self.stress_target = PROPERTIES[self.stress_name]
        self.grad_stress_name = f"{self.grad_prefix}_{self.stress_name}"

        self.grad_rotation_name = "rotational_grad"

        self.extra_properties = []
        for name in heads.keys() - {"energy"}:
            self.extra_properties.append(heads[name].target.fullname)  # type: ignore

        self._system_config = system_config

    @property
    def system_config(self) -> SystemConfig:
        return self._system_config

    @property
    def properties(self):
        """List of names of predicted properties."""
        props = [
            self.energy_name,
            "free_energy",
            self.grad_forces_name,
            self.grad_stress_name,
            self.grad_rotation_name,
        ]
        props.extend(self.extra_properties)
        return props

    def forward(self, batch: base.AtomGraphs) -> Dict[str, torch.Tensor]:
        """Forward pass computing both direct and conservative predictions."""
        vectors, stress_displacement, generator = (
            batch.compute_differentiable_edge_vectors()
        )
        assert stress_displacement is not None
        assert generator is not None
        batch.system_features["stress_displacement"] = stress_displacement
        batch.system_features["generator"] = generator
        batch.edge_features["vectors"] = vectors

        # Get base model features
        out = self.model(batch)
        node_features = out["node_features"]

        energy_head = self.heads[self.energy_name]
        base_energy = energy_head(node_features, batch)
        raw_energy = energy_head.denormalize(base_energy, batch)
        if self.pair_repulsion:
            raw_energy += self.pair_repulsion_fn(batch)["energy"]
        out[self.energy_name] = energy_head.normalize(raw_energy, batch, online=False)

        forces, stress, rotational_grad = compute_gradient_forces_and_stress(
            energy=raw_energy,
            positions=batch.node_features["positions"],
            displacement=batch.system_features["stress_displacement"],
            cell=batch.system_features["cell"],
            training=self.training,
            compute_stress=True,
            generator=batch.system_features["generator"],
        )
        out[self.grad_forces_name] = forces  # eV / A
        out[self.grad_stress_name] = stress  # eV / A^3

        out[self.grad_rotation_name] = rotational_grad
        for name in self.extra_properties:
            out[name] = self.heads[name](node_features, batch)

        return out

    def predict(
        self, batch: base.AtomGraphs, split: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Predict energy, forces, and stress."""
        preds = self(batch)

        out = {}
        out[self.energy_name] = self.heads[self.energy_name].denormalize(
            preds[self.energy_name], batch
        )
        out[self.grad_forces_name] = preds[self.grad_forces_name]
        out[self.grad_stress_name] = preds[self.grad_stress_name]
        out[self.grad_rotation_name] = preds[self.grad_rotation_name]
        for name in self.extra_properties:
            head = self.heads[name]
            if hasattr(head, "denormalize"):
                out[name] = head.denormalize(preds[name], batch)
            elif name == "confidence":
                out[name] = torch.softmax(preds[name], dim=-1)
            else:
                raise ValueError(f"Expected normalizer or confidence head, got {name}.")

        if split:
            for name, pred in out.items():
                out[name] = split_prediction(pred, batch.n_node)

        return out  # type: ignore

    def loss(self, batch: base.AtomGraphs) -> base.ModelOutput:
        """Compute loss including both direct and conservative terms."""
        out = self(batch)

        energy_pred = out[self.energy_name]
        raw_grad_forces_pred = out[self.grad_forces_name]
        grad_forces_pred = self.grad_forces_normalizer(
            raw_grad_forces_pred, online=False
        )
        raw_grad_stress_pred = out[self.grad_stress_name]
        grad_stress_pred = self.grad_stress_normalizer(
            raw_grad_stress_pred, online=False
        )

        # metrics
        metrics = {}

        total_loss = torch.tensor(
            0.0,
            device=batch.positions.device,
            dtype=batch.positions.dtype,
        )

        # Energy
        energy_head = self.heads[self.energy_name]
        loss_out = energy_head.loss(energy_pred, batch)
        loss = self.loss_weights.get(self.energy_name, 1.0) * loss_out.loss  # type: ignore
        loss_out.log[f"{self.energy_name}_loss"] = loss
        total_loss += loss
        metrics.update(loss_out.log)

        # Conservative forces
        loss_out = forces_loss_function(
            pred=grad_forces_pred,
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
        loss_out.log[f"{self.grad_forces_name}_loss"] = loss
        total_loss += loss
        metrics.update({f"{self.grad_prefix}-{k}": v for k, v in loss_out.log.items()})

        # Conservative stress
        loss_out = stress_loss_function(
            pred=grad_stress_pred,
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
        for grad_name, grad_pred in [
            (self.grad_forces_name, raw_grad_forces_pred),
            (self.grad_stress_name, raw_grad_stress_pred),
        ]:
            direct_name = grad_name.replace(self.grad_prefix + "_", "")
            if direct_name in self.extra_properties:
                direct_head = self.heads[direct_name]
                direct_pred = out[direct_name]
                if self.distill_direct_heads:
                    loss_out = direct_head.loss(
                        direct_pred, batch, alternative_target=grad_pred.detach()
                    )
                else:
                    loss_out = direct_head.loss(direct_pred, batch)
                loss = self.loss_weights.get(direct_name, 1.0) * loss_out.loss  # type: ignore
                loss_out.log[f"{direct_name}_loss"] = loss
                total_loss += loss
                metrics.update(loss_out.log)

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
            raw_forces_target = batch.node_targets[self.forces_name]  # type: ignore
            forces_error = torch.abs(raw_grad_forces_pred - raw_forces_target).mean(
                dim=-1
            )
            confidence_logits = out["confidence"]
            loss_out = confidence_head.loss(confidence_logits, forces_error, batch)
            loss = self.loss_weights.get("confidence", 1.0) * loss_out.loss
            loss_out.log["confidence_loss"] = loss
            total_loss += loss
            metrics.update(loss_out.log)

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
        load_forcefield_state_dict(
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
