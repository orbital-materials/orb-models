from typing import Any, Mapping, Optional, Dict

import torch
from torch import nn

from orb_models.forcefield import base
from orb_models.forcefield.gns import MoleculeGNS
from orb_models.forcefield.graph_regressor import (
    _split_prediction,
    _validate_regressor_inputs,
)
from orb_models.forcefield.forcefield_utils import compute_gradient_forces_and_stress
from orb_models.forcefield.load import _load_forcefield_state_dict


class ConservativeForcefieldRegressor(nn.Module):
    """A specialized regressor that handles both direct and conservative predictions.

    This class is used to train a model that produces both the direct and conservative
    predictions of energy, forces, and stress. The conservative force/stress predictions
    are computed using the gradient of the direct predictions.

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
        **kwargs,
    ):
        super().__init__()
        for kwarg in kwargs:
            if kwarg not in self._deprecated_kwargs:
                raise ValueError(
                    f"Unknown kwargs: {kwarg}, expected only backward compatible kwargs "
                    f"from {self._deprecated_kwargs}"
                )

        # Validate required heads are present
        required_heads = {"energy", "forces", "stress"}
        if not required_heads.issubset(heads.keys()):
            missing = required_heads - set(heads.keys())
            raise ValueError(f"Missing required heads: {missing}")

        loss_weights = loss_weights or {}
        loss_weights = {k: v for k, v in loss_weights.items() if v is not None}
        _validate_regressor_inputs(
            heads, loss_weights, ensure_grad_loss_weights=ensure_grad_loss_weights
        )

        self.heads = torch.nn.ModuleDict(heads)
        self.model = model

        self.loss_weights = loss_weights
        self.distill_direct_heads = distill_direct_heads

        # Names for predictions
        self.energy_name = heads["energy"].target.fullname  # type: ignore
        self.forces_name = heads["forces"].target.fullname  # type: ignore
        self.stress_name = heads["stress"].target.fullname  # type: ignore
        self.grad_prefix = "grad"
        self.grad_forces_name = f"{self.grad_prefix}_{self.forces_name}"
        self.grad_stress_name = f"{self.grad_prefix}_{self.stress_name}"
        self.grad_rotation_name = "rotational_grad"

    @property
    def properties(self):
        """List of names of predicted properties."""
        props = [
            self.energy_name,
            "free_energy",
            self.forces_name,
            self.stress_name,
            self.grad_forces_name,
            self.grad_stress_name,
            self.grad_rotation_name,
        ]
        if "confidence" in self.heads:
            props.append("confidence")
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

        out[self.energy_name] = self.heads[self.energy_name](node_features, batch)
        out[self.forces_name] = self.heads[self.forces_name](node_features, batch)
        out[self.stress_name] = self.heads[self.stress_name](node_features, batch)

        if "confidence" in self.heads:
            target_name = self.heads["confidence"].target.fullname
            out[target_name] = self.heads["confidence"](node_features, batch)

        # Compute conservative predictions if needed
        raw_energy = self.heads[self.energy_name].denormalise_prediction(
            pred=out[self.energy_name], batch=batch
        )
        force, stress, rotational_grad = compute_gradient_forces_and_stress(
            energy=raw_energy,
            positions=batch.node_features["positions"],
            displacement=batch.system_features["stress_displacement"],
            cell=batch.system_features["cell"],
            training=self.training,
            compute_stress=True,
            generator=batch.system_features["generator"],
        )

        # Autograd forces/stress are automatically 'raw' (i.e. in ev/A and ev/A^3)
        # and thus need normalising to have the same scale as the direct predictions
        # Normalize using the same normalizers as direct predictions
        out[self.grad_forces_name] = self.heads[self.forces_name].normalizer(
            force, online=False
        )
        out[self.grad_stress_name] = self.heads[self.stress_name].normalizer(
            stress, online=False
        )
        out[self.grad_rotation_name] = rotational_grad

        return out

    def predict(
        self, batch: base.AtomGraphs, split: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Predict energy, forces, and stress."""
        preds = self(batch)

        out = {}
        out[self.energy_name] = self.heads[self.energy_name].denormalise_prediction(
            pred=preds[self.energy_name], batch=batch
        )
        out[self.forces_name] = self.heads[self.forces_name].normalizer.inverse(
            preds[self.forces_name]
        )

        out[self.stress_name] = self.heads[self.stress_name].normalizer.inverse(
            preds[self.stress_name]
        )

        out[self.grad_forces_name] = self.heads[self.forces_name].normalizer.inverse(
            preds[self.grad_forces_name]
        )
        out[self.grad_stress_name] = self.heads[self.stress_name].normalizer.inverse(
            preds[self.grad_stress_name]
        )
        out[self.grad_rotation_name] = preds[self.grad_rotation_name]

        if "confidence" in self.heads:
            target_name = self.heads["confidence"].target.fullname
            out[target_name] = torch.softmax(preds[target_name], dim=-1)

        if split:
            for name, pred in out.items():
                out[name] = _split_prediction(pred, batch.n_node)

        return out  # type: ignore

    def loss(self, batch: base.AtomGraphs) -> base.ModelOutput:
        """Compute loss including both direct and conservative terms."""
        out = self(batch)

        # predictions
        energy_pred = out[self.energy_name]
        grad_forces_pred = out[self.grad_forces_name]
        grad_stress_pred = out[self.grad_stress_name]
        forces_pred = out[self.forces_name]
        stress_pred = out[self.stress_name]

        # heads
        energy_head = self.heads[self.energy_name]
        forces_head = self.heads[self.forces_name]
        stress_head = self.heads[self.stress_name]

        # metrics
        metrics = {}

        # Energy
        energy_loss = energy_head.loss(energy_pred, batch)
        loss = self.loss_weights.get(self.energy_name, 1.0) * energy_loss.loss  # type: ignore
        metrics.update(energy_loss.log)

        # Conservative forces
        grad_forces_loss = forces_head.loss(grad_forces_pred, batch)
        loss += self.loss_weights[self.grad_forces_name] * grad_forces_loss.loss
        metrics.update(
            {f"{self.grad_prefix}-{k}": v for k, v in grad_forces_loss.log.items()}
        )

        # Conservative stress
        grad_stress_loss = stress_head.loss(grad_stress_pred, batch)
        loss += self.loss_weights[self.grad_stress_name] * grad_stress_loss.loss
        metrics.update(
            {f"{self.grad_prefix}-{k}": v for k, v in grad_stress_loss.log.items()}
        )

        # Direct forces
        force_normalizer_is_online = forces_head.normalizer.online
        forces_head.normalizer.online = False
        if self.distill_direct_heads:
            raw_grad_forces = forces_head.normalizer.inverse(grad_forces_pred)
            forces_loss = forces_head.loss(
                forces_pred, batch, alternative_target=raw_grad_forces.detach()
            )
        else:
            forces_loss = forces_head.loss(forces_pred, batch)
        forces_head.normalizer.online = force_normalizer_is_online
        loss += self.loss_weights.get(self.forces_name, 1.0) * forces_loss.loss  # type: ignore
        metrics.update(forces_loss.log)

        # Direct stress
        stress_normalizer_is_online = stress_head.normalizer.online
        stress_head.normalizer.online = False
        if self.distill_direct_heads:
            raw_grad_stress = stress_head.normalizer.inverse(grad_stress_pred)
            stress_loss = stress_head.loss(
                stress_pred, batch, alternative_target=raw_grad_stress.detach()
            )
        else:
            stress_loss = stress_head.loss(stress_pred, batch)
        stress_head.normalizer.online = stress_normalizer_is_online
        loss += self.loss_weights.get(self.stress_name, 1.0) * stress_loss.loss  # type: ignore
        metrics.update(stress_loss.log)

        # Equigrad
        if self.grad_rotation_name in self.loss_weights:
            rotational_grad_rms = torch.linalg.norm(
                out[self.grad_rotation_name],
                dim=(1, 2),
            ).mean()
            loss += self.loss_weights[self.grad_rotation_name] * rotational_grad_rms
            metrics["rotational_grad_rms"] = rotational_grad_rms

        # Confidence
        if "confidence" in self.heads:
            confidence_head = self.heads["confidence"]

            raw_forces_pred = forces_head.normalizer.inverse(forces_pred)
            raw_forces_target = batch.node_targets[self.forces_name]  # type: ignore
            forces_error = torch.abs(raw_forces_pred - raw_forces_target).mean(dim=-1)
            confidence_logits = out["confidence"]
            head_out = confidence_head.loss(confidence_logits, forces_error, batch)
            metrics.update(head_out.log)

            loss += self.loss_weights.get("confidence", 1.0) * head_out.loss

        metrics["loss"] = loss
        return base.ModelOutput(loss=loss, log=metrics)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
        skip_artifact_reference_energy: bool = False,
    ):
        """Load state dict for GraphRegressor."""
        _load_forcefield_state_dict(
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
