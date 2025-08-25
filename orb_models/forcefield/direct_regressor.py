import torch
from typing import Any, Mapping, Optional, Dict, Union

from orb_models.forcefield.pair_repulsion import ZBLBasis
from orb_models.forcefield import base
from orb_models.forcefield.forcefield_utils import (
    split_prediction,
    validate_regressor_inputs,
)
from orb_models.forcefield.gns import MoleculeGNS
from orb_models.forcefield.load import load_forcefield_state_dict
from orb_models.forcefield.atomic_system import SystemConfig


class DirectForcefieldRegressor(torch.nn.Module):
    """Direct Forcefield regressor."""

    _deprecated_kwargs = ["cutoff_layers"]

    def __init__(
        self,
        heads: Mapping[str, torch.nn.Module],
        model: MoleculeGNS,
        loss_weights: Optional[Dict[str, float]] = None,
        pair_repulsion: bool = False,
        model_requires_grad: bool = True,
        heads_require_grad: Optional[Dict[str, bool]] = None,
        system_config: Optional[SystemConfig] = None,
        **kwargs,
    ) -> None:
        """Initializes the DirectForcefieldRegressor.

        Args:
            heads: The regression heads used to predict node/graph properties.
                Null heads are allowed and will be discarded.
            model: A pretrained model to use for transfer learning/finetuning.
            loss_weights: The weight of the energy loss in the total loss.
                Null weights are allowed and will be discarded.
            pair_repulsion: Whether to use ZBL pair repulsion.
            model_requires_grad: Whether the underlying model should be finetuned or not.
            heads_require_grad: Optional dictionary mapping head names to booleans indicating
                whether the parameters of that head should require gradients.
            system_config: The inferencesystem configuration to use for the model.
        """
        super().__init__()
        for kwarg in kwargs:
            if kwarg not in self._deprecated_kwargs:
                raise ValueError(
                    f"Unknown kwargs: {kwarg}, expected only backward compatible kwargs "
                    f"from {self._deprecated_kwargs}"
                )

        loss_weights = loss_weights or {}
        loss_weights = {k: v for k, v in loss_weights.items() if v is not None}
        if isinstance(heads, Mapping):
            heads = {k: v for k, v in heads.items() if v is not None}
        validate_regressor_inputs(heads, loss_weights)

        self.heads = torch.nn.ModuleDict(heads)
        self.loss_weights = loss_weights
        self.model_requires_grad = model_requires_grad

        self.pair_repulsion = pair_repulsion
        if self.pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(
                p=6,
                node_aggregation="sum",
                compute_gradients=True,
            )

        self.model = model
        if not model_requires_grad:
            for param in self.model.parameters():
                param.requires_grad = False

        if heads_require_grad is not None:
            for head_name, requires_grad in heads_require_grad.items():
                assert head_name in self.heads
                for param in self.heads[head_name].parameters():
                    param.requires_grad = requires_grad

        self._system_config = system_config

    @property
    def system_config(self) -> SystemConfig:
        """Get the system config."""
        return self._system_config
    
    @property
    def has_stress(self) -> bool:
        """Check if the model has stress prediction."""
        return "stress" in self.heads

    def forward(
        self, batch: base.AtomGraphs
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass of DirectForcefieldRegressor."""
        out = self.model(batch)
        node_features = out["node_features"]
        for name, head in self.heads.items():
            res = head(node_features, batch)
            out[name] = res

        if self.pair_repulsion:
            out_pair_repulsion = self.pair_repulsion_fn(batch)
            for name, head in self.heads.items():
                raw_repulsion = self._get_raw_repulsion(name, out_pair_repulsion)
                if raw_repulsion is not None:
                    raw = head.denormalize(out[name], batch)
                    out[name] = head.normalize(raw + raw_repulsion, batch, online=False)
        return out

    def predict(
        self, batch: base.AtomGraphs, split: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Predict node and/or graph level attributes."""
        out = self.model(batch)
        node_features = out["node_features"]
        output = {}
        for name, head in self.heads.items():
            output[name] = head.predict(node_features, batch)

        if self.pair_repulsion:
            out_pair_repulsion = self.pair_repulsion_fn(batch)
            for name, head in self.heads.items():
                raw_repulsion = self._get_raw_repulsion(name, out_pair_repulsion)
                if raw_repulsion is not None:
                    output[name] = output[name] + raw_repulsion

        if split:
            for name, pred in output.items():
                output[name] = split_prediction(pred, batch.n_node)

        return output

    def loss(self, batch: base.AtomGraphs) -> base.ModelOutput:
        """Loss function of DirectForcefieldRegressor."""
        out = self(batch)
        total_loss = torch.tensor(
            0.0,
            device=batch.positions.device,
            dtype=batch.positions.dtype,
        )
        metrics: Dict = {}

        for name, head in self.heads.items():
            if name == "confidence":
                continue
            head_out = head.loss(out[name], batch)
            weight = self.loss_weights.get(name, 1.0)
            loss = weight * head_out.loss
            head_out.log[f"{name}_loss"] = loss
            total_loss += loss
            metrics.update(head_out.log)

        # Do confidence separately, because it requires
        # the force predictions and errors.
        if "confidence" in self.heads:
            forces_head = self.heads["forces"]
            forces_name = forces_head.target.fullname
            confidence_head = self.heads["confidence"]

            forces_pred = out[forces_name]
            raw_forces_pred = forces_head.normalizer.inverse(forces_pred)
            raw_forces_target = batch.node_targets[forces_name]
            forces_error = torch.abs(raw_forces_pred - raw_forces_target).mean(dim=-1)
            confidence_logits = out["confidence"]
            head_out = confidence_head.loss(confidence_logits, forces_error, batch)
            loss = self.loss_weights.get("confidence", 1.0) * head_out.loss
            head_out.log["confidence_loss"] = loss
            total_loss += loss
            metrics.update(head_out.log)

        metrics["loss"] = total_loss
        return base.ModelOutput(loss=total_loss, log=metrics)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
        skip_artifact_reference_energy: bool = False,
    ):
        """Load state dict for DirectForcefieldRegressor."""
        load_forcefield_state_dict(
            self,
            state_dict,
            strict=strict,
            assign=assign,
            skip_artifact_reference_energy=skip_artifact_reference_energy,
        )

    @property
    def properties(self):
        """List of names of predicted properties."""
        heads = list(self.heads.keys())
        if "energy" in heads:
            heads.append("free_energy")
        return heads

    def compile(self, *args, **kwargs):
        """Override the default Module.compile method to compile only the GNS model."""
        self.model.compile(*args, **kwargs)

    def is_compiled(self):
        """Check if the model is compiled."""
        return self._compiled_call_impl or self.model._compiled_call_impl

    def _get_raw_repulsion(
        self, name: str, out_pair_repulsion: Dict[str, torch.Tensor]
    ):
        """Extract raw repulsion value based on the property name."""
        property_types = ["energy", "forces", "stress"]
        for prop_type in property_types:
            if prop_type in name and "d3" not in name and "d4" not in name:
                return out_pair_repulsion[prop_type]
        return None
