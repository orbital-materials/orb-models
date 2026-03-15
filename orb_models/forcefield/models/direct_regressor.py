from collections.abc import Mapping
from typing import Any, cast

import torch

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.models import base
from orb_models.common.models.gns import MoleculeGNS
from orb_models.common.models.graph_regressor import _validate_heads_and_loss_weights
from orb_models.common.models.load import load_regressor_state_dict
from orb_models.common.models.segment_ops import split_prediction
from orb_models.forcefield.models.forcefield_heads import ConfidenceHead, ForcefieldHead
from orb_models.forcefield.models.pair_repulsion import ZBLBasis


class DirectForcefieldRegressor(base.RegressorModelMixin[AtomGraphs]):
    """Direct Forcefield regressor."""

    _deprecated_kwargs: list[str] = []

    def __init__(
        self,
        heads: Mapping[str, ForcefieldHead | ConfidenceHead],
        model: MoleculeGNS,
        loss_weights: dict[str, float] | None = None,
        pair_repulsion: bool = False,
        model_requires_grad: bool = True,
        heads_require_grad: dict[str, bool] | None = None,
        cutoff_layers: int | None = None,
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
        _validate_heads_and_loss_weights(heads, loss_weights)

        self.heads = torch.nn.ModuleDict(heads)
        self._stress_disabled = False
        self.loss_weights = loss_weights
        self.model_requires_grad = model_requires_grad
        self.cutoff_layers = cutoff_layers

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

    @property
    def has_stress(self) -> bool:
        """Check if the model has stress prediction and it is enabled."""
        return "stress" in self.heads and not self._stress_disabled

    def enable_stress(self) -> None:
        """Enable stress computation."""
        if "stress" not in self.heads:
            raise ValueError("Cannot enable stress: no stress head exists.")
        self._stress_disabled = False

    def disable_stress(self) -> None:
        """Disable stress computation."""
        self._stress_disabled = True

    def forward(self, batch: AtomGraphs) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Forward pass of DirectForcefieldRegressor."""
        out = self.model(batch)
        node_features = out["node_features"]
        for name, head in self.heads.items():
            if self._stress_disabled and "stress" in name:
                continue
            res = head(node_features, batch)
            out[name] = res

        if self.pair_repulsion:
            out_pair_repulsion = self.pair_repulsion_fn(batch)
            for name, head in self.heads.items():
                raw_repulsion = self._get_raw_repulsion(name, out_pair_repulsion)
                if raw_repulsion is not None:
                    head = cast(ForcefieldHead, head)
                    raw = head.denormalize(out[name], batch)
                    out[name] = head.normalize(raw + raw_repulsion, batch, online=False)
        return out

    def predict(self, batch: AtomGraphs, split: bool = False) -> dict[str, torch.Tensor]:
        """Predict node and/or graph level attributes."""
        out = self.model(batch)
        node_features = out["node_features"]
        output = {}
        for name, head in self.heads.items():
            if self._stress_disabled and "stress" in name:
                continue
            output[name] = cast(ForcefieldHead | ConfidenceHead, head).predict(node_features, batch)

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

    def loss(self, batch: AtomGraphs) -> base.ModelOutput:
        """Loss function of DirectForcefieldRegressor."""
        assert isinstance(batch, AtomGraphs), f"Expected AtomGraphs, got {type(batch)}"
        out = self(batch)
        total_loss = torch.tensor(
            0.0,
            device=batch.positions.device,
            dtype=batch.positions.dtype,
        )
        metrics: dict = {}

        for name, head in self.heads.items():
            if name == "confidence":
                continue
            if self._stress_disabled and "stress" in name:
                continue
            head = cast(ForcefieldHead, head)
            head_out = head.loss(out[name], batch)
            weight = self.loss_weights[name]
            loss = weight * head_out.loss
            total_loss += loss
            metrics.update(head_out.log)
            metrics[f"{name}_loss"] = loss

        # Do confidence separately, because it requires
        # the force predictions and errors.
        if "confidence" in self.heads:
            forces_head = self.heads["forces"]
            forces_head = cast(ForcefieldHead, forces_head)
            confidence_head = self.heads["confidence"]
            confidence_head = cast(ConfidenceHead, confidence_head)

            # Per-atom force MAE
            forces_pred = out["forces"]
            raw_forces_pred = forces_head.denormalize(forces_pred, batch)
            raw_forces_target = batch.node_targets[forces_head.target.fullname]
            forces_error = torch.abs(raw_forces_pred - raw_forces_target).mean(dim=-1)

            # Confidence loss
            confidence_logits = out["confidence"]
            head_out = confidence_head.loss(confidence_logits, forces_error, batch)
            loss = self.loss_weights["confidence"] * head_out.loss
            total_loss += loss
            metrics.update(head_out.log)
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
        """Load state dict for DirectForcefieldRegressor."""
        load_regressor_state_dict(
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

    def _get_raw_repulsion(self, name: str, out_pair_repulsion: dict[str, torch.Tensor]):
        """Extract raw repulsion value based on the property name."""
        property_types = ["energy", "forces", "stress"]
        for prop_type in property_types:
            if prop_type in name and "d3" not in name and "d4" not in name:
                return out_pair_repulsion[prop_type]
        return None
