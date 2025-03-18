"""A GraphRegressor model that combines a pretrained base model with a set of regression heads.

This module also provides the NodeHead and GraphHead classes, which are generic regression heads.
For regression tasks that require custom prediction heads, we define these in their own modules.
For instance, our Energy and Forces prediction heads are defined in the forcefield module.
"""

from typing import Any, List, Literal, Mapping, Optional, Dict, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from orb_models.forcefield import base
from orb_models.forcefield.property_definitions import PROPERTIES, PropertyDefinition
from orb_models.forcefield.nn_util import ScalarNormalizer, build_mlp
from orb_models.forcefield import segment_ops
from orb_models.forcefield.gns import MoleculeGNS
from orb_models.forcefield.load import _load_forcefield_state_dict
from orb_models.forcefield.pair_repulsion import ZBLBasis


class GraphRegressor(nn.Module):
    """Graph Regressor for finetuning.

    The GraphRegressor combines a pretrained base model with a set of regression heads.
    The regression heads are typically MLP transformations, along with a sum/avg pooling
    operation in the case of graph-level targets. The base model can be jointly fine-tuned
    along with the heads, or kept frozen.
    """

    def __init__(
        self,
        heads: Union[Sequence[torch.nn.Module], Mapping[str, torch.nn.Module]],
        model: MoleculeGNS,
        model_requires_grad: bool = True,
        cutoff_layers: Optional[int] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        pair_repulsion: bool = False,
    ) -> None:
        """Initializes the GraphRegressor.

        Args:
            heads: The regression heads used to predict node/graph properties.
                Null heads are allowed and will be discarded.
            model: A pretrained model to use for transfer learning/finetuning.
            model_requires_grad: Whether the underlying model should be finetuned or not.
            cutoff_layers: The number of message passing layers to keep. If None, all layers are kept.
            loss_weights: The weight of the energy loss in the total loss.
                Null weights are allowed and will be discarded.
        """
        super().__init__()
        loss_weights = loss_weights or {}
        loss_weights = {k: v for k, v in loss_weights.items() if v is not None}
        if isinstance(heads, Mapping):
            heads = {k: v for k, v in heads.items() if v is not None}
        _validate_regressor_inputs(heads, loss_weights)

        if isinstance(heads, Sequence):
            # backwards-compatible list format; we now use dicts which are more overrideable in hydra
            self.heads = torch.nn.ModuleDict(
                {head.target.fullname: head for head in heads}  # type: ignore
            )
        else:
            self.heads = torch.nn.ModuleDict(heads)
        self.cutoff_layers = cutoff_layers
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
        if self.cutoff_layers is not None:
            gns = (
                self.model if isinstance(self.model, MoleculeGNS) else self.model.model
            )
            _set_cutoff_layers(gns, self.cutoff_layers)  # type: ignore

        if not model_requires_grad:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(
        self, batch: base.AtomGraphs
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass of GraphRegressor."""
        out = self.model(batch)
        node_features = out["node_features"]
        for name, head in self.heads.items():
            res = head(node_features, batch)
            out[name] = res

        if self.pair_repulsion:
            out_pair_raw = self.pair_repulsion_fn(batch)
            for name, head in self.heads.items():
                raw = out_pair_raw.get(name, None)
                if raw is None:
                    continue
                if name == "energy" and head.atom_avg:
                    raw = (raw / batch.n_node).unsqueeze(1)
                out[name] = out[name] + head.normalizer(
                    raw,
                    online=False,
                )
        return out

    def predict(
        self, batch: base.AtomGraphs, split: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Predict node and/or graph level attributes."""
        out = self.model(batch)
        node_features = out["node_features"]
        output = {}
        for name, head in self.heads.items():
            pred = head.predict(node_features, batch)
            if split:
                output[name] = _split_prediction(pred, batch.n_node)
            else:
                output[name] = pred

        if self.pair_repulsion:
            out_pair_raw = self.pair_repulsion_fn(batch)
            for name, head in self.heads.items():
                raw = out_pair_raw.get(name, None)
                if raw is None:
                    continue
                output[name] = output[name] + raw

        return output

    def loss(self, batch: base.AtomGraphs) -> base.ModelOutput:
        """Loss function of GraphRegressor."""
        out = self(batch)
        loss = torch.tensor(0.0, device=batch.positions.device)
        metrics: Dict = {}

        for name, head in self.heads.items():
            if name == "confidence":
                continue
            head_out = head.loss(out[name], batch)
            metrics.update(head_out.log)
            weight = self.loss_weights.get(name, 1.0)
            loss += weight * head_out.loss

        # Do confidence separately, because it requires
        # the force predictions and errors.
        if "confidence" in self.heads:
            forces_head = self.heads["forces"]
            forces_name = forces_head.target.fullname
            confidence_head = self.heads["confidence"]

            forces_pred = out[forces_name]
            raw_forces_pred = forces_head.normalizer.inverse(forces_pred)
            raw_forces_target = batch.node_targets[forces_name]  # type: ignore
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


class NodeHead(torch.nn.Module):
    """Node prediction head that can be appended to a base model.

    This head could be added to the foundation model to enable
    auxiliary tasks during pretraining, or added afterwards
    during a finetuning step.
    """

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        target: Union[str, PropertyDefinition],
        loss_type: Literal["mae", "mse", "huber_0.01"] = "huber_0.01",
        dropout: Optional[float] = None,
        checkpoint: Optional[str] = None,
        online_normalisation: bool = True,
        activation: str = "ssp",
    ):
        """Initializes the NodeHead MLP.

        Args:
            latent_dim (int): Dimensionality of the incoming latent vector from the base model.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden size.
            target: either the name of a PropertyDefinition or a PropertyDefinition itself.
            loss_type: The type of loss to use. Either "mae", "mse", or "huber_x"
                where x is the delta parameter for the huber loss.
            dropout: The level of dropout to apply.
            checkpoint: Whether to use PyTorch checkpointing.
                None (no checkpointing), 'reentrant' or 'non-reentrant'.
            online_normalisation: Whether to normalise the target online.
            activation: The activation function to use.
        """
        super().__init__()
        self.target = PROPERTIES[target] if isinstance(target, str) else target

        if self.target.domain != "real":
            raise NotImplementedError("Currently only supports real targets.")

        self.normalizer = ScalarNormalizer(
            init_mean=self.target.means,
            init_std=self.target.stds,
            online=online_normalisation,
        )

        self.mlp = build_mlp(
            input_size=latent_dim,
            hidden_layer_sizes=[mlp_hidden_dim] * num_mlp_layers,
            output_size=self.target.dim,
            activation=activation,
            dropout=dropout,
            checkpoint=checkpoint,
        )
        self.loss_type = loss_type

    def forward(
        self, node_features: torch.Tensor, batch: base.AtomGraphs
    ) -> torch.Tensor:
        """Forward pass (without inverse transformations)."""
        pred = self.mlp(node_features)
        return pred

    def predict(
        self, node_features: torch.Tensor, batch: base.AtomGraphs
    ) -> torch.Tensor:
        """Predict node-level attribute."""
        out = self(node_features, batch)
        pred = out
        return self.normalizer.inverse(pred)

    def loss(
        self,
        pred: torch.Tensor,
        batch: base.AtomGraphs,
        alternative_target: Optional[torch.Tensor] = None,
    ):
        """Apply mlp to compute loss and metrics."""
        name = self.target.fullname
        if alternative_target is not None:
            raw_target = alternative_target
        else:
            raw_target = batch.node_targets[name].squeeze(-1)  # type: ignore
        raw_target = raw_target.squeeze(-1)

        target = self.normalizer(raw_target)
        pred = pred.squeeze(-1)
        assert pred.shape == target.shape, f"{pred.shape} != {target.shape}"

        loss = mean_error(pred, target, self.loss_type, batch.n_node)

        raw_pred = self.normalizer.inverse(pred)
        metrics = {}
        metrics[f"{name}_loss"] = loss
        metrics[f"{name}_mae_raw"] = torch.abs(raw_pred - raw_target).mean()
        metrics[f"{name}_mse_raw"] = ((raw_pred - raw_target) ** 2).mean()
        return base.ModelOutput(loss=loss, log=metrics)


class GraphHead(torch.nn.Module):
    """MLP Regression head that can be appended to a base model.

    This head could be added to the foundation model to enable
    auxiliary tasks during pretraining, or added afterwards
    during a finetuning step.
    """

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        target: Union[str, PropertyDefinition],
        node_aggregation: Literal["sum", "mean"] = "mean",
        real_loss_type: Literal["mae", "mse", "huber_0.01"] = "huber_0.01",
        dropout: Optional[float] = None,
        checkpoint: Optional[str] = None,
        online_normalisation: bool = True,
        activation: str = "ssp",
    ):
        """Initializes the GraphHead MLP.

        Args:
            latent_dim (int): Dimensionality of the incoming latent vector from the base model.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden size.
            target: either the name of a PropertyDefinition or a PropertyDefinition itself
            node_aggregation: The method for aggregating the node features
                from the pretrained model representations.
            loss_type: The type of loss to use. Either "mae", "mse", or "huber_x"
                where x is the delta parameter for the huber loss.
            dropout: The level of dropout to apply.
            checkpoint: Whether to use PyTorch checkpointing.
                    None (no checkpointing), 'reentrant' or 'non-reentrant'.
            online_normalisation: Whether to normalise the target online.
            activation: The activation function to use.
        """
        super().__init__()
        self.target = PROPERTIES[target] if isinstance(target, str) else target

        self.normalizer = ScalarNormalizer(
            init_mean=self.target.means,
            init_std=self.target.stds,
            online=online_normalisation,
        )
        self.node_aggregation = node_aggregation
        self.mlp = build_mlp(
            input_size=latent_dim,
            hidden_layer_sizes=[mlp_hidden_dim] * num_mlp_layers,
            output_size=self.target.dim,
            activation=activation,
            dropout=dropout,
            checkpoint=checkpoint,
        )
        activation_dict = {
            "real": torch.nn.Identity,
            "binary": torch.nn.Sigmoid,
            "categorical": torch.nn.Softmax,
        }
        self.output_activation = activation_dict[self.target.domain]()
        self.real_loss_type = real_loss_type

    def forward(
        self, node_features: torch.Tensor, batch: base.AtomGraphs
    ) -> torch.Tensor:
        """Predictions with raw logits (no sigmoid/softmax or any inverse transformations)."""
        input = segment_ops.aggregate_nodes(
            node_features,
            batch.n_node,
            reduction=self.node_aggregation,
        )
        pred = self.mlp(input)
        return pred

    def predict(
        self, node_features: torch.Tensor, batch: base.AtomGraphs
    ) -> torch.Tensor:
        """Predict graph-level attribute."""
        pred = self(node_features, batch)
        logits = pred.squeeze(-1)
        probs = self.output_activation(logits)
        if self.target.domain == "real":
            probs = self.normalizer.inverse(probs)
        return probs

    def loss(
        self,
        pred: torch.Tensor,
        batch: base.AtomGraphs,
        alternative_target: Optional[torch.Tensor] = None,
    ):
        """Apply mlp to compute loss and metrics.

        Depending on whether the target is real/binary/categorical, we
        use an MSE/cross-entropy loss. In the case of cross-entropy, the
        preds are logits (not normalised) to take advantage of numerically
        stable log-softmax.
        """
        name = self.target.fullname
        if alternative_target is not None:
            target = alternative_target
        else:
            target = batch.system_targets[name].squeeze(-1)  # type: ignore
        pred = pred.squeeze(-1)

        if self.target.domain == "categorical":
            expected_shape = target.shape + (self.target.dim,)
            assert pred.shape == expected_shape, f"{pred.shape} != {expected_shape}"
            loss, metrics = cross_entropy_loss(pred, target, name)
        elif self.target.domain == "binary":
            assert pred.shape == target.shape, f"{pred.shape} != {target.shape}"
            loss, metrics = bce_loss(pred, target, name)
        else:
            assert pred.shape == target.shape, f"{pred.shape} != {target.shape}"
            normalized_target = self.normalizer(target)
            loss = mean_error(pred, normalized_target, self.real_loss_type)
            raw_pred = self.normalizer.inverse(pred)
            metrics = {
                f"{name}_loss": loss,
                f"{name}_mae_raw": torch.abs(raw_pred - target).mean(),
                f"{name}_mse_raw": ((raw_pred - target) ** 2).mean(),
            }

        return base.ModelOutput(loss=loss, log=metrics)


def _validate_regressor_inputs(
    heads: Union[Sequence[torch.nn.Module], Mapping[str, torch.nn.Module]],
    loss_weights: Dict[str, float],
    ensure_grad_loss_weights: bool = False,
):
    """Validate the input heads and loss weights."""
    if isinstance(heads, Sequence):
        head_names = [head.target.fullname for head in heads]  # type: ignore
    else:
        head_names = list(heads.keys())
        targets = [getattr(head, "target", None) for head in heads.values()]
        if all(target is not None for target in targets):
            target_names = [target.fullname for target in targets]  # type: ignore
            if head_names != target_names:
                raise ValueError(
                    f"Head names and target names must match; got {head_names} and {target_names}"
                )
    if len(head_names) == 0:
        raise ValueError("Model must have at least one output head.")
    if len(head_names) != len(set(head_names)):
        raise ValueError(f"Head names must be unique; got {head_names}")
    unknown_keys = set(loss_weights.keys()) - set(head_names)  # type: ignore
    if ensure_grad_loss_weights:
        if (
            "grad_forces" not in loss_weights.keys()
            or "grad_stress" not in loss_weights.keys()
        ):
            raise ValueError("grad_forces and grad_stress must be in loss_weights .")
        unknown_keys = unknown_keys - set(["grad_forces", "grad_stress"])
    unknown_keys = unknown_keys - set(["rotational_grad"])  # not associated with a head
    if unknown_keys:
        raise ValueError(f"Loss weights for unknown targets: {unknown_keys}")


def _set_cutoff_layers(gns: MoleculeGNS, cutoff_layers: int):
    """Set the number of message passing layers to keep."""
    if cutoff_layers > gns.num_message_passing_steps:
        raise ValueError(
            f"cutoff_layers ({cutoff_layers}) must be less than or equal to"
            f" the number of message passing steps ({gns.num_message_passing_steps})"
        )
    else:
        gns.gnn_stacks = gns.gnn_stacks[:cutoff_layers]
        gns.num_message_passing_steps = cutoff_layers  # type: ignore


def _split_prediction(pred: torch.Tensor, n_node: torch.Tensor):
    """Split batched prediction back into per-system predictions."""
    if len(pred) == len(n_node):
        return torch.split(pred, 1, dim=0)
    elif len(pred) == n_node.sum():
        return torch.split(pred, n_node.cpu().tolist(), dim=0)
    else:
        raise ValueError(f"Unexpected length of prediction tensor: {len(pred)}")


def mean_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    error_type: Literal["mae", "mse", "huber_0.01"],
    batch_n_node: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute MAE or MSE for node or graph targets.

    Args:
        target: The target tensor.
        pred: The prediction tensor.
        batch_n_node: The number of nodes per graph. If provided, then a
            nested aggregation is performed for node errors i.e. first we
            average across nodes within each graph, then average across graphs.
        error_type: The type of error to compute. Either "mae" or "mse" or "huber_x"
            where x is the delta parameter for the huber loss.

    Returns:
        A scalar error for the whole batch.
    """
    if error_type.startswith("huber"):
        huber_delta = float(error_type.split("_")[1])
        error_type = "huber"  # type: ignore
        assert huber_delta > 0.0, "HUBER_DELTA must be greater than 0.0"

    error_function = {
        "mae": lambda x, y: torch.abs(x - y),
        "mse": lambda x, y: (x - y) ** 2,
        "huber": lambda x, y: F.huber_loss(x, y, reduction="none", delta=huber_delta),
    }[error_type]

    errors = error_function(pred, target)
    errors = errors.mean(dim=-1) if errors.dim() > 1 else errors

    if batch_n_node is not None:
        error = segment_ops.aggregate_nodes(
            errors, batch_n_node, reduction="mean"
        ).mean()
    else:
        error = errors.mean()

    return error


def bucketed_mean_error(
    target: torch.Tensor,
    pred: torch.Tensor,
    bucket_by: Literal["target", "error"],
    thresholds: List[float],
    batch_n_node: Optional[torch.Tensor] = None,
    error_type: Literal["mae", "mse"] = "mae",
) -> Dict[str, torch.Tensor]:
    """Compute MAE or MSE per-bucket, where each bucket is a range defined by thresholds.

    The target can be a node-level or graph target. For node-level
    targets, providing batch_n_node entails a nested-aggregation
    of the error (first by node, then by graph).

    Errors can be bucketed by their value, or by the value of the ground-truth target.
    If bucketing by target, and the target is multi-dimensional, then the L2 norm
    of the target is used to define the buckets.

    Buckets are defined by a set of real-valued thresholds. For example,
    bucket_by='error' and thresholds=[0.1, 10.0] creates 3 buckets:
        - errors < 0.1
        - 0.1 <= errors < 10.0
        - errors >= 10.0

    Args:
        target: The target tensor.
        pred: The prediction tensor.
        bucket_by: The method for assigning buckets.
        thresholds: The bucket edges. -inf and +inf are automatically added.
        batch_n_node: The number of nodes per graph. If None, no nested aggregation is performed.
        error_type: The type of error to compute. Either "mae" or "mse".

    Returns:
        A dictionary of metrics with entries of the form f"{error_type}_{bucket_name}", where
        bucket name is a string representing the bucket edge values.
    """
    error_function = {
        "mae": lambda x, y: torch.abs(x - y),
        "mse": lambda x, y: (x - y) ** 2,
    }[error_type]

    errors = error_function(target, pred)

    # If multi-dimensional, collapse down to a single error per graph/node
    errors = errors.mean(dim=-1) if errors.dim() > 1 else errors

    # Decide what to bucket by
    if bucket_by == "target":
        values_to_bucket_by = target.norm(dim=-1) if target.dim() > 1 else target
    elif bucket_by == "error":
        values_to_bucket_by = errors
    else:
        raise ValueError(f"Unknown bucket_by: {bucket_by}")

    # Assign each element in the batch to a bucket index
    bucket_edges = torch.tensor(
        [-float("inf")] + thresholds + [float("inf")], device=errors.device
    )
    bucket_indices = torch.bucketize(values_to_bucket_by, bucket_edges, right=True) - 1

    # Iterate over each bucket and compute its average error
    metrics = {}
    bucket_names = [
        f"bucket_{p:.2f}-{q:.2f}" for p, q in zip(bucket_edges[:-1], bucket_edges[1:])
    ]
    for i, name in enumerate(bucket_names):
        mask = bucket_indices == i
        current_errors = errors[mask]

        if batch_n_node is not None:
            current_batch_n_node = segment_ops.aggregate_nodes(
                mask.int(), batch_n_node, reduction="sum"
            )
            error = segment_ops.aggregate_nodes(
                current_errors, current_batch_n_node, reduction="mean"
            ).mean()
        else:
            error = current_errors.mean()

        metrics[name] = error

    return metrics


def binary_accuracy(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> float:
    """Calculate binary accuracy between 2 tensors.

    Args:
        pred: the prediction tensor.
        target: the tensor of target values.
        threshold: Binary classification threshold. Default 0.5.

    Returns:
        mean accuracy.
    """
    return ((pred > threshold) == target).to(pred.dtype).mean().item()


def categorical_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate accuracy for K class classification.

    Args:
        pred: the tensor of logits for K classes of shape (..., K)
        target: tensor of integer target values of shape (...)

    Returns:
        mean accuracy.
    """
    pred_labels = torch.argmax(pred, dim=-1)
    return (pred_labels == target.long()).to(pred.dtype).mean().item()


def bce_loss(
    pred: torch.Tensor, target: torch.Tensor, metric_prefix: str = ""
) -> Tuple:
    """Binary cross-entropy loss with accuracy metric."""
    loss = torch.nn.BCEWithLogitsLoss()(pred, target.to(pred.dtype))
    accuracy = binary_accuracy(pred, target)
    return (
        loss,
        {
            f"{metric_prefix}_accuracy": accuracy,
            f"{metric_prefix}_loss": loss.item(),
        },
    )


def cross_entropy_loss(
    pred: torch.Tensor, target: torch.Tensor, metric_prefix: str = ""
) -> Tuple:
    """Cross-entropy loss with accuracy metric."""
    loss = torch.nn.CrossEntropyLoss()(pred, target.long())
    accuracy = categorical_accuracy(pred, target)
    return (
        loss,
        {
            f"{metric_prefix}_accuracy": accuracy,
            f"{metric_prefix}_loss": loss.item(),
        },
    )
