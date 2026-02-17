"""A GraphRegressor model that combines a pretrained base model with a set of regression heads.

This module also provides the NodeHead and GraphHead classes, which are generic regression heads.
"""

import abc
from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch.nn import functional as F

from orb_models.common.atoms.batch.abstract_batch import AbstractAtomBatch
from orb_models.common.dataset.property_definitions import PROPERTIES, PropertyDefinition
from orb_models.common.models import base, segment_ops
from orb_models.common.models.nn_util import ScalarNormalizer, build_mlp


class RegressionHead(torch.nn.Module, abc.ABC):
    """Abstract base class for regression heads."""

    target: "PropertyDefinition"

    @abc.abstractmethod
    def forward(self, node_features: torch.Tensor, batch: AbstractAtomBatch) -> torch.Tensor: ...

    @abc.abstractmethod
    def predict(self, node_features: torch.Tensor, batch: AbstractAtomBatch) -> torch.Tensor: ...

    @abc.abstractmethod
    def loss(
        self,
        pred: torch.Tensor,
        batch: AbstractAtomBatch,
        alternative_target: torch.Tensor | None = None,
    ) -> base.ModelOutput: ...


class GraphHead(RegressionHead):
    """MLP Regression head that can be appended to a base model.

    This head could be added to the foundation model to enable
    auxiliary tasks during pretraining, or added afterwards
    during a finetuning step.

    NOTE: This head is deprecated. Use EnergyHead, ForceHead, or StressHead instead.
    We only support this head for backwards compatibility with orb-v2 models.
    """

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        target: str | PropertyDefinition,
        node_aggregation: Literal["sum", "mean"] = "mean",
        real_loss_type: Literal["mae", "mse", "huber_0.01"] = "huber_0.01",
        dropout: float | None = None,
        checkpoint: str | None = None,
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
            online_normalisation: Whether to normalize the target online.
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

    def forward(self, node_features: torch.Tensor, batch: AbstractAtomBatch) -> torch.Tensor:
        """Predictions with raw logits (no sigmoid/softmax or any inverse transformations)."""
        input = segment_ops.aggregate_nodes(
            node_features,
            batch.n_node,
            reduction=self.node_aggregation,
        )
        pred = self.mlp(input)
        return pred

    def predict(self, node_features: torch.Tensor, batch: AbstractAtomBatch) -> torch.Tensor:
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
        batch: AbstractAtomBatch,
        alternative_target: torch.Tensor | None = None,
    ):
        """Apply mlp to compute loss and metrics.

        Depending on whether the target is real/binary/categorical, we
        use an MSE/cross-entropy loss. In the case of cross-entropy, the
        preds are logits (not normalized) to take advantage of numerically
        stable log-softmax.
        """
        name = self.target.fullname
        if alternative_target is not None:
            target = alternative_target
        else:
            target = batch.system_targets[name].squeeze(-1)
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


def _validate_heads_and_loss_weights(
    heads: Sequence[torch.nn.Module] | Mapping[str, torch.nn.Module],
    loss_weights: dict[str, float],
):
    """Validate heads are unique and that for each loss weight, there is a corresponding head."""
    if isinstance(heads, Sequence):
        head_names = [head.target.fullname for head in heads]  # type: ignore
    else:
        head_names = list(heads.keys())

    if len(head_names) == 0:
        raise ValueError("Model must have at least one output head.")
    if len(head_names) != len(set(head_names)):
        raise ValueError(f"Head names must be unique; got {head_names}")

    unknown_keys = set(loss_weights.keys()) - set(head_names)  # type: ignore

    if unknown_keys:
        raise ValueError(f"Loss weights for unknown targets: {unknown_keys}")


def mean_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    error_type: Literal["mae", "mse", "huber_0.01"],
    batch_n_node: torch.Tensor | None = None,
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
        error = segment_ops.aggregate_nodes(errors, batch_n_node, reduction="mean").mean()
    else:
        error = errors.mean()

    return error


def bucketed_mean_error(
    target: torch.Tensor,
    pred: torch.Tensor,
    bucket_by: Literal["target", "error"],
    thresholds: list[float],
    batch_n_node: torch.Tensor | None = None,
    error_type: Literal["mae", "mse"] = "mae",
) -> dict[str, torch.Tensor]:
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
    bucket_edges = torch.tensor([-float("inf")] + thresholds + [float("inf")], device=errors.device)
    bucket_indices = torch.bucketize(values_to_bucket_by, bucket_edges, right=True) - 1

    # Iterate over each bucket and compute its average error
    metrics = {}
    bucket_names = [
        f"bucket_{p:.2f}-{q:.2f}" for p, q in zip(bucket_edges[:-1], bucket_edges[1:], strict=False)
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


def binary_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
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


def bce_loss(pred: torch.Tensor, target: torch.Tensor, metric_prefix: str = "") -> tuple:
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


def cross_entropy_loss(pred: torch.Tensor, target: torch.Tensor, metric_prefix: str = "") -> tuple:
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
