from typing import Literal, Optional
import torch

from orb_models.forcefield.forcefield_utils import (
    forces_within_threshold,
    remove_fixed_atoms,
)
from orb_models.forcefield import base
from orb_models.forcefield.forcefield_utils import bucketed_mean_error, mean_error
from orb_models.forcefield.nn_util import ScalarNormalizer


def stress_loss_function(
    pred: torch.Tensor,
    raw_target: torch.Tensor,
    raw_gold_target: torch.Tensor,
    name: str,
    normalizer: ScalarNormalizer,
    loss_type: Literal["mae", "mse", "huber_0.01"] = "huber_0.01",
):
    """Stress loss and metrics."""
    pred = pred.squeeze(-1)
    raw_target = raw_target.squeeze(-1)
    assert pred.shape == raw_target.shape, f"{pred.shape} != {raw_target.shape}"

    target = normalizer(raw_target)
    loss = mean_error(pred, target, loss_type)
    raw_pred = normalizer.inverse(pred)
    metrics = {
        f"{name}_loss": loss,
        f"{name}_mae_raw": torch.abs(raw_pred - raw_gold_target).mean(),
        f"{name}_mse_raw": ((raw_pred - raw_gold_target) ** 2).mean(),
    }

    return base.ModelOutput(loss=loss, log=metrics)


def forces_loss_function(
    pred: torch.Tensor,
    raw_target: torch.Tensor,
    raw_gold_target: torch.Tensor,
    name: str,
    normalizer: ScalarNormalizer,
    n_node: torch.Tensor,
    fix_atoms: Optional[torch.Tensor],
    loss_type: Literal["mae", "mse", "huber_0.01", "condhuber_0.01"] = "condhuber_0.01",
    training: bool = True,
):
    """Compute forces loss and metrics."""
    pred = pred.squeeze(-1)
    raw_target = raw_target.squeeze(-1)

    # remove before applying normalizer
    pred, raw_target, batch_n_node = remove_fixed_atoms(
        pred, raw_target, n_node, fix_atoms, training
    )
    target = normalizer(raw_target)
    assert pred.shape == target.shape, f"{pred.shape} != {target.shape}"

    if loss_type.startswith("condhuber"):
        huber_delta = float(loss_type.split("_")[1])
        loss = _conditional_huber_force_loss(pred, target, huber_delta)
    else:
        loss = mean_error(pred, target, loss_type, batch_n_node)  # type: ignore

    raw_pred = normalizer.inverse(pred)

    metrics = force_metrics(raw_pred, raw_gold_target.squeeze(-1), batch_n_node, name)
    metrics[f"{name}_loss"] = loss

    return base.ModelOutput(loss=loss, log=metrics)


def force_metrics(
    raw_pred: torch.Tensor,
    raw_target: torch.Tensor,
    batch_n_node: torch.Tensor,
    name: str,
):
    """Compute force metrics."""
    metrics = {
        f"{name}_mae_raw": mean_error(raw_pred, raw_target, "mae", batch_n_node),
        f"{name}_mse_raw": mean_error(raw_pred, raw_target, "mse", batch_n_node),
        f"{name}_cosine_sim": torch.cosine_similarity(
            raw_pred, raw_target, dim=-1
        ).mean(),
        f"{name}_wt_0.03": forces_within_threshold(raw_pred, raw_target, batch_n_node),
    }
    bucket_metrics = bucketed_mean_error(
        raw_target,
        raw_pred,
        bucket_by="target",
        thresholds=[0.1, 1.0],
        batch_n_node=batch_n_node,
        error_type="mae",
    )
    bucket_metrics = {f"{name}_mae_raw_{k}": v for k, v in bucket_metrics.items()}
    metrics.update(bucket_metrics)
    return metrics


def _conditional_huber_force_loss(
    pred_forces: torch.Tensor, target_forces: torch.Tensor, huber_delta: float
) -> torch.Tensor:
    """MACE conditional huber loss for forces."""
    # Define the multiplication factors for each condition
    factors = [huber_delta * x for x in [1.0, 0.7, 0.4, 0.1]]

    # Apply multiplication factors based on conditions
    c1 = torch.norm(target_forces, dim=-1) < 100
    c2 = (torch.norm(target_forces, dim=-1) >= 100) & (
        torch.norm(target_forces, dim=-1) < 200
    )
    c3 = (torch.norm(target_forces, dim=-1) >= 200) & (
        torch.norm(target_forces, dim=-1) < 300
    )
    c4 = ~(c1 | c2 | c3)

    se = torch.zeros_like(pred_forces)

    se[c1] = torch.nn.functional.huber_loss(
        target_forces[c1], pred_forces[c1], reduction="none", delta=factors[0]
    )
    se[c2] = torch.nn.functional.huber_loss(
        target_forces[c2], pred_forces[c2], reduction="none", delta=factors[1]
    )
    se[c3] = torch.nn.functional.huber_loss(
        target_forces[c3], pred_forces[c3], reduction="none", delta=factors[2]
    )
    se[c4] = torch.nn.functional.huber_loss(
        target_forces[c4], pred_forces[c4], reduction="none", delta=factors[3]
    )

    return torch.mean(se)
