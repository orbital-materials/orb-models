from typing import List, Optional, Union, cast

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

import wandb
from orb_models.finetune_utilities import experiment
from orb_models.finetune_utilities.ema import EMAContextManager as EMA


from core.models import base
from core.training import util
from core.training.metrics import ScalarMetricTracker
from wandb import wandb_run


def gradient_clipping(
    model: torch.nn.Module, clip_value: float
) -> List[torch.utils.hooks.RemovableHandle]:
    """Add gradient clipping hooks to a model.

    This is the correct way to implement gradient clipping, because
    gradients are clipped as gradients are computed, rather than after
    all gradients are computed - this means expoding gradients are less likely,
    because they are "caught" earlier.

    Args:
        model: The model to add hooks to.
        clip_value: The upper and lower threshold to clip the gradients to.

    Returns:
        A list of handles to remove the hooks from the parameters.
    """
    handles = []

    def _clip(grad):
        if grad is None:
            return grad
        return grad.clamp(min=-clip_value, max=clip_value)

    for parameter in model.parameters():
        if parameter.requires_grad:
            h = parameter.register_hook(lambda grad: _clip(grad))
            handles.append(h)

    return handles


def fintune(
    model: base.ModelMixin,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    ema: Optional[EMA] = None,
    lr_scheduler: Optional[_LRScheduler] = None,
    num_steps: Optional[int] = None,
    clip_grad: Optional[float] = None,
    log_freq: float = 500,
    device: torch.device = torch.device("cpu"),
    epoch: int = 0,
    profiler: Optional[torch.profiler.profile] = None,
    gradient_accumulation_steps: int = 1,
):
    """Train for a fixed number of steps.

    Args:
        model: The model to optimize.
        optimizer: The optimizer for the model.
        dataloader: A Pytorch Dataloader, which may be infinite if num_steps is passed.
        ema: Optional, an Exponential Moving Average tracker for saving averaged model weights.
        lr_scheduler: Optional, a Learning rate scheduler for modifying the learning rate.
        num_steps: The number of training steps to take. This is required for distributed training,
            because controlling parallism is easier if all processes take exactly the same number of steps (
            this particularly applies when using dynamic batching).
        clip_grad: Optional, the gradient clipping threshold.
        log_freq: The logging frequency for step metrics.
        device: The device to use for training.
        epoch: The number of epochs the model has been fintuned.

    Returns
        A dictionary of metrics.
    """
    run: Optional[wandb_run.Run] = cast(Optional[wandb_run.Run], wandb.run)

    is_primary_local_rank = util.is_primary_local_rank()

    if clip_grad is not None:
        hook_handles = gradient_clipping(model, clip_grad)

    # Initialize metrics to track metrics
    # over gradient accumulation for each step.
    metrics = ScalarMetricTracker()
    # Initialize epoch_metrics to track metrics
    # and report at the end of an epoch.
    epoch_metrics = ScalarMetricTracker()

    # Set the model to "train" mode.
    model.train()

    # Get tqdm for the training batches
    batch_generator = iter(dataloader)
    num_training_batches: Union[int, float]
    if num_steps is not None:
        num_training_batches = num_steps
    else:
        try:
            num_training_batches = len(dataloader)
        except TypeError:
            raise ValueError("Dataloader has no length, you must specify num_steps.")

    batch_generator_tqdm = batch_generator

    if profiler is not None:
        profiler.start()
    i = 0
    batch_iterator = iter(batch_generator_tqdm)
    while True:
        if num_steps and i == num_steps:
            break

        optimizer.zero_grad(set_to_none=True)

        step_metrics = {
            "batch_size": 0.0,
            "batch_num_edges": 0.0,
            "batch_num_nodes": 0.0,
        }

        # Reset metrics so that it reports raw values for each step but still do averages on
        # the gradient accumulation.
        if i % log_freq == 0:
            metrics.reset()

        for _ in range(gradient_accumulation_steps):
            batch = next(batch_iterator)
            batch = batch.to(device)
            step_metrics["batch_size"] += len(batch.n_node)
            step_metrics["batch_num_edges"] += batch.n_edge.sum()
            step_metrics["batch_num_nodes"] += batch.n_node.sum()

            with torch.cuda.amp.autocast(enabled=False):
                batch_outputs = model.loss(batch)
                loss = batch_outputs.loss
                metrics.update(batch_outputs.log)
            if torch.isnan(loss):
                raise ValueError("nan loss encountered")
            loss.backward()

        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Update moving averages
        if ema is not None:
            ema.update()

        if profiler is not None:
            profiler.step()

        metrics.update(step_metrics)
        # Update epoch_metrics from metrics which is averaged over
        # gradient accumulation step before it get reset for next step.
        # Note it is not averaged across all workers, which will happen later on.
        epoch_metrics.update(metrics.get_metrics())
        # We don't log the very first step, because it skews wallclock metrics;
        # typically the first step takes longer, as kernels are cached, pytorch
        # does it's optimisations etc. We update tqdm only for the primary worker
        # as the other workers wouldn't have one.
        if is_primary_local_rank and i != 0 and i % log_freq == 0:
            metrics_dict = metrics.get_metrics()
            description = util.tqdm_desc_from_metrics(metrics_dict)
            batch_generator_tqdm.set_description(description, refresh=False)

            if run is not None:
                global_step = (epoch * num_training_batches) + i
                if run.sweep_id is not None:
                    # For wandb sweeps, we must log the target variable at the top level.
                    # For our hyperparameter search, this is just loss normally.
                    # This only applies if you are running a wandb sweep explicitly.
                    run.log(
                        {"loss": metrics_dict["loss"]},
                        commit=False,
                    )
                run.log(
                    {"global_step": global_step},
                    commit=False,
                )
                run.log(
                    experiment.prefix_keys(metrics_dict, "train_step"), commit=False
                )
                # Log learning rates.
                run.log(
                    {
                        f"pg_{idx}": group["lr"]
                        for idx, group in enumerate(optimizer.param_groups)
                    },
                )

        # Finished a single full step!
        i += 1

    if clip_grad is not None:
        for h in hook_handles:
            h.remove()

    return epoch_metrics.get_metrics(sync_dist=True)