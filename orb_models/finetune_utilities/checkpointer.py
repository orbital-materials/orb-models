from typing import Optional, Callable, List
import os
import math

import torch
import wandb
import dataclasses
import heapq
import omegaconf
import pathlib
import hydra
from orb_models.finetune_utilities.experiment import WandbArtifactTypes

from orb_models.finetune_utilities.ema import ExponentialMovingAverage as EMA


def unwrap_model(model):
    """Unwrap a model from various pytorch training wrappers."""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    if isinstance(model, torch.nn.parallel.DataParallel):
        return model.module
    return model


@dataclasses.dataclass(order=True)
class CheckpointInfo:
    """Information about Checkpoints.

    NOTE: order=True means this dataclass can be compared,
    e.g < > = etc. It is compared in order of it's arguments,
    *not including* the path and artifact fields (compare=False).

    This means that we get checkpoints sortable by metric and then
    step count (for tie breaks) by default.

    Args:
        metric: The metric to compare.
        step: The global step of training.
        path: The local path the checkpoint is saved to.
        artifact: An optional wandb artifact reference.
    """

    metric: float
    step: int
    path: str = dataclasses.field(compare=False)
    artifact: Optional[wandb.Artifact] = dataclasses.field(compare=False)


class Checkpointer:
    """Save model checkpoints.

    Args:
        checkpoint_dir: The directory to store checkpoints on disk.
        config_path: Path to the config file used for the run.
        mode: "max" or "min", whether better metrics are larger or smaller.
        top_k: The number of checkpoints to store in wandb.
        extension: The extension to store the checkpoint as.
        unwrap_model_fn: A function which extracts the model you want to save.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        config_path: str,
        mode: str = "min",
        top_k: int = 3,
        extension: str = ".ckpt",
        unwrap_model_fn: Callable = unwrap_model,
    ) -> None:
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        sentinel = self.checkpoint_dir / ".checkpoint_sentinel"
        if sentinel.exists():
            raise ValueError(
                "Checkpoint directory already contains sentinel file. .checkpoint_sentinel."
            )
        else:
            sentinel.touch()

        self.config_path = config_path
        self.mode = mode
        self.top_k = top_k
        self.unwrap_model_fn = unwrap_model_fn
        self.extension = extension

        self.logged_checkpoints: List[CheckpointInfo] = []
        self.latest: Optional[CheckpointInfo] = None
        self.last_step = -1

    def checkpoint(
        self,
        model: torch.nn.Module,
        step: int,
        metric: Optional[float] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        ema: Optional[EMA] = None,
    ):
        """Checkpoint a model.

        Args:
            model: The model to checkpoint.
            step: The global step.
            metric: An optional float value to compare checkpoints.
            optimizer: The optimizer associated with this model.
            scaler: The GradScaler associated with this model.
        Returns:
            None
        """
        if step <= self.last_step:
            raise ValueError("Step must strictly increase over time.")

        self.last_step = step

        last_save_path = self._save(model, step, optimizer, lr_scheduler, scaler, ema)
        artifact: Optional[wandb.Artifact] = None
        run = wandb.run
        if run is not None:
            artifact_type = WandbArtifactTypes.MODEL
            artifact = wandb.Artifact(
                name=f"{artifact_type}-{run.id}",
                type=artifact_type,
            )
            artifact.add_file(last_save_path, name="model.ckpt")
            artifact.add_file(self.config_path, name="config.yaml")
            run.log_artifact(artifact, aliases=["latest"])
            # Wait for artifact to log in order to prevent race
            # conditions when checking if we should delete it later.
            if not run.offline:
                artifact.wait()
            hydra.utils.log.info(f"Logged: {artifact.id}, {artifact.name}")

        if metric is not None and self.mode == "min":
            # metric was passed, but heappop returns the *smallest*
            # values, so if min = better, we want to flip the ordering.
            metric = -metric
        if metric is None:
            # if no metric is passed, we want this checkpoint to be
            # "worst by default".
            metric = -math.inf if self.mode == "max" else math.inf

        info = CheckpointInfo(metric, step, last_save_path, artifact)
        # Push the last checkpoint into heap...
        if self.latest is not None:
            heapq.heappush(self.logged_checkpoints, self.latest)

        # ... But keep the latest one separate so that the heap
        # doesn't delete it, even if it's the worst one (in case something
        # goes wrong during training and we want the last saved state.)
        self.latest = info

        self.cleanup()

    def cleanup(self):
        """Cleanup saved artifacts.

        Cleans up saved model artifacts, both on disk and in wandb.
        """
        run = wandb.run
        while len(self.logged_checkpoints) > self.top_k:
            to_remove = self.logged_checkpoints[0]
            try:
                # Sometimes the wandb api 400s. We never want to
                # crash during training because we failed to cleanup
                # artifacts, so we catch a bare exception here.
                if to_remove.artifact is not None and run is not None:
                    if not run.offline:
                        to_remove.artifact.wait()
                    hydra.utils.log.info(
                        f"Removing checkpoint {to_remove.path}, {to_remove.artifact.id}, {to_remove.artifact.name}"
                    )

                    # Careful! Wandb distinguishes between logged and un-logged artifacts, and they have
                    # a different API, despite using the same interface. When we call wait here,
                    # we are getting an instance of wandb.apis.public.Artifact, which is an interface
                    # to logged data. Only this class has the `delete_aliases` arg, whereas the
                    # wandb.wandb_artifacts.Artifact class does not.
                    to_remove.artifact.wait().delete(delete_aliases=True)  # type: ignore
                os.remove(to_remove.path)
                heapq.heappop(self.logged_checkpoints)
            except Exception as e:
                hydra.utils.log.warning(
                    "Failed to remove artifact from wandb. Skipping. Traceback:"
                )
                hydra.utils.log.warning(e)

    def best_metric_so_far(self):
        """Return the best metric so far."""
        if self.latest is None:
            return None
        metrics = [self.latest.metric] + [
            info.metric for info in self.logged_checkpoints
        ]
        return max(metrics) if self.mode == "max" else -max(metrics)

    def _save(
        self,
        model: torch.nn.Module,
        step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        ema: Optional[EMA] = None,
    ):
        # Be defensive about saving - first save to a temp file,
        # then move to the latest path name.
        tmp_save_path = self.checkpoint_dir / ("tmp" + self.extension)
        last_save_path = self.checkpoint_dir / (f"step-{step}" + self.extension)

        save_state = {
            "step": step,
            "state_dict": self.unwrap_model_fn(model).state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "lr_scheduler": (
                lr_scheduler.state_dict() if lr_scheduler is not None else None
            ),
        }
        if scaler is not None:
            save_state["amp"] = scaler.state_dict()
        if ema is not None:
            save_state["ema"] = ema.state_dict()
        torch.save(save_state, tmp_save_path)
        os.replace(tmp_save_path, last_save_path)
        return last_save_path


def from_config(
    checkpoint_dir: str,
    config_path: str,
    config: omegaconf.DictConfig = None,
) -> Optional[Checkpointer]:
    """Construct checkpointers from a config."""
    cfg = config.get("checkpointer")
    if cfg is None:
        return None
    model_checkpointer = Checkpointer(
        checkpoint_dir=checkpoint_dir, config_path=config_path, **cfg
    )
    return model_checkpointer
