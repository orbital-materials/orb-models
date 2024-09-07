"""Experiment utilities."""

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, TypeVar

import dotenv
import numpy
import torch
import wandb
from wandb import wandb_run

from orb_models.forcefield import base

T = TypeVar("T")


_V = TypeVar("_V", int, float, torch.Tensor)

dotenv.load_dotenv(override=True)
PROJECT_ROOT: Path = Path(
    os.environ.get("PROJECT_ROOT", str(Path(__file__).parent.parent))
)
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"

DATA_ROOT: Path = Path(os.environ.get("DATA_ROOT", default=str(PROJECT_ROOT / "data")))
WANDB_ROOT: Path = Path(
    os.environ.get("WANDB_ROOT", default=str(PROJECT_ROOT / "wandb"))
)


def init_device() -> torch.device:
    """Initialize a device.

    Initializes a device, making sure to also
    initialize the process group in a distributed
    setting.
    """
    rank = 0
    if torch.cuda.is_available():
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
    else:
        device = "cpu"
    return torch.device(device)


def ensure_detached(x: base.Metric) -> base.Metric:
    """Ensure that the tensor is detached and on the CPU."""
    if isinstance(x, torch.Tensor):
        return x.detach()
    return x


def to_item(x: base.Metric) -> base.Metric:
    """Convert a tensor to a python scalar."""
    if isinstance(x, torch.Tensor):
        return x.cpu().item()
    return x


def prefix_keys(
    dict_to_prefix: Dict[str, T], prefix: str, sep: str = "/"
) -> Dict[str, T]:
    """Add a prefix to dictionary keys with a seperator."""
    return {f"{prefix}{sep}{k}": v for k, v in dict_to_prefix.items()}


def seed_everything(seed: int, rank: int = 0) -> None:
    """Set the seed for all pseudo random number generators."""
    random.seed(seed + rank)
    numpy.random.seed(seed + rank)
    torch.manual_seed(seed + rank)


def init_wandb_from_config(args, job_type: str) -> wandb_run.Run:
    """Initialise wandb from config."""
    if not hasattr(args, "wandb_name"):
        run_name = f"{job_type}-test"
    else:
        run_name = args.name
    if not hasattr(args, "wandb_project"):
        project = "orb-experiment"
    else:
        project = args.project

    wandb.init(  # type: ignore
        job_type=job_type,
        dir=os.path.join(os.getcwd(), "wandb"),
        name=run_name,
        project=project,
        entity="orbitalmaterials",
        mode="online",
        sync_tensorboard=False,
    )
    assert wandb.run is not None
    return wandb.run


class ScalarMetricTracker:
    """Keep track of average scalar metric values."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the AverageMetrics."""
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, metrics: Mapping[str, base.Metric]) -> None:
        """Update the metric counts with new values."""
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor) and v.nelement() > 1:
                continue  # only track scalar metrics
            if isinstance(v, torch.Tensor) and v.isnan().any():
                continue
            self.sums[k] += ensure_detached(v)
            self.counts[k] += 1

    def get_metrics(self):
        """Get the metric values, possibly reducing across gpu processes."""
        return {k: to_item(v) / self.counts[k] for k, v in self.sums.items()}
