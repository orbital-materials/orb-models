import os
from collections import defaultdict
from pathlib import Path

import dotenv

from typing import TypeVar, Union, Dict, Mapping
import torch
from orb_models.forcefield import base

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


def int_to_device(device: Union[int, torch.device]) -> torch.device:
    """Converts an integer to a torch device."""
    if isinstance(device, torch.device):
        return device
    if device < 0:
        return torch.device("cpu")
    return torch.device(device)


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


def tqdm_desc_from_metrics(metrics: Dict[str, float]) -> str:
    """Create a tqdm progress bar description from a dict of metrics."""
    return (
        ", ".join(["%s: %.4f" % (name, value) for name, value in metrics.items()])
        + " ||"
    )


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
