import os
from collections import defaultdict
from pathlib import Path

import dotenv

from typing import TypeVar, Union, Dict, Mapping
import torch
import torch.distributed as dist
import datetime
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


def is_distributed() -> bool:
    """Checks if the distributed process group is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def is_distributed_env() -> bool:
    """Checks for presence of environment variables for distributed gpu device configuration."""
    return (
        os.environ.get("WORLD_SIZE") is not None and os.environ.get("RANK") is not None
    )


def check_distributed() -> None:
    """Assert that a distributed process group is available."""
    if not is_distributed():
        raise RuntimeError("No distributed process group is available.")


def get_local_rank() -> int:
    """Get's the local rank of the distributed process.

    If called outside a distributed context, just returns 0.
    """
    if is_distributed():
        return dist.get_rank()
    else:
        return 0


def get_world_size() -> int:
    """Get's the world size of the distributed process.

    If called outside a distributed context, just returns 1.
    """
    if is_distributed():
        return dist.get_world_size()
    else:
        return 1


def is_primary_local_rank() -> bool:
    """Checks if this process is the primary local rank.

    If called outside a distributed context, just returns True.
    """
    if is_distributed():
        return dist.get_rank() == 0
    else:
        return True


def distributed_device() -> torch.device:
    """Get the torch.device of the current process."""
    check_distributed()
    return int_to_device(
        -1 if dist.get_backend() != "nccl" else torch.cuda.current_device()
    )


def init_device() -> torch.device:
    """Initialize a device.

    Initializes a device, making sure to also
    initialize the process group in a distributed
    setting.
    """
    rank = 0
    if is_distributed_env():
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Set the NCCL error handling to terminate
        # on subprocess errors, instead of raising an error.
        # See https://pytorch.org/docs/stable/distributed.html#initialization
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=60 * 60),
        )

    if torch.cuda.is_available():
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
    else:
        device = "cpu"
    return torch.device(device)


def maybe_dist_reduce(value: _V, reduce_op) -> _V:
    """Reduces a value across all distributed gpu processes and nodes.

    If called outside of a distributed context, it will just return `value`.

    Args:
        value: The value to reduce across distributed nodes.
        reduce_op: The reduction operation to use.
        (https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp)

    Returns:
        The final value.
    """
    if not is_distributed():
        return value
    device = distributed_device()
    if isinstance(value, torch.Tensor):
        value_tensor = value.clone().to(device)
    else:
        value_tensor = torch.tensor(value, device=device)
    dist.all_reduce(value_tensor, op=reduce_op)

    if isinstance(value, torch.Tensor):
        return value_tensor
    return value_tensor.item()  # type: ignore[return-value]


def maybe_dist_reduce_sum(value: _V) -> _V:
    """Sums the given `value` across distributed gpu processes and nodes.

    This is equivalent to calling `dist_reduce(v, dist.ReduceOp.SUM)`.
    """
    if not is_distributed():
        return value
    return maybe_dist_reduce(value, dist.ReduceOp.SUM)


def tqdm_desc_from_metrics(metrics: Dict[str, float]) -> str:
    """Create a tqdm progress bar description from a dict of metrics."""
    return (
        ", ".join(["%s: %.4f" % (name, value) for name, value in metrics.items()])
        + " ||"
    )


def gather_predictions(preds: Dict[str, torch.Tensor]):
    """Gathers predictions from all distributed processes."""
    if not is_distributed():
        return preds

    all_preds_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_preds_list, preds)

    combined_preds: Dict[str, torch.Tensor] = {}
    for p in all_preds_list:
        assert p is not None
        combined_preds.update(p)

    return combined_preds


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
