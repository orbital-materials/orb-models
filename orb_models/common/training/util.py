import re
import warnings

import torch


def set_torch_precision(precision: str = "float32-high") -> torch.dtype:
    """Set the default dtype for the current process."""
    if precision == "float64":
        torch_dtype = torch.float64
    elif precision == "float32-highest":
        torch_dtype = torch.float32
        torch.set_float32_matmul_precision("highest")
    elif precision == "float32-high":
        torch_dtype = torch.float32
        torch.set_float32_matmul_precision("high")
    else:
        raise ValueError(f"Unknown precision: {precision}")

    warnings.warn(f"Setting global torch default dtype to {torch_dtype}.")
    if precision != "float32-high":
        warnings.warn(
            "Consider passing 'precision=float32-high' for significantly higher (>2x) "
            "model throughput if high precision is not required."
        )

    torch.set_default_dtype(torch_dtype)

    return torch_dtype


def init_device(device_id: int | None = None) -> torch.device:
    """Initialize a device.

    Initializes a device based on the device id provided,
    if not provided, it will use device_id = 0 if GPU is available.
    """
    if not device_id:
        device_id = 0
    if torch.cuda.is_available():
        device = f"cuda:{device_id}"
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
    else:
        device = "cpu"
    return torch.device(device)


def get_optim(
    lr: float, total_steps: int, model: torch.nn.Module
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler | None]:
    """Configure optimizers, LR schedulers and EMA."""
    # Initialize parameter groups
    params = []

    # Split parameters based on the regex
    for name, param in model.named_parameters():
        if re.search(r"(.*bias|.*layer_norm.*|.*batch_norm.*)", name):
            params.append({"params": param, "weight_decay": 0.0})
        else:
            params.append({"params": param})

    # Create the optimizer with the parameter groups
    optimizer = torch.optim.Adam(params, lr=lr)

    # Create the learning rate scheduler
    div_factor = 10  # max lr will be 10 times larger than initial lr
    final_div_factor = 10  # min lr will be 10 times smaller than initial lr
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * div_factor,
        total_steps=total_steps,
        pct_start=0.05,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )

    return optimizer, scheduler  # type: ignore
