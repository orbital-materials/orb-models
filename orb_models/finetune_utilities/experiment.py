"""Experiment utilities."""

import os
import dataclasses
import random
from typing import Dict, TypeVar

import numpy
import torch
import wandb
from wandb import wandb_run

T = TypeVar("T")


@dataclasses.dataclass
class WandbArtifactTypes:
    """Artifact types for wandb."""

    MODEL = "model"
    CONFIG = "config"
    DATASET = "dataset"
    SCREENING = "screening"


def prefix_keys(
    dict_to_prefix: Dict[str, T], prefix: str, sep: str = "/"
) -> Dict[str, T]:
    """Add a prefix to dictionary keys with a seperator."""
    return {f"{prefix}{sep}{k}": v for k, v in dict_to_prefix.items()}


def seed_everything(seed: int, rank: int) -> None:
    """Set the seed for all pseudo random number generators."""
    random.seed(seed + rank)
    numpy.random.seed(seed + rank)
    torch.manual_seed(seed + rank)


def init_wandb_from_config(args, job_type: str) -> wandb_run.Run:
    """Initialise wandb from config."""
    run_name = args.get("name")
    project = args.get("project")
    if not run_name:
        run_name = f"{job_type}-test"
    if not project:
        project = "orb-experiment"
    wandb.init(  # type: ignore
        job_type=job_type,
        dir=os.path.join(os.getcwd(), "wandb"),
        name=run_name,
        project=project,
        entity=args.entity,
        mode=args.mode,
        group=args.get("group"),
        sync_tensorboard=True,
    )
    assert wandb.run is not None
    return wandb.run
