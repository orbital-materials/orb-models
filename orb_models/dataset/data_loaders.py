import dataclasses
import random
import logging
from typing import Dict, List, Optional, Union

import numpy as np
import omegaconf
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
)

import hydra
from orb_models.forcefield.atomic_system import make_property_definitions_from_config
from orb_models.forcefield import base

HAVE_PRINTED_WORKER_INFO = False


def worker_init_fn(id: int):
    """Set seeds per worker, so augmentations etc are not duplicated across workers.

    Unused id arg is a requirement for the Dataloader interface.

    By default, each worker will have its PyTorch seed set to base_seed + worker_id,
    where base_seed is a long generated by main process using its RNG
    (thereby, consuming a RNG state mandatorily) or a specified generator.
    However, seeds for other libraries may be duplicated upon initializing workers,
    causing each worker to return identical random numbers.

    In worker_init_fn, you may access the PyTorch seed set for each worker with either
    torch.utils.data.get_worker_info().seed or torch.initial_seed(), and use it to seed
    other libraries before data loading.
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)
    global HAVE_PRINTED_WORKER_INFO
    if not HAVE_PRINTED_WORKER_INFO:
        print(torch.utils.data.get_worker_info())
        HAVE_PRINTED_WORKER_INFO = True


@dataclasses.dataclass
class DatasetsConfig(omegaconf.DictConfig):
    """Config for Datasets."""

    data_root: str

    train: List[str]
    val: List[str]
    test: List[str]

    extra_kwargs: Dict[str, omegaconf.DictConfig]


@dataclasses.dataclass
class WorkerConfig:
    """Config for Workers."""

    train: int
    val: int
    test: int


@dataclasses.dataclass
class BatchSizeConfig:
    """Config for Dynamic Batch Size."""

    train: Union[int]
    val: int
    test: int


def build_train_loader(
    datasets: DatasetsConfig,
    num_workers: WorkerConfig,
    batch_size_dict: omegaconf.DictConfig,
    system_config_dict: omegaconf.DictConfig,
    target_config_dict: Optional[omegaconf.DictConfig] = None,
    augmentation: Optional[List[str]] = None,
    **kwargs,
) -> DataLoader:
    """Builds the train dataloader from a config file.

    Args:
        datasets: The dataset config.
        num_workers: The number of workers for each dataset.
        batch_size: The batch_size config for each dataset.
        system_config: The system config.
        target_config: The target config.
        temperature: The temperature for temperature sampling.
            Default is None for using random sampler.
        augmentation: If rotation augmentation is used.

    Returns:
        The train Dataloader.
    """
    batch_size = hydra.utils.instantiate(batch_size_dict)
    system_config = hydra.utils.instantiate(system_config_dict)
    target_config = make_property_definitions_from_config(target_config_dict)

    log_train = "Loading train datasets:\n"
    dataset = get_dataset(
        datasets=datasets.train,
        system_config=system_config,
        target_config=target_config,
        mode="train",
        extra_kwargs=datasets.get("extra_kwargs"),
        augmentations=augmentation,
        download=datasets.get("download"),
    )

    log_train += f"Total train dataset size: {len(dataset)} samples"
    logging.info(log_train)

    sampler = RandomSampler(dataset)

    batch_sampler = BatchSampler(
        sampler,
        batch_size=batch_size,
        drop_last=True,
    )

    train_loader: DataLoader = DataLoader(
        dataset,
        num_workers=num_workers.train,
        worker_init_fn=worker_init_fn,
        collate_fn=base.batch_graphs,
        batch_sampler=batch_sampler,
        timeout=10 * 60 if num_workers.train > 0 else 0,
    )
    return train_loader