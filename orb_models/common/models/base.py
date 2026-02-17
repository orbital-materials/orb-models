"""Base Model class."""

from collections.abc import Mapping
from typing import NamedTuple

import torch

from orb_models.common.atoms.batch.abstract_batch import AbstractAtomBatch

Metric = torch.Tensor | int | float
TensorDict = Mapping[str, torch.Tensor | None]


# NOTE: Pytorch DDP does not support dataclasses.
# https://github.com/pytorch/pytorch/issues/41327
# Thus it is very important that we do not use them,
# because we want our models to be usable with DDP.
class ModelOutput(NamedTuple):
    """A model's output."""

    loss: torch.Tensor
    log: Mapping[str, Metric]


class ModelMixin[T: AbstractAtomBatch](torch.nn.Module):
    """Model Mixin for our models.

    This model mixin acts to specify a consistent
    return type for models so that they are compatible
    with a single training and evaluation loop. Any models
    inheriting from and implementing this interface will
    be trainable by `core.training.step.*` functions.
    """

    def loss(self, batch: T) -> ModelOutput:
        """Encodes to latents before message passing."""
        raise NotImplementedError()


class RegressorModelMixin[T: AbstractAtomBatch](ModelMixin[T]):
    """Model Mixin for our regression models."""

    def predict(self, batch: T, split: bool) -> dict[str, torch.Tensor]:
        """Predicts a set of properties."""
        raise NotImplementedError()
