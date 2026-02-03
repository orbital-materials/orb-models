from abc import ABC, abstractmethod
from collections.abc import Iterable, MutableMapping
from typing import (
    Any,
)

import ase
import torch

from orb_models.common.atoms.batch.abstract_batch import AbstractAtomBatch
from orb_models.common.utils import is_periodic


class AbstractAtomsAdapter[T: AbstractAtomBatch](ABC):
    """Adapter for converting ase.Atoms to model-specific AbstractAtomBatch instances.

    The ase.Atoms object is mapped to a batchable, model-specific AbstractAtomBatch instance.
    Subclasses should pass model-specific featurization config to the constructor, and
    implement the following abstract methods:
        - is_compatible_with
        - from_ase_atoms

    Each model adapter should be defined in its own core/atoms/adapter/X_atoms_adapter.py module.
    """

    output_cls: type[T]
    _deprecated_args: set[str] = set()
    _from_ase_atoms_valid_kwargs: set[str] = set()

    def __init__(
        self,
        radius: float | None = None,
        max_num_neighbors: int | None = None,
        extra_features: MutableMapping[Any, Any] | None = None,
        **kwargs,
    ):
        """Initialize the AbstractAtomsAdapter.

        Args:
            radius: The radius of the atoms (for graph-based models).
            max_num_neighbors: The maximum number of neighbors to use (for graph-based models).
            extra_features: Dict of additional features to be stored on the AtomBatch.
                Format: {'node': [...], 'edge': [...], 'graph': [...]}
                Each named feature must exist in core.common.dataset.property_definitions.PROPERTIES,
                where a function to extract the feature from an ase.db.row.AtomsRow is defined.
        """
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        self.extra_features = extra_features

        kwargs = {k: v for k, v in kwargs.items() if k not in self._deprecated_args}
        if kwargs:
            raise ValueError(f"Got invalid kwargs for {self.__class__.__name__}: {kwargs.keys()}")

    @abstractmethod
    def is_compatible_with(self, other: "AbstractAtomsAdapter") -> bool:
        """Check if this AbstractAtomsAdapter is compatible with another.

        Two AbstractAtomsAdapters are incompatible if one cannot be substituted
        for the other at inference time.

        Returns:
            True if the builders are compatible (error otherwise).

        Raises:
            Exception: raises an error for incompatibilities.
        """
        pass

    @abstractmethod
    def from_ase_atoms(
        self,
        atoms: ase.Atoms,
        *,
        wrap: bool = True,
        device: torch.device | str | None = None,
        output_dtype: torch.dtype | None = None,
        system_id: int | None = None,
    ) -> T:
        """Convert an ase.Atoms object into an AbstractAtomBatch instance, ready for use in a model.

        Args:
            atoms: ase.Atoms object
            wrap: whether to wrap atomic positions into the central unit cell (if there is one).
            device: The device to put the tensors on.
            output_dtype: The dtype to use for all floating point tensors stored on the output.
            system_id: Optional index that is relative to a particular dataset.
            **kwargs: Additional keyword arguments.

        Returns:
            AbstractAtomBatch object
        """
        pass

    def from_ase_atoms_list(self, atoms: list[ase.Atoms]) -> T:
        """Convert a list of ase.Atoms into a single batch object."""
        return self.batch([self.from_ase_atoms(a) for a in atoms])

    def batch(self, items: list[T]) -> T:
        """Convert a list of AbstractAtomBatch objects into a single batch object."""
        return self.output_cls.batch(items)

    def _validate_inputs(self, atoms: ase.Atoms, output_dtype: torch.dtype):
        """Validate the inputs to the 'from_ase_atoms' method."""
        # valid_kwargs = self._get_valid_kwargs()
        if isinstance(atoms.pbc, Iterable) and any(atoms.pbc) and not all(atoms.pbc):
            raise NotImplementedError(
                "We do not support periodicity along a subset of axes. Please ensure atoms.pbc is "
                "True/False for all axes and you have padded your systems with sufficient vacuum if necessary."
            )
        # This will raise an error if atoms.pbc is not consistent with atoms.cell
        is_periodic(atoms)

        if output_dtype == torch.float64:
            # when using fp64 precision, we must ensure all features + targets
            # stored in the atoms.info dict are already fp64
            _check_floating_point_tensors_are_fp64(atoms.info)


def _check_floating_point_tensors_are_fp64(obj):
    """Recursively check that all floating point tensors are fp64."""
    if isinstance(obj, torch.Tensor) and torch.is_floating_point(obj):
        if obj.dtype != torch.float64:
            raise ValueError("All torch tensors stored in atoms.info must be fp64")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _check_floating_point_tensors_are_fp64(v)
