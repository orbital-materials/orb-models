from abc import ABC, abstractmethod
from collections.abc import Callable

import ase
import torch
from torch.utils.data import Dataset

from orb_models.common.atoms.abstract_atoms_adapter import AbstractAtomsAdapter
from orb_models.common.atoms.batch.abstract_batch import AbstractAtomBatch
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter


class AtomsDataset(ABC, Dataset):
    """An abstract Pytorch Dataset for loading atomic systems.

    Supports loading atomic systems via:
    - loading ase.Atoms via get_atom()
    - loading AbstractAtomBatch via __get_item__()

    At minimum, subclasses must implement get_atom(), get_atom_and_metadata() and len().

    Overriding __getitem__() is optional, but needed for any non-trivial logic e.g. data augmentation.
    By default, __getitem__() will call get_atom(idx) and convert the result to an AbstractAtomBatch.
    """

    def __init__(
        self,
        name: str,
        atoms_adapter: AbstractAtomsAdapter,
        augmentations: list[Callable[[ase.Atoms], None]] | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialise the AtomsDataset.

        Args:
            name: The name of the dataset.
            atoms_adapter: An instance of AbstractAtomsAdapter for converting ase.Atoms to a batch.
            augmentations: The random augmentations to use.
            dtype: The dtype for floating point tensors in the output.
        """
        super().__init__()
        self.name = name
        self.atoms_adapter = atoms_adapter
        self.augmentations = augmentations or []
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()

    def __getitem__(self, idx: int) -> AbstractAtomBatch:
        """Get an item from the dataset.

        Args:
            idx: The index of the item to get.
        """
        atoms = self.get_atom(idx)
        return self._from_ase_atoms(
            atoms=atoms,
            device=torch.device("cpu"),
            output_dtype=self.dtype,
            system_id=idx,
            half_supercell=False,
            graph_construction_dtype=self.dtype,
        )

    def __getitems__(self, indices: list[int]) -> list[AbstractAtomBatch]:
        """Get a list of items from the dataset."""
        return [self[idx] for idx in indices]

    @abstractmethod
    def get_atom(self, idx: int) -> ase.Atoms:
        """Return the Atoms object for the dataset index."""
        pass

    @abstractmethod
    def get_atom_and_metadata(self, idx: int) -> tuple[ase.Atoms, dict]:
        """Return the Atoms object plus a dict of metadata for the dataset index."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the dataset length."""
        pass

    def __repr__(self) -> str:
        """String representation of class."""
        return f"AtomsDataset({self.name=})"

    def _from_ase_atoms(self, atoms: ase.Atoms, **kwargs) -> AbstractAtomBatch:
        """Convert an ase.Atoms object into an AbstractAtomBatch instance, ready for use in a model."""
        if isinstance(self.atoms_adapter, ForcefieldAtomsAdapter):
            return self.atoms_adapter.from_ase_atoms(
                atoms=atoms,
                edge_method=kwargs["edge_method"],
                half_supercell=kwargs["half_supercell"],
                device=kwargs["device"],
                output_dtype=kwargs["output_dtype"],
                graph_construction_dtype=kwargs["graph_construction_dtype"],
                system_id=kwargs["system_id"],
            )
        else:
            raise ValueError(f"Unknown atoms adapter: {self.atoms_adapter}")
