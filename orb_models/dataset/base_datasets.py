from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import ase
import ase.db
import ase.db.row
from torch.utils.data import Dataset

from orb_models.forcefield import atomic_system
from orb_models.forcefield.base import AtomGraphs



class AtomsDataset(ABC, Dataset):
    """AtomsDataset.

    An abstract Pytorch Dataset for loading atomic systems.
    supports:
        - loading ase.Atoms via get_atom()
        - loading AtomGraphs via __get_item__()

    At a minimum, subclasses must define get_atom_and_metadata(), get_idx_to_natoms() and len().

    Args:
        name: The dataset name.
        system_config: A config for controlling how an atomic system is represented
        target_config: A config for regression/classification targets
        position_override_path: Path to a json file containing position overrides.
        augmentations: A list of augmentation functions to apply to the atoms object.

    Returns:
        An AtomsDataset.
    """

    def __init__(
        self,
        name: str,
        system_config: atomic_system.SystemConfig,
        augmentations: Optional[List[Callable[[ase.Atoms], None]]] = None,
    ):
        super().__init__()
        self.name = name
        self.system_config = system_config
        self.augmentations = augmentations or []

    def __getitem__(self, idx: int) -> AtomGraphs:
        """Fetch an AtomGraphs system.

        Args:
            idx: An index that uniquely specifies a datapoint in this dataset.

        Returns:
            A AtomGraphs object containing everything the model needs as input,
            positions and atom types and other auxillary information, such as
            fine tuning targets, or global graph features.
        """
        raise NotImplementedError

    def __getitems__(self, indices: List[int]) -> List[AtomGraphs]:
        """Get a list of items from the dataset."""
        return [self[idx] for idx in indices]

    @abstractmethod
    def get_atom(self, idx: int) -> ase.Atoms:
        """Return the Atoms object for the dataset index."""
        raise NotImplementedError

    @abstractmethod
    def get_atom_and_metadata(self, idx: int) -> Tuple[ase.Atoms, Dict]:
        """Return the Atoms object plus a dict of metadata for the dataset index."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return the dataset length."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """String representation of class."""
        return f"AtomsDataset({self.name=})"
