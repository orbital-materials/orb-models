from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import ase
import ase.db
import ase.db.row
import torch

from orb_models.forcefield import atomic_system, property_definitions
from orb_models.dataset.base_datasets import (
    AtomsDataset,
)
from orb_models.forcefield.base import AtomGraphs


class AseSqliteDataset(AtomsDataset):
    """AseSqliteDataset.

    A Pytorch Dataset for reading ASE Sqlite serialized Atoms objects.

    Args:
        name: The dataset name.
        path: Local path to read the data from.
        system_config: A config for controlling how an atomic system is represented.
        target_config: A config for regression/classification targets.
        augmentations: A list of augmentation functions to apply to the atoms object.
        dtype: The dtype for floating point tensors in the output.

    Returns:
        An AseSqliteDataset.
    """

    def __init__(
        self,
        name: str,
        path: Union[str, Path],
        system_config: atomic_system.SystemConfig,
        target_config: Optional[property_definitions.PropertyConfig] = None,
        augmentations: Optional[List[Callable[[ase.Atoms], None]]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            name=name,
            system_config=system_config,
            augmentations=augmentations,
        )
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.path = path
        self.db = ase.db.connect(str(self.path), serial=True, type="db")

        self.target_config = (
            target_config
            if target_config is not None
            else property_definitions.PropertyConfig()
        )
        self.constraints = []  # type: ignore[var-annotated]

    def __getitem__(self, idx: int) -> AtomGraphs:
        """Fetch an item from the db.

        Args:
            idx: An index to fetch from the db file and convert to an AtomGraphs.

        Returns:
            A AtomGraphs object containing everything the model needs as input,
            positions and atom types and other auxillary information, such as
            fine tuning targets, or global graph features.
        """
        # Sqlite db is 1 indexed.
        row = self.db.get(idx + 1)
        atoms = row.toatoms()

        # Features and targets are stored in a mutable atoms.info dict.
        # This dict may be modified as part of an augmentation e.g.
        # Force and stress targets are transformed when rotating a system.
        atoms.info = {}
        atoms.info.update(self.target_config.extract(row, self.name, "targets"))

        for augmentation in self.augmentations:
            augmentation(atoms)

        # Remove preexisting constraints and then apply any of our own constraints.
        # Constraints have modelling implications e.g. for partial-diffusion / force-prediction.
        atoms.set_constraint(None)
        for constraint_fn in self.constraints:
            constraint_fn(atoms, row.data, self.name)

        return atomic_system.ase_atoms_to_atom_graphs(
            atoms=atoms,
            system_config=self.system_config,
            edge_method="knn_scipy",
            wrap=True,
            system_id=idx,
            output_dtype=self.dtype,
            graph_construction_dtype=self.dtype,
        )

    def get_atom(self, idx: int) -> ase.Atoms:
        """Return the Atoms object for the dataset index."""
        row = self.db.get(idx + 1)
        return row.toatoms()

    def get_atom_and_metadata(self, idx: int) -> Tuple[ase.Atoms, Dict]:
        """Return the Atoms object plus a dict of metadata for the dataset index."""
        row = self.db.get(idx + 1)
        return row.toatoms(), row.data

    def __len__(self) -> int:
        """Return the dataset length."""
        return len(self.db)

    def __repr__(self) -> str:
        """String representation of class."""
        return f"AseSqliteDataset({self.name=}, {self.path=})"
