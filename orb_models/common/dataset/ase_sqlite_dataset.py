import os
from collections.abc import Callable
from pathlib import Path

import ase
import ase.db
import ase.db.row
import torch

from orb_models.common.atoms.abstract_atoms_adapter import AbstractAtomsAdapter
from orb_models.common.atoms.batch.abstract_batch import AbstractAtomBatch
from orb_models.common.dataset import property_definitions
from orb_models.common.dataset.abstract_dataset import AtomsDataset


class AseSqliteDataset(AtomsDataset):
    """AseSqliteDataset.

    A Pytorch Dataset for reading ASE Sqlite serialized Atoms objects.

    Args:
        name: The dataset name.
        path: Local path to read the dataset from.
        atoms_adapter: Adapter for converting ase.Atoms to model-specific AbstractAtomBatch instances.
        target_config: A config for regression/classification targets.
        augmentations: A list of augmentation functions to apply to the atoms object.
        dtype: The dtype for floating point tensors in the output.
    """

    def __init__(
        self,
        name: str,
        path: str | Path,
        atoms_adapter: AbstractAtomsAdapter,
        target_config: property_definitions.PropertyConfig | None = None,
        augmentations: list[Callable[[ase.Atoms], None]] | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            name=name,
            atoms_adapter=atoms_adapter,
            augmentations=augmentations,
            dtype=dtype,
        )
        self.path = path
        if os.path.exists(self.path):
            self.db = ase.db.connect(str(self.path), serial=True, type="db")
        else:
            raise ValueError(f"Database file {self.path} not found")

        self.feature_config = property_definitions.instantiate_property_config(
            atoms_adapter.extra_features
        )
        self.target_config = target_config or property_definitions.PropertyConfig()
        self.constraints: list[Callable[[ase.Atoms, ase.db.row.FancyDict, str], None]] = []

    def __getitem__(self, idx: int) -> AbstractAtomBatch:
        """Fetch an item from the db.

        Args:
            idx: An index to fetch from the db file and convert to an AtomGraphs.

        Returns:
            A AtomGraphs object containing everything the model needs as input,
            positions and atom types and other auxillary information, such as
            fine tuning targets, or global graph features.
        """
        # Validate index bounds
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        # Sqlite db is 1 indexed.
        row = self.db.get(idx + 1)
        atoms = row.toatoms()

        # Features and targets are stored in a mutable atoms.info dict.
        # This dict may be modified as part of an augmentation e.g.
        # Force and stress targets are transformed when rotating a system.
        # These will add {node,graph,edge}_{features,targets} keys to the atoms.info
        atoms.info = {}
        atoms.info.update(self.feature_config.extract(row, self.name, "features"))
        atoms.info.update(self.target_config.extract(row, self.name, "targets"))

        for augmentation in self.augmentations:
            augmentation(atoms)

        # Remove preexisting constraints and then apply any of our own constraints.
        # Constraints have modelling implications e.g. for partial-diffusion / force-prediction.
        atoms.set_constraint(None)
        for constraint_fn in self.constraints:
            constraint_fn(atoms, row.data, self.name)

        return self._from_ase_atoms(
            atoms=atoms,
            device=torch.device("cpu"),
            output_dtype=self.dtype,
            system_id=idx,
            edge_method="knn_alchemi",
            half_supercell=False,
            graph_construction_dtype=self.dtype,
        )

    def get_atom(self, idx: int) -> ase.Atoms:
        """Return the Atoms object for the dataset index."""
        row = self.db.get(idx + 1)
        return row.toatoms()

    def get_atom_and_metadata(self, idx: int) -> tuple[ase.Atoms, dict]:
        """Return the Atoms object plus a dict of metadata for the dataset index."""
        row = self.db.get(idx + 1)
        return row.toatoms(), row.data

    def __len__(self) -> int:
        """Return the dataset length."""
        return len(self.db)

    def __repr__(self) -> str:
        """String representation of class."""
        return f"AseSqliteDataset({self.name=}, {self.path=})"
