from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import ase
import ase.db
import ase.db.row
import torch
from ase.stress import voigt_6_to_full_3x3_stress
import numpy as np
from e3nn import o3


from orb_models.forcefield import (
    atomic_system,
    property_definitions,
)
from torch.utils.data import Dataset
from orb_models.forcefield.base import AtomGraphs


class AseSqliteDataset(Dataset):
    """AseSqliteDataset.

    A Pytorch Dataset for reading ASE Sqlite serialized Atoms objects.

    Args:
        name: The dataset name.
        path: Local or GCS path to the sqlite file to read, or to a directory containing .db files
            representing shards of a dataset. Shards are read in alphabetically sorted
            order, so it is important to name them in a way that preserves the order
            of the dataset.
        system_config: A config for controlling how an atomic system is represented.
        target_config: A config for regression/classification targets.
        evaluation: Three modes: "eval_with_noise", "eval_no_noise", "train".
        augmentation: If random rotation augmentation is used.
        limit_size: Limit the size of the dataset to this many samples. Useful for debugging.
        masking_args: Arguments for masking function.
        filter_indices_path: Path to a file containing a list of indices to include in the dataset.

    Returns:
        An AseSqliteDataset.
    """

    def __init__(
        self,
        name: str,
        path: str,
        system_config: Optional[atomic_system.SystemConfig] = None,
        target_config: Optional[atomic_system.PropertyConfig] = None,
        augmentation: Optional[bool] = True,
    ):
        super().__init__()
        self.name = name
        self.augmentation = augmentation
        self.path = path
        self.db = ase.db.connect(str(self.path), serial=True, type="db")

        self.feature_config = system_config
        self.target_config = target_config

    def __getitem__(self, idx) -> AtomGraphs:
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
        extra_feats = self._get_row_properties(row, self.feature_config)
        extra_targets = self._get_row_properties(row, self.target_config)

        if self.augmentation:
            atoms, extra_targets = random_rotations_with_properties(atoms, extra_targets)  # type: ignore

        atom_graph = atomic_system.ase_atoms_to_atom_graphs(
            atoms,
            system_id=idx,
            brute_force_knn=False,
        )
        atom_graph = self._add_extra_feats_and_targets(
            atom_graph, extra_feats, extra_targets
        )

        return atom_graph

    def get_atom(self, idx: int) -> ase.Atoms:
        """Return the Atoms object for the dataset index."""
        row = self.db.get(idx + 1)
        return row.toatoms()

    def get_atom_and_metadata(self, idx: int) -> Tuple[ase.Atoms, Dict]:
        """Return the Atoms object plus a dict of metadata for the dataset index."""
        row = self.db.get(idx + 1)
        return row.toatoms(), row.data

    def get_idx_to_natoms(self) -> Dict[int, int]:
        """Return a mapping between dataset index and number of atoms."""
        return self.db.get_idx_to_natoms(zero_index=True)

    def __len__(self) -> int:
        """Return the dataset length."""
        return len(self.db)

    def __repr__(self) -> str:
        """String representation of class."""
        return f"AseSqliteDataset({self.name=}, {self.path=})"

    def _get_row_properties(
        self,
        row: ase.db.row.AtomsRow,
        property_config: Optional[atomic_system.PropertyConfig] = None,
    ) -> Dict:
        """Extract numerical properties from the db as tensors, to be used as features/targets.

        Applies extraction function (e.g. extract from metadata) and normalisation

        Args:
            row: Database row
            property_config: The config specifying how to extract the property/target.

        Returns:
            ExtrinsicProperties containing the tensors for the row.
        """
        if property_config is None:
            return {"node": {}, "edge": {}, "graph": {}}

        def _get_properties(
            property_definitions: Optional[
                Dict[str, property_definitions.PropertyDefinition]
            ],
        ) -> Dict[str, torch.Tensor]:
            kwargs = {}
            if property_definitions is not None:
                for key, definition in property_definitions.items():
                    if definition.row_to_property_fn is not None:
                        property_tensor = definition.row_to_property_fn(
                            row=row, dataset=self.name
                        )
                        kwargs[key] = property_tensor
            return kwargs

        node_properties = _get_properties(property_config.node_properties)
        edge_properties = _get_properties(property_config.edge_properties)
        system_properties = _get_properties(property_config.graph_properties)
        return {
            "node": node_properties,
            "edge": edge_properties,
            "graph": system_properties,
        }

    def _add_extra_feats_and_targets(
        self,
        atom_graph: AtomGraphs,
        extra_feats: Dict[str, Dict],
        extra_targets: Dict[str, Dict],
    ):
        """Add extra features and targets to the AtomGraphs object.

        Args:
            atom_graph: AtomGraphs object to add extra features and targets to.
            extra_feats: Dictionary of extra features with keys
            extra_targets: Dictionary of extra targets to add.
        """
        node_feats = {**atom_graph.node_features, **extra_feats["node"]}
        edge_feats = {**atom_graph.edge_features, **extra_feats["edge"]}

        system_feats = (
            atom_graph.system_features if atom_graph.system_features is not None else {}
        )
        system_feats = {**system_feats, **extra_feats["graph"]}

        node_targets = (
            atom_graph.node_targets if atom_graph.node_targets is not None else {}
        )
        node_targets = {**node_targets, **extra_targets["node"]}

        edge_targets = (
            atom_graph.edge_targets if atom_graph.edge_targets is not None else {}
        )
        edge_targets = {**edge_targets, **extra_targets["edge"]}

        system_targets = (
            atom_graph.system_targets if atom_graph.system_targets is not None else {}
        )
        system_targets = {**system_targets, **extra_targets["graph"]}

        return atom_graph._replace(
            node_features=node_feats,
            edge_features=edge_feats,
            system_features=system_feats,
            node_targets=node_targets if node_targets != {} else None,
            edge_targets=edge_targets if edge_targets != {} else None,
            system_targets=system_targets if system_targets != {} else None,
        )


def get_dataset(
    path: Union[str, Path],
    name: str,
    system_config: atomic_system.SystemConfig,
    target_config: atomic_system.PropertyConfig,
    evaluation: Literal["eval_with_noise", "eval_no_noise", "train"] = "train",
) -> AseSqliteDataset:
    """Dataset factory function."""
    return AseSqliteDataset(
        path=path,
        name=name,
        system_config=system_config,
        target_config=target_config,
        evaluation=evaluation,
    )


def random_rotations_with_properties(
    atoms: ase.Atoms, properties: dict
) -> Tuple[ase.Atoms, dict]:
    """Randomly rotate atoms in ase.Atoms object.

    This exists to handle the case where we also need to rotate properties.
    Currently we only ever do this for random rotations, but it could be extended.

    Args:
        atoms (ase.Atoms): Atoms object to rotate.
        properties (dict): Dictionary of properties to rotate.
    """
    rand_rotation = o3.rand_matrix(1)[0].numpy()
    atoms.positions = atoms.positions @ rand_rotation
    if atoms.cell is not None:
        atoms.set_cell(atoms.cell.array @ rand_rotation)

    new_node_properties = {}
    for key, v in properties["node"].items():
        if tuple(v.shape) == tuple(atoms.positions.shape):
            new_node_properties[key] = v @ rand_rotation
        else:
            new_node_properties[key] = v
    properties["node"] = new_node_properties

    if "stress" in properties["graph"]:
        # Transformation rule of stress tensor
        stress = properties["graph"]["stress"]
        full_stress = voigt_6_to_full_3x3_stress(stress)

        # The featurization code adds a batch dimension, so we need to reshape
        if full_stress.shape != (3, 3):
            full_stress = full_stress.reshape(3, 3)

        transformed = np.dot(np.dot(rand_rotation, full_stress), rand_rotation.T)
        # Back to voigt notation, and shape (1, 6) for consistency with batching
        properties["graph"]["stress"] = torch.tensor(
            [
                transformed[0, 0],
                transformed[1, 1],
                transformed[2, 2],
                transformed[1, 2],
                transformed[0, 2],
                transformed[0, 1],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

    return atoms, properties
