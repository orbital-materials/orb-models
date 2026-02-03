import random
from collections.abc import Mapping
from typing import Any

import ase
import numpy as np
import torch


def replace_prefix_in_keys(dictionary: Mapping[str, Any], old_prefix: str, new_prefix: str) -> None:
    """Mutate dictionary, replacing `old_prefix` with `new_prefix`."""
    for key in list(dictionary.keys()):
        if key.startswith(old_prefix):
            new_key = key.replace(old_prefix, new_prefix, 1)
            assert new_key not in dictionary, f"Key {new_key} already exists."
            dictionary[new_key] = dictionary.pop(key)  # type: ignore


def seed_everything(seed: int, rank: int = 0) -> None:
    """Set the seed for all pseudo random number generators."""
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)


def is_periodic(atoms: ase.Atoms):
    """Check if the atoms object is periodic.

    Args:
        atoms: The atoms object
    Returns:
        True if the atoms object is periodic, False otherwise
    Raises:
        ValueError: If the PBCs are not consistent with the cell
    """
    cell = atoms.cell.array
    pbc = atoms.pbc
    has_cell = np.any(cell != 0)
    has_pbc = np.any(pbc)
    if has_pbc and not has_cell:
        raise ValueError(
            f"Atoms has pbc but cell is zero for {atoms.get_chemical_formula()}. "
            f"cell: {cell}, pbc: {pbc}"
        )
    return has_pbc
