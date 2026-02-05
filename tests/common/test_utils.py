import ase
import pytest

from orb_models.common.utils import is_periodic


def test_is_periodic_with_periodic_atoms():
    """Test is_periodic with atoms that have both cell and pbc."""
    # full periodicity
    atoms = ase.Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms.cell = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    atoms.pbc = [True, True, True]
    assert is_periodic(atoms)

    # Partial periodicity
    atoms = ase.Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms.cell = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    atoms.pbc = [True, True, False]
    assert is_periodic(atoms)


def test_is_periodic_with_non_periodic_atoms():
    """Test is_periodic with atoms that have neither cell nor pbc."""
    # no cell or pbc by default
    atoms = ase.Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    assert not is_periodic(atoms)

    # explicitly set zero cell and no pbc
    atoms = ase.Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms.cell = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    atoms.pbc = [False, False, False]
    assert not is_periodic(atoms)

    # Molecules can have non-zero cells, and are still counted as
    # 'non-periodic' because their pbc attribute is False
    atoms = ase.Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms.cell = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    atoms.pbc = [False, False, False]
    assert not is_periodic(atoms)


def test_is_periodic_inconsistency_raises_error():
    """Test that inconsistent cell and pbc raises ValueError."""
    atoms = ase.Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])

    # Set cell to zeros but pbc to True (inconsistent)
    atoms.cell = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    atoms.pbc = [True, True, True]
    with pytest.raises(ValueError):
        is_periodic(atoms)
