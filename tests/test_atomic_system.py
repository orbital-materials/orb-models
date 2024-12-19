import ase.io
import numpy as np
import torch
from orb_models.forcefield.base import batch_graphs
from orb_models.forcefield.atomic_system import (
    atom_graphs_to_ase_atoms,
    ase_atoms_to_atom_graphs,
)


def test_atoms_to_atom_graphs_invertibility(fixtures_path):
    atoms = ase.Atoms(ase.io.read(fixtures_path / "AFI.cif"))

    atom_graphs = ase_atoms_to_atom_graphs(atoms, wrap=False)
    recovered_atoms = atom_graphs_to_ase_atoms(atom_graphs)[0]

    assert np.allclose(recovered_atoms.positions, atoms.positions)
    assert np.allclose(recovered_atoms.cell, atoms.cell)
    assert (recovered_atoms.numbers == atoms.numbers).all()


def test_atom_graphs_to_ase_atoms_debatches(fixtures_path):
    atoms = ase.Atoms(ase.io.read(fixtures_path / "AFI.cif"))
    graphs = [ase_atoms_to_atom_graphs(atoms, wrap=False) for _ in range(4)]
    batch = batch_graphs(graphs)
    atoms_list = atom_graphs_to_ase_atoms(batch)
    assert len(atoms_list) == 4
    assert (atoms_list[0].positions == atoms_list[1].positions).all()
    assert (atoms_list[0].get_tags() == atoms_list[1].get_tags()).all()


def test_ase_atoms_to_atom_graphs_wraps(fixtures_path):
    atoms_unwrapped = ase.Atoms(ase.io.read(fixtures_path / "AFI.cif"))
    atoms_unwrapped.positions[:10] += 2.0 * atoms_unwrapped.cell.array.max()
    atoms_wrapped = atoms_unwrapped.copy()
    atoms_wrapped.wrap()
    assert not np.allclose(atoms_wrapped.positions, atoms_unwrapped.positions)

    atom_graphs = ase_atoms_to_atom_graphs(atoms_unwrapped, wrap=False)
    assert np.allclose(atom_graphs.positions.numpy(), atoms_unwrapped.positions)

    # Note: this test is slightly indirect. We can't test that wrap=True yields the same
    # results as ase's .wrap(), because of slight numerical differences at the boundaries.
    # Instead, we test that wrap=True for an unwrapped system yields the same results
    # as wrap=True for an ase-wrapped system.
    atom_graphs1 = ase_atoms_to_atom_graphs(atoms_unwrapped, wrap=True)
    atom_graphs2 = ase_atoms_to_atom_graphs(atoms_wrapped, wrap=True)
    assert torch.allclose(atom_graphs1.positions, atom_graphs2.positions)
