"""
This file does not actually run any tests. Instead, it defines generic
tests that can apply to any adapter, which are then called by each
adapter inside e.g. 'tests/latent_diffusion/latent_diffusion_adapter`.
"""

import copy

import numpy as np
import pytest
import torch

from orb_models.common.atoms.batch.graph_batch import AtomGraphs


def adapter_is_compatible_with(adapter):
    adapter_2 = copy.deepcopy(adapter)

    # Modify config2 to make it incompatible
    adapter_2.radius = 100

    with pytest.raises(ValueError):
        adapter.is_compatible_with(adapter_2)

    # Reset config2 to be identical to config1
    adapter_2.radius = adapter.radius
    assert adapter.is_compatible_with(adapter_2)


def adapter_wraps(adapter, atoms):
    atoms_unwrapped = atoms.copy()
    atoms_unwrapped.positions[:10] += 2.0 * atoms_unwrapped.cell.array.max()
    atoms_wrapped = atoms_unwrapped.copy()
    atoms_wrapped.wrap()
    assert not np.allclose(atoms_wrapped.positions, atoms_unwrapped.positions)

    atom_graphs = adapter.from_ase_atoms(atoms_unwrapped, wrap=False)
    assert np.allclose(atom_graphs.positions.numpy(), atoms_unwrapped.positions)

    # Note: this test is slightly indirect. We can't test that wrap=True yields the same
    # results as ase's .wrap(), because of slight numerical differences at the boundaries.
    # Instead, we test that wrap=True for an unwrapped system yields the same results
    # as wrap=True for an ase-wrapped system.
    atom_graphs1 = adapter.from_ase_atoms(atoms_unwrapped, wrap=True)
    atom_graphs2 = adapter.from_ase_atoms(atoms_wrapped, wrap=True)
    assert torch.allclose(atom_graphs1.positions, atom_graphs2.positions)


def adapter_raises_on_mixed_pbc(adapter, atoms):
    atoms = atoms.copy()
    atoms.set_pbc([True, False, True])
    with pytest.raises(NotImplementedError):
        adapter.from_ase_atoms(atoms)

    atoms.set_pbc([True, True, True])
    adapter.from_ase_atoms(atoms)


def atoms_to_atom_graphs_invertibility(adapter, atoms):
    atoms = atoms.copy()
    atom_graphs = adapter.from_ase_atoms(atoms, wrap=False)
    recovered_atoms = atom_graphs.to_ase_atoms()[0]

    assert np.allclose(recovered_atoms.positions, atoms.positions)
    assert np.allclose(recovered_atoms.cell, atoms.cell)
    assert (recovered_atoms.numbers == atoms.numbers).all()


def adapter_debatches(adapter, atoms):
    atoms = atoms.copy()
    graphs = [adapter.from_ase_atoms(atoms) for _ in range(4)]
    batch = AtomGraphs.batch(graphs)
    atoms_list = batch.to_ase_atoms()
    assert len(atoms_list) == 4
    assert (atoms_list[0].positions == atoms_list[1].positions).all()
    assert (atoms_list[0].get_tags() == atoms_list[1].get_tags()).all()
