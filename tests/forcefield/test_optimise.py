import numpy as np
import ase

from orb_models.forcefield.atomic_system import SystemConfig, ase_atoms_to_atom_graphs
from orb_models.forcefield import base
from orb_models.forcefield.calculator import ORBCalculator


system_config = SystemConfig(radius=6.0, max_num_neighbors=20)


def atoms(unit_cell=False):
    nodes = 10
    positions = np.random.randn(nodes, 3)
    atomic_numbers = np.arange(0, nodes)
    atoms = ase.Atoms(
        numbers=atomic_numbers,
        positions=positions,
        cell=np.eye(3) if unit_cell else np.zeros((3, 3)),
        pbc=unit_cell,
    )
    return atoms


def batch(unit_cell=False):
    atoms_list = [atoms(unit_cell), atoms(unit_cell)]
    graphs = [ase_atoms_to_atom_graphs(a, system_config) for a in atoms_list]
    return base.batch_graphs(graphs)


def test_orb_calculator(euclidean_norm):
    a = atoms()
    a.calc = ORBCalculator(euclidean_norm, system_config=system_config)  # type: ignore
    # energy and forces of random initial position should be non-zero
    assert a.get_potential_energy() > 1e-5
    assert np.any(np.abs(a.get_forces()) > 1e-5)

    # energy and force of atoms at globl min should be zero
    minimum = list(euclidean_norm.minimum.numpy())
    a.positions = np.array([minimum] * len(a))
    assert np.abs(a.get_potential_energy()) < 1e-5
    assert np.all(np.abs(a.get_forces()) < 1e-5)
    assert np.all(np.abs(a.get_stress()) < 1e-5)
