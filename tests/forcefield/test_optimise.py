import ase
import numpy as np

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.inference.calculator import ORBCalculator

adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=20)


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
    graphs = [adapter.from_ase_atoms(a) for a in atoms_list]
    return AtomGraphs.batch(graphs)


def test_orb_calculator(euclidean_norm):
    a = atoms()
    device = "cpu"
    a.calc = ORBCalculator(euclidean_norm, atoms_adapter=adapter, device=device)
    # energy and forces of random initial position should be non-zero
    assert a.get_potential_energy() > 1e-5
    assert np.any(np.abs(a.get_forces()) > 1e-5)

    # energy and force of atoms at globl min should be zero
    minimum = list(euclidean_norm.minimum.numpy())
    a.positions = np.array([minimum] * len(a))
    assert np.abs(a.get_potential_energy()) < 1e-5
    assert np.all(np.abs(a.get_forces()) < 1e-5)
    assert np.all(np.abs(a.get_stress()) < 1e-5)
