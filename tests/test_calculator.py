import ase
import numpy as np
import pytest
import torch
import torch.nn as nn

from orb_models.forcefield import segment_ops
from orb_models.forcefield.calculator import ORBCalculator


class EuclideanNormModel(nn.Module):
    def __init__(self, minimum=[-0.5, -2.0, -1.0]):
        super(EuclideanNormModel, self).__init__()
        self.minimum = torch.tensor(minimum)
        self.node_head = torch.nn.Linear(3, 1)  # Dummy head
        self.graph_head = torch.nn.Linear(3, 1)  # Dummy head
        self.stress_head = torch.nn.Linear(3, 1)  # Dummy head

    def forward(self, batch):
        positions = batch.positions
        sqnorm = (torch.norm(positions - self.minimum, dim=1)) ** 2
        energies = segment_ops.aggregate_nodes(sqnorm, batch.n_node, reduction="sum")
        neg_grad = 2 * (self.minimum - positions)  # analytical gradient
        stress = torch.zeros(
            (6,),
            dtype=positions.dtype,
            device=positions.device,
        )
        return energies, neg_grad, stress

    def predict(self, batch):
        out = {}
        energy, forces, stress = self.forward(batch)
        out["graph_pred"] = energy
        out["node_pred"] = forces
        out["stress_pred"] = stress
        return out


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


@pytest.mark.parametrize("brute_force_knn", [True, False])
def test_orb_calculator(brute_force_knn):
    minimum = [-0.5, -2.0, -1.0]
    a = atoms()
    a.calc = ORBCalculator(EuclideanNormModel(minimum), brute_force_knn)  # type: ignore
    # energy and forces of random initial position should be non-zero
    assert a.get_potential_energy() > 1e-5
    assert np.any(np.abs(a.get_forces()) > 1e-5)

    # energy and force of atoms at globl min should be zero
    a.positions = np.array([minimum] * len(a))
    assert np.abs(a.get_potential_energy()) < 1e-5
    assert np.all(np.abs(a.get_forces()) < 1e-5)
    assert np.all(np.abs(a.get_stress()) < 1e-5)
