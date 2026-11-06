import numpy as np
import pytest
import torch
from ase import Atoms

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.models.forcefield_utils import (
    compute_gradient_forces_and_stress,
    torch_voigt_6_to_full_3x3_stress,
)


def test_batch_voigt_conversion():
    from ase.stress import voigt_6_to_full_3x3_stress

    n = 10
    stress_voigt = torch.rand(n, 6, dtype=torch.double)

    stress_matrix = torch_voigt_6_to_full_3x3_stress(stress_voigt)
    for i in range(n):
        assert np.allclose(
            stress_matrix[i].numpy(),
            voigt_6_to_full_3x3_stress(stress_voigt[i].numpy()),
        )


@pytest.mark.parametrize("periodic", [False, True])
def test_stress_loss_preserves_force_gradients_for_molecules(periodic):
    distances = torch.tensor([0.7, 0.9], dtype=torch.float64)
    atoms = [
        Atoms(
            "H2",
            positions=[[0, 0, 0], [distance, 0, 0]],
            cell=[4, 4, 4] if pbc else None,
            pbc=pbc,
        )
        for distance, pbc in zip(distances.tolist(), [False, periodic], strict=True)
    ]
    adapter = ForcefieldAtomsAdapter(radius=2.0, max_num_neighbors=20)
    batch = AtomGraphs.batch(
        [adapter.from_ase_atoms(a, device="cpu", output_dtype=torch.float64) for a in atoms]
    )
    geom = batch.compute_differentiable_edge_vectors()
    stiffness = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    energy = stiffness * geom.edge_vectors.square().sum(-1).reshape(2, -1).sum(-1)
    forces, stress, _ = compute_gradient_forces_and_stress(
        energy,
        batch.positions,
        geom.stress_displacement,
        geom.rotation_generator,
        batch.system_features["cell"],
        training=True,
        compute_stress=True,
    )
    loss = forces.square().sum() + stress.square().sum()
    gradient = torch.autograd.grad(loss, stiffness)[0]

    expected_stress = torch.zeros(2, 6, dtype=torch.float64)
    if periodic:
        expected_stress[1, 0] = 4 * distances[1].square() / 64
    torch.testing.assert_close(stress, expected_stress)
    # Each bond is represented by two directed edges, so F = +/-4*k*r.
    expected_gradient = 64 * distances.square().sum() + 2 * expected_stress.square().sum()
    torch.testing.assert_close(gradient, expected_gradient)
