import numpy as np
import torch
from ase import Atoms
from ase.build import bulk

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.models import segment_ops
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


def test_stress_gradients_are_finite_for_a_batch_mixing_a_molecule_and_a_crystal():
    """The molecule's zero-volume stress is masked to 0, but must not backpropagate 0 * (1 / 0) = NaN."""
    molecule = Atoms("H2O", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.93, 0.0, -0.24]])
    adapter = ForcefieldAtomsAdapter(radius=3.0, max_num_neighbors=20)
    batch = AtomGraphs.batch(
        [adapter.from_ase_atoms(a, device="cpu") for a in (molecule, bulk("Si"))]
    )
    geometry = batch.compute_differentiable_edge_vectors()
    scale = torch.tensor(1.0, requires_grad=True)
    edge_energy = scale * geometry.edge_vectors.norm(dim=-1) ** 2
    energy = segment_ops.aggregate_nodes(edge_energy, batch.n_edge, reduction="sum")

    _, stress, _ = compute_gradient_forces_and_stress(
        energy=energy,
        positions=batch.node_features["positions"],
        displacement=geometry.stress_displacement,
        generator=geometry.rotation_generator,
        cell=batch.system_features["cell"],
        training=True,
        compute_stress=True,
    )
    stress.square().sum().backward()

    assert torch.equal(stress[0], torch.zeros(6))
    assert stress[1].abs().max() > 0
    assert torch.isfinite(scale.grad) and scale.grad != 0
