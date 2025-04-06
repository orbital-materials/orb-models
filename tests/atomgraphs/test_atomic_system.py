import copy
import pytest

from ase import Atoms
import ase.io
import numpy as np
import torch

from orb_models.forcefield.featurization_utilities import rotation_from_generator
from orb_models.forcefield.base import batch_graphs
from orb_models.forcefield import atomic_system
from orb_models.forcefield.atomic_system import (
    atom_graphs_to_ase_atoms,
    ase_atoms_to_atom_graphs,
    SystemConfig,
)


@pytest.fixture
def system_config():
    return atomic_system.SystemConfig(
        radius=6.0,
        max_num_neighbors=20,
    )


def test_ase_atoms_to_atom_graphs_wraps(fixtures_path, system_config):
    atoms_unwrapped = ase.Atoms(ase.io.read(fixtures_path / "AFI.cif"))
    atoms_unwrapped.positions[:10] += 2.0 * atoms_unwrapped.cell.array.max()
    atoms_wrapped = atoms_unwrapped.copy()
    atoms_wrapped.wrap()
    assert not np.allclose(atoms_wrapped.positions, atoms_unwrapped.positions)

    atom_graphs = ase_atoms_to_atom_graphs(
        atoms_unwrapped, system_config=system_config, wrap=False
    )
    assert np.allclose(atom_graphs.positions.numpy(), atoms_unwrapped.positions)

    # Note: this test is slightly indirect. We can't test that wrap=True yields the same
    # results as ase's .wrap(), because of slight numerical differences at the boundaries.
    # Instead, we test that wrap=True for an unwrapped system yields the same results
    # as wrap=True for an ase-wrapped system.
    atom_graphs1 = ase_atoms_to_atom_graphs(
        atoms_unwrapped, system_config=system_config, wrap=True
    )
    atom_graphs2 = ase_atoms_to_atom_graphs(
        atoms_wrapped, system_config=system_config, wrap=True
    )
    assert torch.allclose(atom_graphs1.positions, atom_graphs2.positions)


def test_ase_atoms_to_atom_graphs_raises_on_mixed_pbc(fixtures_path, system_config):
    atoms = ase.Atoms(ase.io.read(fixtures_path / "AFI.cif"))
    atoms.set_pbc([True, False, True])

    with pytest.raises(NotImplementedError):
        ase_atoms_to_atom_graphs(atoms, system_config=system_config)

    atoms.set_pbc([True, True, True])
    ase_atoms_to_atom_graphs(atoms, system_config=system_config)


def test_atoms_to_atom_graphs_invertibility(fixtures_path, system_config):
    atoms = ase.Atoms(ase.io.read(fixtures_path / "AFI.cif"))

    atom_graphs = ase_atoms_to_atom_graphs(
        atoms, system_config=system_config, wrap=False
    )
    recovered_atoms = atom_graphs_to_ase_atoms(atom_graphs)[0]

    assert np.allclose(recovered_atoms.positions, atoms.positions)
    assert np.allclose(recovered_atoms.cell, atoms.cell)
    assert (recovered_atoms.numbers == atoms.numbers).all()


def test_atom_graphs_to_ase_atoms_debatches(fixtures_path, system_config):
    atoms = ase.Atoms(ase.io.read(fixtures_path / "AFI.cif"))

    graphs = [
        ase_atoms_to_atom_graphs(atoms, system_config=system_config) for _ in range(4)
    ]
    batch = batch_graphs(graphs)
    atoms_list = atom_graphs_to_ase_atoms(batch)
    assert len(atoms_list) == 4
    assert (atoms_list[0].positions == atoms_list[1].positions).all()
    assert (atoms_list[0].get_tags() == atoms_list[1].get_tags()).all()


def test_input_rotation():
    n = 100
    generator = 10 * torch.rand(size=(n, 3, 3), requires_grad=True)
    rotations = rotation_from_generator(generator)
    np.testing.assert_allclose(  # check orthonormality
        torch.matmul(rotations, torch.transpose(rotations, dim0=1, dim1=2))
        .detach()
        .numpy(),
        torch.eye(3).repeat(n, 1, 1).numpy(),
        atol=1e-5,
    )

    periodic = Atoms(
        "H2O",
        positions=np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]),
        pbc=True,
        cell=np.diag([5, 5, 5]),
    )
    molecule = Atoms(
        "H2O",
        positions=np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]),
        pbc=False,
    )
    system_config = SystemConfig(radius=6.0, max_num_neighbors=100)

    graphs = []
    graphs_ = []
    for atoms in [periodic, molecule]:
        g = ase_atoms_to_atom_graphs(atoms, system_config=system_config)
        vectors, _, generator = g.compute_differentiable_edge_vectors()
        g.edge_features["vectors"] = vectors
        g.system_features["generator"] = generator
        graphs.append(g)
        graphs_.append(ase_atoms_to_atom_graphs(atoms, system_config=system_config))

    for i in range(2):  # periodic and molecule
        for k, v in graphs[i].node_features.items():
            if type(v) is torch.Tensor:
                np.testing.assert_allclose(
                    graphs[i].node_features[k].detach().numpy(),
                    graphs_[i].node_features[k].detach().numpy(),
                )
        for k, v in graphs[i].edge_features.items():
            if type(v) is torch.Tensor:
                np.testing.assert_allclose(
                    graphs[i].edge_features[k].detach().numpy(),
                    graphs_[i].edge_features[k].detach().numpy(),
                )

    def predict_invariant(graph):
        return torch.linalg.norm(graph.edge_features["vectors"], dim=1)

    def predict_covariant(graph):
        return torch.sum(graph.edge_features["vectors"][:, 0] ** 2)

    for i, graph in enumerate(graphs):
        invariant = predict_invariant(graph)
        gradient = torch.autograd.grad(
            outputs=[invariant],
            inputs=[graph.system_features["generator"]],
            grad_outputs=torch.ones_like(invariant),
            retain_graph=True,
        )[0]
        np.testing.assert_allclose(gradient.numpy(), 0.0, atol=1e-5)
        covariant = predict_covariant(graph)
        gradient = torch.autograd.grad(
            outputs=covariant,
            inputs=graph.system_features["generator"],
        )[0]
        assert torch.linalg.norm(gradient) > 0.0
