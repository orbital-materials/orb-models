import ase.io
import numpy as np
import pytest
import torch
from ase import Atoms

from orb_models.common.atoms.featurization import rotation_from_generator
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from tests.common.atoms import common_adapter_tests

try:
    import torch_sim as ts

    _TORCH_SIM_AVAILABLE = True
except ImportError:
    ts = None  # type: ignore[assignment]
    _TORCH_SIM_AVAILABLE = False

requires_torch_sim = pytest.mark.skipif(
    not _TORCH_SIM_AVAILABLE,
    reason="torch_sim is required for this test",
)


@pytest.fixture
def AFI_atoms(shared_fixtures_path):
    return ase.Atoms(ase.io.read(shared_fixtures_path / "structures" / "AFI.cif"))


def test_basic_adapter_functionality(AFI_atoms):
    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=20)
    common_adapter_tests.adapter_is_compatible_with(adapter)
    common_adapter_tests.adapter_wraps(adapter, AFI_atoms)
    common_adapter_tests.adapter_raises_on_mixed_pbc(adapter, AFI_atoms)
    common_adapter_tests.atoms_to_atom_graphs_invertibility(adapter, AFI_atoms)
    common_adapter_tests.adapter_debatches(adapter, AFI_atoms)


@pytest.mark.parametrize("max_num_neighbors", [10, None])
def test_forcefield_adapter_edge_reg(max_num_neighbors):
    """
    Test that when max_num_neighbors is None then adapter.max_num_neighbors is used.
    """
    atoms = ase.Atoms(
        "C" * 20,
        positions=np.random.rand(20, 3) * 10,
        cell=np.eye(3) * 10,
        pbc=True,
    )
    adapter = ForcefieldAtomsAdapter(
        radius=6.0,
        max_num_neighbors=20,
        min_num_neighbors=2,
        max_num_neighbors_alpha=2.0,
    )
    atom_graphs = adapter.from_ase_atoms(atoms, max_num_neighbors=max_num_neighbors)
    if max_num_neighbors is not None:
        assert atom_graphs.max_num_neighbors.item() == max_num_neighbors
    else:
        assert atom_graphs.max_num_neighbors.item() == adapter.max_num_neighbors


def test_forcefield_adapter_equigrad():
    n = 100
    generator = 10 * torch.rand(size=(n, 3, 3), requires_grad=True)
    rotations = rotation_from_generator(generator)
    np.testing.assert_allclose(  # check orthonormality
        torch.matmul(rotations, torch.transpose(rotations, dim0=1, dim1=2)).detach().numpy(),
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
    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=100)

    graphs = []
    graphs_ = []
    for atoms in [periodic, molecule]:
        g = adapter.from_ase_atoms(atoms)
        vectors, _, generator = g.compute_differentiable_edge_vectors()
        g.edge_features["vectors"] = vectors
        g.system_features["generator"] = generator
        graphs.append(g)
        graphs_.append(adapter.from_ase_atoms(atoms))

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


def test_forcefield_adapter_parses_spin_and_charge():
    """Test that ForcefieldAtomsAdapter correctly parses spin and charge from atoms.info."""
    atoms = Atoms(
        "H2O",
        positions=np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]),
        pbc=True,
        cell=np.diag([5, 5, 5]),
    )
    atoms.info["charge"] = 1
    atoms.info["spin"] = 2

    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=100)
    graph = adapter.from_ase_atoms(atoms)

    assert "total_charge" in graph.system_features
    assert "spin_multiplicity" in graph.system_features
    assert graph.system_features["total_charge"].item() == 1.0
    assert graph.system_features["spin_multiplicity"].item() == 2.0


def test_forcefield_adapter_no_spin_charge_when_absent():
    """Test that total_charge and spin_multiplicity are not present when not specified."""
    atoms = Atoms(
        "H2O",
        positions=np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]),
        pbc=True,
        cell=np.diag([5, 5, 5]),
    )

    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=100)
    graph = adapter.from_ase_atoms(atoms)

    assert "total_charge" not in graph.system_features
    assert "spin_multiplicity" not in graph.system_features


def test_forcefield_adapter_requires_both_spin_and_charge():
    """Test that adapter raises when only one of charge/spin is provided."""
    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=100)

    atoms_charge_only = Atoms(
        "H2O",
        positions=np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]),
        pbc=True,
        cell=np.diag([5, 5, 5]),
    )
    atoms_charge_only.info["charge"] = 1

    with pytest.raises(AssertionError, match="Charge and spin must be present together"):
        adapter.from_ase_atoms(atoms_charge_only)

    atoms_spin_only = Atoms(
        "H2O",
        positions=np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]),
        pbc=True,
        cell=np.diag([5, 5, 5]),
    )
    atoms_spin_only.info["spin"] = 2

    with pytest.raises(AssertionError, match="Charge and spin must be present together"):
        adapter.from_ase_atoms(atoms_spin_only)


def test_from_ase_atoms_list_parallel_equivalence():
    """Test that from_ase_atoms_list produces equivalent results to sequential processing."""
    atoms_list = [
        Atoms("H2O", positions=np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]) + i * 0.1, pbc=True, cell=np.diag([5, 5, 5]))
        for i in range(4)
    ]
    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=20)

    batch_result = adapter.from_ase_atoms_list(atoms_list)
    from orb_models.common.atoms.batch.graph_batch import AtomGraphs

    sequential_batch = AtomGraphs.batch([adapter.from_ase_atoms(a) for a in atoms_list])

    assert torch.equal(batch_result.n_node.cpu(), sequential_batch.n_node.cpu())
    assert torch.equal(batch_result.n_edge.cpu(), sequential_batch.n_edge.cpu())
    assert torch.allclose(batch_result.positions.cpu(), sequential_batch.positions.cpu(), atol=1e-5)
    assert torch.equal(
        batch_result.node_features["atomic_numbers"].cpu(),
        sequential_batch.node_features["atomic_numbers"].cpu(),
    )
    # Edge ordering may differ between batched and sequential, so compare sorted distances
    for sys_idx in range(len(atoms_list)):
        start = batch_result.n_edge[:sys_idx].sum().item() if sys_idx > 0 else 0
        end = start + batch_result.n_edge[sys_idx].item()
        bd = batch_result.edge_features["vectors"][start:end].cpu().norm(dim=1).sort()[0]
        sd = sequential_batch.edge_features["vectors"][start:end].cpu().norm(dim=1).sort()[0]
        assert torch.allclose(bd, sd, atol=1e-4)


def test_from_ase_atoms_list_nonperiodic():
    """Test from_ase_atoms_list with non-periodic systems."""
    atoms_list = [
        Atoms("H2", positions=np.array([[0, 0, 0], [0, 0.74, 0]]) + i * 0.1, pbc=False)
        for i in range(3)
    ]
    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=20)
    batch = adapter.from_ase_atoms_list(atoms_list)

    assert batch.n_node.tolist() == [2, 2, 2]
    assert batch.positions.shape == (6, 3)


def test_from_ase_atoms_list_single_atom_fallback():
    """Test that from_ase_atoms_list falls back to from_ase_atoms for a single atom."""
    atoms = Atoms("H2O", positions=np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]), pbc=True, cell=np.diag([5, 5, 5]))
    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=20)

    single_result = adapter.from_ase_atoms_list([atoms])
    direct_result = adapter.from_ase_atoms(atoms)

    assert torch.equal(single_result.n_node.cpu(), direct_result.n_node.cpu())
    assert torch.allclose(single_result.positions.cpu(), direct_result.positions.cpu(), atol=1e-6)


def test_from_ase_atoms_list_empty_raises():
    """Test that from_ase_atoms_list raises ValueError for an empty list."""
    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=20)
    with pytest.raises(ValueError, match="atoms list must not be empty"):
        adapter.from_ase_atoms_list([])


def test_from_ase_atoms_list_with_charge_and_spin():
    """Test that from_ase_atoms_list correctly handles charge and spin."""
    atoms_list = []
    for i in range(3):
        a = Atoms("H2O", positions=np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]) + i * 0.1, pbc=True, cell=np.diag([5, 5, 5]))
        a.info["charge"] = float(i)
        a.info["spin"] = float(i + 1)
        atoms_list.append(a)

    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=20)
    batch = adapter.from_ase_atoms_list(atoms_list)

    assert "total_charge" in batch.system_features
    assert "spin_multiplicity" in batch.system_features
    torch.testing.assert_close(
        batch.system_features["total_charge"].cpu(),
        torch.tensor([0.0, 1.0, 2.0]),
    )
    torch.testing.assert_close(
        batch.system_features["spin_multiplicity"].cpu(),
        torch.tensor([1.0, 2.0, 3.0]),
    )


@requires_torch_sim
def test_forcefield_adapter_parses_spin_and_charge_from_simstate():
    """Test that ForcefieldAtomsAdapter correctly parses spin and charge from SimState."""
    state = ts.SimState(
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        masses=torch.tensor([1.0, 1.0, 16.0]),
        cell=torch.diag(torch.tensor([5.0, 5.0, 5.0])).unsqueeze(0),
        pbc=True,
        atomic_numbers=torch.tensor([1, 1, 8]),
        charge=torch.tensor([1.0]),
        spin=torch.tensor([2.0]),
    )

    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=100)
    graph = adapter.from_torchsim_state(state)

    assert "total_charge" in graph.system_features
    assert "spin_multiplicity" in graph.system_features
    assert graph.system_features["total_charge"].item() == 1.0
    assert graph.system_features["spin_multiplicity"].item() == 2.0


@requires_torch_sim
def test_forcefield_adapter_parses_spin_and_charge_from_batched_simstate():
    """Test that ForcefieldAtomsAdapter correctly parses spin and charge from batched SimState."""
    # Two systems: first has 3 atoms (H2O), second has 2 atoms (H2)
    state = ts.SimState(
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],  # System 0: H2O
                [0.0, 0.0, 0.0],
                [0.0, 0.74, 0.0],  # System 1: H2
            ]
        ),
        masses=torch.tensor([1.0, 1.0, 16.0, 1.0, 1.0]),
        cell=torch.stack(
            [
                torch.diag(torch.tensor([5.0, 5.0, 5.0])),
                torch.diag(torch.tensor([5.0, 5.0, 5.0])),
            ]
        ),
        pbc=True,
        atomic_numbers=torch.tensor([1, 1, 8, 1, 1]),
        system_idx=torch.tensor([0, 0, 0, 1, 1]),
        charge=torch.tensor([1.0, -1.0]),
        spin=torch.tensor([2.0, 0.0]),
    )

    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=100)
    graph = adapter.from_torchsim_state(state)

    assert "total_charge" in graph.system_features
    assert "spin_multiplicity" in graph.system_features
    torch.testing.assert_close(graph.system_features["total_charge"], torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(graph.system_features["spin_multiplicity"], torch.tensor([2.0, 0.0]))
