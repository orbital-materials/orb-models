import pytest
import torch
from orb_models.forcefield.graph_regressor import (
    remove_net_torque,
    selectively_remove_net_torque_for_nonpbc_systems,
)


@pytest.fixture
def positions_forces_n_nodes():
    # Single graph with 5 atoms
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [3.0, 2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    # net force is [0, 0, 0]
    forces = torch.tensor(
        [
            [0.2, -0.1, 0.4],
            [-0.2, -0.1, -0.3],
            [0.0, 0.3, -0.2],
            [0.1, -0.2, 0.2],
            [-0.1, 0.1, -0.1],
        ],
        dtype=torch.float32,
    )
    n_nodes = torch.tensor([5], dtype=torch.long)
    return positions, forces, n_nodes


@pytest.fixture
def random_batched_positions_forces_n_nodes():
    num_graphs = 100
    positions_list = []
    forces_list = []
    n_nodes_list = []
    for i in range(num_graphs):
        n = int(torch.randint(4, 200, (1,)))
        p = torch.randn((n, 3), dtype=torch.float32)
        f = torch.randn((n, 3), dtype=torch.float32)
        f -= f.mean(dim=0)
        positions_list.append(p)
        forces_list.append(f)
        n_nodes_list.append(n)

    positions = torch.cat(positions_list)
    forces = torch.cat(forces_list)
    n_nodes = torch.tensor(n_nodes_list, dtype=torch.long)
    return positions, forces, n_nodes


def _net_force_and_torque(positions, forces):
    net_force = forces.sum(dim=0)
    relative_positions = positions - positions.mean(dim=0)
    net_torque = torch.cross(relative_positions, forces, dim=-1).sum(dim=0)
    return net_force, net_torque


def test_adjust_forces_single_graph(positions_forces_n_nodes):
    """
    Test with a single graph to ensure it works for simple cases.
    """
    positions, forces, n_nodes = positions_forces_n_nodes

    # initial net force is zero, net torque nonzero
    net_force, net_torque = _net_force_and_torque(positions, forces)
    assert torch.allclose(net_force, torch.zeros(3), atol=1e-6)
    assert not torch.allclose(net_torque, torch.zeros(3), atol=1e-6)

    adjusted_forces = remove_net_torque(positions, forces, n_nodes)

    # Assert net force is still zero and net torque is now also zero
    net_force, net_torque = _net_force_and_torque(positions, adjusted_forces)
    assert torch.allclose(net_force, torch.zeros(3), atol=1e-6)
    assert torch.allclose(net_torque, torch.zeros(3), atol=1e-6)


def test_adjust_forces_multiple_graphs(positions_forces_n_nodes):
    """
    Test with multiple graphs to ensure it works for batched inputs.
    """
    # create a 2 graph batch
    positions1, forces1, n_nodes = positions_forces_n_nodes
    perm = torch.randperm(len(positions1))
    positions2 = positions1[perm]
    forces2 = forces1[perm]
    positions = torch.cat([positions1, positions2])
    forces = torch.cat([forces1, forces2])
    n_nodes = torch.cat([n_nodes, n_nodes])
    batch_indices = torch.repeat_interleave(torch.arange(2), n_nodes)

    # net force is zero, net torque is *not* zero
    for i in range(2):
        idx = batch_indices == i
        net_force = forces[idx].sum(dim=0)
        relative_positions = positions[idx] - positions[idx].mean(dim=0)
        net_torque = torch.cross(relative_positions, forces[idx], dim=-1).sum(dim=0)
        assert torch.allclose(net_force, torch.zeros(3), atol=1e-6)
        assert not torch.allclose(net_torque, torch.zeros(3), atol=1e-6)

    adjusted_forces = remove_net_torque(positions, forces, n_nodes)

    # net force is still zero, net torque is now zero
    for i in range(2):
        idx = batch_indices == i
        net_force = adjusted_forces[idx].sum(dim=0)
        relative_positions = positions[idx] - positions[idx].mean(dim=0)
        net_torque = torch.cross(relative_positions, adjusted_forces[idx], dim=-1).sum(dim=0)
        assert torch.allclose(net_force, torch.zeros(3), atol=1e-6)
        assert torch.allclose(net_torque, torch.zeros(3), atol=1e-6)


def test_adjust_forces_random_graphs(random_batched_positions_forces_n_nodes):
    """
    Test with random graphs to ensure it works for any configuration.
    """
    positions, forces, n_nodes = random_batched_positions_forces_n_nodes
    num_graphs = n_nodes.size(0)
    batch_indices = torch.repeat_interleave(torch.arange(num_graphs), n_nodes)

    adjusted_forces = remove_net_torque(positions, forces, n_nodes)

    # Check net force and net torque per graph
    for i in range(num_graphs):
        idx = batch_indices == i
        net_force, net_torque = _net_force_and_torque(
            positions[idx], adjusted_forces[idx]
        )
        assert torch.allclose(net_force, torch.zeros(3), atol=1e-4)
        # higher atol of 1e-4 required here, but that's still fine
        assert torch.allclose(net_torque, torch.zeros(3), atol=1e-4)


def test_adjust_forces_zero_nodes():
    """
    Test with zero nodes to ensure it handles empty graphs.
    """
    positions = torch.empty((0, 3), dtype=torch.float32)
    forces = torch.empty((0, 3), dtype=torch.float32)
    n_nodes = torch.tensor([0], dtype=torch.long)

    adjusted_forces = remove_net_torque(positions, forces, n_nodes)

    # Adjusted forces should be empty
    assert adjusted_forces.shape == (0, 3)


def test_adjust_forces_singular_matrix():
    """
    Test with a configuration that may lead to a singular matrix M.
    """
    positions = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    forces = torch.randn((4, 3), dtype=torch.float32)
    forces -= forces.mean(dim=0)
    n_nodes = torch.tensor([4], dtype=torch.long)

    adjusted_forces = remove_net_torque(positions, forces, n_nodes)

    # Even if M is singular, the function should handle it
    net_force, net_torque = _net_force_and_torque(positions, adjusted_forces)
    assert torch.allclose(net_force, torch.zeros(3), atol=1e-6)
    assert torch.allclose(net_torque, torch.zeros(3), atol=1e-6)


def test_adjust_forces_no_adjustment_needed():
    """
    Test when the net force and net torque are already zero.
    """
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    forces = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    n_nodes = torch.tensor([2], dtype=torch.long)

    adjusted_forces = remove_net_torque(positions, forces, n_nodes)

    # Forces should remain unchanged
    assert torch.allclose(adjusted_forces, forces, atol=1e-6)


@pytest.mark.parametrize("remove_mean", [True, False])
def test_adjust_forces_large_values(remove_mean):
    """
    Test with large force values to check numerical stability.
    Test w/wo removing mean to check if it works in both cases.
    """
    positions = torch.randn((5, 3), dtype=torch.float32)
    forces = torch.randn((5, 3), dtype=torch.float32) * 1e3
    if remove_mean:
        forces -= forces.mean(dim=0)
    n_nodes = torch.tensor([5], dtype=torch.long)

    adjusted_forces = remove_net_torque(positions, forces, n_nodes)

    net_force = adjusted_forces.sum(dim=0)
    if remove_mean:
        # net force should still be zero
        assert torch.allclose(net_force, torch.zeros(3), atol=1e-2)
    else:
        # net force should not be zero
        assert not torch.allclose(net_force, torch.zeros(3), atol=1e-2)

    # net torque should be zero
    relative_positions = positions - positions.mean(dim=0)
    net_torque = torch.cross(relative_positions, adjusted_forces, dim=-1).sum(dim=0)
    assert torch.allclose(net_torque, torch.zeros(3), atol=1e-2)


def test_adjust_forces_dtype(positions_forces_n_nodes):
    """
    Test it works with different data types.
    """
    positions, forces, n_nodes = positions_forces_n_nodes
    positions = positions.to(torch.float64)
    forces = forces.to(torch.float64)

    adjusted_forces = remove_net_torque(positions, forces, n_nodes)

    assert adjusted_forces.dtype == torch.float64

    net_force, net_torque = _net_force_and_torque(positions, adjusted_forces)
    assert torch.allclose(net_force, torch.zeros(3, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(net_torque, torch.zeros(3, dtype=torch.float64), atol=1e-6)


@pytest.mark.parametrize("batch_type", ["all_pbc", "no_pbc", "mixed"])
def test_selectively_remove_net_torque_for_nonpbc_systems(
    batch_type, positions_forces_n_nodes
):
    positions, forces, n_nodes = positions_forces_n_nodes
    num_graphs = 2  # Create a batch with 2 graphs for simplicity

    # Duplicate the single graph to create a batch of 2 graphs
    positions_batch = torch.cat([positions, positions], dim=0)
    forces_batch = torch.cat([forces, forces], dim=0)
    n_nodes_batch = torch.cat([n_nodes, n_nodes], dim=0)

    # Define cell tensors based on the batch_type
    if batch_type == "all_pbc":
        # Both graphs have non-zero cells (PBC)
        cell = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            ],
            dtype=torch.float32,
        )
    elif batch_type == "no_pbc":
        # Both graphs have zero cells (non-PBC)
        cell = torch.zeros((2, 3, 3), dtype=torch.float32)
    elif batch_type == "mixed":
        # First graph has PBC, second graph has no PBC
        cell = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )

    # Compute net torques before adjustment
    net_torques_before = []
    start = 0
    for n in n_nodes_batch:
        pos = positions_batch[start : start + n]
        force = forces_batch[start : start + n]
        _, net_tau = _net_force_and_torque(pos, force)
        net_torques_before.append(net_tau)
        start += n
    net_torques_before = torch.stack(net_torques_before)  # type: ignore

    # Apply the function under test
    adjusted_pred = selectively_remove_net_torque_for_nonpbc_systems(
        pred=forces_batch,
        positions=positions_batch,
        cell=cell,
        n_node=n_nodes_batch,
    )

    # Compute net torques after adjustment
    net_torques_after = []
    start = 0
    for n in n_nodes_batch:
        pos = positions_batch[start : start + n]
        force = adjusted_pred[start : start + n]
        _, net_tau = _net_force_and_torque(pos, force)
        net_torques_after.append(net_tau)
        start += n
    net_torques_after = torch.stack(net_torques_after)  # type: ignore

    # Verify the results based on batch_type
    for i in range(num_graphs):
        if batch_type == "all_pbc":
            assert torch.allclose(
                net_torques_before[i], net_torques_after[i], atol=1e-6
            ), f"Net torque changed for PBC system {i} in 'all_pbc' batch."
        elif batch_type == "no_pbc":
            assert torch.allclose(
                net_torques_after[i], torch.zeros(3), atol=1e-6
            ), f"Net torque not zeroed for non-PBC system {i} in 'no_pbc' batch."
        elif batch_type == "mixed":
            if i == 0:
                assert torch.allclose(
                    net_torques_before[i], net_torques_after[i], atol=1e-6
                ), f"Net torque changed for PBC system {i} in 'mixed' batch."
            else:
                assert torch.allclose(
                    net_torques_after[i], torch.zeros(3), atol=1e-6
                ), f"Net torque not zeroed for non-PBC system {i} in 'mixed' batch."
