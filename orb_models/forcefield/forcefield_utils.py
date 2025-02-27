from typing import Optional, Tuple

import torch

from orb_models.forcefield import base, segment_ops


def remove_fixed_atoms(
    pred_node: torch.Tensor,
    node_target: torch.Tensor,
    batch_n_node: torch.Tensor,
    fix_atoms: Optional[torch.Tensor],
    training: bool,
):
    """We use inf targets on purpose to designate nodes for removal."""
    assert len(pred_node) == len(node_target)
    if fix_atoms is not None and not training:
        pred_node = pred_node[~fix_atoms]
        node_target = node_target[~fix_atoms]
        batch_n_node = segment_ops.aggregate_nodes(
            (~fix_atoms).int(), batch_n_node, reduction="sum"
        )
    return pred_node, node_target, batch_n_node


def forces_within_threshold(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch_num_nodes: torch.Tensor,
    threshold: float = 0.03,
) -> torch.Tensor:
    """Calculate MAE between batched graph tensors within a threshold.

    The predictions for a graph are counted as being within the threshold
    only if all nodes in the graph have predictions within the threshold.
    """
    error = torch.abs(pred - target)  # (batch_num_nodes, 3)
    largest_dim_fwt = error.max(-1).values < threshold  # (batch_num_nodes)
    count_within_threshold = segment_ops.aggregate_nodes(
        largest_dim_fwt.to(error.dtype), batch_num_nodes, reduction="sum"
    )
    # count equals batch_num_nodes if all nodes within threshold
    return (count_within_threshold == batch_num_nodes).to(error.dtype).mean()


def maybe_remove_net_force_and_torque(
    batch: base.AtomGraphs,
    force_pred: torch.Tensor,
    remove_mean: bool,
    remove_torque: bool,
) -> torch.Tensor:
    """Maybe remove the mean and net torque from the predicted forces."""
    if remove_mean:
        system_means = segment_ops.aggregate_nodes(
            force_pred, batch.n_node, reduction="mean"
        )
        node_broadcasted_means = torch.repeat_interleave(
            system_means, batch.n_node, dim=0
        )
        force_pred = force_pred - node_broadcasted_means

    if remove_torque:
        force_pred = _selectively_remove_net_torque_for_nonpbc_systems(
            force_pred, batch.positions, batch.system_features["cell"], batch.n_node
        )

    return force_pred


def _selectively_remove_net_torque_for_nonpbc_systems(
    pred: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    n_node: torch.Tensor,
):
    """Remove net torque from non-PBC-system forces, but preserve PBC-system forces.

    Args:
        pred: The predicted forces of shape (n_atoms_in_batch, 3).
        positions: The positions of shape (n_atoms_in_batch, 3).
        cell: The cell of shape (n_batch, 3, 3).
        n_node: The number of nodes per graph, of shape (n_batch,).
    """
    nopbc_graph = torch.all(cell == 0.0, dim=(1, 2))
    if torch.any(nopbc_graph):
        if torch.all(nopbc_graph):
            pred = _remove_net_torque(positions, pred, n_node)
        else:
            # Handle a mixed batch of pbc and non-pbc systems
            batch_indices = torch.repeat_interleave(
                torch.arange(cell.size(0), device=n_node.device), n_node
            )
            nopbc_atom = nopbc_graph[batch_indices]
            adjusted_pred_non_pbc = _remove_net_torque(
                positions[nopbc_atom], pred[nopbc_atom], n_node[nopbc_graph]
            )
            pred = pred.clone()
            pred[nopbc_atom] = adjusted_pred_non_pbc

    return pred


def _remove_net_torque(
    positions: torch.Tensor,
    forces: torch.Tensor,
    n_nodes: torch.Tensor,
) -> torch.Tensor:
    """Adjust the predicted forces to eliminate net torque for each graph in the batch.

    The mathematical derivation of this function is given here:
    https://www.notion.so/orbitalmaterials/Net-torque-removal-11f56117b79780dfbbb9ce78e245be38?pvs=4

    The naming conventions here match those in the derivation.

    Args:
        positions : torch.Tensor of shape (N, 3)
            Positions of atoms (concatenated for all graphs in the batch).
        forces : torch.Tensor of shape (N, 3)
            Predicted forces on atoms.
        n_nodes : torch.Tensor of shape (B,)
            Number of nodes in each graph, where B is the number of graphs in the batch.

    Returns:
        adjusted_forces : torch.Tensor of shape (N, 3)
            Adjusted forces with zero net torque and net force for each graph.
    """
    B = n_nodes.shape[0]
    tau_total, r = _compute_net_torque(positions, forces, n_nodes)

    # Compute scalar s per graph: sum_i ||r_i||^2
    r_squared = torch.sum(r**2, dim=1)  # Shape: (N,)
    s = segment_ops.aggregate_nodes(r_squared, n_nodes, "sum")  # Shape: (B,)

    # Compute matrix S per graph: sum_i outer(r_i, r_i)
    r_unsqueezed = r.unsqueeze(2)  # Shape: (N, 3, 1)
    r_T_unsqueezed = r.unsqueeze(1)  # Shape: (N, 1, 3)
    outer_products = r_unsqueezed @ r_T_unsqueezed  # Shape: (N, 3, 3)
    S = segment_ops.aggregate_nodes(outer_products, n_nodes, "sum")  # Shape: (B, 3, 3)

    # Compute M = S - sI
    I = (  # noqa: E741
        torch.eye(3, device=positions.device).unsqueeze(0).expand(B, -1, -1)
    )  # Shape: (B, 3, 3)
    M = S - (s.view(-1, 1, 1)) * I  # Shape: (B, 3, 3)

    # Right-hand side vector b per graph
    b = -tau_total  # Shape: (B, 3)

    # Solve M * mu = b for mu per graph
    try:
        mu = torch.linalg.solve(M, b.unsqueeze(2)).squeeze(2)  # Shape: (B, 3)
    except RuntimeError:
        # Handle singular matrix M by using the pseudo-inverse
        M_pinv = torch.linalg.pinv(M)  # Shape: (B, 3, 3)
        mu = torch.bmm(M_pinv, b.unsqueeze(2)).squeeze(2)  # Shape: (B, 3)

    # Compute adjustments to forces
    mu_batch = torch.repeat_interleave(mu, n_nodes, dim=0)  # Shape: (N, 3)
    forces_delta = torch.linalg.cross(r, mu_batch)  # Shape: (N, 3)

    # Adjusted forces
    adjusted_forces = forces + forces_delta  # Shape: (N, 3)

    return adjusted_forces


def _compute_net_torque(
    positions: torch.Tensor,
    forces: torch.Tensor,
    n_nodes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the net torque on a system of particles."""
    com = segment_ops.aggregate_nodes(positions, n_nodes, "mean")
    com_repeat = torch.repeat_interleave(com, n_nodes, dim=0)  # Shape: (N, 3)
    com_relative_positions = positions - com_repeat  # Shape: (N, 3)
    torques = torch.linalg.cross(com_relative_positions, forces)  # Shape: (N, 3)
    net_torque = segment_ops.aggregate_nodes(torques, n_nodes, "sum")
    return net_torque, com_relative_positions


def compute_gradient_forces_and_stress(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    generator: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute forces and stress from energy using autograd."""
    inputs = [positions, displacement, generator]
    grads = torch.autograd.grad(
        outputs=[energy],  # (n_graphs,)
        inputs=inputs,  # (n_nodes, 3)
        grad_outputs=[torch.ones_like(energy)],
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )
    forces = grads[0]
    virials = grads[1]
    rotational_grad = grads[2]

    if forces is None:
        raise ValueError(
            "Forces are None. The computational graph between energy and "
            "positions has been broken. Make sure the positions tensor has "
            "not been replaced since calling compute_differentiable_edge_vectors()"
        )
    if virials is None:
        raise ValueError(
            "Virials are None. The computational graph between energy and "
            "displacement has been broken. Make sure the displacement tensor has "
            "not been replaced since calling compute_differentiable_edge_vectors()"
        )
    if rotational_grad is None:
        raise ValueError(
            "rotational gradients are None. The computational graph between energy and "
            "rotation generator has been broken. Make sure the generator tensor has "
            "not been replaced since calling compute_differentiable_edge_vectors()"
        )

    stress = torch.zeros_like(displacement)
    if compute_stress:
        cell = cell.view(-1, 3, 3)
        volume = torch.linalg.det(cell).abs().unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
        stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
    voigt_stress = torch_full_3x3_to_voigt_6_stress(stress)
    return -1 * forces, voigt_stress, rotational_grad


def torch_full_3x3_to_voigt_6_stress(stress_matrix: torch.Tensor) -> torch.Tensor:
    """Convert a batch of 3x3 stress tensors to a 6-component stress tensor in Voigt notation.

    Args:
        stress_matrix (torch.Tensor): A tensor of shape (..., 3, 3) representing the stress matrices.

    Returns:
        torch.Tensor: A tensor of shape (..., 6) representing the stress vectors in Voigt notation.
    """
    # Extract the normal stress components
    s00 = stress_matrix[..., 0, 0]
    s11 = stress_matrix[..., 1, 1]
    s22 = stress_matrix[..., 2, 2]

    # Extract and average the shear stress components
    s12 = (stress_matrix[..., 1, 2] + stress_matrix[..., 2, 1]) / 2
    s02 = (stress_matrix[..., 0, 2] + stress_matrix[..., 2, 0]) / 2
    s01 = (stress_matrix[..., 0, 1] + stress_matrix[..., 1, 0]) / 2

    # Stack the components into a Voigt vector
    voigt = torch.stack([s00, s11, s22, s12, s02, s01], dim=-1)

    return voigt


def torch_voigt_6_to_full_3x3_stress(stress_voigt: torch.Tensor) -> torch.Tensor:
    """Convert a batch of 6-component stress tensors to a full 3x3 stress tensor.

    Args:
        stress_voigt (torch.Tensor): A tensor of shape (..., 6) representing the stress matrices.

    Returns:
        torch.Tensor: A tensor of shape (..., 3, 3) representing the stress vectors in Voigt notation.
    """
    indices_row_major = [0, 5, 4, 5, 1, 3, 4, 3, 2]
    stress_matrix = torch.stack(
        [stress_voigt[..., i] for i in indices_row_major],
        dim=-1,
    )
    return stress_matrix.reshape(*stress_voigt.shape[:-1], 3, 3)


def conditional_huber_force_loss(
    pred_forces: torch.Tensor, target_forces: torch.Tensor, huber_delta: float
) -> torch.Tensor:
    """MACE conditional huber loss for forces."""
    # Define the multiplication factors for each condition
    factors = [huber_delta * x for x in [1.0, 0.7, 0.4, 0.1]]

    # Apply multiplication factors based on conditions
    c1 = torch.norm(target_forces, dim=-1) < 100
    c2 = (torch.norm(target_forces, dim=-1) >= 100) & (
        torch.norm(target_forces, dim=-1) < 200
    )
    c3 = (torch.norm(target_forces, dim=-1) >= 200) & (
        torch.norm(target_forces, dim=-1) < 300
    )
    c4 = ~(c1 | c2 | c3)

    se = torch.zeros_like(pred_forces)

    se[c1] = torch.nn.functional.huber_loss(
        target_forces[c1], pred_forces[c1], reduction="none", delta=factors[0]
    )
    se[c2] = torch.nn.functional.huber_loss(
        target_forces[c2], pred_forces[c2], reduction="none", delta=factors[1]
    )
    se[c3] = torch.nn.functional.huber_loss(
        target_forces[c3], pred_forces[c3], reduction="none", delta=factors[2]
    )
    se[c4] = torch.nn.functional.huber_loss(
        target_forces[c4], pred_forces[c4], reduction="none", delta=factors[3]
    )

    return torch.mean(se)
