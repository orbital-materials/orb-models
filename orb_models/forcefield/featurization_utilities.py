"""Featurization utilities for molecular models."""

from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch

# TODO(Mark): Make pynanoflann optional
from pynanoflann import KDTree as NanoKDTree
from scipy.spatial import KDTree as SciKDTree

DistanceFeaturizer = Callable[[torch.Tensor], torch.Tensor]


def get_device(
    requested_device: Optional[Union[torch.device, str, int]] = None
) -> torch.device:
    """Get a torch device, defaulting to gpu if available."""
    if requested_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def gaussian_basis_function(
    scalars: torch.Tensor,
    num_bases: Union[torch.Tensor, int],
    radius: Union[torch.Tensor, float],
    scale: Union[torch.Tensor, float] = 1.0,
) -> torch.Tensor:
    """Gaussian basis function applied to a tensor of scalars.

    Args:
        scalars (torch.Tensor): Scalars to compute the gbf on. Shape [num_scalars].
        num_bases (torch.Tensor): The number of bases. An Int.
        radius (torch.Tensor): The largest centre of the bases. A Float.
        scale (torch.Tensor, optional): The width of the gaussians. Defaults to 1.

    Returns:
        torch.Tensor: A tensor of shape [num_scalars, num_bases].
    """
    assert len(scalars.shape) == 1
    gaussian_means = torch.arange(
        0, float(radius), float(radius) / num_bases, device=scalars.device
    )
    return torch.exp(
        -(scale**2) * (scalars.unsqueeze(1) - gaussian_means.unsqueeze(0)).abs() ** 2
    )


def featurize_edges(
    edge_vectors: torch.Tensor, distance_featurization: DistanceFeaturizer
) -> torch.Tensor:
    """Featurizes edge features, provides concatented unit vector along with featurized distances.

    Args:
        edge_vectors (torch.tensor): Edge vectors to featurize. Shape [num_edge, 3]
        distance_featurization (DistanceFeaturization): A function that featurizes the distances of the vectors.

    Returns:
        torch.tensor: Edge features, shape [num_edge, num_edge_features].
    """
    edge_features = []
    edge_norms = torch.linalg.norm(edge_vectors, axis=1)
    featurized_edge_norms = distance_featurization(edge_norms)
    unit_vectors = edge_vectors / edge_norms.unsqueeze(1)
    unit_vectors = torch.nan_to_num(unit_vectors, nan=0, posinf=0, neginf=0)
    edge_features.append(featurized_edge_norms)
    edge_features.append(unit_vectors)
    return torch.cat(edge_features, dim=-1).to(torch.float32)


def compute_edge_vectors(
    edge_index: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Computes edge vectors from positions.

    Args:
        edge_index (torch.tensor): The edge index. First position the senders, second
            position the receivers. Shape [2, num_edge].
        positions (torch.tensor): Positions of each node. Shape [num_nodes, 3]

    Returns:
        torch.tensor: The vectors of each edge.
    """
    senders = edge_index[0]
    receivers = edge_index[1]
    return positions[receivers] - positions[senders]


# These are offsets applied to coordinates to create a 3x3x3
# tiled periodic image of the input structure.
OFFSETS = np.array(
    [
        [-1.0, 1.0, -1.0],
        [0.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [-1.0, -1.0, -1.0],
        [0.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [-1.0, -1.0, 1.0],
        [0.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
    ]
)

NUM_OFFSETS = len(OFFSETS)


def _compute_img_positions_torch(
    positions: torch.Tensor, periodic_boundaries: torch.Tensor
) -> torch.Tensor:
    """Computes the positions of the periodic images of the input structure.

    Consider the following 2D periodic boundary image.
    + --- + --- + --- +
    |     |     |     |
    + --- + --- + --- +
    |     |  x  |     |
    + --- + --- + --- +
    |     |     |     |
    + --- + --- + --- +

    Each tile in this has an associated translation to translate
    'x'. For example, the top left would by (-1, +1). These are
    the 'OFFSETS', but OFFSETS are for a 3x3x3 grid.

    This is complicated by the fact that our periodic
    boundaries are not orthogonal to each other, and so we form a new
    translation by taking a linear combination of the unit cell axes.

    Args:
        positions (torch.Tensor): Positions of the atoms. Shape [num_atoms, 3].
        periodic_boundaries (torch.Tensor): Periodic boundaries of the unit cell.
            This can be 2 shapes - [3, 3] or [num_atoms, 3, 3]. If the shape is
            [num_atoms, 3, 3], it is assumed that the PBC has been repeat_interleaved
            for each atom, i.e this function is agnostic as to whether it is computing
            with respect to a batch or not.
    Returns:
        torch.Tensor: The positions of the periodic images. Shape [num_atoms, 27, 3].
    """
    num_positions = len(positions)

    has_unbatched_pbc = periodic_boundaries.shape == (3, 3)
    if has_unbatched_pbc:
        periodic_boundaries = periodic_boundaries.unsqueeze(0)
        periodic_boundaries = periodic_boundaries.expand(num_positions, 3, 3)

    # This section *assumes* we have already repeat_interleaved the periodic
    # boundaries to be the same size as the positions. e.g:
    # (batch_size, 3, 3) -> (batch_n_node, 3, 3)
    assert periodic_boundaries.shape[0] == positions.shape[0]
    # First, create a tensor of offsets where the first axis
    # is the number of particles
    # Shape (27, 3)
    offsets = torch.tensor(OFFSETS, device=positions.device, dtype=positions.dtype)
    # Shape (1, 27, 3)
    offsets = torch.unsqueeze(offsets, 0)
    # Shape (batch_n_node, 27, 3)
    repeated_offsets = offsets.expand(num_positions, NUM_OFFSETS, 3)
    # offsets is now size (batch_n_node, 27, 3). Now we want a translation which is
    # a linear combination of the pbcs which is currently shape (batch_n_node, 3, 3).
    # Make offsets shape (batch_n_node, 27, 3, 1)
    repeated_offsets = torch.unsqueeze(repeated_offsets, 3)
    # Make pbcs shape (batch_n_node, 1, 3, 3)
    periodic_boundaries = torch.unsqueeze(periodic_boundaries, 1)
    # Take the linear combination.
    # Shape (batch_n_node, 27, 3, 3)
    translations = repeated_offsets * periodic_boundaries
    # Shape (batch_n_node, 27, 3)
    translations = translations.sum(2)

    # Expand the positions so we can broadcast add the translations per PBC image.
    # Shape (batch_n_node, 1, 3)
    expanded_positions = positions.unsqueeze(1)
    # Broadcasted addition. Shape (batch_n_node, 27, 3)
    translated_positions = expanded_positions + translations
    return translated_positions


def brute_force_knn(
    img_positions: torch.Tensor, positions: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Brute force k-nearest neighbors.

    Args:
        img_positions (torch.Tensor): The positions of the images. Shape [num_atoms * 27, 3].
        positions (torch.Tensor): The positions of the query atoms. Shape [num_atoms, 3].
        k (int): The number of nearest neighbors to find.

    Returns:
        torch.return_types.topk: The indices of the nearest neighbors. Shape [num_atoms, k].
    """
    dist = torch.cdist(positions, img_positions)
    return torch.topk(dist, k, largest=False, sorted=True)


def compute_pbc_radius_graph(
    *,
    positions: torch.Tensor,
    periodic_boundaries: torch.Tensor,
    radius: Union[float, torch.Tensor],
    max_number_neighbors: int = 20,
    brute_force: Optional[bool] = None,
    library: str = "pynanoflann",
    n_workers: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes periodic condition radius graph from positions.

    Args:
        positions (torch.Tensor): 3D positions of particles. Shape [num_particles, 3].
        periodic_boundaries (torch.Tensor): A 3x3 matrix where the periodic boundary axes are rows or columns.
        radius (Union[float, torch.tensor]): The radius within which to connect atoms.
        max_number_neighbors (int, optional): The maximum number of neighbors for each particle. Defaults to 20.
        brute_force (bool, optional): Whether to use brute force knn. Defaults to None, in which case brute_force
            is used if GPU is available (2-6x faster), but not on CPU (1.5x faster - 4x slower, depending on
            system size).
        library (str, optional): The KDTree library to use. Currently, either 'scipy' or 'pynanoflann'.
        n_workers (int, optional): The number of workers to use for KDTree construction. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A 2-Tuple. First, an edge_index tensor, where the first index are the
        sender indices and the second are the receiver indices. Second, the vector displacements between edges.
    """
    device = get_device()
    if brute_force is None:
        # use brute force if positions are already on gpu
        brute_force = device.type != "cpu"

    if brute_force:
        # use gpu if available
        positions = positions.to(device)
        periodic_boundaries = periodic_boundaries.to(device)

    device = positions.device

    if torch.any(periodic_boundaries != 0.0):
        # Shape (num_positions, 27, 3)
        supercell_positions = _compute_img_positions_torch(
            positions=positions, periodic_boundaries=periodic_boundaries
        )
        # CRITICALLY IMPORTANT: We need to reshape the supercell_positions to be
        # flat, so we can use them for the nearest neighbors. The *way* in which
        # they are flattened is important, because we need to be able to map the
        # indices returned from the nearest neighbors to the original positions.
        # The easiest way to do this is to transpose, so that when we flatten, we
        # have:
        # [
        #   img_0_atom_0,
        #   img_0_atom_1,
        #   ...,
        #   img_0_atom_N,
        #   img_1_atom_0,
        #   img_1_atom_1,
        #   ...,
        #   img_N_atom_N,
        #   etc
        # ]
        # This way, we can take the mod of the indices returned from the nearest
        # neighbors to get the original indices.
        # Shape (27, num_positions, 3)
        supercell_positions = supercell_positions.transpose(0, 1)
        supercell_positions = supercell_positions.reshape(-1, 3)
    else:
        supercell_positions = positions

    num_positions = positions.shape[0]

    if brute_force:
        # Brute force
        distance_values, nearest_img_neighbors = brute_force_knn(
            supercell_positions,
            positions,
            min(max_number_neighbors + 1, len(supercell_positions)),
        )

        # remove distances greater than radius, and exclude self
        within_radius = distance_values[:, 1:] < (radius + 1e-6)

        num_neighbors_per_position = within_radius.sum(-1)
        # remove the self node which will be closest
        index_array = nearest_img_neighbors[:, 1:]

        senders = torch.repeat_interleave(
            torch.arange(num_positions, device=device), num_neighbors_per_position
        )
        receivers_imgs = index_array[within_radius]

        receivers = receivers_imgs % num_positions
        vectors = supercell_positions[receivers_imgs] - positions[senders]
        stacked = torch.stack((senders, receivers), dim=0)
        return stacked, vectors

    else:
        # Build a KDTree from the supercell positions.
        # Query that KDTree just for the positions in the central cell.
        tree_data = supercell_positions.clone().detach().cpu().numpy()
        tree_query = positions.clone().detach().cpu().numpy()
        distance_upper_bound = np.array(radius) + 1e-8
        if library == "scipy":
            tree = SciKDTree(tree_data, leafsize=100)
            _, nearest_img_neighbors = tree.query(
                tree_query,
                max_number_neighbors + 1,
                distance_upper_bound=distance_upper_bound,
                workers=n_workers,
                p=2,
            )
            # Remove the self-edge that will be closest
            index_array = np.array(nearest_img_neighbors)[:, 1:]  # type: ignore
            # Remove any entry that equals len(supercell_positions), which are negative hits
            receivers_imgs = index_array[index_array != len(supercell_positions)]
            num_neighbors_per_position = (index_array != len(supercell_positions)).sum(
                -1
            )
        elif library == "pynanoflann":
            tree = NanoKDTree(
                n_neighbors=min(max_number_neighbors + 1, len(supercell_positions)),
                radius=radius,
                leaf_size=100,
                metric="l2",
            )
            tree.fit(tree_data)
            distance_values, nearest_img_neighbors = tree.kneighbors(
                tree_query, n_jobs=n_workers
            )
            nearest_img_neighbors = nearest_img_neighbors.astype(np.int32)  # type: ignore

            # remove the self node which will be closest
            index_array = nearest_img_neighbors[:, 1:]
            # remove distances greater than radius
            within_radius = distance_values[:, 1:] < (radius + 1e-6)
            receivers_imgs = index_array[within_radius]
            num_neighbors_per_position = within_radius.sum(-1)

        # We construct our senders and receiver indexes.
        senders = np.repeat(np.arange(num_positions), list(num_neighbors_per_position))  # type: ignore
        receivers_img_torch = torch.tensor(receivers_imgs, device=positions.device)
        # Map back to indexes on the central image.
        receivers = receivers_img_torch % num_positions
        senders_torch = torch.tensor(senders, device=positions.device)

        # Finally compute the vector displacements between senders and receivers.
        vectors = supercell_positions[receivers_img_torch] - positions[senders_torch]
        return torch.stack((senders_torch, receivers), dim=0), vectors


def batch_map_to_pbc_cell(
    positions: torch.Tensor,
    periodic_boundary_conditions: torch.Tensor,
    num_atoms: torch.Tensor,
) -> torch.Tensor:
    """Maps positions to within a periodic boundary cell, for a batched system.

    Args:
        positions (torch.Tensor): The positions to be mapped. Shape [num_particles, 3]
        periodic_boundary_conditions (torch.Tensor): The periodic boundary conditions. Shape [num_batches, 3, 3]
        num_atoms (torch.LongTensor): The number of atoms in each batch. Shape [num_batches]
    """
    dtype = positions.dtype
    positions = positions.double()
    periodic_boundary_conditions = periodic_boundary_conditions.double()

    pbc_nodes = torch.repeat_interleave(periodic_boundary_conditions, num_atoms, dim=0)

    # To use the stable torch.linalg.solve, we need to mask batch elements which don't
    # have periodic boundaries. We do this by adding the identity matrix as their PBC,
    # because we need the PBCs to be non-singular.
    # Shape (batch_n_atoms,)
    null_pbc = pbc_nodes.abs().sum(dim=[1, 2]) == 0
    # Shape (3, 3)
    identity = torch.eye(3, dtype=torch.bool, device=pbc_nodes.device)
    # Broadcast the identity to the elements of the batch that have a null pbc.
    # Shape (batch_n_atoms, 3, 3)
    null_pbc_identity_mask = null_pbc.view(-1, 1, 1) & identity.view(1, 3, 3)
    # Shape (batch_n_atoms, 3, 3)
    pbc_nodes_masked = pbc_nodes + null_pbc_identity_mask.double()

    # Shape (batch_n_atoms, 3)
    lattice_coords = torch.linalg.solve(pbc_nodes_masked.transpose(1, 2), positions)
    frac_coords = lattice_coords % 1.0

    cartesian = torch.einsum("bi,bij->bj", frac_coords, pbc_nodes)
    return torch.where(null_pbc.unsqueeze(1), positions, cartesian).to(dtype)


def batch_compute_pbc_radius_graph(
    *,
    positions: torch.Tensor,
    periodic_boundaries: torch.Tensor,
    radius: Union[float, torch.Tensor],
    image_idx: torch.Tensor,
    max_number_neighbors: int = 20,
    brute_force: Optional[bool] = None,
    library: str = "scipy",
):
    """Computes batched periodic boundary condition radius graph from positions.

    This function is optimised for computation on CPU, and work work on device. GPU implementations
    are likely to be significantly slower because of the irregularly sized tensor computations and the
    lack of extremely fast GPU knn routines.

    Args:
        positions (torch.Tensor): 3D positions of a batch of particles. Shape [num_particles, 3].
        periodic_boundaries (torch.Tensor): A batch where each element 3x3 matrix where the periodic boundary axes
            are rows or columns.
        radius (Union[float, torch.tensor]): The radius within which to connect atoms.
        image_idx (torch.Tensor): A vector where each element indicates the number of particles in each element of
            the batch. Of size len(batch).
        max_number_neighbors (int, optional): The maximum number of neighbors for each particle. Defaults to 20.
        brute_force (bool, optional): Whether to use brute force knn. Defaults to None, in which case brute_force
            is used if we are on GPU (2-6x faster), but not on CPU (1.5x faster - 4x slower).
        library (str, optional): The KDTree library to use. Currently, either 'scipy' or 'pynanoflann'.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A 2-Tuple. First, an edge_index tensor, where the first index are the
        sender indices and the second are the receiver indices. Second, the vector displacements between edges.
    """
    idx = 0
    all_edges = []
    all_vectors = []
    num_edges = []

    device = positions.device
    for p, pbc in zip(
        torch.tensor_split(positions, torch.cumsum(image_idx, 0)[:-1].cpu()),
        periodic_boundaries,
    ):
        edges, vectors = compute_pbc_radius_graph(
            positions=p,
            periodic_boundaries=pbc,
            radius=radius,
            max_number_neighbors=max_number_neighbors,
            brute_force=brute_force,
            library=library,
        )
        if idx == 0:
            offset = 0
        else:
            offset += image_idx[idx - 1]  # type: ignore
        all_edges.append(edges + offset)
        all_vectors.append(vectors)
        num_edges.append(len(edges[0]))
        idx += 1

    all_edges = torch.concatenate(all_edges, 1)  # type: ignore
    all_vectors = torch.concatenate(all_vectors, 0)  # type: ignore
    num_edges = torch.tensor(num_edges, dtype=torch.int64, device=device)  # type: ignore
    return all_edges, all_vectors, num_edges
