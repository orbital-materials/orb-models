"""Featurization utilities for molecular models."""

import typing
import gc
from typing import Optional, Tuple, Union, Literal, List

import ase
import numpy as np
import torch

try:
    import cuml
except ImportError:
    cuml = None

from scipy.spatial import KDTree as SciKDTree


TORCH_FLOAT_DTYPES = [
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
]
ATOM_TYPE_K = 5

EdgeCreationMethod = Literal[
    "knn_brute_force", "knn_scipy", "knn_cuml_brute", "knn_cuml_rbc"
]


def get_device(
    requested_device: Optional[Union[torch.device, str, int]],
) -> torch.device:
    """Get a torch device, defaulting to gpu if available."""
    if requested_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def get_default_edge_method(
    device: torch.device, num_atoms: int, is_periodic: bool
) -> EdgeCreationMethod:
    """Get the default edge method for a given device and number of atoms."""
    if device.type != "cpu":
        if (
            cuml is None
            or (is_periodic and num_atoms < 5_000)
            or (not is_periodic and num_atoms < 30_000)
        ):
            edge_method = "knn_brute_force"
        else:
            edge_method = "knn_cuml_rbc"
    else:
        edge_method = "knn_scipy"
    assert edge_method in typing.get_args(EdgeCreationMethod)
    return edge_method  # type: ignore


def get_atom_embedding(atoms: ase.Atoms, k_hot: bool = False) -> torch.Tensor:
    """Get an atomic embedding."""
    atomic_numbers = torch.from_numpy(atoms.numbers).to(torch.long)
    n_atoms = len(atomic_numbers)
    if k_hot:
        atom_type_embedding = torch.ones(n_atoms, 118) * -ATOM_TYPE_K
        atom_type_embedding[torch.arange(n_atoms), atomic_numbers] = ATOM_TYPE_K
    else:
        atom_type_embedding = torch.nn.functional.one_hot(
            atomic_numbers, num_classes=118
        )
    return atom_type_embedding.to(torch.get_default_dtype())


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


def torch_lexsort(
    keys: List[torch.Tensor],
    dim: int = -1,
    descending: bool = False,
) -> torch.Tensor:
    r"""Performs an indirect stable sort using a sequence of keys.

    This is a torch version of numpy.lexsort, adapted from
    https://pytorch-geometric.readthedocs.io/en/2.4.0/_modules/torch_geometric/utils/lexsort.html

    Given multiple sorting keys, returns an array of integer indices that
    describe their sort order.
    The last key in the sequence is used for the primary sort order, the
    second-to-last key for the secondary sort order, and so on.

    Args:
        keys ([torch.Tensor]): The :math:`k` different columns to be sorted.
            The last key is the primary sort key.
        dim (int, optional): The dimension to sort along. (default: :obj:`-1`)
        descending (bool, optional): Controls the sorting order (ascending or
            descending). (default: :obj:`False`)

    Returns:
        torch.Tensor: The indices that would sort the keys.
    """
    assert len(keys) >= 1
    kwargs = dict(dim=dim, descending=descending, stable=True)
    out = keys[0].argsort(**kwargs)  # type: ignore
    for k in keys[1:]:
        out = out.gather(dim, k.gather(dim, out).argsort(**kwargs))  # type: ignore

    return out


def _integer_lattice(
    start: int, stop: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Generate a 3D integer lattice of shape (N+1, N+1, N+1) for N = stop - start."""
    n_vals = torch.arange(start, stop + 1, device=device, dtype=dtype)
    N1, N2, N3 = torch.meshgrid(n_vals, n_vals, n_vals, indexing="ij")
    integer_offsets = torch.stack((N1, N2, N3), dim=-1).reshape(-1, 3)
    return integer_offsets


def find_minimal_supercell_translations(
    cell: torch.Tensor,
    cutoff: Union[float, torch.Tensor],
    N: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find the all neighbouring unit cells within the cutoff.

    Two cells, A & B, are 'within a cutoff' iff:
                min_{x in A, y in B} || x - y || < cutoff

    Computing this minimum would be very difficult for arbitrary sets A and B.
    However, as they are convex parallelpipeds, it suffices to consider only their
    vertices, which is computationally tractable.

    Our approach is as follows: we enumerate all integers (n1, n2, n3) in range [-N, N] s.t.
            ||n1*a1 + n2*a2 + n3*a3|| <= cutoff.
    Where (a1, a2, a3) denote the lattice vectors.

    Each such (n1, n2, n3) is then added to the 27 integer offsets (-1/0/1, -1/0/1, -1/0/1).

    Finally, we discard duplicate integer triplets to get a unique set of offsets.
    Multiplying these unique integer offsets by the lattice vectors gives us the real-space translations.

    Args:
        cell (torch.Tensor): Shape (3, 3). Rows are the lattice vectors a1, a2, a3.
        cutoff (float): The real-space distance cutoff. If None, then we build a supercell of size 3x3x3.
        N (int, optional): The integer range to consider [-N, N] when searching for neighbouring cells.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - translations: Shape (K, 3). The real-space translation vectors
              from the central cell to the neighbouring cells.
              K is variable and depends on the cutoff and cell shape.
            - integer_offsets: Shape (K, 3). The integer lattice offsets from
              the central cell (0, 0, 0) to the neighbouring cells.
              K is variable and depends on the cutoff and cell shape.
    """
    assert N >= 4, "We have no realistic use-case for N<4, and it risks missing edges."
    if cutoff > 6.0 and N == 4:
        raise NotImplementedError(
            "For a cutoff beyond 6.0, we have not validated whether N=4 is sufficient. "
            "Since it is hard to obtain a valid N analytically; we advise an empirical approach. "
            "Run this function with a large N (e.g. 10) across Alexandria validation set, "
            "record the largest integer_offsets.abs().max() seen in practice, and set N to be "
            "at least one more than this."
        )

    # Generate all (n1, n2, n3) in [-N, N] such that ||n1*a1 + n2*a2 + n3*a3|| <= cutoff
    # (2N+1, 3)
    integer_offsets = _integer_lattice(-N, N, cell.device, cell.dtype)
    # (2N+1, 3)
    translations = integer_offsets.mm(cell)
    # (2N+1,)
    dist = translations.norm(dim=1)
    mask = dist <= cutoff + 1e-4
    # (M, 3)
    cutoff_integer_offsets = integer_offsets[mask]  # integer combos

    # We are currently missing an 'outer shell' and hence need to
    # broadcast add [-1/0/1, -1/0/1, -1/0/1] to cutoff_integer_offsets.
    # Why are we missing an outer shell? Above, we found lattice points
    # within cutoff *of the origin*, but really we want all lattice points
    # within cutoff of *any point in the central cell*

    # Shape (27, 3)
    central_integer_offsets = _integer_lattice(
        -1, 1, device=cell.device, dtype=cell.dtype
    )
    # (M, 27, 3)
    central_integer_offsets = central_integer_offsets.unsqueeze(0).expand(
        len(cutoff_integer_offsets), 27, 3
    )
    # (27, 1, 3)
    cutoff_integer_offsets = cutoff_integer_offsets.unsqueeze(1)
    # (27, M, 3)
    all_integer_offsets = cutoff_integer_offsets + central_integer_offsets
    # (27*M, 3)
    all_integer_offsets = all_integer_offsets.reshape(-1, 3)

    # The broadcast addition above will generate duplicates which need removing
    # (K, 3)
    unique_integer_offsets = torch.unique(all_integer_offsets, dim=0)
    # (K, 3)
    unique_translations = unique_integer_offsets.mm(cell)

    return unique_translations, unique_integer_offsets


def construct_minimal_supercell(
    positions: torch.Tensor,
    cell: torch.Tensor,
    cutoff: Optional[Union[float, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute positions for the smallest supercell needed to capture all interactions within the cutoff.

    By default, we build a 3x3x3 supercell when cutoff is None. More generally, the size of the
    supercell depends on the combination of cutoff distance and size/shape of the cell.

    Args:
        positions (torch.Tensor): Positions of the atoms. Shape [num_atoms, 3].
        cell (torch.Tensor): [3, 3] unit cell.
        cutoff (Optional[Union[float, torch.Tensor]]): The cutoff distance for atomic interactions.
            If None, then we build a supercell of size 3x3x3.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - supercell_positions: Shape [num_atoms, num_unit_cells, 3].
            - integer_offsets: Shape [num_unit_cells, 3].

    Example:
        Suppose we have a 'wide' unit cell (in 2D) of shape:
                          + -------- +
                         /          /
                        + -------- +
        In this case, we will need to pad more in the vertical direction than in the
        horizontal direction. For instance, we may need to build a supercell like so:
                     +--------- +--------- +--------- +
                    /  [-1, 2] / [0, 2]   / [1, 2]   /
                   + -------- + -------- + -------- +
                  /  [-1, 1] / [0, 1]   / [1, 1]   /
                 + -------- + -------- + -------- +
                /  [-1, 0] / [0, 0]   / [1, 0]   /
               + -------- + -------- + -------- +
              /  [-1,-1] / [0,-1]   / [1,-1]   /
             + -------- + -------- + -------- +
            /  [-1,-2] / [0,-2]   / [1,-2]   /
           + -------- + -------- + -------- +
        Where we have labelled each cell with the integer offsets relative to the central cell.

        Determining the smallest necessary supercell is a slightly tricky integer-search problem,
        but can be done fairly efficiently if we limit the integer offsets to e.g. < 5, which is
        likely sufficient for all realistic 3D atomic systems when the cutoff is small e.g. 6 Å.
    """
    if cell.shape != (3, 3):
        raise ValueError("Cell must be a 3x3 matrix. Batched PBCs are not supported.")
    n_positions = len(positions)

    if cutoff is None:
        # construct a 3x3x3 supercell
        integer_offsets = _integer_lattice(-1, 1, device=cell.device, dtype=cell.dtype)
        translations = integer_offsets.mm(cell)  # shape: (27, 3)
    else:
        translations, integer_offsets = find_minimal_supercell_translations(
            cell, cutoff
        )

    # broadcasted positions + translations
    n_offsets = len(integer_offsets)
    expanded_translations = translations.unsqueeze(0).expand(n_positions, n_offsets, 3)
    expanded_positions = positions.unsqueeze(1)

    # Shape (n_positions, n_offsets, 3)
    supercell_positions = expanded_positions + expanded_translations

    return supercell_positions, integer_offsets


# NOTE: it's crucial that [0, 0, 0] is the first entry so that the indices
# of atoms in the central image are preserved. This assumption is needed
# for _copy_and_reverse_boundary_crossing_edges to work correctly.
HALF_OFFSETS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 0.0],
        [1.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, -1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ]
)


def construct_half_3x3x3_supercell(
    positions: torch.Tensor, cell: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find the minimal half-supercell needed to capture all interactions within the cutoff."""
    if cell.shape != (3, 3):
        raise ValueError("Cell must be a 3x3 matrix. Batched PBCs are not supported.")
    n_positions = len(positions)
    volume = (cell[:, 0] * torch.linalg.cross(cell[:, 1], cell[:, 2])).sum(-1)
    assert volume > 1000, (
        "half_supercell is currently only supported for "
        f"large cells with volume > 1000, but got volume: {volume}."
    )

    integer_offsets = torch.tensor(
        HALF_OFFSETS, device=positions.device, dtype=positions.dtype
    )
    # Map unit offsets to real-space translation vectors.
    translations = integer_offsets.mm(cell)

    # broadcasted positions + translations
    n_offsets = len(integer_offsets)
    expanded_translations = translations.unsqueeze(0).expand(n_positions, n_offsets, 3)
    expanded_positions = positions.unsqueeze(1)

    # Shape (n_positions, n_offsets, 3)
    supercell_positions = expanded_positions + expanded_translations

    return supercell_positions, integer_offsets


def combine_edge_sets(
    senders1: torch.Tensor,
    senders2: torch.Tensor,
    receivers1: torch.Tensor,
    receivers2: torch.Tensor,
    vectors1: torch.Tensor,
    vectors2: torch.Tensor,
    integer_offsets1: torch.Tensor,
    integer_offsets2: torch.Tensor,
    num_neighbors1: torch.Tensor,
    num_neighbors2: torch.Tensor,
    max_number_neighbors: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine two sets of edges into one set.

    This operation is a trivial concatenation when we only use a max_radius to
    restrict the set of edges. However, when max_number_neighbors is specified,
    we must re-sort edges based on their lengths and eliminate any excess edges.
    """
    senders = torch.cat([senders1, senders2])
    receivers = torch.cat([receivers1, receivers2])
    vectors = torch.cat([vectors1, vectors2])
    integer_offsets = torch.cat([integer_offsets1, integer_offsets2])
    num_neighbors = num_neighbors1 + num_neighbors2

    if max_number_neighbors is not None:
        # sort by sender, then by length
        vec_lengths = vectors.norm(dim=-1)
        sort_order = torch_lexsort([vec_lengths, senders])

        # extract the top max_num_neighbors
        clamped_num_neighbors = torch.clamp(num_neighbors, max=max_number_neighbors)
        offset = torch.clamp(num_neighbors - clamped_num_neighbors, min=0)  # (n_node,)
        offset = torch.cat(
            [torch.tensor([0], device=offset.device), torch.cumsum(offset, dim=0)[:-1]]
        )  # (n_node,)
        index_mapping = torch.arange(
            clamped_num_neighbors.sum().item(), device=num_neighbors.device
        ) + torch.repeat_interleave(offset, clamped_num_neighbors)

        senders = senders[sort_order][index_mapping]
        receivers = receivers[sort_order][index_mapping]
        vectors = vectors[sort_order][index_mapping]
        integer_offsets = integer_offsets[sort_order][index_mapping]
        num_neighbors = clamped_num_neighbors

    return senders, receivers, vectors, integer_offsets, num_neighbors


def _copy_and_reverse_half_supercell_edges(
    integer_offsets: torch.Tensor,
    senders: torch.Tensor,
    half_supercell_receivers: torch.Tensor,
    vectors: torch.Tensor,
    num_neighbors_per_sender: torch.Tensor,
    natoms: int,
    max_number_neighbors: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy and reverse the directions of edges that cross a pbc boundary.

    This function is the main component of the 'half supercell trick'. We assume
    that the input edges to this function were computed between atoms in a
    central unit cell and half of the possible neighbour cells i.e. those
    indicated by an O in the diagram below:
                            + --- + --- + --- +
                            |  O  |  O  |  O  |
                            + --- + --- + --- +
                            |     |  x  |  O  |
                            + --- + --- + --- +
                            |     |     |     |
                            + --- + --- + --- +

    Our goal is to obtain the 'missing edges' going from the central cell (X) to the blank cells.
    These edges are obtained by copying the edges from X to each O, and reversing their directions.
    After this copy operation, we may have extra edges (>max_num_neighbor) and hence must re-sort
    all our edges by their length and delete the longest edges to ensure the max_num_neighbor
    contstraint is obeyed.
    """
    receivers = half_supercell_receivers % natoms  # map to central cell
    integer_offsets = torch.tensor(
        integer_offsets, device=vectors.device, dtype=vectors.dtype
    )
    per_edge_integer_offsets = integer_offsets[half_supercell_receivers // natoms]
    crosses_boundary = half_supercell_receivers > natoms
    non_central_receivers = half_supercell_receivers[crosses_boundary]

    extra_senders = non_central_receivers % natoms  # map to central cell
    extra_receivers = senders[crosses_boundary]
    extra_vectors = -vectors[crosses_boundary]  # flip
    extra_per_edge_integer_offsets = -integer_offsets[non_central_receivers // natoms]
    extra_num_neighbors_per_sender = torch.bincount(extra_senders, minlength=natoms)

    return combine_edge_sets(
        senders1=senders,
        senders2=extra_senders,
        receivers1=receivers,
        receivers2=extra_receivers,
        vectors1=vectors,
        vectors2=extra_vectors,
        integer_offsets1=per_edge_integer_offsets,
        integer_offsets2=extra_per_edge_integer_offsets,
        num_neighbors1=num_neighbors_per_sender,
        num_neighbors2=extra_num_neighbors_per_sender,
        max_number_neighbors=max_number_neighbors,
    )


def compute_pbc_radius_graph(
    *,
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    radius: Union[float, torch.Tensor],
    max_number_neighbors: int,
    edge_method: Optional[EdgeCreationMethod] = None,
    n_workers: int = 1,
    device: Optional[Union[torch.device, str, int]] = None,
    half_supercell: Optional[bool] = None,
    float_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes graph edges within a max radius and num_neighbors, accounting for periodic-boundary conditions.

    Args:
        positions (torch.Tensor): 3D positions of particles. Shape [num_particles, 3].
        cell (torch.Tensor): A 3x3 matrix where the lattice vectors are rows or columns.
        pbc (torch.Tensor): A boolean tensor of shape [3] indicating which directions are periodic.
        radius (Union[float, torch.tensor]): The radius within which to connect atoms.
        max_number_neighbors (int): The maximum number of neighbors for each particle.
        edge_method (EdgeCreationMethod, optional): The method to use for graph edge construction.
            Defaults to None, in which case edge method is chosen as follows:
            * knn_brute_force: If device is not CPU, and cuML is not installed or num_atoms is < 5000 (PBC)
                or < 30000 (non-PBC).
            * knn_cuml_rbc: If device is not CPU, and cuML is installed, and num_atoms is >= 5000 (PBC) or
                >= 30000 (non-PBC).
            * knn_scipy: If device is CPU.
            On GPU, for num_atoms ≲ 5000 (PBC) or ≲ 30000 (non-PBC), knn_brute_force is faster than knn_cuml_*,
            but uses more memory. For num_atoms ≳ 5000 (PBC) or ≳ 30000 (non-PBC), knn_cuml_* is faster and uses
            less memory, but requires cuML to be installed. knn_scipy is typically fastest on the CPU.
        n_workers (int, optional): The number of workers for KDTree construction in knn_scipy. Defaults to 1.
        device (Union[torch.device, str, int], optional): The device to use for computation.
            Defaults to None, in which case GPU is used if available.
        half_supercell (bool): Whether to use half the supercell for graph construction, and then symmetrize.
            This flag does not affect the resulting graph; it is purely an optimization that can double
            throughput and half memory for very large cells (e.g. 5k+ atoms). For smaller systems, it can harm
            performance due to additional computation to enforce max_num_neighbors.
        float_dtype (torch.dtype): The dtype to use for floating point tensors in the graph construction.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A 3-Tuple. First, an edge_index tensor,
        where the first index are the sender indices and the second are the receiver indices.
        Second, the vector displacements between edges. Third, the unit shifts for each edge.
    """
    if float_dtype not in TORCH_FLOAT_DTYPES:
        raise ValueError(
            f"float_dtype must be one of {TORCH_FLOAT_DTYPES}, got {float_dtype}"
        )

    positions = positions.to(dtype=float_dtype)
    cell = cell.to(dtype=float_dtype)

    natoms = positions.shape[0]
    half_supercell = (
        len(positions) >= 5_000 if half_supercell is None else half_supercell
    )
    half_supercell = half_supercell and bool(torch.any(cell != 0.0))
    is_periodic = bool(torch.any(cell != 0.0).item() and torch.any(pbc).item())

    device = get_device(requested_device=device)
    edge_method = edge_method or get_default_edge_method(device, natoms, is_periodic)
    if edge_method == "knn_brute_force" or edge_method.startswith("knn_cuml_"):
        # if knn_brute_force or knn_cuml_*, then try to place tensors on the gpu if device is not provided
        positions = positions.to(device)
        cell = cell.to(device)

    if is_periodic:
        if half_supercell:
            supercell_positions, integer_offsets = construct_half_3x3x3_supercell(
                positions=positions, cell=cell
            )
        else:
            supercell_positions, integer_offsets = construct_minimal_supercell(
                positions=positions, cell=cell, cutoff=radius
            )
        # supercell_positions: Shape (natoms, num_unit_cells, 3)
        # integer_offsets: Shape (num_unit_cells, 3)

        # NOTE: We need to reshape the supercell_positions to be flat, so we can use them
        # to build a nearest neighbor tree. The *way* in which they are flattened is important in
        # order to ensure that we can subsequently map supercell indices to unit cell indices
        # via a simple modulus operation. Specifically, we use a transpose and reshape to get:
        # [
        #   cell_0_atom_0,
        #   ...,
        #   cell_0_atom_N,
        #   cell_1_atom_0,
        #   ...,
        #   cell_M_atom_N,
        # ]
        supercell_positions = supercell_positions.transpose(0, 1)
        supercell_positions = supercell_positions.reshape(-1, 3)
    else:
        supercell_positions = positions
        integer_offsets = torch.tensor(
            [[0, 0, 0]], device=positions.device, dtype=positions.dtype
        )

    # For the half_supercell method, we (temporarily) need slack in the max_num_neighbor threshold.
    # 2x is a heuristic that gives exact results for all realistic systems tested.
    # The original max_num_neighbor cutoff is enforced later in _copy_and_reverse_half_supercell_edges.
    k = 2 * max_number_neighbors if half_supercell else max_number_neighbors

    senders, supercell_receivers, vectors, num_neighbors_per_sender = (
        compute_supercell_neighbors(
            central_cell_positions=positions,
            supercell_positions=supercell_positions,
            radius=radius,
            max_num_neighbors=k,
            edge_method=edge_method,
            n_workers=n_workers,
        )
    )
    if half_supercell:
        (
            senders,
            receivers,
            vectors,
            per_edge_integer_offsets,
            num_neighbors_per_sender,
        ) = _copy_and_reverse_half_supercell_edges(
            integer_offsets=integer_offsets,
            senders=senders,
            half_supercell_receivers=supercell_receivers,
            vectors=vectors,
            num_neighbors_per_sender=num_neighbors_per_sender,
            natoms=natoms,
            max_number_neighbors=max_number_neighbors,
        )
    else:
        receivers = supercell_receivers % natoms  # map to central cell
        per_edge_integer_offsets = integer_offsets[
            supercell_receivers // natoms
        ]  # (n_edges, 3)

    return torch.stack((senders, receivers), dim=0), vectors, per_edge_integer_offsets


def compute_supercell_neighbors(
    central_cell_positions: torch.Tensor,
    supercell_positions: torch.Tensor,
    radius: Union[float, torch.Tensor],
    max_num_neighbors: int,
    edge_method: EdgeCreationMethod,
    n_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute directed edges from atoms in central unit cell to neighbours in the supercell.

    Args:
        positions (torch.Tensor): 3D positions of particles. Shape [num_particles, 3].
        supercell_positions (torch.Tensor): 3D positions of particles in the supercell.
            NOTE: for non-pbc systems, this tensor will be identical to 'positions'.
            If we are using the 'half_supercell' trick, then only half of the supercell is passed here.
        radius (Union[float, torch.tensor]): The radius within which to connect atoms.
        max_number_neighbors (int, optional): The maximum number of neighbors for each particle. Defaults to 20.
        edge_method (EdgeCreationMethod): The method to use for graph edge construction:
            - knn_brute_force: Use brute force knn implementation: compute all pairwise distances between
            positions and supercell_positions, and subsequently filter edges based on radius and max_num_neighbors.
            - knn_cuml_rbc: Use cuML's random-ball algorithm implementation.
            - knn_cuml_brute: Use cuML's brute force implementation.
            - knn_scipy: Use scipy's KDTree implementation.
        n_workers (int, optional): The number of workers to use for KDTree construction. Defaults to 1.
    """
    if edge_method == "knn_brute_force":

        # Always use float64 for distance calculations, because
        # torch.cdist can be quite inprecise for float32 when use_mm_for_euclid_dist is True.
        # This can lead to incorrect edge selection.
        original_dtype = central_cell_positions.dtype
        central_cell_positions_f64 = central_cell_positions.to(torch.float64)
        supercell_positions_f64 = supercell_positions.to(torch.float64)
        distances = torch.cdist(central_cell_positions_f64, supercell_positions_f64)
        k = min(max_num_neighbors + 1, len(supercell_positions))
        distances, supercell_receivers = torch.topk(
            distances, k=k, largest=False, sorted=True
        )
        distances = distances.to(original_dtype)
        # remove self-edges and edges beyond radius
        within_radius = distances[:, 1:] < (radius + 1e-6)
        num_neighbors_per_sender = within_radius.sum(-1)
        supercell_receivers = supercell_receivers[:, 1:][within_radius]
    elif edge_method.startswith("knn_cuml_"):
        if cuml is None:
            raise ImportError(
                "cuML is not installed. Please install cuML: https://docs.rapids.ai/install/."
            )
        assert (
            supercell_positions.device.type == "cuda"
            and central_cell_positions.device.type == "cuda"
        ), "cuML KNN is only supported on CUDA devices"
        algorithm = edge_method.split("_")[-1]
        k = min(max_num_neighbors + 1, len(supercell_positions))
        knn = cuml.neighbors.NearestNeighbors(
            n_neighbors=k,
            algorithm=algorithm,
            metric="euclidean",
        )
        knn.fit(supercell_positions)
        distances, supercell_receivers = knn.kneighbors(
            central_cell_positions, return_distance=True
        )

        # Repeated use of cuML methods causes memory leaks:
        # https://github.com/rapidsai/cuml/issues/5666
        # https://github.com/rapidsai/cuml/issues/4068
        # https://github.com/rapidsai/cuml/issues/4759
        # To mitigate this, we de-reference the knn object after use, and force garbage collection.
        # NOTE: we use gc.collect(0) to specifically collect short-lived objects.
        # This is faster than calling gc.collect(), which defaults to gc.collect(2)
        # that scans through all objects, including long-lived objects, which is very slow.
        del knn
        gc.collect(0)

        # Convert from CuPy arrays to PyTorch tensors
        distances = torch.as_tensor(distances)
        supercell_receivers = torch.as_tensor(supercell_receivers)
        # remove self-edges and edges beyond radius
        within_radius = distances[:, 1:] < (radius + 1e-6)
        num_neighbors_per_sender = within_radius.sum(-1)
        supercell_receivers = supercell_receivers[:, 1:][within_radius]
    elif edge_method == "knn_scipy":
        tree_data = supercell_positions.clone().detach().cpu().numpy()
        tree_query = central_cell_positions.clone().detach().cpu().numpy()
        distance_upper_bound = np.array(radius) + 1e-8
        tree = SciKDTree(tree_data, leafsize=100)
        _, supercell_receivers = tree.query(
            x=tree_query,
            k=min(max_num_neighbors + 1, len(supercell_positions)),
            distance_upper_bound=distance_upper_bound,
            workers=n_workers,
            p=2,
        )
        if len(supercell_receivers.shape) == 1:
            supercell_receivers = supercell_receivers[None, :]

        # Remove the self-edge that will be closest
        supercell_receivers = np.array(supercell_receivers)[:, 1:]  # type: ignore

        # Remove any entry that equals len(supercell_positions), which are negative hits
        valid_hits = supercell_receivers != len(supercell_positions)
        supercell_receivers = torch.tensor(
            supercell_receivers[valid_hits], device=central_cell_positions.device
        )
        num_neighbors_per_sender = torch.tensor(
            valid_hits.sum(-1), device=central_cell_positions.device
        )
    else:
        raise ValueError(f"Invalid graph edge creation method: {edge_method}")

    natoms = central_cell_positions.shape[0]
    senders = torch.repeat_interleave(
        torch.arange(natoms, device=central_cell_positions.device),
        num_neighbors_per_sender,
    )
    vectors = supercell_positions[supercell_receivers] - central_cell_positions[senders]

    return senders, supercell_receivers, vectors, num_neighbors_per_sender


def map_to_pbc_cell(positions: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Maps positions to within a periodic boundary cell.

    Args:
        positions (torch.Tensor): The positions to be mapped. Shape [num_particles, 3]
        cell (torch.Tensor): The matrix of lattice vectors. Shape 3x3.

    Returns:
        torch.Tensor: Positions mapped to within a periodic boundary cell.
    """
    assert positions.dtype == cell.dtype
    original_type = positions.dtype
    # Inverses are a lot more reliable in double precision, so we'll do the whole
    # thing in double then go back to single.
    positions = positions.double()
    cell = cell.double()
    # The strategy here is to map our positions to fractional or internal coordinates.
    # Then we take the modulo, then map back to euclidian co-ordinates.
    fractional_pos = torch.linalg.solve(cell.T, positions.T).T
    fractional_pos = fractional_pos % 1.0
    return (fractional_pos @ cell).to(original_type)


def batch_map_to_pbc_cell(
    positions: torch.Tensor, cell: torch.Tensor, n_node: torch.Tensor
) -> torch.Tensor:
    """Maps positions to within a periodic boundary cell, for a batched system.

    Args:
        positions (torch.Tensor): The positions to be mapped. Shape [num_particles, 3]
        cell (torch.Tensor): The matrices of lattice vectors. Shape [num_batches, 3, 3]
        n_node (torch.LongTensor): The number of atoms in each graph. Shape [num_batches]
    """
    dtype = positions.dtype
    positions = positions.double()
    cell = cell.double()

    pbc_nodes = torch.repeat_interleave(cell, n_node, dim=0)

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
    cells: torch.Tensor,
    pbc: torch.Tensor,
    radius: Union[float, torch.Tensor],
    n_node: torch.Tensor,
    max_number_neighbors: torch.Tensor,
    edge_method: Optional[EdgeCreationMethod] = None,
    half_supercell: bool = False,
    device: Optional[Union[torch.device, str, int]] = None,
):
    """Computes edges within a max radius and num_neighbors, accounting for periodic-boundary conditions.

    This function is optimised for computation on CPU, and work work on device. GPU implementations
    are likely to be significantly slower because of the irregularly sized tensor computations and the
    lack of extremely fast GPU knn routines.

    Args:
        positions (torch.Tensor): 3D positions of a batch of particles. Shape [num_particles, 3].
        cells (torch.Tensor): A batch of 3x3 matrices where the lattice vectors are rows.
        pbc (torch.Tensor): A batch of boolean tensors of shape [3] indicating which directions are periodic.
        radius (Union[float, torch.tensor]): The radius within which to connect atoms.
        n_node (torch.Tensor): A vector where each element indicates the number of particles in each element of
            the batch. Of size len(batch).
        max_number_neighbors (torch.Tensor): The maximum number of neighbors for each particle.
        edge_method (EdgeCreationMethod, optional): The method to use for graph edge construction.
            Defaults to None, in which case edge method is chosen as follows:
            * knn_brute_force: If device is not CPU, and cuML is not installed or num_atoms is < 5000 (PBC)
                or < 30000 (non-PBC).
            * knn_cuml_rbc: If device is not CPU, and cuML is installed, and num_atoms is >= 5000 (PBC) or
                >= 30000 (non-PBC).
            * knn_scipy: If device is CPU.
            On GPU, for num_atoms ≲ 5000 (PBC) or ≲ 30000 (non-PBC), knn_brute_force is faster than knn_cuml_*,
            but uses more memory. For num_atoms ≳ 5000 (PBC) or ≳ 30000 (non-PBC), knn_cuml_* is faster and uses
            less memory, but requires cuML to be installed. knn_scipy is typically fastest on the CPU.
        half_supercell (bool): Whether to use half the supercell for graph construction, and then symmetrize.
            This flag does not affect the resulting graph; it is purely an optimization that can double
            throughput and half memory for very large cells (e.g. 10k+ atoms). For smaller systems, it can harm
            performance due to additional computation to enforce max_num_neighbors.
        device (Optional[Union[torch.device, str, int]], optional): The device to use for computation.
            Defaults to None, in which case GPU is used if available.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A 4-Tuple.
        First, an edge_index tensor, where the first index are the sender indices and the
        Second are the receiver indices. Second, the vector displacements between edges.
        Third, the unit shifts between the sender and receiver.
        Fourth, the number of neighbors for each sender.
    """
    idx = 0
    all_edges = []
    all_vectors = []
    num_edges = []
    all_unit_shifts = []

    for p, cell, pbc, mn in zip(
        torch.tensor_split(positions, torch.cumsum(n_node, 0)[:-1].cpu()),
        cells,
        pbc,
        max_number_neighbors,
        strict=True,
    ):
        edges, vectors, unit_shifts = compute_pbc_radius_graph(
            positions=p,
            cell=cell,
            pbc=pbc,
            radius=radius,
            max_number_neighbors=int(mn),
            edge_method=edge_method,
            half_supercell=half_supercell,
            device=device,
        )
        if idx == 0:
            offset = 0
        else:
            offset += n_node[idx - 1]  # type: ignore
        all_edges.append(edges + offset)
        all_vectors.append(vectors)
        all_unit_shifts.append(unit_shifts)
        num_edges.append(len(edges[0]))
        idx += 1

    all_edges = torch.concatenate(all_edges, 1)  # type: ignore
    all_vectors = torch.concatenate(all_vectors, 0)  # type: ignore
    all_unit_shifts = torch.cat(all_unit_shifts, 0)  # type: ignore
    num_edges = torch.tensor(num_edges, dtype=torch.int64, device=device)  # type: ignore
    return all_edges, all_vectors, all_unit_shifts, num_edges


def rotation_from_generator(generator: torch.Tensor) -> torch.Tensor:
    """Uses generator to create unitary (rotation) matrix.

    generator -> skew-symmetric matrix S -> R = exp(S)
    S has imaginary eigenvalues, therefore R is unitary

    generator is a (..., 3, 3) tensor.

    """
    return torch.matrix_exp(generator - torch.transpose(generator, dim0=-2, dim1=-1))
