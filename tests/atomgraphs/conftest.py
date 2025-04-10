from typing import Dict, Tuple, Union
import ase
import numpy as np
import pytest
import torch

from orb_models.dataset import ase_sqlite_dataset
from orb_models.forcefield import atomic_system
from orb_models.forcefield.property_definitions import PropertyConfig
from orb_models.forcefield import base, featurization_utilities
from orb_models.forcefield.featurization_utilities import EdgeCreationMethod


def one_hot(x):
    return torch.nn.functional.one_hot(x, num_classes=118).to(torch.float64)


@pytest.fixture()
def dataset_and_loader(fixtures_path):
    dataset_config = atomic_system.SystemConfig(
        radius=6.0,
        max_num_neighbors=20,
    )

    dataset = ase_sqlite_dataset.AseSqliteDataset(
        name="test_dataset",
        path=str(fixtures_path / "test_dataset.db"),
        system_config=dataset_config,
        target_config=PropertyConfig(),
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=5,
        collate_fn=base.batch_graphs,
    )
    return (dataset, loader)


@pytest.fixture()
def graph():
    nodes, edges = 10, 6
    positions = torch.randn((nodes, 3))
    atomic_numbers = torch.arange(0, nodes)
    vectors = torch.randn((edges, 3))
    lengths = vectors.norm(dim=1)
    return base.AtomGraphs(
        senders=torch.tensor([0, 1, 2, 1, 2, 0]),
        receivers=torch.tensor([1, 0, 1, 2, 0, 2]),
        n_node=torch.tensor([nodes]),
        n_edge=torch.tensor([edges]),
        node_features=dict(
            atomic_numbers=atomic_numbers,
            atomic_numbers_embedding=one_hot(atomic_numbers),
            positions=positions,
        ),
        edge_features=dict(
            vectors=vectors,
            r=lengths,
            unit_shifts=torch.zeros_like(vectors),
        ),
        system_features={
            "cell": torch.eye(3).unsqueeze(0),
            "prior_loss": torch.tensor([0.0]),
        },
        node_targets={"noise_target": torch.randn_like(positions)},
        edge_targets={},
        system_targets={"graph_target": torch.tensor([[23.3]])},
        fix_atoms=torch.tensor([1, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=torch.bool),
        system_id=None,
        tags=None,
        radius=6.0,
        max_num_neighbors=20,
    )


def get_zeolites(fixtures_path) -> Dict[str, ase.Atoms]:
    """Function to get a zeolite framework."""
    zeos = {}
    data = torch.load(fixtures_path / "zeo_test.pkl", weights_only=False)
    for sample in data:
        zeos[sample.data.attributes.zeolite_code] = sample.toatoms()
    return zeos


def get_CO2():
    """Function to get CO2."""

    positions = [
        (-1.16, 0, 0),  # O on the left
        (0, 0, 0),  # C in the middle
        (1.16, 0, 0),  # O on the right
    ]
    CO2 = ase.Atoms("OCO", positions=positions)
    return CO2


def compute_pbc_radius_graph_nequip(
    positions: torch.Tensor,
    cell: torch.Tensor,
    radius: Union[float, torch.Tensor],
    max_number_neighbors: int = 20,
    self_interaction: bool = False,
    strict_self_interaction: bool = True,
    pbc: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Create neighbor list and neighbor vectors based on radial cutoff.

    Adapted from here: https://github.com/mir-group/nequip/blob/main/nequip/data/AtomicData.py#L683.
    Create neighbor list (``edge_index``) and relative vectors
    (``edge_attr``) based on radial cutoff.

    Edges are given by the following convention:
    - ``edge_index[0]`` is the *source* (convolution center).
    - ``edge_index[1]`` is the *target* (neighbor).

    Thus, ``edge_index`` has the same convention as the relative vectors:
    :math:`\\vec{r}_{source, target}`

    If the input positions are a tensor with ``requires_grad == True``,
    the output displacement vectors will be correctly attached to the inputs
    for autograd.

    All outputs are Tensors on the same device as ``pos``; this allows future
    optimization of the neighbor list on the GPU.

    Args:
        positions (shape [N, 3]): Positional coordinate; Tensor or numpy array. If Tensor, must be on CPU.
        cell (numpy shape [3, 3]): Cell for periodic boundary conditions.
            Ignored if ``pbc == False``.
        radius (float): Radial cutoff distance for neighbor finding.
        max_number_neighbors (int): maximum number os neighbors for each node.
        self_interaction (bool): Whether or not to include same periodic image self-edges in the neighbor list.
        strict_self_interaction (bool): Whether to include *any* self interaction edges in the graph, even if
            the two instances of the atom are in different periodic images. Defaults to True, should be True
            for most applications.
        pbc (bool or 3-tuple of bool): Whether the system is periodic in each of the three cell dimensions.

    Returns:
        edge_index_top_k (torch.tensor shape [2, num_edges]): List of edges determined by max_number_neighbors.
        distance_vector_top_k (torch.tensor shape [num_edges, 3]): Relative cell shift
            vectors. Returned only if cell is not None.
    """
    if torch.any(cell != 0.0):
        pbc = True
    if isinstance(pbc, bool):
        pbc = (pbc,) * 3  # type: ignore[assignment]

    # Either the position or the cell may be on the GPU as tensors
    if isinstance(positions, torch.Tensor):
        temp_pos = positions.detach().cpu().numpy()
        out_device = positions.device
        out_dtype = positions.dtype
    else:
        temp_pos = np.asarray(positions)
        out_device = torch.device("cpu")
        out_dtype = torch.get_default_dtype()

    # Get a cell on the CPU no matter what
    if isinstance(cell, torch.Tensor):
        temp_cell = cell.detach().cpu().numpy()
    elif cell is not None:
        temp_cell = np.asarray(cell)
    else:
        # ASE will "complete" this correctly.
        temp_cell = np.zeros((3, 3), dtype=temp_pos.dtype)

    # ASE dependent part
    temp_cell = ase.geometry.complete_cell(temp_cell)

    first_idex, second_idex, distance_vector = ase.neighborlist.primitive_neighbor_list(
        "ijD",
        pbc,
        temp_cell,
        temp_pos,
        cutoff=float(radius),
        self_interaction=strict_self_interaction,
        use_scaled_positions=False,
        max_nbins=5,
    )

    # Eliminate true self-edges that don't cross periodic boundaries
    if not self_interaction:
        bad_edge = first_idex == second_idex
        bad_edge &= np.all(distance_vector == 0, axis=1)
        keep_edge = ~bad_edge
        if not np.any(keep_edge):
            raise ValueError(
                f"Every single atom has no neighbors within the cutoff radius={radius}"
            )
        first_idex = first_idex[keep_edge]
        second_idex = second_idex[keep_edge]
        distance_vector = distance_vector[keep_edge]

    # Get distances based on distance_vector
    distances = np.sqrt(np.sum(distance_vector * distance_vector, axis=1))
    # Get list of counts for the sender nodes
    unique_counts = np.unique(first_idex, return_counts=True)[1].tolist()
    # Get top k smallest distance indices, with k = max_number_neighbors
    start_index = 0
    top_k_smallest_indices = []
    for sender_node in range(len(unique_counts)):
        end_index = unique_counts[sender_node] + start_index
        # If max_number_neighbors is larger than the number of edges, it will keep the original edges unchanged.
        distance_indices = (
            distances[start_index:end_index].argsort()[:max_number_neighbors]
            + start_index
        )
        top_k_smallest_indices.append(distance_indices)
        start_index = end_index
    top_k_smallest_indices = np.concatenate(top_k_smallest_indices)
    # Get top k edge indices and distance_vector based on top k indices
    first_idex_top_k = first_idex[top_k_smallest_indices].flatten()
    second_idex_top_k = second_idex[top_k_smallest_indices].flatten()
    distance_vector_top_k = torch.Tensor(distance_vector[top_k_smallest_indices]).view(
        -1, 3
    )

    # Build output:
    edge_index_top_k = torch.vstack(
        (torch.LongTensor(first_idex_top_k), torch.LongTensor(second_idex_top_k))
    ).to(device=out_device)

    distance_vector_top_k = torch.as_tensor(
        distance_vector_top_k,
        dtype=out_dtype,
        device=out_device,
    )
    return edge_index_top_k, distance_vector_top_k


def assert_edges_match_nequips(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    edge_method: EdgeCreationMethod,
    half_supercell: bool = False,
    max_num_neighbors: int = 20,
    max_radius: float = 6.0,
):
    (
        edge_index,
        edge_vectors,
        _,
    ) = featurization_utilities.compute_pbc_radius_graph(
        positions=positions,
        # pbc[0] here is because we have added an outer dim for batching,
        # but a bunch of internal stuff assumesthat the pbc is 3x3 exactly.
        cell=cell[0],
        pbc=pbc,
        radius=max_radius,
        max_number_neighbors=max_num_neighbors,  # set to s large value so that all neighbours are considered
        edge_method=edge_method,
        half_supercell=half_supercell,
    )
    edge_index = edge_index.to(positions.device)
    edge_vectors = edge_vectors.to(positions.device)

    (
        edge_index_nequip,
        edge_vectors_nequip,
    ) = compute_pbc_radius_graph_nequip(
        positions=positions,
        cell=cell[0],
        radius=max_radius,
        max_number_neighbors=max_num_neighbors,
        self_interaction=False,
        strict_self_interaction=True,
        pbc=True,
    )

    # get list of counts for the sender nodes
    unique_counts = torch.unique(edge_index[0], return_counts=True)[1].tolist()
    start_index = 0
    for sender_node in range(len(unique_counts)):
        # for each sender_nodes in the unique_counts list,
        # get the receivers node and count the number of appearances from
        # our edge_index calculation and edge_index_nequip from nequip library
        end_index = unique_counts[sender_node] + start_index
        assert torch.all(
            torch.unique(edge_index[1][start_index:end_index], return_counts=True)[1]
            == torch.unique(
                edge_index_nequip[1][start_index:end_index], return_counts=True
            )[1]
        ), (
            f"Receiver edge indices of edge_index and edge_index_nequip for the start_index: "
            f"{start_index} and end_index: {end_index} are not equal."
        )
        # As the edge vectors may have opposite directions, we compare their norms here.
        edge_vector_norm = torch.norm(edge_vectors[start_index:end_index])
        edge_vector_norm_nequip = torch.norm(edge_vectors_nequip[start_index:end_index])
        torch.testing.assert_close(
            edge_vector_norm,
            edge_vector_norm_nequip,
            msg=(
                f"Edge vector norm: {edge_vector_norm} and edge vector norm nequip: "
                f"{edge_vector_norm_nequip} are not close."
            ),
        )
        start_index = end_index
