"""Tests featurization utilities."""

import ase
import ase.io
import numpy as np
import pytest
import torch
import torch.testing
from scipy.spatial.transform import Rotation

from orb_models.common.atoms import graph_featurization
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from tests.common.atoms.conftest import (
    assert_edges_match_nequips,
    get_CO2,
    get_zeolites,
)


def _matrix_to_set_of_vectors(matrix):
    ndarray_to_tuple = lambda x: tuple(map(tuple, x))
    return set([ndarray_to_tuple(x) for x in np.split(matrix, len(matrix))])


def test_pbc_graph_raises_error_for_periodic_zero_cell():
    adsorbate = get_CO2()
    positions = torch.tensor(adsorbate.positions, dtype=torch.float32)
    periodic_boundaries = torch.zeros((3, 3), dtype=torch.float32)
    pbc = torch.tensor([True, True, True], dtype=torch.bool)
    with pytest.raises(ValueError, match="'pbc' is True, but 'cell' is all zeros!"):
        graph_featurization.compute_pbc_radius_graph(
            positions=positions,
            radius=6.0,
            max_number_neighbors=20,
            cell=periodic_boundaries,
            pbc=pbc,
        )


@pytest.mark.parametrize(
    "edge_method",
    [
        None,
        "knn_brute_force",
        "knn_scipy",
        "knn_alchemi",
        pytest.param(
            "knn_cuml_brute",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
        pytest.param(
            "knn_cuml_rbc",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
    ],
)
def test_artificial_pbc_graph(edge_method):
    """Tests construct a multi graph using periodic boundary conditions."""
    original_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")

    # CASE I
    positions = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
    pbc = torch.tensor([True, True, True], dtype=torch.bool)
    radius = 1.1
    cell = torch.eye(3, dtype=torch.float32)
    out = graph_featurization.compute_pbc_radius_graph(
        positions=positions,
        radius=radius,
        max_number_neighbors=20,
        cell=cell,
        pbc=pbc,
        edge_method=edge_method,
    )
    edge_index, vectors, _ = out
    # There should be 6 edges, because the radius doesn't reach the corners of the grid.
    assert len(edge_index[0]) == 6
    # All edges should be pointing to the same index
    assert torch.all(edge_index == 0)
    # All edges should be of length 1.
    assert torch.all(torch.linalg.norm(vectors, axis=1) == 1.0)

    # CASE II
    # Make the unit cell a 30 degree rotation.
    cell = torch.tensor(
        Rotation.from_euler("xyz", (30, 30, 30), degrees=True).as_matrix(),
        dtype=torch.float32,
    ).T.contiguous()

    positions = positions @ cell
    out = graph_featurization.compute_pbc_radius_graph(
        positions=positions,
        radius=radius + 1e-7,
        max_number_neighbors=20,
        cell=cell,
        pbc=pbc,
        edge_method=edge_method,
    )
    edge_index, pred_rotated_vectors, _ = out
    # There should be 6 edges, because the radius doesn't reach the corners of the grid.
    assert len(edge_index[0]) == 6
    # All edges should be pointing to the same index
    assert torch.all(edge_index == 0)
    # All edges should be of length 1.
    assert torch.all((torch.linalg.norm(pred_rotated_vectors, axis=1) - 1.0) < 1e-6)
    # Rotated vectors should be the original vector but just rotated.
    # Sort to make sure aligning edge index
    pred_rotated_vectors = pred_rotated_vectors[pred_rotated_vectors[:, 0].argsort()]
    rotated_vectors = vectors @ cell.to(vectors.device)
    rotated_vectors = rotated_vectors[rotated_vectors[:, 0].argsort()]

    torch.testing.assert_close(pred_rotated_vectors, rotated_vectors)
    torch.set_float32_matmul_precision(original_precision)


@pytest.mark.parametrize(
    "edge_method",
    [
        None,
        "knn_brute_force",
        "knn_scipy",
        "knn_alchemi",
        pytest.param(
            "knn_cuml_brute",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
        pytest.param(
            "knn_cuml_rbc",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
    ],
)
def test_CO2_graph(edge_method):
    adsorbate = get_CO2()
    positions = torch.tensor(adsorbate.positions, dtype=torch.float32)
    cell = torch.zeros((3, 3), dtype=torch.float32)
    pbc = torch.tensor([False], dtype=torch.bool)
    out = graph_featurization.compute_pbc_radius_graph(
        positions=positions,
        radius=6.0,
        max_number_neighbors=20,
        cell=cell,
        pbc=pbc,
        edge_method=edge_method,
    )
    edge_index, vectors, _ = out

    assert edge_index.shape[1] == len(vectors) == 6

    for i in range(edge_index.shape[1]):
        matches = (
            edge_index[:, i].cpu() == torch.tensor([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]])
        ).all(-1)
        assert matches.any()

        expected_vectors = torch.tensor(
            [
                [1.1600, 0.0000, 0.0000],
                [2.3200, 0.0000, 0.0000],
                [-1.1600, 0.0000, 0.0000],
                [1.1600, 0.0000, 0.0000],
                [-2.3200, 0.0000, 0.0000],
                [-1.1600, 0.0000, 0.0000],
            ],
            dtype=torch.float32,
        )
        expected_vector = expected_vectors[matches]
        torch.testing.assert_close(
            vectors[i].cpu(),
            expected_vector[0],
        )


@pytest.mark.parametrize(
    "zeolite_framework",
    [
        "STW",
        "IFR",
        "VSV",
        "SAV",
    ],
)
@pytest.mark.parametrize(
    "edge_method",
    [
        None,
        "knn_brute_force",
        "knn_scipy",
        "knn_alchemi",
        pytest.param(
            "knn_cuml_brute",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
        pytest.param(
            "knn_cuml_rbc",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
    ],
)
@pytest.mark.parametrize("half_supercell", [False, True])
def test_zeolite_graphs(fixtures_path, half_supercell, edge_method, zeolite_framework):
    """
    Compare our graph construction to nequips on real zeolite systems.
    """
    original_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")

    zeos = get_zeolites(fixtures_path)
    adsorbate = get_CO2()
    framework = zeos[zeolite_framework]
    atoms = framework + adsorbate
    positions = torch.from_numpy(atoms.positions).to(torch.float)
    cell = torch.Tensor(atoms.cell.array[None, ...]).to(torch.float)
    pbc = torch.tensor(atoms.get_pbc())

    assert_edges_match_nequips(
        positions,
        cell,
        pbc,
        edge_method,
        half_supercell,
        max_num_neighbors=120,
        max_radius=6.0,
    )
    torch.set_float32_matmul_precision(original_precision)


@pytest.mark.parametrize(
    "edge_method",
    [
        None,
        "knn_brute_force",
        "knn_scipy",
        "knn_alchemi",
        pytest.param(
            "knn_cuml_brute",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
        pytest.param(
            "knn_cuml_rbc",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
    ],
)
def test_thin_mp_traj_graphs(fixtures_path, edge_method):
    """
    Compare our graph construction to nequips on thin mptraj systems that have interactions
    requiring a large supercell, greater than 3x3x3 when max_num_neighbors is very large (120).

    These are mp-traj val systems with indices:
    3171, 13172, 13173, 13174, 13175, 13176, 13177, 13178, 13179, 13180, 13181
    """
    db = ase.db.connect(fixtures_path / "10_thin_mp_traj_systems.db")
    for row in db.select():
        atoms = row.toatoms()
        positions = torch.from_numpy(atoms.positions).to(torch.float)
        cell = torch.Tensor(atoms.cell.array[None, ...]).to(torch.float)
        pbc = torch.tensor(atoms.get_pbc())
        assert_edges_match_nequips(
            positions,
            cell,
            pbc,
            edge_method,
            max_num_neighbors=120,
            max_radius=6.0,
        )


@pytest.mark.parametrize(
    "edge_method",
    [
        None,
        "knn_brute_force",
        "knn_scipy",
        "knn_alchemi",
        pytest.param(
            "knn_cuml_brute",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
        pytest.param(
            "knn_cuml_rbc",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
    ],
)
def test_artificial_batched_graph(edge_method):
    """Tests batched pbc radius graph construction."""
    radius = 1.0 + 1e-6
    positions = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=torch.float32)
    cells = torch.broadcast_to(torch.eye(3, dtype=torch.float32).unsqueeze(0), (2, 3, 3))
    pbc = torch.tensor([[True, True, True], [True, True, True]], dtype=torch.bool)
    out = graph_featurization.batch_compute_pbc_radius_graph(
        positions=positions,
        radius=radius,
        max_number_neighbors=torch.tensor([120, 120]),
        cells=cells,
        pbcs=pbc,
        n_node=torch.Tensor([1, 1]).long(),
        node_batch_index=torch.tensor([0, 1]).long(),
        edge_method=edge_method,
    )
    edge_index, vectors, _, _ = out
    edge_index = edge_index.cpu()
    vectors = vectors.cpu()
    # There should be 6 edges per element of batch, because the radius doesn't
    # reach the corners of the grid.
    assert len(edge_index[0]) == 12
    # All edges should be pointing to the same index
    assert torch.all(edge_index[0][:6] == 0)
    assert torch.all(edge_index[0][6:] == 1)
    assert torch.all(edge_index[1][:6] == 0)
    assert torch.all(edge_index[1][6:] == 1)
    # All edges should be of length 1.
    assert torch.all(torch.linalg.norm(vectors, axis=1) == 1.0)
    # Make the unit cell a 30 degree rotation.
    cells = (
        torch.tensor(
            np.broadcast_to(
                Rotation.from_euler("xyz", (30, 30, 30), degrees=True).as_matrix(),
                (2, 3, 3),
            ),
            dtype=torch.float32,
        )
        .transpose(1, 2)
        .contiguous()
    )

    # Positions should be rotated too.
    positions = positions @ cells[0]
    out = graph_featurization.batch_compute_pbc_radius_graph(
        positions=positions,
        radius=radius + 1e-4,
        max_number_neighbors=torch.tensor([120, 120]),
        cells=cells,
        pbcs=pbc,
        n_node=torch.Tensor([1, 1]).long(),
        node_batch_index=torch.tensor([0, 1]).long(),
        edge_method=edge_method,
    )
    edge_index, rotated_vectors, _, _ = out
    edge_index = edge_index.cpu()
    rotated_vectors = rotated_vectors.cpu()
    # All edges should be of length 1.
    assert torch.all((torch.linalg.norm(rotated_vectors, axis=1) - 1.0) < 1e-6)
    # Rotated vectors should be the original vector but just rotated.
    out_rotated_back_to_origin = rotated_vectors @ cells[0].T
    # Round because of numerics and make an ndarray
    out_rotated_back_to_origin = np.around(np.array(out_rotated_back_to_origin), 5)
    # Get the sets of the two graphs
    out_rotated_1 = _matrix_to_set_of_vectors(np.round(out_rotated_back_to_origin[:6], decimals=3))
    out_rotated_2 = _matrix_to_set_of_vectors(np.round(out_rotated_back_to_origin[6:], decimals=3))
    # Same for original vectors
    vectors = vectors.numpy()
    vectors_1 = _matrix_to_set_of_vectors(np.round(vectors[:6], decimals=3))
    vectors_2 = _matrix_to_set_of_vectors(np.round(vectors[6:], decimals=3))
    assert (out_rotated_1 - vectors_1) == set()
    assert (out_rotated_2 - vectors_2) == set()


@pytest.mark.parametrize(
    "edge_method",
    [
        None,
        "knn_brute_force",
        "knn_scipy",
        "knn_alchemi",
        pytest.param(
            "knn_cuml_brute",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
        pytest.param(
            "knn_cuml_rbc",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
    ],
)
def test_batched_and_unbatched_equivalence(shared_fixtures_path, edge_method):
    """Tests batched pbc radius graph."""
    original_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")

    with (shared_fixtures_path / "structures/atom_ocp22.json").open("r") as f:
        atoms = ase.Atoms(ase.io.read(f))

    positions = torch.tensor(atoms.get_positions()).float()
    cell = torch.tensor(atoms.get_cell()).float()
    pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool)
    radius = 6.0
    max_num_neighbors = 120
    natoms = len(atoms)

    idx, vectors, _ = graph_featurization.compute_pbc_radius_graph(
        positions=positions,
        cell=cell,
        pbc=pbc,
        radius=radius,
        max_number_neighbors=max_num_neighbors,
        edge_method=edge_method,
    )

    node_0_connections = idx[1][idx[0] == 0]
    cell2 = cell.clone() * 0.9  # alter the cell, which alters the edges
    idx_null, vectors_null, _ = graph_featurization.compute_pbc_radius_graph(
        positions=positions,
        cell=cell2,
        pbc=pbc,
        radius=radius,
        max_number_neighbors=max_num_neighbors,
        edge_method=edge_method,
    )

    batched_positions = torch.cat([positions] * 2)
    batched_cells = torch.stack([cell, cell2])
    batched_pbc = torch.stack([pbc, pbc])
    out = graph_featurization.batch_compute_pbc_radius_graph(
        positions=batched_positions,
        radius=radius,
        cells=batched_cells,
        pbcs=batched_pbc,
        n_node=torch.Tensor([natoms, natoms]).long(),
        node_batch_index=torch.tensor([0] * natoms + [1] * natoms).long(),
        max_number_neighbors=torch.tensor([max_num_neighbors, max_num_neighbors]),
        edge_method=edge_method,
    )
    idx2, vectors2, _, _ = out
    node_0_connections2 = idx2[1][idx2[0] == 0]
    assert set(node_0_connections2.tolist()) == set(node_0_connections.tolist())
    torch.testing.assert_close(vectors, vectors2[: len(vectors)])
    torch.testing.assert_close(vectors_null, vectors2[len(vectors) :])
    torch.set_float32_matmul_precision(original_precision)


def test_graph_vectors_are_consistent(dataset_and_loader):
    """
    Check the differentiable edge computation is consistent with the
    non-differentiable computation for a small dataset of pbc systems.
    """
    dataloader = dataset_and_loader[1]
    batch = next(iter(dataloader))
    torch.testing.assert_close(
        batch.compute_differentiable_edge_vectors()[0],
        batch.edge_features["vectors"],
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("zeolite_framework", ["STW", "IFR", "VSV", "SAV"])
@pytest.mark.parametrize("half_supercell", [False, True])
def test_graph_vectors_are_consistent_for_zeolites(
    fixtures_path, half_supercell, zeolite_framework
):
    """Same as test_graph_vectors_are_consistent, but for a range of zeolites."""
    zeos = get_zeolites(fixtures_path)
    adsorbate = get_CO2()
    framework = zeos[zeolite_framework]
    system = framework + adsorbate

    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=120)
    atom_graph = adapter.from_ase_atoms(system, half_supercell=half_supercell)
    torch.testing.assert_close(
        atom_graph.compute_differentiable_edge_vectors()[0],
        atom_graph.edge_features["vectors"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_artificial_pbc_graph_creates_supercell():
    """Tests construct a multi graph using periodic boundary conditions."""
    positions = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
    pbc = torch.tensor([True, True, True], dtype=torch.bool)
    radius = 1.0 + 1e-6
    cell = torch.eye(3, dtype=torch.float32)
    out = graph_featurization.compute_pbc_radius_graph(
        positions=positions,
        radius=radius,
        max_number_neighbors=120,
        cell=cell,
        pbc=pbc,
    )
    edge_index, vectors, _ = out

    assert len(vectors) == 6


def test_artificial_nonpbc_graph_doesnt_create_supercell():
    """Tests construct a multi graph using periodic boundary conditions."""
    positions = torch.tensor([[0.5, 0.5, 0.8], [0.5, 0.2, 0.5]], dtype=torch.float32)
    pbc = torch.tensor([False, False, False], dtype=torch.bool)
    radius = 1.0
    cell = torch.eye(3, dtype=torch.float32)
    out = graph_featurization.compute_pbc_radius_graph(
        positions=positions,
        radius=radius,
        max_number_neighbors=120,
        cell=cell,
        pbc=pbc,
    )
    edge_index, vectors, _ = out

    # One edge from each atom to the other
    assert len(vectors) == 2


@pytest.mark.parametrize(
    "edge_method",
    [
        None,
        "knn_brute_force",
        "knn_scipy",
        "knn_alchemi",
        pytest.param(
            "knn_cuml_brute",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
        pytest.param(
            "knn_cuml_rbc",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
    ],
)
def test_max_num_neighbors_selects_nearest(edge_method):
    """Test that max_num_neighbors returns the nearest neighbors, not random ones."""
    # Central atom at origin, neighbors at known distances along x-axis
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Central atom (index 0)
            [5.0, 0.0, 0.0],  # Neighbor at distance 5.0
            [2.0, 0.0, 0.0],  # Neighbor at distance 2.0
            [3.0, 0.0, 0.0],  # Neighbor at distance 3.0
            [1.0, 0.0, 0.0],  # Neighbor at distance 1.0
            [4.0, 0.0, 0.0],  # Neighbor at distance 4.0
        ],
        dtype=torch.float32,
    )

    cell = torch.zeros((3, 3), dtype=torch.float32)
    pbc = torch.tensor([False, False, False], dtype=torch.bool)
    radius = 6.0  # All 5 neighbors within radius
    max_num_neighbors = 3  # Only keep 3

    edge_index, vectors, _ = graph_featurization.compute_pbc_radius_graph(
        positions=positions,
        cell=cell,
        pbc=pbc,
        radius=radius,
        max_number_neighbors=max_num_neighbors,
        edge_method=edge_method,
    )

    # Filter edges where central atom (0) is the sender
    central_as_sender_mask = edge_index[0] == 0

    # Get distances for edges FROM the central atom
    central_sender_vectors = vectors[central_as_sender_mask]
    central_sender_distances = torch.linalg.norm(central_sender_vectors, dim=-1)

    # Should have exactly 3 neighbors (the nearest)
    assert len(central_sender_distances) == max_num_neighbors

    # All distances should be <= 3.0 (the 3 nearest neighbors at 1.0, 2.0, 3.0)
    assert torch.all(central_sender_distances <= 3.0 + 1e-6)

    # Verify the actual distances are 1.0, 2.0, 3.0
    sorted_distances = torch.sort(central_sender_distances).values
    expected_distances = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    torch.testing.assert_close(sorted_distances, expected_distances, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "edge_method",
    [
        None,
        "knn_brute_force",
        "knn_scipy",
        "knn_alchemi",
        pytest.param(
            "knn_cuml_brute",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
        pytest.param(
            "knn_cuml_rbc",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
    ],
)
def test_max_num_neighbors_selects_nearest_batched(edge_method):
    """Test that max_num_neighbors returns the nearest neighbors in batched case."""
    # Batch 1: Central atom at origin, neighbors at shuffled distances
    # Batch 2: Same setup but with different distances
    positions = torch.tensor(
        [
            # Batch 1 (6 atoms)
            [0.0, 0.0, 0.0],  # Central atom (index 0)
            [5.0, 0.0, 0.0],  # Neighbor at distance 5.0
            [2.0, 0.0, 0.0],  # Neighbor at distance 2.0
            [3.0, 0.0, 0.0],  # Neighbor at distance 3.0
            [1.0, 0.0, 0.0],  # Neighbor at distance 1.0
            [4.0, 0.0, 0.0],  # Neighbor at distance 4.0
            # Batch 2 (6 atoms)
            [0.0, 0.0, 0.0],  # Central atom (index 6)
            [4.5, 0.0, 0.0],  # Neighbor at distance 4.5
            [1.5, 0.0, 0.0],  # Neighbor at distance 1.5
            [3.5, 0.0, 0.0],  # Neighbor at distance 3.5
            [0.5, 0.0, 0.0],  # Neighbor at distance 0.5
            [2.5, 0.0, 0.0],  # Neighbor at distance 2.5
        ],
        dtype=torch.float32,
    )

    cells = torch.zeros((2, 3, 3), dtype=torch.float32)
    pbcs = torch.tensor([[False, False, False], [False, False, False]], dtype=torch.bool)
    radius = 6.0  # All neighbors within radius
    max_num_neighbors = 3  # Only keep 3 per atom

    edge_index, vectors, _, _ = graph_featurization.batch_compute_pbc_radius_graph(
        positions=positions,
        cells=cells,
        pbcs=pbcs,
        radius=radius,
        max_number_neighbors=torch.tensor([max_num_neighbors, max_num_neighbors]),
        n_node=torch.tensor([6, 6]),
        node_batch_index=torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        edge_method=edge_method,
    )

    # Check batch 1: central atom is index 0
    central_1_mask = edge_index[0] == 0
    central_1_vectors = vectors[central_1_mask]
    central_1_distances = torch.linalg.norm(central_1_vectors, dim=-1)

    assert len(central_1_distances) == max_num_neighbors
    sorted_distances_1 = torch.sort(central_1_distances).values
    expected_distances_1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    torch.testing.assert_close(sorted_distances_1, expected_distances_1, atol=1e-5, rtol=1e-5)

    # Check batch 2: central atom is index 6
    central_2_mask = edge_index[0] == 6
    central_2_vectors = vectors[central_2_mask]
    central_2_distances = torch.linalg.norm(central_2_vectors, dim=-1)

    assert len(central_2_distances) == max_num_neighbors
    sorted_distances_2 = torch.sort(central_2_distances).values
    expected_distances_2 = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)
    torch.testing.assert_close(sorted_distances_2, expected_distances_2, atol=1e-5, rtol=1e-5)
