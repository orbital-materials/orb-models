"""Tests featurization utilities."""

import ase
import ase.io
import ase.neighborlist
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from orb_models.forcefield.atomic_system import SystemConfig, ase_atoms_to_atom_graphs
from orb_models.forcefield import featurization_utilities
from tests.atomgraphs.conftest import assert_edges_match_nequips, get_CO2, get_zeolites


def _matrix_to_set_of_vectors(matrix):
    ndarray_to_tuple = lambda x: tuple(map(tuple, x))
    return set([ndarray_to_tuple(x) for x in np.split(matrix, len(matrix))])


@pytest.mark.parametrize(
    "edge_method",
    [
        None,
        "knn_brute_force",
        "knn_scipy",
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

    positions = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
    pbc = torch.tensor([True, True, True], dtype=torch.bool)
    radius = 1.0
    cell = torch.eye(3, dtype=torch.float32)
    out = featurization_utilities.compute_pbc_radius_graph(
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
    # Make the unit cell a 30 degree rotation.
    cell = torch.tensor(
        Rotation.from_euler("xyz", (30, 30, 30), degrees=True).as_matrix(),
        dtype=torch.float32,
    ).T

    positions = positions @ cell
    out = featurization_utilities.compute_pbc_radius_graph(
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
    periodic_boundaries = torch.zeros((3, 3), dtype=torch.float32)
    pbc = torch.tensor([True, True, True], dtype=torch.bool)
    out = featurization_utilities.compute_pbc_radius_graph(
        positions=positions,
        radius=6.0,
        max_number_neighbors=20,
        cell=periodic_boundaries,
        pbc=pbc,
        edge_method=edge_method,
    )
    edge_index, vectors, _ = out

    assert edge_index.shape[1] == len(vectors) == 6

    for i in range(edge_index.shape[1]):
        matches = (
            edge_index[:, i].cpu()
            == torch.tensor([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]])
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


@pytest.mark.parametrize("zeolite_framework", ["STW", "IFR", "VSV", "SAV"])
@pytest.mark.parametrize(
    "edge_method",
    [
        None,
        "knn_brute_force",
        "knn_scipy",
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
        max_num_neighbors=20,
        max_radius=6.0,
    )
    torch.set_float32_matmul_precision(original_precision)


@pytest.mark.parametrize(
    "edge_method",
    [
        None,
        "knn_brute_force",
        "knn_scipy",
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
    radius = 1.0
    positions = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=torch.float32)
    cells = torch.broadcast_to(
        torch.eye(3, dtype=torch.float32).unsqueeze(0), (2, 3, 3)
    )
    pbc = torch.tensor([[True, True, True], [True, True, True]], dtype=torch.bool)
    out = featurization_utilities.batch_compute_pbc_radius_graph(
        positions=positions,
        radius=radius,
        max_number_neighbors=torch.tensor([6, 6]),
        cells=cells,
        pbc=pbc,
        n_node=torch.Tensor([1, 1]).long(),
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
    cells = torch.tensor(
        np.broadcast_to(
            Rotation.from_euler("xyz", (30, 30, 30), degrees=True).as_matrix(),
            (2, 3, 3),
        ),
        dtype=torch.float32,
    ).transpose(1, 2)

    # Positions should be rotated too.
    positions = positions @ cells[0]
    out = featurization_utilities.batch_compute_pbc_radius_graph(
        positions=positions,
        radius=radius + 1e-4,
        max_number_neighbors=torch.tensor([6, 6]),
        cells=cells,
        pbc=pbc,
        n_node=torch.Tensor([1, 1]).long(),
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
    out_rotated_1 = _matrix_to_set_of_vectors(
        np.round(out_rotated_back_to_origin[:6], decimals=3)
    )
    out_rotated_2 = _matrix_to_set_of_vectors(
        np.round(out_rotated_back_to_origin[6:], decimals=3)
    )
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

    with (shared_fixtures_path / "atom_ocp22.json").open("r") as f:
        atoms = ase.Atoms(ase.io.read(f))
    positions = torch.tensor(atoms.get_positions()).float()
    periodic_boundaries = torch.tensor(atoms.get_cell()).float()
    pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool)
    radius = 6.0
    max_num_neighbors = 20

    idx, vectors, _ = featurization_utilities.compute_pbc_radius_graph(
        positions=positions,
        cell=periodic_boundaries,
        pbc=pbc,
        radius=radius,
        max_number_neighbors=max_num_neighbors,
        edge_method=edge_method,
    )

    node_0_connections = idx[1][idx[0] == 0]
    idx_null, vectors_null, _ = featurization_utilities.compute_pbc_radius_graph(
        positions=positions,
        cell=torch.eye(3),
        pbc=pbc,
        radius=radius,
        max_number_neighbors=max_num_neighbors,
        edge_method=edge_method,
    )

    positions2 = torch.cat([positions] * 2)
    periodic_boundaries2 = torch.stack([periodic_boundaries, torch.eye(3)])
    pbc2 = torch.stack([pbc, pbc])
    out = featurization_utilities.batch_compute_pbc_radius_graph(
        positions=positions2,
        radius=radius,
        cells=periodic_boundaries2,
        pbc=pbc2,
        n_node=torch.Tensor([96, 96]).long(),
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

    atom_graph = ase_atoms_to_atom_graphs(
        system, system_config=SystemConfig(6.0, 20), half_supercell=half_supercell
    )
    torch.testing.assert_close(
        atom_graph.compute_differentiable_edge_vectors()[0],
        atom_graph.edge_features["vectors"],
        atol=1e-5,
        rtol=1e-5,
    )
