"""Tests featurization utilities."""

import functools

import ase
import ase.io
import ase.neighborlist
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from orb_models.forcefield import featurization_utilities


def test_gaussian_basis_function():
    """Tests gaussian basis function."""
    in_scalars = torch.tensor([0.0, 9.0])
    out = featurization_utilities.gaussian_basis_function(
        in_scalars, num_bases=10, radius=10
    )
    assert out[0][0] == 1.0
    assert out[1][-1] == 1.0
    assert out.bool().all()


def test_featurize_edges():
    """Tests edge featurization."""
    sqrt_point_3 = torch.sqrt(torch.tensor(0.3))
    in_vectors = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0],
            [0.5 / sqrt_point_3, 0.2 / sqrt_point_3, 0.1 / sqrt_point_3],
            [2.0, 0.0, 0.0],
        ]
    )
    distance_fn = functools.partial(
        featurization_utilities.gaussian_basis_function, radius=10, num_bases=10
    )
    out = featurization_utilities.featurize_edges(in_vectors, distance_fn)
    unit_vectors = out[:, -3:]
    gbfs = out[:, :-3]
    assert torch.allclose(unit_vectors[0], in_vectors[0])
    assert torch.allclose(unit_vectors[1], in_vectors[1])
    assert torch.allclose(unit_vectors[2], in_vectors[2])
    assert torch.allclose(unit_vectors[3], in_vectors[1])
    assert gbfs[0][1] == 1.0
    assert gbfs[3][2] == 1.0


@pytest.mark.parametrize("brute_force", [True, False])
@pytest.mark.parametrize("library", ["scipy", "pynanoflann"])
def test_pbc_radius_graph(library, brute_force):
    """Tests construct a multi graph using periodic boundary conditions."""
    positions = torch.tensor([[0.5, 0.5, 0.5]])
    radius = 1.0
    periodic_boundaries = torch.eye(3)
    out = featurization_utilities.compute_pbc_radius_graph(
        positions=positions,
        radius=radius,
        max_number_neighbors=20,
        periodic_boundaries=periodic_boundaries,
        brute_force=brute_force,
        library=library,
    )
    edge_index, vectors = out
    # There should be 6 edges, because the radius doesn't reach the corners of the grid.
    assert len(edge_index[0]) == 6
    # # All edges should be pointing to the same index
    assert torch.all(edge_index == 0)
    # # All edges should be of length 1.
    assert torch.all(torch.linalg.norm(vectors, axis=1) == 1.0)
    # Make the unit cell a 30 degree rotation.
    periodic_boundaries = torch.tensor(
        Rotation.from_euler("xyz", (30, 30, 30), degrees=True).as_matrix()
    ).T

    positions = positions.double() @ periodic_boundaries
    out = featurization_utilities.compute_pbc_radius_graph(
        positions=positions,
        radius=radius + 1e-7,
        max_number_neighbors=20,
        periodic_boundaries=periodic_boundaries.float(),
        brute_force=brute_force,
        library=library,
    )
    edge_index, pred_rotated_vectors = out
    # There should be 6 edges, because the radius doesn't reach the corners of the grid.
    assert len(edge_index[0]) == 6
    # All edges should be pointing to the same index
    assert torch.all(edge_index == 0)
    # All edges should be of length 1.
    assert torch.all((torch.linalg.norm(pred_rotated_vectors, axis=1) - 1.0) < 1e-6)
    # Rotated vectors should be the original vector but just rotated.
    # Sort to make sure aligning edge index
    pred_rotated_vectors = pred_rotated_vectors[pred_rotated_vectors[:, 0].argsort()]
    rotated_vectors = vectors.double() @ periodic_boundaries
    rotated_vectors = rotated_vectors[rotated_vectors[:, 0].argsort()]

    assert torch.allclose(pred_rotated_vectors, rotated_vectors)


@pytest.mark.parametrize("brute_force", [True, False])
@pytest.mark.parametrize("library", ["scipy", "pynanoflann"])
def test_pbc_radius_graph_molecule(library, brute_force):
    adsorbate = _get_CO2()
    positions = torch.tensor(adsorbate.positions, dtype=torch.float32)
    periodic_boundaries = torch.zeros((3, 3), dtype=torch.float32)
    out = featurization_utilities.compute_pbc_radius_graph(
        positions=positions,
        radius=10.0,
        max_number_neighbors=20,
        periodic_boundaries=periodic_boundaries,
        brute_force=brute_force,
        library=library,
    )
    edge_index, vectors = out

    assert edge_index.shape[1] == len(vectors) == 6

    for i in range(edge_index.shape[1]):
        matches = (
            edge_index[:, i]
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
        assert torch.allclose(
            vectors[i],
            expected_vector,
        )


@pytest.mark.parametrize("brute_force", [True, False])
@pytest.mark.parametrize("library", ["scipy", "pynanoflann"])
def test_batch_pbc_radius_graph_equivalence(fixtures_path, library, brute_force):
    """Tests batched pbc radius graph."""
    with (fixtures_path / "atom_ocp22.json").open("r") as f:
        atoms = ase.io.read(f)
    positions = torch.tensor(atoms.get_positions()).float()
    periodic_boundaries = torch.tensor(atoms.get_cell()).float()
    radius = 5.0

    idx, vectors = featurization_utilities.compute_pbc_radius_graph(
        positions=positions,
        periodic_boundaries=periodic_boundaries,
        radius=radius,
        brute_force=brute_force,
        library=library,
    )

    node_0_connections = idx[1][idx[0] == 0]
    idx_null, vectors_null = featurization_utilities.compute_pbc_radius_graph(
        positions=positions,
        periodic_boundaries=torch.eye(3),
        radius=radius,
        brute_force=brute_force,
        library=library,
    )

    positions2 = torch.cat([positions] * 2)
    periodic_boundaries2 = torch.stack([periodic_boundaries, torch.eye(3)])
    out = featurization_utilities.batch_compute_pbc_radius_graph(
        positions=positions2,
        radius=radius,
        periodic_boundaries=periodic_boundaries2,
        image_idx=torch.Tensor([96, 96]).long(),
        brute_force=brute_force,
        library=library,
    )
    idx2, vectors2, _ = out
    node_0_connections2 = idx2[1][idx2[0] == 0]
    assert set(node_0_connections2.tolist()) == set(node_0_connections.tolist())
    assert torch.allclose(vectors, vectors2[: len(vectors)])
    assert torch.allclose(vectors_null, vectors2[len(vectors) :])


def _matrix_to_set_of_vectors(matrix):
    ndarray_to_tuple = lambda x: tuple(map(tuple, x))
    return set([ndarray_to_tuple(x) for x in np.split(matrix, len(matrix))])


@pytest.mark.parametrize("brute_force", [True, False])
@pytest.mark.parametrize("library", ["scipy", "pynanoflann"])
def test_batch_pbc_radius_graph(library, brute_force):
    """Tests batched pbc radius graph."""
    positions = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    radius = 1.0
    periodic_boundaries = torch.broadcast_to(torch.eye(3).unsqueeze(0), (2, 3, 3))
    out = featurization_utilities.batch_compute_pbc_radius_graph(
        positions=positions,
        radius=radius,
        max_number_neighbors=6,
        periodic_boundaries=periodic_boundaries.float(),
        image_idx=torch.Tensor([1, 1]).long(),
        brute_force=brute_force,
        library=library,
    )
    edge_index, vectors, _ = out
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
    periodic_boundaries = torch.tensor(
        np.broadcast_to(
            Rotation.from_euler("xyz", (30, 30, 30), degrees=True).as_matrix(),
            (2, 3, 3),
        )
    ).transpose(1, 2)
    # Positions should be rotated too.
    positions = positions.double() @ periodic_boundaries[0]
    out = featurization_utilities.batch_compute_pbc_radius_graph(
        positions=positions,
        radius=radius + 1e-4,
        max_number_neighbors=6,
        periodic_boundaries=periodic_boundaries.float(),
        image_idx=torch.Tensor([1, 1]).long(),
        brute_force=brute_force,
        library=library,
    )
    edge_index, rotated_vectors, _ = out
    # All edges should be of length 1.
    assert torch.all((torch.linalg.norm(rotated_vectors, axis=1) - 1.0) < 1e-6)
    # Rotated vectors should be the original vector but just rotated.
    out_rotated_back_to_origin = rotated_vectors @ periodic_boundaries[0].T
    # Round because of numerics and make an ndarray
    out_rotated_back_to_origin = np.around(np.array(out_rotated_back_to_origin), 5)
    # Get the sets of the two graphs
    out_rotated_1 = _matrix_to_set_of_vectors(out_rotated_back_to_origin[:6])
    out_rotated_2 = _matrix_to_set_of_vectors(out_rotated_back_to_origin[6:])
    # Same for original vectors
    vectors_1 = _matrix_to_set_of_vectors(np.array(vectors[:6]))
    vectors_2 = _matrix_to_set_of_vectors(np.array(vectors[6:]))
    assert (out_rotated_1 - vectors_1) == set()
    assert (out_rotated_2 - vectors_2) == set()


def test_batch_map_to_pbc_cell():
    """Test a batch where the first should be mapped and the second should be left alone."""
    unit_cell = torch.eye(3)
    null_pbc = torch.zeros((3, 3))
    position = torch.tensor([-0.5, 0.0, 0.0])
    positions = torch.stack([position] * 2)
    unit_cells = torch.stack([unit_cell, null_pbc])
    out = featurization_utilities.batch_map_to_pbc_cell(
        positions, unit_cells, torch.Tensor([1, 1]).long()
    )
    assert torch.allclose(out, torch.tensor([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]))


def test_compute_img_positions_columnar_torch():
    positions = torch.tensor([[0.5, 0.5, 0.5]])
    unit_cell = torch.eye(3)

    imgs = featurization_utilities._compute_img_positions_torch(positions, unit_cell)

    offsets = torch.from_numpy(featurization_utilities.OFFSETS).float()
    # For a identity unit cell, we should get the original position plus the offsets
    assert torch.allclose(imgs, positions + offsets)


def test_compute_img_positions_columnar_torch_batched():
    n_node = torch.tensor([3, 7])
    positions = torch.randn(10, 3)
    unit_cell = torch.eye(3).repeat(2, 1, 1)
    unit_cell[1] = unit_cell[1] * 2

    # TODO move this inside the function
    repeated_unit_cell = unit_cell.repeat_interleave(n_node, dim=0)

    imgs = featurization_utilities._compute_img_positions_torch(
        positions, repeated_unit_cell
    )
    offsets = torch.from_numpy(featurization_utilities.OFFSETS).float()

    expanded = offsets.unsqueeze(0) + positions.unsqueeze(1)
    assert torch.allclose(imgs[:3], expanded[:3])


def _get_CO2():
    """Function to get CO2."""

    positions = [
        (-1.16, 0, 0),  # O on the left
        (0, 0, 0),  # C in the middle
        (1.16, 0, 0),  # O on the right
    ]
    CO2 = ase.Atoms("OCO", positions=positions)
    return CO2
