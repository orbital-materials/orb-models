import itertools

import ase
import numpy as np
import pytest
import torch

from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from tests.common.atoms.utils import _get_edge_sets


def _get_real_system():
    # System MP-1234 from Materials Project
    atoms = ase.Atoms(
        symbols="Lu2Al4",
        pbc=True,
        cell=np.array(
            [
                [4.72281782, -0.0, 2.72672144],
                [1.5742712700000001, 4.45271696, 2.72672144],
                [0.0, 0.0, 5.45344188],
            ]
        ),
        positions=np.array(
            [
                [5.50995295, 3.89612734, 9.54352417],
                [0.78713614, 0.55658962, 1.3633606],
                [3.14854455, 2.22635848, 5.45344238],
                [3.14854455, 2.22635848, 2.72672144],
                [0.78713564, 2.22635848, 4.09008166],
                [2.36140891, 0.0, 4.09008166],
            ]
        ),
    )
    return atoms


@pytest.mark.parametrize(
    "edge_method",
    [
        "knn_brute_force",
        pytest.param(
            "knn_cuml_brute",
            marks=[
                pytest.mark.xfail(reason="knn_cuml_brute is currently not translation invariant"),
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA is not available for CuML KNN.",
                ),
            ],
        ),
        pytest.param(
            "knn_cuml_rbc",
            marks=[
                pytest.mark.xfail(reason="knn_cuml_rbc is currently not translation invariant"),
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA is not available for CuML KNN.",
                ),
            ],
        ),
        "knn_scipy",
        "knn_alchemi",
    ],
)
def test_featurization_is_translation_invariant_with_real_system(edge_method):
    original_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")

    atoms = _get_real_system()

    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=120)
    atom_graphs = adapter.from_ase_atoms(atoms, edge_method=edge_method)
    edges = _get_edge_sets(atom_graphs, with_vectors=True, precision=3)

    # Test a grid of translations
    grid_size = 5
    for x, y, z in itertools.product(range(grid_size), repeat=3):
        atoms = atoms.copy()
        # Normalize the shift to be between -1 and 1
        shift = (np.array([x, y, z]) / grid_size) * 2 - 1
        atoms.positions += shift
        shifted_atom_graphs = adapter.from_ase_atoms(atoms, edge_method=edge_method)
        shifted_edges = _get_edge_sets(shifted_atom_graphs, with_vectors=True, precision=3)
        assert edges == shifted_edges

    torch.set_float32_matmul_precision(original_precision)


@pytest.mark.parametrize(
    "edge_method",
    [
        "knn_alchemi",
        "knn_brute_force",
        pytest.param(
            "knn_cuml_brute",
            marks=[
                pytest.mark.xfail(
                    reason="knn_cuml_brute is currently not translation invariant with geometric systems"
                ),
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA is not available for CuML KNN.",
                ),
            ],
        ),
        pytest.param(
            "knn_cuml_rbc",
            marks=[
                pytest.mark.xfail(
                    reason="knn_cuml_rbc is currently not translation invariant with geometric systems"
                ),
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA is not available for CuML KNN.",
                ),
            ],
        ),
        "knn_scipy",
    ],
)
def test_featurization_is_translation_invariant_with_geometric_system(edge_method):
    original_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")

    atoms = ase.Atoms(
        "C" * 5,
        positions=np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [4, 4, 0], [0, 0, 4]]),
        cell=np.eye(3) * 4.01,
        pbc=True,
    )
    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=120)
    atom_graphs = adapter.from_ase_atoms(atoms, edge_method=edge_method)
    edges = _get_edge_sets(atom_graphs, with_vectors=True, precision=3)

    # Test a grid of translations
    grid_size = 5
    for x, y, z in itertools.product(range(grid_size), repeat=3):
        atoms = atoms.copy()
        # Normalize the shift to be between -1 and 1
        shift = (np.array([x, y, z]) / grid_size) * 2 - 1
        atoms.positions += shift
        shifted_atom_graphs = adapter.from_ase_atoms(atoms, edge_method=edge_method)
        shifted_edges = _get_edge_sets(shifted_atom_graphs, with_vectors=True, precision=3)
        assert edges == shifted_edges

    torch.set_float32_matmul_precision(original_precision)


@pytest.mark.parametrize(
    "edge_method,max_num_neighbors",
    [
        ("knn_brute_force", 120),
        ("knn_scipy", 120),
        ("knn_alchemi", 120),
        pytest.param(
            "knn_cuml_brute",
            120,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
        pytest.param(
            "knn_cuml_rbc",
            120,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available for CuML KNN.",
            ),
        ),
    ],
)
def test_featurization_is_rotation_invariant(edge_method, max_num_neighbors):
    original_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")

    atoms = _get_real_system()
    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=max_num_neighbors)
    atom_graphs = adapter.from_ase_atoms(atoms, edge_method=edge_method)
    edges = _get_edge_sets(atom_graphs, with_vectors=False, precision=3)

    # Test a grid of rotations
    grid_size = 6
    for x, y, z in itertools.product(range(1, grid_size), repeat=3):
        atoms = atoms.copy()

        rotation_angle = 360 / grid_size
        rotation_vec = np.array([x, y, z])
        atoms.rotate(
            rotation_angle,
            v=rotation_vec,
            center=atoms.get_center_of_mass(),
            rotate_cell=True,
        )
        rotated_atom_graphs = adapter.from_ase_atoms(atoms, edge_method=edge_method)
        rotated_edges = _get_edge_sets(rotated_atom_graphs, with_vectors=False, precision=3)
        assert edges == rotated_edges

        with pytest.raises(AssertionError):
            # This should fail because the vectors are not invariant
            rotated_edges_with_vectors = _get_edge_sets(
                rotated_atom_graphs, with_vectors=True, precision=3
            )
            assert edges == rotated_edges_with_vectors

    torch.set_float32_matmul_precision(original_precision)
