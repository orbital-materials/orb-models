import itertools
import pytest
from collections import defaultdict

import ase.io
import numpy as np
import torch

from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs, SystemConfig


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


def _get_edge_sets(atom_graphs, with_vectors=False, precision=4):
    edges_with_counts = defaultdict(int)
    for s, r, vec in zip(
        atom_graphs.senders,
        atom_graphs.receivers,
        atom_graphs.edge_features["vectors"],
    ):
        vec_tuple = tuple([round(x.item(), precision) for x in vec])
        norm = round(vec.norm(dim=-1).item(), precision)
        if with_vectors:
            edges_with_counts[(s.item(), r.item(), norm, vec_tuple)] += 1
        else:
            edges_with_counts[(s.item(), r.item(), norm)] += 1
    return dict(edges_with_counts)


@pytest.mark.parametrize(
    "edge_method",
    [
        pytest.param(
            "knn_brute_force",
            marks=pytest.mark.xfail(
                reason="Brute-force knn is currently not translation invariant"
            ),
        ),
        pytest.param(
            "knn_cuml_brute",
            marks=[
                pytest.mark.xfail(
                    reason="knn_cuml_brute is currently not translation invariant"
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
                    reason="knn_cuml_rbc is currently not translation invariant"
                ),
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA is not available for CuML KNN.",
                ),
            ],
        ),
        pytest.param(
            "knn_scipy",
            marks=pytest.mark.xfail(reason="Scipy is not translation invariant"),
        ),
    ],
)
def test_featurization_is_translation_invariant_with_real_system(edge_method):

    original_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")

    atoms = _get_real_system()
    system_config = SystemConfig(
        radius=6.0,
        max_num_neighbors=20,
    )
    atom_graphs = ase_atoms_to_atom_graphs(
        atoms, system_config=system_config, edge_method=edge_method
    )
    edges = _get_edge_sets(atom_graphs, with_vectors=True, precision=3)

    # Test a grid of translations
    grid_size = 5
    for x, y, z in itertools.product(range(grid_size), repeat=3):
        atoms = atoms.copy()
        # Normalize the shift to be between -1 and 1
        shift = (np.array([x, y, z]) / grid_size) * 2 - 1
        atoms.positions += shift
        shifted_atom_graphs = ase_atoms_to_atom_graphs(
            atoms, system_config, edge_method=edge_method
        )
        shifted_edges = _get_edge_sets(
            shifted_atom_graphs, with_vectors=True, precision=3
        )
        assert edges == shifted_edges

    torch.set_float32_matmul_precision(original_precision)


@pytest.mark.parametrize(
    "edge_method",
    [
        pytest.param(
            "knn_brute_force",
            marks=pytest.mark.xfail(
                reason="Brute-force knn is currently not translation invariant with geometric systems"
            ),
        ),
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
        pytest.param(
            "knn_scipy",
            marks=pytest.mark.xfail(
                reason="Scipy is not translation invariant with geometric systems"
            ),
        ),
    ],
)
def test_featurization_is_translation_invariant_with_geometric_system(edge_method):
    original_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")

    atoms = ase.Atoms(
        "C" * 5,
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]]),
        cell=np.eye(3) * 1.01,
        pbc=True,
    )
    system_config = SystemConfig(
        radius=6.0,
        max_num_neighbors=120,
    )
    atom_graphs = ase_atoms_to_atom_graphs(
        atoms, system_config=system_config, edge_method=edge_method
    )
    edges = _get_edge_sets(atom_graphs, with_vectors=True, precision=3)

    # Test a grid of translations
    grid_size = 5
    for x, y, z in itertools.product(range(grid_size), repeat=3):
        atoms = atoms.copy()
        # Normalize the shift to be between -1 and 1
        shift = (np.array([x, y, z]) / grid_size) * 2 - 1
        atoms.positions += shift
        shifted_atom_graphs = ase_atoms_to_atom_graphs(
            atoms, system_config, edge_method=edge_method
        )
        shifted_edges = _get_edge_sets(
            shifted_atom_graphs, with_vectors=True, precision=3
        )
        assert edges == shifted_edges

    torch.set_float32_matmul_precision(original_precision)


@pytest.mark.parametrize(
    "edge_method,max_num_neighbors",
    [
        ("knn_brute_force", 120),
        ("knn_scipy", 120),
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
        pytest.param(
            "knn_brute_force",
            20,
            marks=pytest.mark.xfail(
                reason="Brute-force knn is currently not perfectly rotation "
                "invariant due to random selection of equidistant neighbors"
            ),
        ),
        pytest.param(
            "knn_scipy",
            20,
            marks=pytest.mark.xfail(
                reason="Scipy is not perfectly rotation invariant due to random "
                "selection of equidistant neighbors"
            ),
        ),
        pytest.param(
            "knn_cuml_brute",
            20,
            marks=[
                pytest.mark.xfail(
                    reason="knn_cuml_brute is currently not perfectly rotation invariant"
                ),
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA is not available for CuML KNN.",
                ),
            ],
        ),
        pytest.param(
            "knn_cuml_rbc",
            20,
            marks=[
                pytest.mark.xfail(
                    reason="knn_cuml_rbc is currently not perfectly rotation invariant"
                ),
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA is not available for CuML KNN.",
                ),
            ],
        ),
    ],
)
def test_featurization_is_rotation_invariant(edge_method, max_num_neighbors):
    original_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")

    atoms = _get_real_system()
    system_config = SystemConfig(radius=6.0, max_num_neighbors=max_num_neighbors)
    atom_graphs = ase_atoms_to_atom_graphs(
        atoms, system_config=system_config, edge_method=edge_method
    )
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
        rotated_atom_graphs = ase_atoms_to_atom_graphs(
            atoms,
            system_config=system_config,
            edge_method=edge_method,
        )
        rotated_edges = _get_edge_sets(
            rotated_atom_graphs, with_vectors=False, precision=3
        )
        assert edges == rotated_edges

        with pytest.raises(AssertionError):
            # This should fail because the vectors are not invariant
            rotated_edges_with_vectors = _get_edge_sets(
                rotated_atom_graphs, with_vectors=True, precision=3
            )
            assert edges == rotated_edges_with_vectors

    torch.set_float32_matmul_precision(original_precision)
