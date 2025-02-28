"""Tests featurization utilities."""

import ase
import ase.io
import ase.neighborlist
import numpy as np
import torch

from orb_models.forcefield import featurization_utilities


def test_gaussian_basis_function():
    """Tests gaussian basis function."""
    in_scalars = torch.tensor([0.0, 9.0])
    out = featurization_utilities.gaussian_basis_function(
        in_scalars, num_bases=10, radius=10.0
    )
    assert out[0][0] == 1.0
    assert out[1][-1] == 1.0
    assert out.bool().all()


def test_map_to_pbc_cell():
    """Tests map to pbc cell."""
    unit_cell = torch.eye(3)
    position = torch.tensor([[-0.5, 0.0, 0.0]])
    out = featurization_utilities.map_to_pbc_cell(position, unit_cell)
    assert torch.allclose(out, torch.tensor([0.5, 0.0, 0.0]))
    position = torch.tensor([[-0.2, 0.0, 0.0]])
    out = featurization_utilities.map_to_pbc_cell(position, unit_cell)
    assert torch.allclose(out, torch.tensor([0.8, 0.0, 0.0]))
    unit_cell = unit_cell / 2.0
    out = featurization_utilities.map_to_pbc_cell(position, unit_cell)
    assert torch.allclose(out, torch.tensor([0.3, 0.0, 0.0]))


def test_map_to_pbc_cell_ase(shared_fixtures_path):
    with (shared_fixtures_path / "atom_ocp22.json").open("r") as f:
        atoms = ase.Atoms(ase.io.read(f))
    positions = torch.tensor(atoms.get_positions())
    cell = torch.tensor(atoms.get_cell())

    positions = positions + (20 * torch.randn_like(positions))
    out = featurization_utilities.map_to_pbc_cell(positions, cell)
    wrapped = ase.geometry.wrap_positions(positions, cell, eps=0.0)
    wrapped_tensor = torch.tensor(wrapped)
    assert torch.allclose(out, wrapped_tensor)

    # batched, one system
    out = featurization_utilities.batch_map_to_pbc_cell(
        positions, cell.unsqueeze(0), torch.Tensor([positions.shape[0]]).long()
    )
    assert torch.allclose(out, wrapped_tensor.unsqueeze(0))

    # batched, one system
    out = featurization_utilities.batch_map_to_pbc_cell(
        torch.cat([positions, positions], dim=0),
        torch.stack([cell, cell], dim=0),
        torch.Tensor([positions.shape[0], positions.shape[0]]).long(),
    )
    assert torch.allclose(out[: positions.shape[0]], wrapped_tensor)
    assert torch.allclose(out[positions.shape[0] :], wrapped_tensor)


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


def test_minimal_supercell_defaults_to_3x3x3():
    """Test the minimal supercell construction."""
    positions = torch.tensor([[5.0, 5.0, 5.0]])
    cell = torch.eye(3)
    supercell_positions, integer_offsets = (
        featurization_utilities.construct_minimal_supercell(positions, cell)
    )
    assert supercell_positions.shape == (1, 27, 3)
    assert integer_offsets.shape == (27, 3)


def test_minimal_supercell():
    """Test the minimal supercell construction."""
    positions = torch.tensor([[5.0, 5.0, 5.0]])

    # We test two types of unit cell. Both have length-10 lattice vectors, and are
    # axis-aligned along the x and y directions. But we vary the angle of the final
    # lattice vector:
    # - pi/2 is completely axis-aligned and hence cubic. A 3x3x3 supercell is sufficient.
    # - pi/6 yields a 'thin diamond' unit cell. The minimal supercell is irregularly shaped,
    #   containing 3 * (3 + 4 + 5 + 4 + 3) = 57 unit cells.
    angles = [np.pi / 2, np.pi / 6]
    expected_sizes = [27, 57]

    for angle, expected_size in zip(angles, expected_sizes):
        x = [10, 0, 0]
        y = [0, 10, 0]
        z = [10 * np.cos(angle), 0, 10 * np.sin(angle)]
        cell = torch.tensor([x, y, z])
        supercell_positions, integer_offsets = (
            featurization_utilities.construct_minimal_supercell(
                positions, cell, cutoff=6.0
            )
        )
        assert supercell_positions.shape == (1, expected_size, 3)
        assert integer_offsets.shape == (expected_size, 3)
