import ase
import torch
from ase import constraints

from orb_models.common.torch_utils import replace_tensor_elements_within_tolerance

ATOM_TYPE_K = 5


def get_atom_embedding(atoms: ase.Atoms, k_hot: bool = False) -> torch.Tensor:
    """Get an atomic embedding."""
    atomic_numbers = torch.from_numpy(atoms.numbers).to(torch.long)
    n_atoms = len(atomic_numbers)
    if k_hot:
        atom_type_embedding = (
            torch.ones(n_atoms, 118, dtype=torch.get_default_dtype()) * -ATOM_TYPE_K
        )
        atom_type_embedding[torch.arange(n_atoms), atomic_numbers] = ATOM_TYPE_K
    else:
        atom_type_embedding = torch.nn.functional.one_hot(atomic_numbers, num_classes=118).to(
            torch.get_default_dtype()
        )
    return atom_type_embedding.to(torch.get_default_dtype())


def get_ase_tags(atoms: ase.Atoms) -> torch.Tensor:
    """Get tags from ase.Atoms object."""
    tags = atoms.get_tags()
    tags = torch.Tensor(tags) if tags is not None else torch.zeros(len(atoms))
    return tags


def ase_fix_atoms_to_tensor(atoms: ase.Atoms) -> torch.Tensor | None:
    """Get fixed atoms from ase.Atoms object."""
    fixed_atoms = None
    if atoms.constraints is not None and len(atoms.constraints) > 0:
        constraint = atoms.constraints[0]
        if isinstance(constraint, constraints.FixAtoms):
            fixed_atoms = torch.zeros((len(atoms)), dtype=torch.bool)
            fixed_atoms[constraint.index] = True
    return fixed_atoms


def gaussian_basis_function(
    scalars: torch.Tensor,
    num_bases: torch.Tensor | int,
    radius: torch.Tensor | float,
    scale: torch.Tensor | float = 1.0,
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
        0, float(radius), float(radius / num_bases), device=scalars.device
    )
    return torch.exp(-(scale**2) * (scalars.unsqueeze(1) - gaussian_means.unsqueeze(0)).abs() ** 2)


def map_to_pbc_cell(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Maps positions to within a periodic boundary cell.

    Args:
        positions (torch.Tensor): The positions to be mapped. Shape [num_particles, 3]
        cell (torch.Tensor): The matrix of lattice vectors. Shape [3, 3]
        pbc: (torch.Tensor): The periodic boundaries. Shape [3,]

    Returns:
        torch.Tensor: Positions mapped to within a periodic boundary cell.
    """
    assert positions.dtype == cell.dtype
    is_periodic = bool(torch.any(pbc).item())
    if not is_periodic:
        return positions

    original_type = positions.dtype
    # Inverses are a lot more reliable in double precision, so we'll do the whole
    # thing in double then go back to single.
    positions = positions.double()
    cell = cell.double()
    # The strategy here is to map our positions to fractional or internal coordinates.
    # Then we take the modulo, then map back to Euclidean co-ordinates.
    fractional_pos = torch.linalg.solve(cell.T, positions.T).T
    fractional_pos = fractional_pos % 1.0

    # Due to numerical precision, some inputs do not wrap around even when they should.
    # To fix this we use a small tolerance to the check values close to 1 and wrap them to 0.
    fractional_pos = replace_tensor_elements_within_tolerance(
        fractional_pos,
        from_val=1.0,
        to_val=0.0,
        rtol=1e-12,
        atol=1e-12,
    )
    return (fractional_pos @ cell).to(original_type)


def batch_map_to_pbc_cell(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    n_node: torch.Tensor,
) -> torch.Tensor:
    """Maps positions to within a periodic boundary cell, for a batched system.

    Args:
        positions (torch.Tensor): The positions to be mapped. Shape [num_particles, 3]
        cell (torch.Tensor): The matrices of lattice vectors. Shape [num_batches, 3, 3]
        pbc (torch.Tensor): The periodic boundaries. Shape [num_batches, 3]
        n_node (torch.LongTensor): The number of atoms in each graph. Shape [num_batches]
    """
    is_periodic = torch.any(pbc, dim=1)
    if not torch.all(is_periodic == is_periodic[0]):
        # TODO: we should temporarily set the non-periodic cells to
        # zero for the rest of the function to work appropriately
        raise NotImplementedError("Mixed periodic and non-periodic systems are not yet supported")
    # exit early if non-periodic
    if not torch.any(is_periodic):
        return positions

    dtype = positions.dtype
    positions = positions.double()
    cell = cell.double()

    cells_repeated = torch.repeat_interleave(cell, n_node, dim=0)

    # To use the stable torch.linalg.solve, we need to mask batch elements which don't
    # have periodic boundaries. We do this by adding the identity matrix as their PBC,
    # because we need the PBCs to be non-singular.
    # Shape (batch_n_atoms,)
    null_pbc = cells_repeated.abs().sum(dim=[1, 2]) == 0
    # Shape (3, 3)
    identity = torch.eye(3, dtype=torch.bool, device=cells_repeated.device)
    # Broadcast the identity to the elements of the batch that have a null pbc.
    # Shape (batch_n_atoms, 3, 3)
    null_pbc_identity_mask = null_pbc.view(-1, 1, 1) & identity.view(1, 3, 3)
    # Shape (batch_n_atoms, 3, 3)
    pbc_nodes_masked = cells_repeated + null_pbc_identity_mask.double()

    # Shape (batch_n_atoms, 3)
    lattice_coords = torch.linalg.solve(pbc_nodes_masked.transpose(1, 2), positions)
    frac_coords = lattice_coords % 1.0

    cartesian = torch.einsum("bi,bij->bj", frac_coords, cells_repeated)
    return torch.where(null_pbc.unsqueeze(1), positions, cartesian).to(dtype)


def rotation_from_generator(generator: torch.Tensor) -> torch.Tensor:
    """Uses generator to create unitary (rotation) matrix.

    generator -> skew-symmetric matrix S -> R = exp(S)
    S has imaginary eigenvalues, therefore R is unitary

    generator is a (..., 3, 3) tensor.

    """
    return torch.matrix_exp(generator - torch.transpose(generator, dim0=-2, dim1=-1))
