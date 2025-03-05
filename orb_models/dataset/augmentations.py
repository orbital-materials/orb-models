import math

from ase.stress import voigt_6_to_full_3x3_stress
import ase
import numpy as np
import torch

from orb_models.utils import to_numpy


def rand_matrix(*shape, requires_grad=False, dtype=None, device=None):
    r"""Random rotation matrix.

    Sourced from https://github.com/e3nn/e3nn/blob/0.4.4/e3nn/o3/_rotation.py

    Parameters
    ----------
    *shape : int

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(\mathrm{shape}, 3, 3)`
    """
    R = angles_to_matrix(*rand_angles(*shape, dtype=dtype, device=device))
    return R.detach().requires_grad_(requires_grad)


def rand_angles(*shape, requires_grad=False, dtype=None, device=None):
    r"""Random rotation angles.

    Sourced from https://github.com/e3nn/e3nn/blob/0.4.4/e3nn/o3/_rotation.py

    Parameters
    ----------
    *shape : int

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(\mathrm{shape})`

    beta : `torch.Tensor`
        tensor of shape :math:`(\mathrm{shape})`

    gamma : `torch.Tensor`
        tensor of shape :math:`(\mathrm{shape})`
    """
    alpha, gamma = 2 * math.pi * torch.rand(2, *shape, dtype=dtype, device=device)
    beta = torch.rand(shape, dtype=dtype, device=device).mul(2).sub(1).acos()
    alpha = alpha.detach().requires_grad_(requires_grad)
    beta = beta.detach().requires_grad_(requires_grad)
    gamma = gamma.detach().requires_grad_(requires_grad)
    return alpha, beta, gamma


def angles_to_matrix(alpha, beta, gamma):
    r"""Conversion from angles to matrix.

    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    return matrix_y(alpha) @ matrix_x(beta) @ matrix_y(gamma)


def matrix_x(angle: torch.Tensor) -> torch.Tensor:
    r"""Matrix of rotation around X axis.

    Sourced from https://github.com/e3nn/e3nn/blob/0.4.4/e3nn/o3/_rotation.py

    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack(
        [
            torch.stack([o, z, z], dim=-1),
            torch.stack([z, c, -s], dim=-1),
            torch.stack([z, s, c], dim=-1),
        ],
        dim=-2,
    )


def matrix_y(angle: torch.Tensor) -> torch.Tensor:
    r"""Matrix of rotation around Y axis.

    Sourced from https://github.com/e3nn/e3nn/blob/0.4.4/e3nn/o3/_rotation.py

    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack(
        [
            torch.stack([c, z, s], dim=-1),
            torch.stack([z, o, z], dim=-1),
            torch.stack([-s, z, c], dim=-1),
        ],
        dim=-2,
    )


def matrix_z(angle: torch.Tensor) -> torch.Tensor:
    r"""Matrix of rotation around Z axis.

    Sourced from https://github.com/e3nn/e3nn/blob/0.4.4/e3nn/o3/_rotation.py

    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack(
        [
            torch.stack([c, -s, z], dim=-1),
            torch.stack([s, c, z], dim=-1),
            torch.stack([z, z, o], dim=-1),
        ],
        dim=-2,
    )


def rotate_randomly(atoms: ase.Atoms) -> None:
    """Randomly rotate atoms in ase.Atoms object.

    NOTE: We look for forces / stress in the `ase.info` dict,
    since AseSqliteDataset expects features/targets to live there.
    """
    # sample a rotation matrix and store in the atoms.info dict
    rand_rotation = rand_matrix(1, dtype=torch.float64)[0].numpy()
    graph_dict = atoms.info.setdefault("graph_features", {})
    graph_dict["random_rotation_matrix"] = torch.from_numpy(rand_rotation)

    atoms.positions = atoms.positions @ rand_rotation
    if atoms.cell is not None:
        atoms.set_cell(atoms.cell.array @ rand_rotation)

    for type in ["features", "targets"]:
        node_dict = atoms.info.get(f"node_{type}", {})
        for key, v in node_dict.items():
            if "forces" in key:
                node_dict[key] = v @ rand_rotation

        graph_dict = atoms.info.get(f"graph_{type}", {})
        for key, v in graph_dict.items():
            original_dtype = v.dtype
            if "stress" in key:
                # Transformation rule of stress tensor,
                # see https://en.wikipedia.org/wiki/Cauchy_stress_tensor#Transformation_rule_of_the_stress_tensor
                full_stress = voigt_6_to_full_3x3_stress(to_numpy(v))

                # Our featurization code adds a batch dimension, so we need to reshape
                if full_stress.shape != (3, 3):
                    full_stress = full_stress.reshape(3, 3)

                transformed = np.dot(
                    np.dot(rand_rotation.T, full_stress), rand_rotation
                )
                # Back to voigt notation
                graph_dict[key] = torch.tensor(
                    [
                        transformed[0, 0],
                        transformed[1, 1],
                        transformed[2, 2],
                        transformed[1, 2],
                        transformed[0, 2],
                        transformed[0, 1],
                    ],
                    dtype=original_dtype,
                )
