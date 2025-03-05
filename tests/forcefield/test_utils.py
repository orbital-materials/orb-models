import numpy as np
import torch

from orb_models.forcefield.forcefield_utils import torch_voigt_6_to_full_3x3_stress


def test_batch_voigt_conversion():
    from ase.stress import voigt_6_to_full_3x3_stress

    n = 10
    stress_voigt = torch.rand(n, 6, dtype=torch.double)

    stress_matrix = torch_voigt_6_to_full_3x3_stress(stress_voigt)
    for i in range(n):
        assert np.allclose(
            stress_matrix[i].numpy(),
            voigt_6_to_full_3x3_stress(stress_voigt[i].numpy()),
        )
