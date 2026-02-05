import ase
import ase.db
import numpy as np
import torch
from ase.stress import voigt_6_to_full_3x3_stress

from orb_models.common.dataset.augmentations import geometric_augmentations


def test_stress_properties_are_rotated(fixtures_path):
    fixtures_path = fixtures_path.parent.parent / "fixtures"
    atoms = ase.db.connect(str(fixtures_path / "test_with_target.db")).get(1).toatoms()

    # 1 for batch dimension
    stress = np.random.rand(1, 6)
    atoms.info["graph_targets"] = {"stress": torch.from_numpy(stress)}
    geometric_augmentations.rotate_randomly(atoms)
    rotated_stress = atoms.info["graph_targets"]["stress"]

    assert not np.allclose(rotated_stress.numpy(), stress)

    # Principal stresses (eigenvalues of full stress tensor) should be the same
    eigvals_orig = np.linalg.eigvals(voigt_6_to_full_3x3_stress(stress.squeeze(0)))
    eigvals_new = np.linalg.eigvals(voigt_6_to_full_3x3_stress(rotated_stress.squeeze(0).numpy()))
    eigvals_new.sort()
    eigvals_orig.sort()
    assert np.allclose(eigvals_orig, eigvals_new)
