"""End-to-end smoke test for orbmol_v2() against the published HF checkpoint.

Downloads ~100 MB on first run (cached_path caches it). Disabled by default to keep
the unit-test suite fast — set ORB_RUN_NETWORK_TESTS=1 to enable.
"""

import os

import numpy as np
import pytest
import torch
from ase.build import bulk, molecule

# Reference values for the released OrbMol-v2 checkpoint, CPU fp32.
# atol=1e-5 on energy / forces / stress.
H2O_ENERGY_GOLD = np.array(-2079.86339)
H2O_FORCES_0_GOLD = np.array([-1.0472e-04, 2.5031e-04, -4.8726e-01])
CU_ENERGY_GOLD = np.array(-178549.38604)
CU_STRESS_GOLD = np.array([-0.49615, -0.49357, -0.49229, 0.00097, 0.00205, -0.00068])

NETWORK_TESTS_ENABLED = bool(os.getenv("ORB_RUN_NETWORK_TESTS"))
SKIP_REASON = (
    "Network test (downloads ~100 MB orbmol-v2 checkpoint). "
    "Set ORB_RUN_NETWORK_TESTS=1 to enable."
)


@pytest.mark.skipif(not NETWORK_TESTS_ENABLED, reason=SKIP_REASON)
def test_orbmol_v2_h2o_and_cu_match_gold():
    """Loads the public orbmol-v2 weights from HF and verifies H2O + Cu predictions."""
    from orb_models.forcefield.pretrained import orbmol_v2

    model, adapter = orbmol_v2(device="cpu", compile=False)
    model.eval()

    # H2O — non-periodic, no PME
    atoms = molecule("H2O")
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    batch = adapter.from_ase_atoms(atoms, device="cpu")
    out = model.predict(batch)
    energy = out["energy"][0].detach().numpy() if out["energy"].ndim else out["energy"].detach().numpy()
    forces = out["grad_forces"][0].detach().numpy()
    np.testing.assert_allclose(energy, H2O_ENERGY_GOLD, atol=1e-5)
    np.testing.assert_allclose(forces, H2O_FORCES_0_GOLD, atol=1e-5)

    # Cu fcc — periodic, exercises PME via nvalchemiops
    model.enable_stress()
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    batch = adapter.from_ase_atoms(atoms, device="cpu")
    out = model.predict(batch)
    energy = out["energy"][0].detach().numpy() if out["energy"].ndim else out["energy"].detach().numpy()
    stress = out["grad_stress"][0].detach().numpy()
    # Cu absolute energy is ~1.8e5 eV; relative-to-magnitude check is more meaningful than atol.
    assert abs(float(energy) - float(CU_ENERGY_GOLD)) / abs(float(CU_ENERGY_GOLD)) < 1e-7
    np.testing.assert_allclose(stress, CU_STRESS_GOLD, atol=1e-5)


@pytest.mark.skipif(not NETWORK_TESTS_ENABLED, reason=SKIP_REASON)
def test_orbmol_v2_returns_fp64_energy():
    """absolute_energy promotes to fp64 to preserve kJ/mol resolution against OMol-scale refs."""
    from orb_models.forcefield.pretrained import orbmol_v2

    model, adapter = orbmol_v2(device="cpu", compile=False)
    model.eval()

    atoms = molecule("H2O")
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    batch = adapter.from_ase_atoms(atoms, device="cpu")
    out = model.predict(batch)
    assert out["energy"].dtype == torch.float64
