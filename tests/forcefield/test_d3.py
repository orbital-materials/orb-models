"""Test correctness of AlchemiDFTD3 by comparing with TorchDFTD for PBE-BJ functional."""

import numpy as np
import pytest
import torch
from ase.build import bulk, molecule
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.inference.d3_model import AlchemiDFTD3


@pytest.fixture
def forcefield_adapter():
    return ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=100)


@pytest.fixture
def alchemi_d3():
    return AlchemiDFTD3(functional="PBE", damping="BJ", cutoff=50.2, has_stress=True)


@pytest.fixture
def torchdftd_calculator():
    """TorchDFTD3Calculator with matching PBE-BJ parameters (float32 to match AlchemiDFTD3)."""
    return TorchDFTD3Calculator(
        device="cpu",
        damping="bj",
        xc="pbe",
        cutoff=50.2,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("mol_name", ["H2O", "CH4"])
def test_pbe_bj_molecule_agreement(
    mol_name,
    alchemi_d3,
    torchdftd_calculator,
    forcefield_adapter,
):
    """Test that AlchemiDFTD3 and TorchDFTD produce similar energy and forces for molecules."""
    atoms = molecule(mol_name)
    atoms.center(vacuum=5.0)
    atoms.set_cell([15.0, 15.0, 15.0])
    atoms.set_pbc(False)

    # Get AlchemiDFTD3 predictions
    graph = forcefield_adapter.from_ase_atoms(atoms)
    batch = AtomGraphs.batch([graph])
    alchemi_out = alchemi_d3.predict(batch)

    # Get TorchDFTD predictions (use ASE interface to populate results)
    atoms.calc = torchdftd_calculator
    torchdftd_energy = atoms.get_potential_energy()
    torchdftd_forces = atoms.get_forces()

    np.testing.assert_allclose(
        alchemi_out["energy"].detach().cpu().numpy(),
        torchdftd_energy,
        atol=1e-6,
        err_msg=f"Energy mismatch for {mol_name}",
    )

    np.testing.assert_allclose(
        alchemi_out["forces"].detach().cpu().numpy(),
        torchdftd_forces,
        atol=1e-6,
        err_msg=f"Forces mismatch for {mol_name}",
    )


def test_pbe_bj_periodic_agreement(
    alchemi_d3,
    torchdftd_calculator,
    forcefield_adapter,
):
    """Test that AlchemiDFTD3 and TorchDFTD produce similar results for periodic bulk Si."""
    atoms = bulk("Si", cubic=True)

    # Perturb positions to get non-zero forces (equilibrium structure has ~zero forces)
    rng = np.random.default_rng(seed=20260129)
    atoms.positions += rng.normal(scale=0.1, size=atoms.positions.shape)

    # Get AlchemiDFTD3 predictions
    graph = forcefield_adapter.from_ase_atoms(atoms)
    batch = AtomGraphs.batch([graph])
    alchemi_out = alchemi_d3.predict(batch)

    # Get TorchDFTD predictions (use ASE interface to populate results)
    atoms.calc = torchdftd_calculator
    torchdftd_energy = atoms.get_potential_energy()
    torchdftd_forces = atoms.get_forces()
    torchdftd_stress = atoms.get_stress()

    # Use relative tolerance for periodic systems due to different neighbor list implementations
    alchemi_energy = alchemi_out["energy"].detach().cpu().numpy().squeeze()
    np.testing.assert_allclose(
        alchemi_energy,
        torchdftd_energy,
        rtol=1e-3,
    )
    alchemi_forces = alchemi_out["forces"].detach().cpu().numpy().squeeze()
    np.testing.assert_allclose(
        alchemi_forces,
        torchdftd_forces,
        rtol=0.1,
        atol=1e-6,
    )
    alchemi_stress = alchemi_out["stress"].detach().cpu().numpy().squeeze()
    np.testing.assert_allclose(
        alchemi_stress,
        torchdftd_stress,
        rtol=0.1,
        atol=1e-6,
    )
