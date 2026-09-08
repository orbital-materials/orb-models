import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import PropertyNotImplementedError

from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.inference.calculator import ORBCalculator


def test_conservative_calculator(conservative_regressor, mptraj_10_systems_db):
    atoms = mptraj_10_systems_db.get_atoms(1)
    conservative_calc = ORBCalculator(
        model=conservative_regressor,
        atoms_adapter=ForcefieldAtomsAdapter(6.0, 20),
    )
    conservative_calc.calculate(atoms)

    assert "energy" in conservative_calc.results
    assert "forces" in conservative_calc.results
    assert "stress" in conservative_calc.results


def test_calc_non_conservative_defaults(direct_regressor):
    calc = ORBCalculator(model=direct_regressor, atoms_adapter=ForcefieldAtomsAdapter(6.0, 20))
    assert set(calc.implemented_properties) == {
        "energy",
        "free_energy",
        "forces",
        "stress",
    }


def test_conservative_stress_disabled(conservative_regressor, mptraj_10_systems_db):
    conservative_regressor.disable_stress()
    calc = ORBCalculator(
        model=conservative_regressor,
        atoms_adapter=ForcefieldAtomsAdapter(6.0, 20),
    )
    assert "stress" not in calc.implemented_properties
    atoms = mptraj_10_systems_db.get_atoms(1)
    calc.calculate(atoms)
    assert "stress" not in calc.results
    assert "forces" in calc.results


def test_conservative_stress_enabled(conservative_regressor, mptraj_10_systems_db):
    conservative_regressor.disable_stress()
    conservative_regressor.enable_stress()
    calc = ORBCalculator(
        model=conservative_regressor,
        atoms_adapter=ForcefieldAtomsAdapter(6.0, 20),
    )
    assert "stress" in calc.implemented_properties
    atoms = mptraj_10_systems_db.get_atoms(1)
    calc.calculate(atoms)
    assert "stress" in calc.results
    assert "forces" in calc.results


def test_direct_stress_disabled(direct_regressor, mptraj_10_systems_db):
    direct_regressor.disable_stress()
    calc = ORBCalculator(
        model=direct_regressor,
        atoms_adapter=ForcefieldAtomsAdapter(6.0, 20),
    )
    assert "stress" not in calc.implemented_properties
    atoms = mptraj_10_systems_db.get_atoms(1)
    calc.calculate(atoms)
    assert "stress" not in calc.results
    assert "forces" in calc.results


def test_direct_stress_enabled(direct_regressor, mptraj_10_systems_db):
    direct_regressor.disable_stress()
    direct_regressor.enable_stress()
    calc = ORBCalculator(
        model=direct_regressor,
        atoms_adapter=ForcefieldAtomsAdapter(6.0, 20),
    )
    assert "stress" in calc.implemented_properties
    atoms = mptraj_10_systems_db.get_atoms(1)
    calc.calculate(atoms)
    assert "stress" in calc.results
    assert "forces" in calc.results


@pytest.mark.parametrize(
    ("atoms", "total_charge"),
    [
        (molecule("H2O"), 1),
        # Single atom: guards against to_numpy collapsing (1,) to a Python float.
        (Atoms("H", positions=[[0.0, 0.0, 0.0]]), 0),
    ],
)
def test_charges(conservative_regressor, atoms, total_charge):
    """Charges are exposed to ASE as (n_atoms,) and sum to the requested total."""
    atoms.info["charge"] = total_charge
    atoms.info["spin"] = 1
    calc = ORBCalculator(
        model=conservative_regressor,
        atoms_adapter=ForcefieldAtomsAdapter(6.0, 20),
        use_experimental_charges=True,
    )
    assert "charges" in calc.implemented_properties
    atoms.calc = calc

    charges = atoms.get_charges()
    assert charges.shape == (len(atoms),)
    assert np.isfinite(charges).all()
    assert charges.sum() == pytest.approx(total_charge, abs=1e-5)


def test_dipole(conservative_regressor, mptraj_10_systems_db):
    """Dipole is the point-charge sum, and only available for non-periodic systems."""
    adapter = ForcefieldAtomsAdapter(6.0, 20)

    atoms = molecule("H2O")
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.calc = ORBCalculator(
        model=conservative_regressor, atoms_adapter=adapter, use_experimental_charges=True
    )

    dipole = atoms.get_dipole_moment()
    assert dipole.shape == (3,)
    np.testing.assert_allclose(dipole, atoms.get_charges() @ atoms.get_positions(), rtol=1e-6)

    periodic = mptraj_10_systems_db.get_atoms(1)
    periodic.info["charge"] = 0
    periodic.info["spin"] = 1
    calc = ORBCalculator(
        model=conservative_regressor, atoms_adapter=adapter, use_experimental_charges=True
    )
    calc.calculate(periodic)
    assert "charges" in calc.results
    assert "dipole" not in calc.results


def test_charges_require_opt_in(conservative_regressor):
    """Charges and dipole are absent unless use_experimental_charges is set."""
    atoms = molecule("H2O")
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    calc = ORBCalculator(
        model=conservative_regressor, atoms_adapter=ForcefieldAtomsAdapter(6.0, 20)
    )
    assert "charges" not in calc.implemented_properties
    assert "dipole" not in calc.implemented_properties

    calc.calculate(atoms)
    assert "charges" not in calc.results
    assert "dipole" not in calc.results

    atoms.calc = calc
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_charges()
