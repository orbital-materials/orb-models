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
