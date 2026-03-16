import numpy as np

from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.inference.calculator import ORBCalculator


def test_conservative_calculator(conservative_regressor, mptraj_10_systems_db):
    atoms = mptraj_10_systems_db.get_atoms(1)
    conservative_calc = ORBCalculator(
        model=conservative_regressor,
        atoms_adapter=ForcefieldAtomsAdapter(6.0, 20),
    )
    conservative_calc.calculate(atoms)

    # Test that setting 'conservative=True' correctly relabels the keys in the results dict.
    assert np.allclose(
        conservative_calc.results["forces"],
        conservative_calc.results[conservative_calc.model.grad_forces_name],
    )
    assert np.allclose(
        conservative_calc.results["stress"],
        conservative_calc.results[conservative_calc.model.grad_stress_name],
    )


def test_calc_conservative_defaults(conservative_regressor):
    # Conservative model should use conservative forces by default
    calc = ORBCalculator(
        model=conservative_regressor,
        atoms_adapter=ForcefieldAtomsAdapter(6.0, 20),
    )
    assert calc.conservative is True
    assert set(calc.implemented_properties) == set(
        [
            "energy",
            "free_energy",
            "forces",
            "stress",
            "grad_forces",
            "grad_stress",
            "rotational_grad",
        ]
    )


def test_calc_non_conservative_defaults(direct_regressor):
    calc = ORBCalculator(model=direct_regressor, atoms_adapter=ForcefieldAtomsAdapter(6.0, 20))
    assert calc.conservative is False
    assert set(calc.implemented_properties) == {
        "energy",
        "free_energy",
        "forces",
        "stress",
    }


class TestStressToggle:
    """Test that enable_stress/disable_stress controls stress in calculator results."""

    def test_conservative_stress_disabled(self, conservative_regressor, mptraj_10_systems_db):
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

    def test_conservative_stress_enabled(self, conservative_regressor, mptraj_10_systems_db):
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

    def test_direct_stress_disabled(self, direct_regressor, mptraj_10_systems_db):
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

    def test_direct_stress_enabled(self, direct_regressor, mptraj_10_systems_db):
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
