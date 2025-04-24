import ase
import numpy as np
import pytest
import torch

from orb_models.forcefield.calculator import ORBCalculator
from orb_models.forcefield.atomic_system import SystemConfig


def test_conservative_calculator(conservative_regressor, shared_fixtures_path):
    db = ase.db.connect(shared_fixtures_path / "10_mptraj_systems.db")
    atoms = db.get_atoms(1)

    conservative_calc = ORBCalculator(
        model=conservative_regressor,
        system_config=SystemConfig(6.0, 20),
        conservative=True,
    )
    conservative_calc.calculate(atoms)

    nonconservative_calc = ORBCalculator(
        model=conservative_regressor,
        system_config=SystemConfig(6.0, 20),
        conservative=False,
    )
    nonconservative_calc.calculate(atoms)

    # Test that setting 'conservative=True' correctly relabels the keys in the results dict.
    # Whilst this relabelling may seem pointless, it is important because "forces" and
    # "stress" are used by ASE internally (e.g. in geometry optimization).
    assert np.allclose(
        conservative_calc.results["forces"],
        nonconservative_calc.results[conservative_calc.model.grad_forces_name],
    )
    assert np.allclose(
        conservative_calc.results["stress"],
        nonconservative_calc.results[conservative_calc.model.grad_stress_name],
    )

    # Check that the direct forces and stress are not present in the results dict.
    # This is a sanity check as we do not have multi-headed models.
    assert "direct_forces" not in conservative_calc.results
    assert "direct_stress" not in conservative_calc.results

def test_calc_conservative_defaults(conservative_regressor):

    # Conservative model should use conservative forces by default
    calc = ORBCalculator(
        model=conservative_regressor, system_config=SystemConfig(6.0, 20)
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
    # non-conservative model should raise error if conservative=True
    with pytest.raises(ValueError):
        ORBCalculator(
            model=torch.nn.Linear(10, 1),
            system_config=SystemConfig(6.0, 20),
            conservative=True,
        )


def test_calc_non_conservative_defaults(graph_regressor):
    calc = ORBCalculator(model=graph_regressor, system_config=SystemConfig(6.0, 20))
    assert calc.conservative is False
    assert set(calc.implemented_properties) == {
        "energy",
        "free_energy",
        "forces",
        "stress",
    }



def test_calc_produces_non_singleton_dims_for_forces(conservative_regressor, graph_regressor):

    # system with 1 atom
    atoms = ase.Atoms("H", positions=np.array([[0, 0, 0]]), cell=np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]]), pbc=True)
    calc = ORBCalculator(model=conservative_regressor, system_config=SystemConfig(6.0, 20))
    calc.calculate(atoms)
    assert calc.results["forces"].shape == (1, 3)
    assert calc.results["stress"].shape == (6,)


    atoms = ase.Atoms("H", positions=np.array([[0, 0, 0]]), cell=np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]]), pbc=True)
    calc = ORBCalculator(model=graph_regressor, system_config=SystemConfig(6.0, 20))
    calc.calculate(atoms)
    assert calc.results["forces"].shape == (1, 3)
    assert calc.results["stress"].shape == (6,)
