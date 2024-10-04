import pytest
import numpy as np
from ase.build import bulk
from ase.optimize import BFGS
from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.calculator import ORBCalculator


@pytest.mark.parametrize("model_fn", [pretrained.orb_v1, pretrained.orb_v2])
def test_energy_forces_stress_prediction(model_fn):
    """Tests model compatibility on energy, forces and stress."""
    orbff = model_fn(device="cpu")
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    graph = atomic_system.ase_atoms_to_atom_graphs(atoms, device="cpu")
    result = orbff.predict(graph)

    energy = result["graph_pred"][0]
    forces = result["node_pred"][0]
    stress = result["stress_pred"][0]

    if model_fn == pretrained.orb_v1:
        energy_gold = np.array(-16.3437)
        forces_gold = np.array([9.3569e-05, -4.0413e-05, 6.0380e-05])
        stress_gold = np.array(
            [-3.2064e-02, -3.2115e-02, -3.1964e-02, -6.2145e-07, 2.5603e-07, 1.6000e-06]
        )
    elif model_fn == pretrained.orb_v2:
        energy_gold = np.array(-16.3459)
        forces_gold = np.array([-1.0881e-06, 7.6937e-08, -1.7336e-06])
        stress_gold = np.array(
            [-3.2431e-02, -3.2185e-02, -3.1791e-02, 1.5402e-06, 1.9556e-06, 6.9951e-07]
        )

    assert np.isclose(energy, energy_gold, atol=1e-4)
    assert np.allclose(forces, forces_gold, atol=1e-6)
    assert np.allclose(stress, stress_gold, atol=1e-6)


@pytest.mark.parametrize("model_fn", [pretrained.orb_v1, pretrained.orb_v2])
def test_optimization(model_fn):
    """Tests model compatibility on optimization."""
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    orbff = model_fn(device="cpu")
    calc = ORBCalculator(orbff, device="cpu")
    atoms.set_calculator(calc)
    atoms.rattle(0.5)
    rattled_energy = atoms.get_potential_energy()
    dyn = BFGS(atoms)
    dyn.run(fmax=0.01)
    optimized_energy = atoms.get_potential_energy()
    if model_fn == pretrained.orb_v1:
        gold_rattled_energy = -11.943148
        gold_optimized_energy = -16.345758
    elif model_fn == pretrained.orb_v2:
        gold_rattled_energy = -12.034759
        gold_optimized_energy = -16.348310

    assert np.isclose(rattled_energy, gold_rattled_energy, atol=1e-6)
    assert np.isclose(optimized_energy, gold_optimized_energy, atol=1e-6)
