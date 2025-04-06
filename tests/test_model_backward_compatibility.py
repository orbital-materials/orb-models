import pytest
import numpy as np
import torch
import random
from ase.build import bulk
from ase.optimize import BFGS
from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.calculator import ORBCalculator


def set_seed(seed):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


@pytest.mark.parametrize("model_fn", [pretrained.orb_v2])
def test_energy_forces_stress_prediction(model_fn):
    """Tests model compatibility on energy, forces and stress."""
    orb = model_fn(device="cpu")
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    graph = atomic_system.ase_atoms_to_atom_graphs(atoms, system_config=orb.system_config)
    result = orb.predict(graph)
    energy = result["energy"][0]
    forces = result["forces"][0]
    stress = result["stress"][0]
    energy_gold = np.array(-16.3510)
    forces_gold = np.array([1.7524e-06, -1.2913e-06, -1.0884e-06])
    stress_gold = np.array(
        [-3.4152e-02, -3.3998e-02, -3.3992e-02, -2.7855e-07, -1.6083e-06, -1.1105e-06]
    )
    assert np.isclose(energy, energy_gold, atol=1e-4)
    np.testing.assert_allclose(forces, forces_gold, atol=1e-5)
    np.testing.assert_allclose(stress, stress_gold, atol=1e-4)


@pytest.mark.parametrize("model_fn", [pretrained.orb_v2])
def test_optimization(model_fn):
    """Test that we haven't changed the optimization behaviour of orb-v2."""
    set_seed(42)
    orb = model_fn(device="cpu")
    calc = ORBCalculator(orb, device=torch.device("cpu"))
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    atoms.calc = calc
    atoms.rattle(0.5, seed=42)
    rattled_energy = atoms.get_potential_energy()
    dyn = BFGS(atoms, logfile=None)
    dyn.run(fmax=0.01)
    optimized_energy = atoms.get_potential_energy()
    gold_rattled_energy = -11.975013
    gold_optimized_energy = -16.347759
    assert np.isclose(rattled_energy, gold_rattled_energy, atol=1e-5)
    assert np.isclose(optimized_energy, gold_optimized_energy, atol=1e-5)
