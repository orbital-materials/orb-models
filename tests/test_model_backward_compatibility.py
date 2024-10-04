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


@pytest.mark.parametrize("model_fn", [pretrained.orb_v1, pretrained.orb_v2])
def test_energy_forces_stress_prediction(model_fn):
    """Tests model compatibility on energy, forces and stress."""
    set_seed(42)
    orbff = model_fn(device="cpu")
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    graph = atomic_system.ase_atoms_to_atom_graphs(atoms, device="cpu")
    result = orbff.predict(graph)

    energy = result["graph_pred"][0]
    forces = result["node_pred"][0]
    stress = result["stress_pred"][0]

    if model_fn == pretrained.orb_v1:
        energy_gold = np.array(-16.3437)
        forces_gold = np.array([9.2324e-05, -3.9214e-05, 6.3257e-05])
        stress_gold = np.array(
            [-3.2026e-02, -3.2077e-02, -3.1928e-02, -6.2547e-07, 2.4863e-07, 1.6014e-06]
        )
    elif model_fn == pretrained.orb_v2:
        energy_gold = np.array(-16.3459)
        forces_gold = np.array([1.1065e-07, 6.6851e-08, -3.3196e-07])
        stress_gold = np.array(
            [-3.2460e-02, -3.2213e-02, -3.1816e-02, 1.5517e-06, 1.9550e-06, 6.9791e-07]
        )

    assert np.isclose(energy, energy_gold, atol=1e-4)
    assert np.allclose(forces, forces_gold, atol=1e-6)
    assert np.allclose(stress, stress_gold, atol=1e-6)


@pytest.mark.parametrize("model_fn", [pretrained.orb_v1, pretrained.orb_v2])
def test_optimization(model_fn):
    """Tests model compatibility on optimization."""
    set_seed(42)
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
