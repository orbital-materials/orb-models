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
    set_seed(42)
    orbff = model_fn(device="cpu")
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    graph = atomic_system.ase_atoms_to_atom_graphs(atoms, device="cpu")
    result = orbff.predict(graph)

    energy = result["graph_pred"][0].numpy()
    forces = result["node_pred"][0].numpy()
    stress = result["stress_pred"][0].numpy()

    energy_gold = np.array(-16.3459)
    assert np.isclose(energy, energy_gold, atol=1e-2)
    assert np.all(forces < 1e-5)
    assert np.all(stress < 1e-2)


@pytest.mark.parametrize("model_fn", [pretrained.orb_v2])
def test_optimization(model_fn):
    """Tests model compatibility on optimization."""
    set_seed(42)
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    orbff = model_fn(device="cpu")
    calc = ORBCalculator(orbff, device="cpu")
    atoms.calc = calc
    atoms.rattle(0.5)
    rattled_energy = atoms.get_potential_energy()
    dyn = BFGS(atoms)
    dyn.run(fmax=0.01)
    optimized_energy = atoms.get_potential_energy()

    gold_optimized_energy_upper_bound = -16.00

    assert optimized_energy < rattled_energy
    assert optimized_energy < gold_optimized_energy_upper_bound
