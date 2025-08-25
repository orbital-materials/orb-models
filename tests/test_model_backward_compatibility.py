import pytest
import numpy as np
import torch
import random
from ase.build import bulk, molecule
from ase.optimize import BFGS
from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.calculator import ORBCalculator


def set_seed(seed):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def test_orb_v2_predictions():
    """Test that we haven't changed the predictions of orb-v2."""
    orb = pretrained.orb_v2()
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    graph = atomic_system.ase_atoms_to_atom_graphs(atoms, orb.system_config)
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


def test_orbv2_optimization():
    """Test that we haven't changed the optimization behaviour of orb-v2."""
    orb = pretrained.orb_v2()
    calc = ORBCalculator(orb, system_config=orb.system_config, device=torch.device("cpu"))
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


def test_orb_v3_direct_omat_predictions():
    """Test that we haven't changed the predictions of orb-v3-direct."""
    orb = pretrained.orb_v3_direct_inf_omat()
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    graph = atomic_system.ase_atoms_to_atom_graphs(atoms, orb.system_config)
    result = orb.predict(graph)
    energy = result["energy"][0]
    forces = result["forces"][0]
    stress = result["stress"][0]
    energy_gold = np.array(-14.9910)
    forces_gold = np.array([5.1031e-07, 3.7791e-07, 3.0662e-07])
    stress_gold = np.array(
        [-2.5220e-02, -2.5239e-02, -2.5593e-02, 6.7905e-05, 3.3452e-04, 1.5985e-04]
    )
    assert np.isclose(energy, energy_gold, atol=1e-4)
    np.testing.assert_allclose(forces, forces_gold, atol=1e-5)
    np.testing.assert_allclose(stress, stress_gold, atol=1e-4)


def test_orbv3_direct_omat_optimization():
    """Test that we haven't changed the optimization behaviour of orb-v3-direct."""
    orb = pretrained.orb_v3_direct_inf_omat()
    calc = ORBCalculator(orb, system_config=orb.system_config, device=torch.device("cpu"))
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    atoms.calc = calc
    atoms.rattle(0.5, seed=42)
    rattled_energy = atoms.get_potential_energy()
    dyn = BFGS(atoms, logfile=None)
    dyn.run(fmax=0.01)
    optimized_energy = atoms.get_potential_energy()
    gold_rattled_energy = -10.578043937683105
    gold_optimized_energy = -14.991006851196289
    assert np.isclose(rattled_energy, gold_rattled_energy, atol=1e-5)
    assert np.isclose(optimized_energy, gold_optimized_energy, atol=1e-5)


def test_orb_v3_con_omat_predictions():
    """Test that we haven't changed the predictions of orb-v3-conservative."""
    orb = pretrained.orb_v3_conservative_inf_omat()
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    graph = atomic_system.ase_atoms_to_atom_graphs(atoms, orb.system_config)
    result = orb.predict(graph)
    energy = result["energy"][0].detach().numpy()
    forces = result["grad_forces"][0].detach().numpy()
    stress = result["grad_stress"][0].detach().numpy()
    energy_gold = np.array(-14.9525)
    forces_gold = np.array([1.3970e-09, 1.0419e-08, -1.0361e-08])
    stress_gold = np.array(
        [-2.5326e-02, -2.5321e-02, -2.5328e-02, 1.2287e-06, 1.1870e-06, 6.1788e-06]
    )
    assert np.isclose(energy, energy_gold, atol=1e-4)
    np.testing.assert_allclose(forces, forces_gold, atol=1e-5)
    np.testing.assert_allclose(stress, stress_gold, atol=1e-4)


def test_orbv3_con_omat_optimization():
    """Test that we haven't changed the optimization behaviour of orb-v3-conservative."""
    orb = pretrained.orb_v3_conservative_inf_omat()
    calc = ORBCalculator(orb, system_config=orb.system_config, device=torch.device("cpu"))
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    atoms.calc = calc
    atoms.rattle(0.5, seed=42)
    rattled_energy = atoms.get_potential_energy()
    dyn = BFGS(atoms, logfile=None)
    dyn.run(fmax=0.01)
    optimized_energy = atoms.get_potential_energy()
    gold_rattled_energy = -10.517223358154297
    gold_optimized_energy = -14.952497482299805
    assert np.isclose(rattled_energy, gold_rattled_energy, atol=1e-5)
    assert np.isclose(optimized_energy, gold_optimized_energy, atol=1e-5)


def test_orb_v3_con_omol_predictions():
    """Test that we haven't changed the predictions of orb-v3-conservative-omol."""
    orb = pretrained.orb_v3_conservative_omol()
    atoms = molecule("C6H6")
    atoms.info["charge"] = 1.0
    atoms.info["spin"] = 0.0
    energy_gold = np.array(-6316.6646)
    forces_gold = np.array(([0.14257583, -0.2070809, 0.01658938]))

    # First: check the model.predict() is correct
    graph = atomic_system.ase_atoms_to_atom_graphs(atoms, orb.system_config)
    result = orb.predict(graph)
    energy = result["energy"][0].detach().numpy()
    forces = result["grad_forces"][0].detach().numpy()
    assert np.isclose(energy, energy_gold, atol=1e-4)
    np.testing.assert_allclose(forces, forces_gold, atol=1e-5)

    # Second: check the ase calculator interface is correct
    calc = ORBCalculator(orb, system_config=orb.system_config, device=torch.device("cpu"))
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()[0]
    assert np.isclose(energy, energy_gold, atol=1e-4)
    np.testing.assert_allclose(forces, forces_gold, atol=1e-5)


def test_orbv3_con_omol_optimization():
    """Test that we haven't changed the optimization behaviour of orb-v3-conservative."""
    orb = pretrained.orb_v3_conservative_omol()
    calc = ORBCalculator(orb, system_config=orb.system_config, device=torch.device("cpu"))
    atoms = molecule("C6H6")
    atoms.calc = calc
    atoms.rattle(0.5, seed=42)

    atoms.info["charge"] = 1.0
    atoms.info["spin"] = 0.0
    rattled_energy = atoms.get_potential_energy()
    dyn = BFGS(atoms, logfile=None)
    dyn.run(fmax=0.01)
    optimized_energy = atoms.get_potential_energy()

    gold_rattled_energy = -6280.37646484375
    gold_optimized_energy = -6311.3916015625
    assert np.isclose(rattled_energy, gold_rattled_energy, atol=1e-5)
    assert np.isclose(optimized_energy, gold_optimized_energy, atol=1e-5)

    atoms = molecule("C6H6")
    atoms.calc = calc
    with pytest.raises(
        ValueError, match="atoms.info must contain both 'charge' and 'spin'"
    ):
        atoms.get_potential_energy()
