import numpy as np
import pytest
import torch
from ase.build import bulk, molecule
from ase.optimize import BFGS

from orb_models.forcefield.inference.calculator import ORBCalculator


def test_orb_v2_predictions(orb_v2_and_config):
    """Test that we haven't changed the predictions of orb-v2."""
    orb, adapter = orb_v2_and_config
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    graph = adapter.from_ase_atoms(atoms)
    result = orb.predict(graph)
    energy = result["energy"][0]
    forces = result["forces"][0]
    stress = result["stress"][0]
    energy_gold = np.array(-16.3510)
    forces_gold = np.array([1.7524e-06, -1.2913e-06, -1.0884e-06])
    stress_gold = np.array(
        [-3.4152e-02, -3.3998e-02, -3.3992e-02, -2.7855e-07, -1.6083e-06, -1.1105e-06]
    )
    np.testing.assert_allclose(energy, energy_gold, atol=1e-4)
    np.testing.assert_allclose(forces, forces_gold, atol=1e-5)
    np.testing.assert_allclose(stress, stress_gold, atol=1e-4)


def test_orbv2_optimization(orb_v2_and_config):
    """Test that we haven't changed the optimization behaviour of orb-v2."""
    orb, adapter = orb_v2_and_config
    calc = ORBCalculator(orb, adapter, device=torch.device("cpu"))
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


def test_orb_v3_direct_omat_predictions(orb_v3_direct_omat_and_config):
    """Test that we haven't changed the predictions of orb-v3-direct."""
    orb, adapter = orb_v3_direct_omat_and_config
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    graph = adapter.from_ase_atoms(atoms)
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
    np.testing.assert_allclose(stress, stress_gold, atol=1e-5)


def test_orbv3_direct_omat_optimization(orb_v3_direct_omat_and_config):
    """Test that we haven't changed the optimization behaviour of orb-v3-direct."""
    orb, adapter = orb_v3_direct_omat_and_config
    calc = ORBCalculator(orb, adapter, device=torch.device("cpu"))
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


def test_orb_v3_con_omat_predictions(orb_v3_conservative_omat_and_config):
    """Test that we haven't changed the predictions of orb-v3-conservative."""
    orb, adapter = orb_v3_conservative_omat_and_config
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    graph = adapter.from_ase_atoms(atoms)
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
    np.testing.assert_allclose(stress, stress_gold, atol=1e-5)


def test_orbv3_con_omat_optimization(orb_v3_conservative_omat_and_config):
    """Test that we haven't changed the optimization behaviour of orb-v3-conservative."""
    orb, adapter = orb_v3_conservative_omat_and_config
    calc = ORBCalculator(orb, adapter, device=torch.device("cpu"))
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


def test_orbmol_v1_con_predictions(orb_v3_conservative_omol_and_config):
    """Test that we haven't changed the predictions of orbmol-v1."""
    orb, adapter = orb_v3_conservative_omol_and_config
    atoms = molecule("C6H6")
    atoms.info["charge"] = 1.0
    atoms.info["spin"] = 0.0
    energy_gold = np.array(-6316.6646)
    forces_gold = np.array([0.14257583, -0.2070809, 0.01658938])

    # First: check the model.predict() is correct
    graph = adapter.from_ase_atoms(atoms)
    result = orb.predict(graph)
    energy = result["energy"][0].detach().numpy()
    forces = result["grad_forces"][0].detach().numpy()
    assert np.isclose(energy, energy_gold, atol=1e-5)
    np.testing.assert_allclose(forces, forces_gold, atol=1e-5)

    # Second: check the ase calculator interface is correct
    calc = ORBCalculator(orb, adapter, device=torch.device("cpu"))
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()[0]
    assert np.isclose(energy, energy_gold, atol=1e-5)
    np.testing.assert_allclose(forces, forces_gold, atol=1e-5)


def test_orbmol_v1_con_omol_optimization(orb_v3_conservative_omol_and_config):
    """Test that we haven't changed the optimization behaviour of orbmol-v1."""
    orb, adapter = orb_v3_conservative_omol_and_config
    calc = ORBCalculator(orb, adapter, device=torch.device("cpu"))
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
    with pytest.raises(ValueError, match="atoms.info must contain both 'charge' and 'spin'"):
        atoms.get_potential_energy()


@pytest.mark.parametrize(
    "fixture_name",
    [
        "orb_v3_conservative_omat_and_config",
        "orb_v3_conservative_omol_and_config",
    ],
)
def test_pre_cutoff_conservative_models_use_mean_pair_repulsion(fixture_name, request):
    """Pre-cutoff conservative models must keep ZBL mean-aggregation via load_model override.

    The default for newly-trained ConservativeForcefieldRegressor is
    node_aggregation="sum"; load_model switches it back to "mean" for
    artifacts created before _CONSERVATIVE_PAIR_REPULSION_SUM_CUTOFF. All
    shipped orb-v3 conservative models (and orbmol-v2) predate that cutoff.
    """
    orb, _ = request.getfixturevalue(fixture_name)
    assert orb.pair_repulsion, f"{fixture_name} should have pair_repulsion enabled"
    assert orb.pair_repulsion_fn.node_aggregation == "mean", (
        f"{fixture_name}: expected mean aggregation for pre-cutoff model, "
        f"got {orb.pair_repulsion_fn.node_aggregation}"
    )


def test_orbmol_v2_predictions(orbmol_v2_and_config):
    """Test that we haven't changed the predictions of orbmol-v2 (electrostatics model)."""
    orb, adapter = orbmol_v2_and_config

    # Non-periodic: H2O
    atoms = molecule("H2O")
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    graph = adapter.from_ase_atoms(atoms)
    result = orb.predict(graph)
    energy = result["energy"][0].detach().numpy()
    forces = result["grad_forces"][0].detach().numpy()
    h2o_energy_gold = np.array(-2079.86339)
    h2o_forces_gold = np.array([-1.0472e-04, 2.5031e-04, -4.8726e-01])
    assert np.isclose(energy, h2o_energy_gold, atol=1e-5)
    np.testing.assert_allclose(forces, h2o_forces_gold, atol=1e-5)

    # Periodic: Cu
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    graph = adapter.from_ase_atoms(atoms)
    result = orb.predict(graph)
    energy = result["energy"][0].detach().numpy()
    stress = result["grad_stress"][0].detach().numpy()
    cu_energy_gold = np.array(-178549.3860)
    cu_stress_gold = np.array([-0.49615, -0.49357, -0.49229, 0.00097, 0.00205, -0.00068])
    assert np.isclose(energy, cu_energy_gold, atol=1e-5)
    np.testing.assert_allclose(stress, cu_stress_gold, atol=1e-5)

    # Also verify via ASE calculator interface
    calc = ORBCalculator(orb, adapter, device=torch.device("cpu"))
    h2o = molecule("H2O")
    h2o.info["charge"] = 0
    h2o.info["spin"] = 1
    h2o.calc = calc
    energy = h2o.get_potential_energy()
    forces = h2o.get_forces()[0]
    assert np.isclose(energy, h2o_energy_gold[()], atol=1e-5)
    np.testing.assert_allclose(forces, h2o_forces_gold, atol=1e-5)
