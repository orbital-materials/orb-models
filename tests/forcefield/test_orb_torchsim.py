import pytest

pytest.importorskip("torch_sim", reason="torch_sim is required for these tests")

import torch
import torch_sim as ts
from torch_sim.models.interface import validate_model_outputs
from torch_sim.testing import (
    CONSISTENCY_SIMSTATES,
    SIMSTATE_GENERATORS,
    assert_model_calculator_consistency,
)

from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.inference.calculator import ORBCalculator
from orb_models.forcefield.inference.orb_torchsim import OrbTorchSimModel

DEVICE = torch.device("cpu")
DTYPE = torch.float64


# Use the TorchSim consistency check harness for external models
@pytest.mark.parametrize("sim_state_name", CONSISTENCY_SIMSTATES)
@pytest.mark.parametrize(
    "edge_method",
    ["knn_scipy", "knn_alchemi"],
)
def test_orb_torchsim_consistency(sim_state_name, edge_method, conservative_regressor):
    adapter = ForcefieldAtomsAdapter(6.0, 120)
    calculator = ORBCalculator(
        model=conservative_regressor,
        atoms_adapter=adapter,
        edge_method=edge_method,
    )
    sim_model = OrbTorchSimModel(conservative_regressor, adapter, edge_method=edge_method)

    sim_state = SIMSTATE_GENERATORS[sim_state_name](DEVICE, DTYPE)
    assert_model_calculator_consistency(sim_model, calculator, sim_state)


def test_orb_torchsim_validate_outputs(conservative_regressor):
    adapter = ForcefieldAtomsAdapter(6.0, 120)
    sim_model = OrbTorchSimModel(conservative_regressor, adapter, dtype=DTYPE)
    validate_model_outputs(sim_model, device=DEVICE, dtype=DTYPE)


class TestOrbTorchSimStressToggle:
    """Test that enable_stress/disable_stress controls stress in OrbTorchSimModel results."""

    def test_conservative_stress_disabled(self, conservative_regressor, mptraj_10_systems_db):
        conservative_regressor.disable_stress()
        atoms_list = [mptraj_10_systems_db.get_atoms(1)]
        adapter = ForcefieldAtomsAdapter(6.0, 120)
        sim_state = ts.io.atoms_to_state(atoms_list, "cpu", torch.get_default_dtype())
        sim_model = OrbTorchSimModel(conservative_regressor, adapter)
        results = sim_model(sim_state)
        assert "stress" not in results
        assert "forces" in results

    def test_conservative_stress_enabled(self, conservative_regressor, mptraj_10_systems_db):
        conservative_regressor.disable_stress()
        conservative_regressor.enable_stress()
        atoms_list = [mptraj_10_systems_db.get_atoms(1)]
        adapter = ForcefieldAtomsAdapter(6.0, 120)
        sim_state = ts.io.atoms_to_state(atoms_list, "cpu", torch.get_default_dtype())
        sim_model = OrbTorchSimModel(conservative_regressor, adapter)
        results = sim_model(sim_state)
        assert "stress" in results
        assert "forces" in results

    def test_direct_stress_disabled(self, direct_regressor, mptraj_10_systems_db):
        direct_regressor.disable_stress()
        atoms_list = [mptraj_10_systems_db.get_atoms(1)]
        adapter = ForcefieldAtomsAdapter(6.0, 120)
        sim_state = ts.io.atoms_to_state(atoms_list, "cpu", torch.get_default_dtype())
        sim_model = OrbTorchSimModel(direct_regressor, adapter)
        results = sim_model(sim_state)
        assert "stress" not in results
        assert "forces" in results

    def test_direct_stress_enabled(self, direct_regressor, mptraj_10_systems_db):
        direct_regressor.disable_stress()
        direct_regressor.enable_stress()
        atoms_list = [mptraj_10_systems_db.get_atoms(1)]
        adapter = ForcefieldAtomsAdapter(6.0, 120)
        sim_state = ts.io.atoms_to_state(atoms_list, "cpu", torch.get_default_dtype())
        sim_model = OrbTorchSimModel(direct_regressor, adapter)
        results = sim_model(sim_state)
        assert "stress" in results
        assert "forces" in results
