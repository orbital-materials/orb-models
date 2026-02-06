import pytest

pytest.importorskip("torch_sim", reason="torch_sim is required for these tests")

import torch
import torch_sim as ts
from torch_sim.elastic import full_3x3_to_voigt_6_stress

from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.inference.calculator import ORBCalculator
from orb_models.forcefield.inference.orb_torchsim import OrbTorchSimModel


@pytest.mark.parametrize(
    "edge_method",
    ["knn_scipy", "knn_alchemi"],
)
def test_orb_torchsim_interface(edge_method, conservative_regressor, mptraj_10_systems_db):
    atoms_list = [mptraj_10_systems_db.get_atoms(i) for i in range(1, 5)]

    adapter = ForcefieldAtomsAdapter(6.0, 120)
    calculator = ORBCalculator(
        model=conservative_regressor,
        atoms_adapter=adapter,
        edge_method=edge_method,
    )

    # TorchSim results
    sim_state = ts.io.atoms_to_state(atoms_list, "cpu", torch.get_default_dtype())
    sim_model = OrbTorchSimModel(conservative_regressor, adapter, edge_method=edge_method)
    model_results = sim_model(sim_state)

    # ASE calculator results
    calc_energy_list = []
    calc_force_list = []
    calc_stress_list = []
    for atoms in atoms_list:
        atoms.calc = calculator
        calc_energy = atoms.get_potential_energy()
        calc_forces = torch.tensor(
            atoms.get_forces(),
            dtype=model_results["forces"].dtype,
        )
        calc_stress = torch.tensor(
            atoms.get_stress(),
            dtype=model_results["stress"].dtype,
        )
        calc_energy_list.append(calc_energy)
        calc_force_list.append(calc_forces)
        calc_stress_list.append(calc_stress)

    # Test consistency with specified tolerances
    torch.testing.assert_close(
        model_results["energy"],
        torch.tensor(calc_energy_list),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        model_results["forces"],
        torch.cat(calc_force_list),
        rtol=1e-5,
        atol=1e-5,
    )
    if "stress" in model_results:
        torch.testing.assert_close(
            full_3x3_to_voigt_6_stress(model_results["stress"]),
            torch.stack(calc_stress_list),
            rtol=1e-5,
            atol=1e-5,
            equal_nan=True,
        )
