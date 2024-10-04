import numpy as np
from ase.optimize import BFGS
from ase.build import bulk
from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.calculator import ORBCalculator


def test_energy_and_forces_prediction():
    orbff = pretrained.orb_v1(device="cpu")
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    graph = atomic_system.ase_atoms_to_atom_graphs(atoms, device="cpu")
    result = orbff.predict(graph)
    energy_gold = np.array(-16.3437)
    energy = result["graph_pred"][0]
    print(energy)
    forces_gold = np.array([9.3569e-05, -4.0413e-05, 6.0380e-05])
    forces = result["node_pred"][0][0]
    print(forces)
    stress_gold = np.array(
        [-3.2064e-02, -3.2115e-02, -3.1964e-02, -6.2145e-07, 2.5603e-07, 1.6000e-06]
    )
    stress = result["stress_pred"][0]
    print(stress)


def test_optimization():
    # Rattle the atoms to get them out of the minimum energy configuration
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    orbff = pretrained.orb_v1(device="cpu")
    calc = ORBCalculator(orbff, device="cpu")
    atoms.set_calculator(calc)
    atoms.rattle(0.5)
    rattled_energy = atoms.get_potential_energy()
    print(rattled_energy)
    dyn = BFGS(atoms)
    dyn.run(fmax=0.01)
    optimized_energy = atoms.get_potential_energy()
    print(optimized_energy)
