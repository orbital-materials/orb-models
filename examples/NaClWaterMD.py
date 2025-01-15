import torch
import numpy as np
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units
from orb_models.forcefield import atomic_system, pretrained
from ase.calculators.calculator import Calculator, all_properties
from ase.build import molecule, make_supercell
from ase.md import MDLogger
from ase.io import write

# Device setup this should enable use of both NVIDIA and Mac GPUs
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Basic configuration
input_file = "NaClWater.xyz"
cell_size = 25.25  

# Define the OrbD3Calculator class
class OrbD3Calculator(Calculator):
    def __init__(self, model, **kwargs):
        self.implemented_properties = ['energy', 'forces']
        super().__init__(**kwargs)
        self.model = model.to(device)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_properties):
        super().calculate(atoms, properties, system_changes)
        graph = atomic_system.ase_atoms_to_atom_graphs(atoms).to(device)
        result = self.model.predict(graph)
        self.results['energy'] = float(result["graph_pred"].cpu().detach())
        self.results['forces'] = result["node_pred"].cpu().detach().numpy()

# Read in the system from file and set the cell size and pbc
atoms = read(input_file)
atoms.set_cell([cell_size] * 3)
atoms.set_pbc([True] * 3)

# Set the calculator
atoms.calc = OrbD3Calculator(model=pretrained.orb_d3_v2())

# Adjust simulation parameters
temperature_K = 300
timestep = 0.5 * units.fs

# Set the initial velocities
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

# Set the dynamics
dyn = Langevin(atoms, timestep, temperature_K=temperature_K, friction=0.01 / units.fs)

# Define output functions and attach to dynamics
dyn.attach(lambda: write('NaClWaterMD.xyz', atoms, append=True), interval=20)
dyn.attach(MDLogger(dyn, atoms, "md_nvt.log"), interval=1)

# Run the dynamics
dyn.run(steps=100000)