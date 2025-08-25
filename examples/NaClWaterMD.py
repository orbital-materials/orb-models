import torch
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units
from ase.md import MDLogger

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator


def setup_device():
    """Set up and return the appropriate compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def run_md_simulation(
    input_file: str = "NaClWater.xyz",
    cell_size: float = 25.25,
    temperature_K: float = 300,
    timestep: float = 0.5 * units.fs,
    friction: float = 0.01 / units.fs,
    total_steps: int = 100,
    traj_interval: int = 20,
    log_interval: int = 1,
):
    """Run molecular dynamics simulation with specified parameters.

    Args:
        input_file: Path to input XYZ file
        cell_size: Size of cubic simulation cell
        temperature_K: Temperature in Kelvin
        timestep: MD timestep
        friction: Langevin friction coefficient
        total_steps: Total number of MD steps
        traj_interval: Interval for trajectory writing
        log_interval: Interval for log writing
    """
    # Set up device
    device = setup_device()

    # Read in the system from file and set the cell size and pbc
    atoms = read(input_file)
    atoms.set_cell([cell_size] * 3)
    atoms.set_pbc([True] * 3)

    # Set charge and spin multiplicity for OrbMol models
    atoms.info["charge"] = 0.0  # total charge
    atoms.info["spin"] = 1.0  # multiplicity (2S+1)

    # Set the calculator
    # Note: If you encounter compilation errors (e.g., Triton issues on clusters),
    # you can disable compilation by adding compile=False:
    # orbff = pretrained.orb_v3_conservative_omol(device=device, compile=False)
    orbff = pretrained.orb_v3_conservative_omol(device=device)
    atoms.calc = ORBCalculator(orbff, device=device)

    # Set the initial velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

    # Set the dynamics
    dyn = Langevin(atoms, timestep, temperature_K=temperature_K, friction=friction)

    # Define output functions and attach to dynamics
    dyn.attach(
        lambda: write("NaClWaterMD.xyz", atoms, append=True), interval=traj_interval
    )
    dyn.attach(MDLogger(dyn, atoms, "md_nvt.log"), interval=log_interval)

    # Run the dynamics
    dyn.run(steps=total_steps)


def main():
    """Main entry point for the script."""
    run_md_simulation()


if __name__ == "__main__":
    main()
