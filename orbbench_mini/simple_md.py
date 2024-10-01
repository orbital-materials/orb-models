from ase.md.langevin import Langevin
from ase.build import molecule
from ase.units import fs
import numpy as np
import wandb
from ase.calculators.calculator import Calculator


def water_drift(
    calc: Calculator,
    mol="H2O",
    n_runs: int = 3,
    n_steps: int = 1_000,
    timestep: float = 1.0 * fs,
    temperature: float = 300,
    friction: float = 0.002,
):
    """Run MD on a water molecule, tracking drift and momentum (averaging over several runs)."""
    water = molecule(mol)
    init_com = water.get_positions().mean(0)  # centre-of-mass
    water.calc = calc

    md_drifts = []
    for _ in range(n_runs):
        md = Langevin(water, timestep, temperature_K=temperature, friction=friction)
        md.run(n_steps)
        final_com = water.get_positions().mean(0)
        md_drift = np.abs(init_com - final_com).mean()
        md_drifts.append(md_drift)

    # Calculate means and standard deviations
    drift_mean = np.mean(md_drifts)
    drift_std = np.std(md_drifts)
    return {
        "md_drift_mean": drift_mean,
        "md_drift_std": drift_std,
    }


def main(calc: Calculator):
    """Run simple MD tests, like drift of a water molecule."""
    for mol in ["H2O", "CO2"]:
        results = water_drift(calc, mol=mol)
        wandb.log({f"simple-md/{mol}-drift": results})
