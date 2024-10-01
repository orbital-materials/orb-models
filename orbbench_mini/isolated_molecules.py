from collections import defaultdict
from ase.calculators.calculator import Calculator
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import wandb
from orb_models.dataset.ase_dataset import random_rotations_with_properties
from ase.io import read


def eval_small_molecule_maes(calc: Calculator):
    """Compute MAEs on (trajectories of) various isolated molecules and plot them.

    The data was obtained as follows:
        - We took the 163 molecules from https://wiki.fysik.dtu.dk/ase/ase/build/build.html#ase.build.molecule
        - Ran DFT optimisations on those molecules, using MPRelaxSet settings

    Exactly 100 optimisations geo-converged as documented here:
    https://linear.app/orbital-materials/issue/FND-365/orbbench-mini#comment-c33451c1

    Note that in this function we evaluate on every 3rd step of the geometry optimizations for efficiency.
    """
    db = read("trajectories.db", index=":")
    seen_counts: dict = defaultdict(int)
    mol_to_energy_ae = defaultdict(list)
    mol_to_force_ae = defaultdict(list)
    system_force = 0.0
    for row in tqdm(db.select(), total=len(db)):
        mol = row.toatoms()
        formula = mol.get_chemical_formula()
        if seen_counts[formula] != 0:
            continue
        seen_counts[formula] = (seen_counts[formula] + 1) % 3

        mol.set_calculator(calc)
        pred_energy = mol.get_potential_energy()
        energy_mae = np.abs(pred_energy - row["energy"])
        pred_forces = mol.get_forces()
        force_mae = np.abs(pred_forces - row["forces"]).mean()

        mol_to_energy_ae[formula].append(energy_mae)
        mol_to_force_ae[formula].append(force_mae)
        system_force += float(np.linalg.norm(pred_forces.sum(0)))

    system_force = system_force / len(db)
    mol_names = list(mol_to_energy_ae.keys())
    mol_names.sort()
    energy_maes = [np.mean(mol_to_energy_ae[k]) for k in mol_names]
    force_maes = [np.mean(mol_to_force_ae[k]) for k in mol_names]

    sns.set_theme()
    num_plots = len(mol_names) // 20 + 1
    fig, axs = plt.subplots(num_plots, 2, figsize=(20, 5 * num_plots))
    for i, (ax1, ax2) in enumerate(axs):
        start = i * 20
        end = min((i + 1) * 20, len(mol_names))
        x = np.arange(start, end)
        ax1.bar(x, energy_maes[start:end])
        ax1.set_xticks(x)
        ax1.set_xticklabels(mol_names[start:end], rotation=90)
        ax1.set_title("Energy MAE")
        ax2.bar(x, force_maes[start:end])
        ax2.set_xticks(x)
        ax2.set_xticklabels(mol_names[start:end], rotation=90)
        ax2.set_title("Force MAE")
    fig.tight_layout()

    return fig, np.mean(energy_maes), np.mean(force_maes), system_force


def eval_small_molecule_equivariance(
    calc: Calculator,
    molecules_to_eval: tuple = ("C5H5N", "CH3NO2", "AlCl3", "H6Si2", "C2H3FO"),
    num_rotations: int = 10,
):
    """Evaluate a model's energy-invariance and force-equivariance under rotations.

    By default, we evaluate 10 random rotations of 5 molecules.
    These molecules are unoptimized and have non-neglible forces.
    """
    db = read("step1.db", index=":")
    per_mol_energy_deviations: dict = defaultdict(list)
    per_mol_force_deviations: dict = defaultdict(list)
    for row in db.select():
        mol = row.toatoms()
        formula = mol.get_chemical_formula()
        if formula not in molecules_to_eval or per_mol_energy_deviations[formula]:
            continue
        mol.set_calculator(calc)
        base_energy = mol.get_potential_energy()
        base_force = mol.get_forces()
        energy_deviations = []
        force_deviations = []
        for _ in range(num_rotations):
            rmol, props = random_rotations_with_properties(
                mol.copy(), {"node": {"forces": base_force}, "graph": {}}
            )
            rmol.set_calculator(calc)
            expected_rotated_forces = props["node"]["forces"]

            # invariant energy
            pred_energy = rmol.get_potential_energy()
            energy_deviations.append(np.abs(pred_energy - base_energy))

            # equivariant force
            pred_rotated_forces = rmol.get_forces()
            force_deviations.append(
                np.linalg.norm(pred_rotated_forces - expected_rotated_forces)
            )

        per_mol_energy_deviations[formula].extend(energy_deviations)
        per_mol_force_deviations[formula].extend(force_deviations)

    # bar-charts of deviations per molecule
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    for i, (ax, deviations) in enumerate(
        zip(axs, [per_mol_energy_deviations, per_mol_force_deviations])
    ):
        ax.bar(
            deviations.keys(),
            [np.mean(v) for v in deviations.values()],
            yerr=[np.std(v) for v in deviations.values()],
        )
        ax.set_title(
            "Energy invariance error" if i == 0 else "Force equivariance error"
        )
        ax.set_ylabel("Mean Absolute Error")
        ax.set_xticklabels(deviations.keys(), rotation=90)
    fig.tight_layout()

    # average deviations across mols
    mean_energy_deviation = np.mean(
        [np.mean(v) for v in per_mol_energy_deviations.values()]
    )
    mean_force_deviation = np.mean(
        [np.mean(v) for v in per_mol_force_deviations.values()]
    )
    return fig, mean_energy_deviation, mean_force_deviation


def main(calc: Calculator):
    """Measure several types of performance on various isolated molecules.

    Note: we use wandb.Image to avoids buggy xtick labels.
    """
    fig, energy_mae, force_mae, system_force = eval_small_molecule_maes(calc)
    wandb.log({"isolated_molecules/energy_mae": energy_mae})
    wandb.log({"isolated_molecules/force_mae": force_mae})
    wandb.log({"isolated_molecules/system_force": system_force})
    wandb.log({"isolated_molecules/mae_img": wandb.Image(fig)})

    fig, energy_invariance_error, force_equivariance_error = (
        eval_small_molecule_equivariance(calc)
    )
    wandb.log({"isolated_molecules/energy_invariance_error": energy_invariance_error})
    wandb.log({"isolated_molecules/force_equivariance_error": force_equivariance_error})
    wandb.log({"isolated_molecules/equivariance_img": wandb.Image(fig)})
