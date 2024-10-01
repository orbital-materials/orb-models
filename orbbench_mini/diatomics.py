from typing import Optional
import ase
import cached_path
import json
import numpy as np
import pandas as pd
import wandb
from matplotlib import pyplot as plt
from ase.calculators.calculator import Calculator


def get_energy(atoms: ase.Atoms, calc: Calculator):
    """Get the energy of an atoms object."""
    atoms.calc = calc
    return atoms.get_potential_energy()


def get_forces(atoms: ase.Atoms, calc: Calculator):
    """Get the forces on an atoms object."""
    atoms.calc = calc
    return atoms.get_forces()


def add_horizontal_line(
    ax: plt.Axes,
    y_value: float,
    color: str = "r",
    linestyle: str = "--",
    label: Optional[str] = None,
):
    """Add a horizontal line to a plot."""
    ax.axhline(y=y_value, color=color, linestyle=linestyle, label=label)
    if label:
        ax.legend()
    return ax


def calculate_bifurcated_mae(
    true_values: np.ndarray, pred_values: np.ndarray, lower: int = -10, upper: int = 10
):
    """Calculate MAE for values within and outside a specified range."""
    in_range_mask = (true_values >= lower) & (true_values <= upper)
    out_range_mask = ~in_range_mask

    mae_in = (
        np.abs(true_values[in_range_mask] - pred_values[in_range_mask]).mean()
        if in_range_mask.any()
        else None
    )
    mae_out = (
        np.abs(true_values[out_range_mask] - pred_values[out_range_mask]).mean()
        if out_range_mask.any()
        else None
    )
    return mae_in, mae_out


def predict_diatomics(
    calc: Calculator,
    element: str,
    dists: np.ndarray = np.linspace(0.01, 10, 100),
):
    """Predict the energy and force of two atoms of the same type at varying distances."""
    energies = []
    forces = []
    base_energy = get_energy(ase.Atoms(element), calc)
    for x in dists:
        m = ase.Atoms(f"2{element}", positions=[[0, 0, 0], [x, 0, 0]])
        energy = get_energy(m, calc)
        energies.append(energy)
        f = get_forces(m, calc)
        forces.append(f[:, 0])  # x-component of forces on each atom
    pred_energy = np.array(energies) - (2 * base_energy)
    pred_forces = np.array(forces)[:, 1]
    return pred_energy, pred_forces


def eval_diatomics(calc: Calculator, element: str):
    """Plot the energy and force of a two atoms of the same type at varying distances."""
    energy_path = cached_path.cached_path(
        f"gs://orb-data-artifacts/datasets/homonuclear-diatomics/{element}_distenergy_pairs.json"
    )
    with open(energy_path, "r") as fpath:
        data = json.load(fpath)
        df = pd.DataFrame(data[element], columns=["distance", "energy"])
    force_path = cached_path.cached_path(
        f"gs://orb-data-artifacts/datasets/homonuclear-diatomics/{element}_distforce_pairs.json"
    )
    with open(force_path, "r") as fpath:
        data = json.load(fpath)
        df["force"] = [x[1] for x in data[element]]

    # make predictions on the coarse grid for which we have DFT values
    coarse_dists = df["distance"].values
    pred_energy_coarse, pred_force_coarse = predict_diatomics(
        calc, element, coarse_dists
    )

    # Calculate energy MAEs
    energy_mae_in, energy_mae_out = calculate_bifurcated_mae(
        df["energy"].values, pred_energy_coarse
    )
    # Calculate force MAEs
    force_mae_in, force_mae_out = calculate_bifurcated_mae(
        df["force"].values, pred_force_coarse
    )

    # make predictions on a fine grid so we can plot a high-res curve
    fine_dists = np.linspace(0.01, 10, 100)
    pred_energy, pred_force = predict_diatomics(calc, element, fine_dists)

    fig, axs = plt.subplots(1, 2)

    # Plot energies
    df.plot.scatter(x="distance", y="energy", ax=axs[0], color="b")
    axs[0].scatter(coarse_dists, pred_energy_coarse, color="r", label="Predicted", s=20)
    axs[0].plot(fine_dists, pred_energy, color="r")
    ax = add_horizontal_line(axs[0], 0)
    ax.set_title(f"{element}-{element} energy")
    legend = [f"MAE (|x|<10): {energy_mae_in:.3f} eV"]
    if energy_mae_out is not None:
        legend += [f"MAE (|x|>10): {energy_mae_out:.3f} eV"]
    ax.legend(legend, fontsize=13)

    # Plot forces
    df.plot.scatter(x="distance", y="force", ax=axs[1], color="b")
    axs[1].scatter(coarse_dists, pred_force_coarse, color="r", label="Predicted", s=20)
    axs[1].plot(fine_dists, pred_force, color="r")
    ax = add_horizontal_line(axs[1], 0)
    ax.set_title(f"{element}-{element} force")
    legend = [f"MAE (|x|<10): {force_mae_in:.3f} eV/A"]
    if force_mae_out is not None:
        legend += [f"MAE (|x|>10): {force_mae_out:.3f} eV/A"]
    ax.legend(legend, fontsize=13)

    for ax in axs:
        ax.set_ylim(-10, 10)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.grid(0.5, alpha=0.3)

    return fig, energy_mae_in, energy_mae_out, force_mae_in, force_mae_out


def main(calc: Calculator):
    """Run the diatomics benchmark."""
    energy_maes_in, energy_maes_out = [], []
    force_maes_in, force_maes_out = [], []
    for element in ["C", "O", "H", "Si", "Al", "Na", "K", "Li", "Mg", "Ca"]:
        fig, energy_mae_in, energy_mae_out, force_mae_in, force_mae_out = (
            eval_diatomics(calc, element)
        )
        wandb.log({f"diatomics/{element}": fig})
        fig.clf()
        energy_maes_in.append(energy_mae_in)
        if energy_mae_out is not None:
            energy_maes_out.append(energy_mae_out)
        force_maes_in.append(force_mae_in)
        if force_mae_out is not None:
            force_maes_out.append(force_mae_out)

    wandb.log({"diatomics/inlier_energy_mae": np.mean(energy_maes_in)})
    wandb.log({"diatomics/outlier_energy_mae": np.mean(energy_maes_out)})
    wandb.log({"diatomics/inlier_force_mae": np.mean(force_maes_in)})
    wandb.log({"diatomics/outlier_force_mae": np.mean(force_maes_out)})
