from ase.calculators.calculator import Calculator
from ase.vibrations import Vibrations
from ase.build import molecule
from ase.optimize import LBFGS
from matplotlib import pyplot as plt
import numpy as np
import wandb


def vibrate_water(calc: Calculator, box_size: float, fmax=0.01) -> float:
    """Vibrate a water molecule and return the average error compared to gold standard frequencies."""
    water = molecule("H2O")
    water.calc = calc
    water.set_cell([box_size] * 3)
    water.set_pbc(True)
    water.center()

    # Relax the structure using LBFGS minimization
    LBFGS(water).run(fmax, steps=1000)

    # Run the vibrational analysis
    vib = Vibrations(water)
    vib.clean()
    vib.run()

    # Compare predicted and true frequencies
    predicted_freqs = np.array(vib.get_frequencies())[-3:].real

    # https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Units=SI&Mask=1880#Electronic-Spec
    gold_freqs = np.array([1595, 3657, 3756])

    errors = np.abs(predicted_freqs - gold_freqs)
    average_error = np.mean(errors)

    vib.clean()
    return average_error


def vibrate_water_for_various_box_sizes(calc: Calculator):
    """Vibrate a water molecule and return the average error compared to gold standard frequencies."""
    box_sizes = [6.0, 10.0, 14.0, 18.0]
    errors = []
    for box_size in box_sizes:
        error = vibrate_water(calc, box_size)
        errors.append(error)

    fig, ax = plt.subplots()
    ax.plot(box_sizes, errors)
    ax.set_xlabel("Box size")
    ax.set_ylabel("Error")
    return fig, np.mean(errors), np.std(errors)


def main(calc: Calculator):
    """Measure accuracy of vibration frequencies on a few small molecules."""
    fig, water_mae, water_std = vibrate_water_for_various_box_sizes(calc)
    wandb.log({"vibration/water_freq_mae": water_mae}, commit=False)
    wandb.log({"vibration/water_freq_std": water_std}, commit=False)
    wandb.log({"vibration/freq_mae_vs_boxsize": fig}, commit=True)
