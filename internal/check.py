"""Integration tests to check compatibility of outputs with internal OM models."""

import argparse

import ase
import torch
import numpy as np

from core.dataset import atomic_system as core_atomic_system
from core.models import load

from orb_models.forcefield import atomic_system, pretrained


def main(model: str, core_model: str):
    """Compare outputs of pretrained model with internal model.

    NOTE: This script is for internal use only.

    Args:
        model: Name of the pretrained model to use.
        core_model: Path to the core model.
    """
    original_orbff, _, sys_config = load.load_model(core_model, precision="float32-high")

    atoms = ase.Atoms(
        "H2O",
        positions=[[0, 0, 0], [0, 0, 1.1], [0, 1.1, 0]],
        cell=np.eye(3) * 2,
        pbc=True,
    )

    graph_orig = core_atomic_system.ase_atoms_to_atom_graphs(atoms, sys_config)
    graph = atomic_system.ase_atoms_to_atom_graphs(atoms, sys_config)

    pred_orig = original_orbff.predict(graph_orig)

    orbff = pretrained.ORB_PRETRAINED_MODELS[model](precision="float32-high")
    pred = orbff.predict(graph)

    forces_key = "grad_forces" if "grad_forces" in pred else "forces"
    assert torch.allclose(pred[forces_key], pred_orig[forces_key], atol=1e-4)
    assert torch.allclose(pred["energy"], pred_orig["energy"], atol=1e-4)
    print("Model outputs are identical!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model outputs")
    parser.add_argument(
        "--model",
        type=str,
        default="orb-v1",
        help="Name of the pretrained model to use",
    )
    parser.add_argument(
        "--core_model",
        type=str,
        default="orbitalmaterials/model-registry/orbFF:v1",
        help="Path to the core model",
    )
    args = parser.parse_args()

    main(args.model, args.core_model)
