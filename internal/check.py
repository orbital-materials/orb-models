"""Integration tests to check compatibility of outputs with internal OM models."""

import argparse
import ase
import torch
import numpy as np

from core.models import load
from orb_models.forcefield import atomic_system, pretrained


def main(model: str, core_model: str):
    """Compare outputs of pretrained model with internal model.

    NOTE: This script is for internal use only.

    Args:
        model: Name of the pretrained model to use.
        core_model: Path to the core model.
    """
    original_orbff, _, atoms_adapter = load.load_model(core_model, precision="float32-high")
    sys_config = atomic_system.SystemConfig(
        radius=atoms_adapter.radius,
        max_num_neighbors=atoms_adapter.max_num_neighbors
    )
    atoms = ase.Atoms(
        "H2O",
        positions=[[0, 0, 0], [0, 0, 1.1], [0, 1.1, 0]],
        cell=np.eye(3) * 2,
        pbc=True,
    )
    atoms.info["charge"] = 1
    atoms.info["spin"] = 0

    graph_orig = atoms_adapter.from_ase_atoms(atoms)
    graph = atomic_system.ase_atoms_to_atom_graphs(atoms=atoms, system_config=sys_config)

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
        default="orb-v3-conservative-omol",
        help="Name of the pretrained model to use",
    )
    parser.add_argument(
        "--core_model",
        type=str,
        default="orbital-materials/wandb-registry-model/orb-v3-conservative-120-omol:v1",
        help="Path to the core model",
    )
    args = parser.parse_args()

    main(args.model, args.core_model)


# TODO (BEN):
# - check calculator as well as model
# - stress must be optional everywhere in the codebase now
# - update readme