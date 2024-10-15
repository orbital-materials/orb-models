"""Integration tests to check compatibility of outputs with internal OM models."""

import argparse

import ase
import torch
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
    original_orbff, _, sys_config = load.load_model(core_model)

    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1.1], [0, 1.1, 0]])

    graph_orig = core_atomic_system.ase_atoms_to_atom_graphs(atoms, sys_config)
    graph = atomic_system.ase_atoms_to_atom_graphs(atoms)

    pred_orig = original_orbff.predict(graph_orig)

    orbff = pretrained.ORB_PRETRAINED_MODELS[model]()
    pred = orbff.predict(graph)

    assert torch.allclose(pred["graph_pred"], pred_orig["graph_pred"])
    assert torch.allclose(pred["node_pred"], pred_orig["node_pred"])
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
