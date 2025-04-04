"""Utilities to load models."""

from typing import Any, Mapping

import torch.nn as nn

from orb_models.forcefield.gns import MoleculeGNS


def load_forcefield_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    strict: bool = True,
    assign: bool = False,
    skip_artifact_reference_energy: bool = False,
):
    """Load a state dict into the DirectForcefieldRegressor or ConservativeForcefieldRegressor.

    This method overrides the generic nn.Module load_state_dict method in order
    to handle the following cases:
        - The state_dict comes from a legacy (orb-v2) GraphRegressor.
        - The state_dict comes from a DirectForcefieldRegressor or ConservativeForcefieldRegressor
            with a different set of heads. In this case, we only load weights for the common heads.
        - The state_dict contains a reference energy key, which we skip loading
            if skip_artifact_reference_energy is True.

    NOTE: We assume that the presence of the prefix "heads." in any key of the
    state_dict implies that the state_dict comes from a DirectForcefieldRegressor 
    or ConservativeForcefieldRegressor.
    """
    state_dict = dict(state_dict)  # Shallow copy

    if skip_artifact_reference_energy:
        keys_to_remove = [k for k in state_dict.keys() if "reference" in k]
        for k in keys_to_remove:
            del state_dict[k]

    # BC-compatbility for orb-v2 era models
    replace_prefix(state_dict, "node_head.", "heads.forces.")
    replace_prefix(state_dict, "graph_head.", "heads.energy.")
    replace_prefix(state_dict, "stress_head.", "heads.stress.")

    loading_regressor = any(key.startswith("heads.") for key in state_dict.keys())
    if not loading_regressor:
        # emulate a headless regressor
        replace_prefix(state_dict, "", "model.")

    # Edit state dict of Denoiser/DiffusionModel so it can be loaded
    loading_regressor_with_denoising_base = any(
        key.startswith("model.model.") for key in state_dict.keys()
    )
    if loading_regressor_with_denoising_base and isinstance(model.model, MoleculeGNS):
        # Delete all params except the GNS params
        for key in list(state_dict.keys()):
            if key.startswith("model.") and not key.startswith("model.model."):
                del state_dict[key]

        replace_prefix(state_dict, "model.model.", "model.")

    if getattr(model, "cutoff_layers", None) is not None:
        # If cutoff_layers is specified, remove extra GNS layers from state dict
        cutoff = model.cutoff_layers
        for key in list(state_dict.keys()):
            if key.startswith("model.gnn_stacks."):
                layer_num = int(key.split(".")[2])
                if layer_num >= cutoff:  # type: ignore
                    del state_dict[key]

    # Call the parent class's load_state_dict method, which is nn.Module.load_state_dict
    bad_keys = super(type(model), model).load_state_dict(  # type: ignore
        state_dict, strict=False, assign=assign
    )

    if strict:
        # Ensure bad keys are purely due to missing/extra heads
        head_keys = [k for k in model.state_dict().keys() if k.startswith("heads.")]
        for key in bad_keys.missing_keys:
            if key not in head_keys:
                raise RuntimeError(f"Missing key in state_dict: {key}")
        for key in bad_keys.unexpected_keys:
            if not key.startswith("heads."):
                raise RuntimeError(f"Unexpected key in state_dict: {key}")

    return bad_keys


def replace_prefix(
    dictionary: Mapping[str, Any], old_prefix: str, new_prefix: str
) -> None:
    """Mutate dictionary, replacing `old_prefix` with `new_prefix`."""
    for key in list(dictionary.keys()):
        if key.startswith(old_prefix):
            new_key = key.replace(old_prefix, new_prefix, 1)
            assert new_key not in dictionary, f"Key {new_key} already exists."
            dictionary[new_key] = dictionary.pop(key)  # type: ignore
