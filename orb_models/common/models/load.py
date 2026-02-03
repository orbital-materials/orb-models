"""Utilities to load models."""

import warnings
from collections.abc import Mapping
from typing import Any

from orb_models.common.models import base
from orb_models.common.models.gns import MoleculeGNS
from orb_models.common.utils import replace_prefix_in_keys


def load_regressor_state_dict(
    model: base.ModelMixin,
    state_dict: Mapping[str, Any],
    strict: bool = True,
    assign: bool = False,
    skip_artifact_reference_energy: bool = False,
):
    """Load a state dict into a RegressorModelMixin.

    This method overrides the generic nn.Module load_state_dict method in order
    to handle the following cases:
        - The state_dict comes from a GNS/DiffusionModel.
        - The state_dict comes from a legacy (orb-v2) GraphRegressor.
        - The state_dict comes from a forcefield with a different set of heads.
            In this case, we only load weights for the common heads.
        - The state_dict contains a reference energy key, which we skip loading
            if skip_artifact_reference_energy is True.
    NOTE:
    - We assume that the presence of the prefix "heads." in any key of the
        state_dict implies that the state_dict comes from a forcefield.
    - We allow diffusion models to be loaded into a forcefield.
      This is because some diffusion models lack time embeddings and are thus null
      wrappers around a GNS model and this wrapper can simply be discarded.
    """
    state_dict = dict(state_dict)  # Shallow copy

    if skip_artifact_reference_energy:
        keys_to_remove = [k for k in state_dict if "reference" in k]
        for k in keys_to_remove:
            del state_dict[k]

    # BC-compatbility for orb-v2 era models
    replace_prefix_in_keys(state_dict, "node_head.", "heads.forces.")
    replace_prefix_in_keys(state_dict, "graph_head.", "heads.energy.")
    replace_prefix_in_keys(state_dict, "stress_head.", "heads.stress.")

    # BC-compatibility
    replace_prefix_in_keys(
        state_dict,
        "heads.stress.diag_normaliser.",
        "heads.stress.diag_normalizer.",
    )
    replace_prefix_in_keys(
        state_dict,
        "heads.stress.offdiag_normaliser.",
        "heads.stress.offdiag_normalizer.",
    )

    loading_regressor = any(key.startswith("heads.") for key in state_dict)
    if not loading_regressor:
        # emulate a headless regressor
        replace_prefix_in_keys(state_dict, "", "model.")

    # Edit state dict of Denoiser/DiffusionModel so it can be loaded
    loading_regressor_with_denoising_base = any(
        key.startswith("model.model.") for key in state_dict
    )
    if loading_regressor_with_denoising_base and isinstance(model.model, MoleculeGNS):
        # Delete all params except the GNS params
        for key in list(state_dict.keys()):
            if key.startswith("model.") and not key.startswith("model.model."):
                del state_dict[key]

        replace_prefix_in_keys(state_dict, "model.model.", "model.")

    if getattr(model, "cutoff_layers", None) is not None:
        # If cutoff_layers is specified, remove extra GNS layers from state dict
        cutoff = model.cutoff_layers
        for key in list(state_dict.keys()):
            if key.startswith("model.gnn_stacks."):
                layer_num = int(key.split(".")[2])
                if layer_num >= cutoff:  # type: ignore
                    del state_dict[key]

    # Call the parent class's load_state_dict method, which is nn.Module.load_state_dict
    bad_keys = super(type(model), model).load_state_dict(state_dict, strict=False, assign=assign)

    if strict:
        # Ensure bad keys are purely due to missing/extra heads
        def is_head_key(key: str) -> bool:
            return (
                key.startswith("heads.")
                or key.startswith("grad_forces_normalizer")
                or key.startswith("grad_stress_normalizer")
            )

        head_keys = [k for k in model.state_dict() if is_head_key(k)]
        for key in bad_keys.missing_keys:
            if key not in head_keys:
                raise RuntimeError(f"Found key in model, missing in state_dict: {key}")
            else:
                # warning if the key is a head key
                warnings.warn(
                    f"Found key in model, missing in state_dict: {key}",
                    UserWarning,
                )
        for key in bad_keys.unexpected_keys:
            if not is_head_key(key):
                raise RuntimeError(f"Found key in state_dict, unexpected in model: {key}")
            else:
                # warning if the key is a head key
                warnings.warn(
                    f"Found key in state_dict, unexpected in model: {key}",
                    UserWarning,
                )

    return bad_keys
