# flake8: noqa: E501
from typing import Union, Optional
from functools import partial

import torch
from cached_path import cached_path

from orb_models.forcefield.featurization_utilities import (
    get_device,
    gaussian_basis_function,
)
from orb_models.forcefield.gns import MoleculeGNS
from orb_models.forcefield.graph_regressor import (
    GraphHead,
    GraphRegressor,
)
from orb_models.forcefield.forcefield_heads import EnergyHead, ForceHead
from orb_models.utils import set_torch_precision


def load_model_for_inference(
    model: GraphRegressor,
    weights_path: str,
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> GraphRegressor:
    """Load a pretrained model from a local path or a wandb artifact.

    The model class is inferred from the hydra config. This config is expected
    to contain a `model` key that, when instantiated, is a valid `base.ModelMixin`.

    Args:
        model: The model to load the weights into.
        weights_path: The name of the weights file.
        device: Defaults to GPU if available, but specific CUDA devices can be specified via device index.
        precision: The floating point precision to use for the model.
            - "float32-high" means torch.set_float32_matmul_precision("high") will be called.
            - "float32-highest" means torch.set_float32_matmul_precision("highest") will be called.
            - "float64" means the model will use double precision.
        compile: Whether to torch.compile the model. Defaults to None, which will compile the model
            if the device is not MPS.

    Returns:
        model: The pretrained model
    """
    # Set the precision for the model
    dtype = set_torch_precision(precision)
    model = model.to(dtype)

    # Load the weights
    local_path = cached_path(weights_path)
    state_dict = torch.load(local_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    # Move the model to the device
    device = get_device(device)
    model = model.to(device)
    model = model.eval()

    # Compile the model
    if compile is None:
        compile = device.type != "mps"
    assert not (
        device.type == "mps" and compile
    ), "Model compilation is not supported on MPS."
    if compile:
        model.compile(mode="default", dynamic=True)

    # Freeze the parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


def orb_v2_architecture(
    num_message_passing_steps: int = 15,
    device: Optional[Union[torch.device, str]] = None,
) -> GraphRegressor:
    """Orb-v2 architecture."""
    model = GraphRegressor(
        heads=[
            EnergyHead(
                latent_dim=256,
                num_mlp_layers=1,
                mlp_hidden_dim=256,
                predict_atom_avg=True,
                activation="ssp",
            ),
            ForceHead(
                latent_dim=256,
                num_mlp_layers=1,
                mlp_hidden_dim=256,
                remove_mean=True,
                remove_torque_for_nonpbc_systems=True,
                activation="ssp",
            ),
            GraphHead(
                latent_dim=256,
                num_mlp_layers=1,
                mlp_hidden_dim=256,
                target="stress",
                node_aggregation="mean",
                activation="ssp",
            ),
        ],
        model=MoleculeGNS(
            num_node_in_features=256,
            num_node_out_features=3,
            num_edge_in_features=23,
            latent_dim=256,
            interaction_params={"distance_cutoff": True, "attention_gate": "sigmoid"},
            num_message_passing_steps=num_message_passing_steps,
            num_mlp_layers=2,
            mlp_hidden_dim=512,
            rbf_transform=partial(gaussian_basis_function, num_bases=20, radius=10.0),
            use_embedding=True,
            node_feature_names=["feat"],
            edge_feature_names=["feat"],
            activation="ssp",
            mlp_norm="layer_norm",
        ),
    )
    device = get_device(device)
    if device is not None and device != torch.device("cpu"):
        model.cuda(device)
    else:
        model = model.cpu()

    return model


def orb_v2(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v2-20241011.ckpt",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> GraphRegressor:
    """Load ORB v2."""
    model = orb_v2_architecture(device=device)
    model = load_model_for_inference(
        model, weights_path, device, precision=precision, compile=compile
    )

    return model


def orb_mptraj_only_v2(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-mptraj-only-v2-20241014.ckpt",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
):
    """Load ORB MPTraj Only v2."""

    return orb_v2(weights_path, device, precision=precision, compile=compile)


def orb_d3_v2(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-d3-v2-20241011.ckpt",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> GraphRegressor:
    """Load ORB D3 v2."""
    model = orb_v2_architecture(device=device)
    model = load_model_for_inference(
        model, weights_path, device, precision=precision, compile=compile
    )

    return model


def orb_d3_sm_v2(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-d3-sm-v2-20241011.ckpt",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> GraphRegressor:
    """Load ORB D3 v2."""
    model = orb_v2_architecture(num_message_passing_steps=10, device=device)
    model = load_model_for_inference(
        model, weights_path, device, precision=precision, compile=compile
    )

    return model


def orb_d3_xs_v2(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-d3-xs-v2-20241011.ckpt",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> GraphRegressor:
    """Load ORB D3 xs v2."""
    model = orb_v2_architecture(num_message_passing_steps=5, device=device)
    model = load_model_for_inference(
        model, weights_path, device, precision=precision, compile=compile
    )

    return model


def _deprecated_model(model_name: str):
    """Deprecated model."""

    raise ValueError(
        f"{model_name} is deprecated. Please use orb-v2 instead."
        "Orb V2 models are more accurate, more robust under simulation, and run faster."
    )


def orb_v1(
    weights_path: Optional[str] = None,
    device: Union[torch.device, str, None] = None,
):
    """Deprecated model."""

    _deprecated_model("orb-v1")


def orb_d3_v1(
    weights_path: Optional[str] = None,
    device: Union[torch.device, str, None] = None,
):
    """Deprecated model."""

    _deprecated_model("orb-d3-v1")


def orb_d3_sm_v1(
    weights_path: Optional[str] = None,
    device: Union[torch.device, str, None] = None,
):
    """Deprecated model."""

    _deprecated_model("orb-d3-sm-v1")


def orb_d3_xs_v1(
    weights_path: Optional[str] = None,
    device: Union[torch.device, str, None] = None,
):
    """Deprecated model."""
    _deprecated_model("orb-d3-xs-v1")


def orb_v1_mptraj_only(
    weights_path: Optional[str] = None,
    device: Union[torch.device, str, None] = None,
):
    """Deprecated model."""
    _deprecated_model("orb-mptraj-only-v1")


ORB_PRETRAINED_MODELS = {
    "orb-v2": orb_v2,
    "orb-d3-v2": orb_d3_v2,
    "orb-d3-sm-v2": orb_d3_sm_v2,
    "orb-d3-xs-v2": orb_d3_xs_v2,
    "orb-mptraj-only-v2": orb_mptraj_only_v2,
    # Deprecated models
    "orb-v1": orb_v1,
    "orb-d3-v1": orb_d3_v1,
    "orb-d3-sm-v1": orb_d3_sm_v1,
    "orb-d3-xs-v1": orb_d3_xs_v1,
    "orb-v1-mptraj-only": orb_v1_mptraj_only,
}
