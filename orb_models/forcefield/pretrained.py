# flake8: noqa: E501
from typing import Tuple, Union, Optional
from functools import partial

import torch
from cached_path import cached_path

from orb_models.forcefield.angular import SphericalHarmonics
from orb_models.forcefield.atomic_system import SystemConfig
from orb_models.forcefield.conservative_regressor import ConservativeForcefieldRegressor
from orb_models.forcefield.direct_regressor import DirectForcefieldRegressor
from orb_models.forcefield.featurization_utilities import (
    get_device,
    gaussian_basis_function,
)
from orb_models.forcefield.gns import MoleculeGNS
from orb_models.forcefield.forcefield_heads import ConfidenceHead, EnergyHead, ForceHead, GraphHead, StressHead
from orb_models.forcefield.rbf import BesselBasis
from orb_models.utils import set_torch_precision


def load_model_for_inference(
    model: Union[DirectForcefieldRegressor, ConservativeForcefieldRegressor],
    weights_path: str,
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> Union[DirectForcefieldRegressor, ConservativeForcefieldRegressor]:
    """Load a pretrained model from a local path or a wandb artifact.

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
    system_config: Optional[SystemConfig] = None,
) -> DirectForcefieldRegressor:
    """Orb-v2 architecture."""
    model = DirectForcefieldRegressor(
        heads={
            "energy": EnergyHead(
                latent_dim=256,
                num_mlp_layers=1,
                mlp_hidden_dim=256,
                predict_atom_avg=True,
                activation="ssp",
            ),
            "forces": ForceHead(
                latent_dim=256, 
                num_mlp_layers=1,
                mlp_hidden_dim=256,
                remove_mean=True,
                remove_torque_for_nonpbc_systems=True,
                activation="ssp",
            ),
            "stress": GraphHead(
                latent_dim=256,
                num_mlp_layers=1,
                mlp_hidden_dim=256,
                target="stress",
                node_aggregation="mean",
                activation="ssp",
            ),
        },
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
        pair_repulsion=False,
        system_config=system_config,
    )
    device = get_device(device)
    if device is not None and device != torch.device("cpu"):
        model.cuda(device)
    else:
        model = model.cpu()

    return model


def orb_v3_conservative_architecture(
    latent_dim: int = 256,
    base_mlp_hidden_dim: int = 1024,
    base_mlp_depth: int = 2,
    head_mlp_hidden_dim: int = 256,
    head_mlp_depth: int = 1,
    num_message_passing_steps: int = 5,
    activation: str = "silu",
    device: Optional[torch.device] = None,
    system_config: Optional[SystemConfig] = None,
) -> ConservativeForcefieldRegressor:
    """The orb-v3 conservative architecture."""
    model = ConservativeForcefieldRegressor(
        heads={
            "energy": EnergyHead(
                latent_dim=latent_dim,
                num_mlp_layers=head_mlp_depth,
                mlp_hidden_dim=head_mlp_hidden_dim,
                predict_atom_avg=True,
                activation=activation,
            ),
            "confidence": ConfidenceHead(
                latent_dim=latent_dim,
                num_mlp_layers=head_mlp_depth,
                mlp_hidden_dim=head_mlp_hidden_dim,
                activation=activation,
            ),
        },
        model=MoleculeGNS(
            latent_dim=latent_dim,
            num_message_passing_steps=num_message_passing_steps,
            num_mlp_layers=base_mlp_depth,
            mlp_hidden_dim=base_mlp_hidden_dim,
            rbf_transform=BesselBasis(
                r_max=6.0,
                num_bases=8,
            ),
            angular_transform=SphericalHarmonics(
                lmax=3,
                normalize=True,
                normalization="component",
            ),
            outer_product_with_cutoff=True,
            use_embedding=True,
            interaction_params={
                "distance_cutoff": True,
                "attention_gate": "sigmoid",
            },
            node_feature_names=["feat"],
            edge_feature_names=["feat"],
            activation=activation,
            mlp_norm="rms_norm",
        ),
        ensure_grad_loss_weights=False,
        pair_repulsion=True,
        system_config=system_config,
    )
    if device is not None and device != torch.device("cpu"):
        model.cuda(device)
    else:
        model = model.cpu()

    return model


def orb_v3_direct_architecture(
    latent_dim: int = 256,
    base_mlp_hidden_dim: int = 1024,
    base_mlp_depth: int = 2,
    head_mlp_hidden_dim: int = 256,
    head_mlp_depth: int = 1,
    num_message_passing_steps: int = 5,
    activation: str = "silu",
    device: Optional[torch.device] = None,
    system_config: Optional[SystemConfig] = None,
) -> DirectForcefieldRegressor:
    """The orb-v3 architecture, defaulting to a direct model."""
    model = DirectForcefieldRegressor(
        heads={
            "energy": EnergyHead(
                latent_dim=latent_dim,
                num_mlp_layers=head_mlp_depth,
                mlp_hidden_dim=head_mlp_hidden_dim,
                predict_atom_avg=True,
                activation=activation,
            ),
            "forces": ForceHead(
                latent_dim=latent_dim,
                num_mlp_layers=head_mlp_depth,
                mlp_hidden_dim=head_mlp_hidden_dim,
                remove_mean=True,
                remove_torque_for_nonpbc_systems=True,
                activation=activation,
            ),
            "stress": StressHead(
                latent_dim=latent_dim,
                num_mlp_layers=head_mlp_depth,
                mlp_hidden_dim=head_mlp_hidden_dim,
                node_aggregation="mean",
                activation=activation,
            ),
            "confidence": ConfidenceHead(
                latent_dim=latent_dim,
                num_mlp_layers=head_mlp_depth,
                mlp_hidden_dim=head_mlp_hidden_dim,
                activation=activation,
            ),
        },
        model=MoleculeGNS(
            latent_dim=latent_dim,
            num_message_passing_steps=num_message_passing_steps,
            num_mlp_layers=base_mlp_depth,
            mlp_hidden_dim=base_mlp_hidden_dim,
            rbf_transform=BesselBasis(
                r_max=6.0,
                num_bases=8,
            ),
            angular_transform=SphericalHarmonics(
                lmax=3,
                normalize=True,
                normalization="component",
            ),
            outer_product_with_cutoff=True,
            use_embedding=True,
            interaction_params={
                "distance_cutoff": True,
                "attention_gate": "sigmoid",
            },
            node_feature_names=["feat"],
            edge_feature_names=["feat"],
            activation=activation,
            mlp_norm="rms_norm",
        ),
        pair_repulsion=True,
        system_config=system_config,
    )
    if device is not None and device != torch.device("cpu"):
        model.cuda(device)
    else:
        model = model.cpu()

    return model

def orb_v3_conservative_20_omat(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-20-omat-20250404.ckpt",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> ConservativeForcefieldRegressor:
    """Load ORB v3 Conservative 20 max neighbors OMAT."""

    system_config = SystemConfig(radius=6.0, max_num_neighbors=20)
    model = orb_v3_conservative_architecture(device=device, system_config=system_config)
    model = load_model_for_inference(
        model, weights_path, device, precision=precision, compile=compile
    )

    return model

def orb_v3_conservative_inf_omat(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-inf-omat-20250404.ckpt",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> ConservativeForcefieldRegressor:
    """Load ORB v3 Conservative with effectively unlimited neighbors, trained on OMAT.

    'Effectively unlimited' means that the model will use all neighbors within 6A
    the cutoff radius. Empirically, for the training distribution, 120 is sufficient.
    """
    system_config = SystemConfig(radius=6.0, max_num_neighbors=120)
    model = orb_v3_conservative_architecture(device=device, system_config=system_config)
    model = load_model_for_inference(
        model, weights_path, device, precision=precision, compile=compile
    )

    return model

def orb_v3_direct_20_omat(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-direct-20-omat-20250404.ckpt",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> DirectForcefieldRegressor:
    """Load ORB v3 Direct 20 max neighbors OMAT."""
    system_config = SystemConfig(radius=6.0, max_num_neighbors=20)
    model = orb_v3_direct_architecture(device=device, system_config=system_config)
    model = load_model_for_inference(
        model, weights_path, device, precision=precision, compile=compile
    )

    return model

def orb_v3_direct_inf_omat(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-direct-inf-omat-20250404.ckpt",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> DirectForcefieldRegressor:
    """Load ORB v3 Direct with effectively unlimited neighbors, trained on OMAT.

    'Effectively unlimited' means that the model will use all neighbors within 6A
    the cutoff radius. Empirically, for the training distribution, 120 is sufficient.
    """
    system_config = SystemConfig(radius=6.0, max_num_neighbors=120)
    model = orb_v3_direct_architecture(device=device, system_config=system_config)
    model = load_model_for_inference(
        model, weights_path, device, precision=precision, compile=compile
    )

    return model

def orb_v3_conservative_20_mpa(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-20-mpa-20250404.ckpt",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> ConservativeForcefieldRegressor:
    """Load ORB v3 Conservative 20 max neighbors MPTraj + Alexandria."""
    system_config = SystemConfig(radius=6.0, max_num_neighbors=20)
    model = orb_v3_conservative_architecture(device=device, system_config=system_config)
    model = load_model_for_inference(
        model, weights_path, device, precision=precision, compile=compile
    )

    return model

def orb_v3_conservative_inf_mpa(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-inf-mpa-20250404.ckpt",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> ConservativeForcefieldRegressor:
    """Load ORB v3 Conservative with effectively unlimited neighbors, trained on MPTraj + Alexandria.

    'Effectively unlimited' means that the model will use all neighbors within 6A
    the cutoff radius. Empirically, for the training distribution, 120 is sufficient.
    """
    system_config = SystemConfig(radius=6.0, max_num_neighbors=120)
    model = orb_v3_conservative_architecture(device=device, system_config=system_config)
    model = load_model_for_inference(
        model, weights_path, device, precision=precision, compile=compile
    )

    return model

def orb_v3_direct_20_mpa(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-direct-20-mpa-20250404.ckpt",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> DirectForcefieldRegressor:
    """Load ORB v3 Direct 20 max neighbors MPTraj + Alexandria."""
    system_config = SystemConfig(radius=6.0, max_num_neighbors=20)
    model = orb_v3_direct_architecture(device=device, system_config=system_config)
    model = load_model_for_inference(
        model, weights_path, device, precision=precision, compile=compile
    )

    return model

def orb_v3_direct_inf_mpa(
    weights_path: str = "",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> DirectForcefieldRegressor:
    """Load ORB v3 Direct with effectively unlimited neighbors, trained on MPTraj + Alexandria.

    'Effectively unlimited' means that the model will use all neighbors within 6A
    the cutoff radius. Empirically, for the training distribution, 120 is sufficient.
    """
    system_config = SystemConfig(radius=6.0, max_num_neighbors=120)
    model = orb_v3_direct_architecture(device=device, system_config=system_config)
    model = load_model_for_inference(
        model, weights_path, device, precision=precision, compile=compile
    )

    return model


def orb_v2(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v2-20241011.ckpt",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> DirectForcefieldRegressor:
    """Load ORB v2 Direct with 20 max neighbors, trained on MPTraj + Alexandria."""
    system_config = SystemConfig(radius=6.0, max_num_neighbors=20)
    model = orb_v2_architecture(device=device, system_config=system_config)
    model = load_model_for_inference(
        model, weights_path, device, precision=precision, compile=compile
    )

    return model


def orb_mptraj_only_v2(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-mptraj-only-v2-20241014.ckpt",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> DirectForcefieldRegressor:
    """Load ORB MPTraj Only v2 Direct with 20 max neighbors, trained on MPTraj."""
    system_config = SystemConfig(radius=6.0, max_num_neighbors=20)
    model = orb_v2_architecture(device=device, system_config=system_config)
    model = load_model_for_inference(
        model, weights_path, device, precision=precision, compile=compile
    )

    return model


def orb_d3_v2(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-d3-v2-20241011.ckpt",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> DirectForcefieldRegressor:
    """Load ORB D3 v2 Direct with 20 max neighbors, trained on MPTraj + Alexandria."""
    system_config = SystemConfig(radius=6.0, max_num_neighbors=20)
    model = orb_v2_architecture(device=device, system_config=system_config)
    model = load_model_for_inference(
        model, weights_path, device, precision=precision, compile=compile
    )

    return model


def orb_d3_sm_v2(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-d3-sm-v2-20241011.ckpt",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> DirectForcefieldRegressor:
    """Load ORB D3 small v2 with 20 max neighbors, trained on MPTraj + Alexandria."""
    system_config = SystemConfig(radius=6.0, max_num_neighbors=20)
    model = orb_v2_architecture(num_message_passing_steps=10, device=device, system_config=system_config)
    model = load_model_for_inference(
        model, weights_path, device, precision=precision, compile=compile
    )

    return model


def orb_d3_xs_v2(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-d3-xs-v2-20241011.ckpt",  # noqa: E501
    device: Union[torch.device, str, None] = None,
    precision: str = "float32-high",
    compile: Optional[bool] = None,
) -> DirectForcefieldRegressor:
    """Load ORB D3 xs v2 with 20 max neighbors, trained on MPTraj + Alexandria."""
    system_config = SystemConfig(radius=6.0, max_num_neighbors=20)
    model = orb_v2_architecture(num_message_passing_steps=5, device=device, system_config=system_config)
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
    # most performant orb-v3 omat models
    "orb-v3-conservative-20-omat": orb_v3_conservative_20_omat, 
    "orb-v3-conservative-inf-omat": orb_v3_conservative_inf_omat,
    "orb-v3-direct-20-omat": orb_v3_direct_20_omat,
    "orb-v3-direct-inf-omat": orb_v3_direct_inf_omat,
    # orb-v3 mptraj + alexandria models
    "orb-v3-conservative-20-mpa": orb_v3_conservative_20_mpa,
    "orb-v3-conservative-inf-mpa": orb_v3_conservative_inf_mpa,
    "orb-v3-direct-20-mpa": orb_v3_direct_20_mpa,
    "orb-v3-direct-inf-mpa": orb_v3_direct_inf_mpa,
    # less performant orb-v2 models
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
