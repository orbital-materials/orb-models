# flake8: noqa: E501
from typing import Union, Optional

import torch
from cached_path import cached_path
from functools import partial

from orb_models.forcefield.featurization_utilities import (
    get_device,
    gaussian_basis_function,
)
from orb_models.forcefield.gns import MoleculeGNS
from orb_models.forcefield.graph_regressor import (
    EnergyHead,
    GraphHead,
    GraphRegressor,
    NodeHead,
)
from orb_models.forcefield.rbf import ExpNormalSmearing

global HAS_MESSAGED_FOR_TF32_MATMUL
HAS_MESSAGED_FOR_TF32_MATMUL = False

torch.set_float32_matmul_precision("high")


def get_base(
    latent_dim: int = 256,
    mlp_hidden_dim: int = 512,
    num_message_passing_steps: int = 15,
    num_edge_in_features: int = 23,
    distance_cutoff: bool = True,
    attention_gate: str = "sigmoid",
    rbf_transform: str = "gaussian",
) -> MoleculeGNS:
    """Define the base pretrained model architecture."""
    return MoleculeGNS(
        num_node_in_features=256,
        num_node_out_features=3,
        num_edge_in_features=num_edge_in_features,
        latent_dim=latent_dim,
        interactions="simple_attention",
        interaction_params={
            "distance_cutoff": distance_cutoff,
            "polynomial_order": 4,
            "cutoff_rmax": 6,
            "attention_gate": attention_gate,
        },
        num_message_passing_steps=num_message_passing_steps,
        num_mlp_layers=2,
        mlp_hidden_dim=mlp_hidden_dim,
        rbf_transform=(
            ExpNormalSmearing(num_rbf=50, cutoff_upper=10.0)
            if rbf_transform == "exp_normal_smearing"
            else partial(gaussian_basis_function, num_bases=20, radius=10.0)
        ),
        use_embedding=True,
        node_feature_names=["feat"],
        edge_feature_names=["feat"],
    )


def load_model_for_inference(
    model: torch.nn.Module,
    weights_path: str,
    device: Union[torch.device, str] = None,
) -> torch.nn.Module:
    """Load a pretrained model in inference mode, using GPU if available."""
    global HAS_MESSAGED_FOR_TF32_MATMUL
    local_path = cached_path(weights_path)
    state_dict = torch.load(local_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    device = get_device(device)
    if (
        device.type == "cuda"
        and torch.get_float32_matmul_precision() == "high"
        and not HAS_MESSAGED_FOR_TF32_MATMUL
    ):
        print(
            "GPU tensorfloat matmuls precision set to 'high'. "
            "This can achieve up to 2x speedup on Nvidia A100 and H100 devices."
        )
        HAS_MESSAGED_FOR_TF32_MATMUL = True
    model = model.to(device)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def orb_v2(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v2-20241011.ckpt",  # noqa: E501
    device: Union[torch.device, str] = None,
):
    """Load ORB v2."""
    base = get_base()

    model = GraphRegressor(
        graph_head=EnergyHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="energy",
            node_aggregation="mean",
            reference_energy_name="vasp-shifted",
            train_reference=True,
            predict_atom_avg=True,
        ),
        node_head=NodeHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="forces",
            remove_mean=True,
        ),
        stress_head=GraphHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="stress",
            compute_stress=True,
        ),
        model=base,
    )

    model = load_model_for_inference(model, weights_path, device)

    return model


def orb_mptraj_only_v2(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-mptraj-only-v2-20241014.ckpt",  # noqa: E501
    device: Union[torch.device, str] = None,
):
    """Load ORB MPTraj Only v2."""

    return orb_v2(weights_path, device)


def orb_d3_v2(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-d3-v2-20241011.ckpt",  # noqa: E501
    device: Union[torch.device, str] = None,
):
    """Load ORB D3 v2."""
    base = get_base()

    model = GraphRegressor(
        graph_head=EnergyHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="energy",
            node_aggregation="mean",
            reference_energy_name="vasp-shifted",
            train_reference=True,
            predict_atom_avg=True,
        ),
        node_head=NodeHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="forces",
            remove_mean=True,
        ),
        stress_head=GraphHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="stress",
            compute_stress=True,
        ),
        model=base,
    )

    model = load_model_for_inference(model, weights_path, device)

    return model


def orb_d3_sm_v2(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-d3-sm-v2-20241011.ckpt",  # noqa: E501
    device: Union[torch.device, str] = None,
):
    """Load ORB D3 v2."""
    base = get_base(
        num_message_passing_steps=10,
    )

    model = GraphRegressor(
        graph_head=EnergyHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="energy",
            node_aggregation="mean",
            reference_energy_name="vasp-shifted",
            train_reference=True,
            predict_atom_avg=True,
        ),
        node_head=NodeHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="forces",
            remove_mean=True,
        ),
        stress_head=GraphHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="stress",
            compute_stress=True,
        ),
        model=base,
    )

    model = load_model_for_inference(model, weights_path, device)

    return model


def orb_d3_xs_v2(
    weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-d3-xs-v2-20241011.ckpt",  # noqa: E501
    device: Union[torch.device, str] = None,
):
    """Load ORB D3 xs v2."""
    base = get_base(
        num_message_passing_steps=5,
    )

    model = GraphRegressor(
        graph_head=EnergyHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="energy",
            node_aggregation="mean",
            reference_energy_name="vasp-shifted",
            train_reference=True,
            predict_atom_avg=True,
        ),
        node_head=NodeHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="forces",
            remove_mean=True,
        ),
        stress_head=GraphHead(
            latent_dim=256,
            num_mlp_layers=1,
            mlp_hidden_dim=256,
            target="stress",
            compute_stress=True,
        ),
        model=base,
    )

    model = load_model_for_inference(model, weights_path, device)

    return model


def _deprecated_model(model_name: str):
    """Deprecated model."""

    raise ValueError(
        f"{model_name} is deprecated. Please use orb-v2 instead."
        "Orb V2 models are more accurate, more robust under simulation, and run faster."
    )


def orb_v1(
    weights_path: Optional[str] = None,
    device: Union[torch.device, str] = None,
):
    """Deprecated model."""

    _deprecated_model("orb-v1")


def orb_d3_v1(
    weights_path: Optional[str] = None,
    device: Union[torch.device, str] = None,
):
    """Deprecated model."""

    _deprecated_model("orb-d3-v1")


def orb_d3_sm_v1(
    weights_path: Optional[str] = None,
    device: Union[torch.device, str] = None,
):
    """Deprecated model."""

    _deprecated_model("orb-d3-sm-v1")


def orb_d3_xs_v1(
    weights_path: Optional[str] = None,
    device: Union[torch.device, str] = None,
):
    """Deprecated model."""
    _deprecated_model("orb-d3-xs-v1")


def orb_v1_mptraj_only(
    weights_path: Optional[str] = None,
    device: Union[torch.device, str] = None,
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
