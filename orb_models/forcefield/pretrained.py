# flake8: noqa: E501
from typing import Union

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
    num_edge_in_features: int = 53,
    distance_cutoff: bool = False,
    attention_gate: str = "softmax",
    rbf_transform: str = "exp_normal_smearing",
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


def orb_v1(
    weights_path: str = "https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/orbff-v1-20240827.ckpt",  # noqa: E501
    device: Union[torch.device, str] = None,
):
    """Load ORB v1."""
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


def orb_v2(
    weights_path: str = "https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/orbff-v2-20240930.ckpt",  # noqa: E501
    device: Union[torch.device, str] = None,
):
    """Load ORB v1."""
    base = get_base(
        num_edge_in_features=23,
        distance_cutoff=True,
        attention_gate="sigmoid",
        rbf_transform="gaussian",
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


def orb_d3_v1(
    weights_path: str = "https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/orb-d3-v1-20240902.ckpt",
    device: Union[torch.device, str] = None,
):
    """ORB v1 with D3 corrections."""
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


def orb_d3_sm_v1(
    weights_path: str = "https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/orb-d3-sm-v1-20240902.ckpt",
    device: Union[torch.device, str] = None,
):
    """A 10 layer model pretrained on bulk data."""
    base = get_base(num_message_passing_steps=10)

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


def orb_d3_xs_v1(
    weights_path: str = "https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/orb-d3-xs-v1-20240902.ckpt",
    device: Union[torch.device, str] = None,
):
    """A 5 layer model pretrained on bulk data."""
    base = get_base(num_message_passing_steps=5)
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


def orb_v1_mptraj_only(
    weights_path: str = "https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/orbff-mptraj-only-v1-20240827.ckpt",
    device: Union[torch.device, str] = None,
):
    """A 10 layer model pretrained on bulk data."""
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


ORB_PRETRAINED_MODELS = {
    "orb-v1": orb_v1,
    "orb-v2": orb_v2,
    "orb-d3-v1": orb_d3_v1,
    "orb-d3-sm-v1": orb_d3_sm_v1,
    "orb-d3-xs-v1": orb_d3_xs_v1,
    "orb-v1-mptraj-only": orb_v1_mptraj_only,
}
