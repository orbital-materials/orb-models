# flake8: noqa: E501
import torch
from cached_path import cached_path
from orb_models.forcefield.graph_regressor import (
    EnergyHead,
    NodeHead,
    GraphHead,
    GraphRegressor,
)
from orb_models.forcefield.gns import MoleculeGNS
from orb_models.forcefield.rbf import ExpNormalSmearing


def get_base(
    latent_dim: int = 256,
    mlp_hidden_dim: int = 512,
    num_message_passing_steps: int = 15,
) -> MoleculeGNS:
    """Define the base pretrained model architecture."""
    return MoleculeGNS(
        num_node_in_features=256,
        num_node_out_features=3,
        num_edge_in_features=53,
        latent_dim=latent_dim,
        interactions="simple_attention",
        num_message_passing_steps=num_message_passing_steps,
        num_mlp_layers=2,
        mlp_hidden_dim=mlp_hidden_dim,
        rbf_transform=ExpNormalSmearing(num_rbf=50, cutoff_upper=10.0),
        use_embedding=True,
        node_feature_names=["feat"],
        edge_feature_names=["feat"],
    )


def orb_v1(
    weights_path: str = "https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/orbff-v1-20240827.ckpt",  # noqa: E501
    # NOTE: Use https scheme for weights so that folks can download without gcloud auth.
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

    local_path = cached_path(weights_path)
    state_dict = torch.load(local_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    return model


def orb_d3_v1(
    weights_path: str = "https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/orb-d3-v1-20240902.ckpt",
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

    local_path = cached_path(weights_path)
    state_dict = torch.load(local_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    return model


def orb_d3_sm_v1(
    weights_path: str = "https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/orb-d3-sm-v1-20240902.ckpt",
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

    local_path = cached_path(weights_path)
    state_dict = torch.load(local_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    return model


def orb_d3_xs_v1(
    weights_path: str = "https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/orb-d3-xs-v1-20240902.ckpt",
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

    local_path = cached_path(weights_path)
    state_dict = torch.load(local_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model


def orb_v1_mptraj_only(
    weights_path: str = "https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/orbff-mptraj-only-v1-20240827.ckpt",
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

    local_path = cached_path(weights_path)
    state_dict = torch.load(local_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    return model


ORB_PRETRAINED_MODELS = {
    "orb-v1": orb_v1,
    "orb-d3-v1": orb_d3_v1,
    "orb-d3-sm-v1": orb_d3_sm_v1,
    "orb-d3-xs-v1": orb_d3_xs_v1,
    "orb-v1-mptraj-only": orb_v1_mptraj_only,
}
