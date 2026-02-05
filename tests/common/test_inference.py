import pytest
import torch

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.models import gns
from orb_models.common.models.rbf import ExpNormalSmearing
from orb_models.forcefield.models.direct_regressor import DirectForcefieldRegressor
from orb_models.forcefield.models.forcefield_heads import EnergyHead, ForceHead


def graph(use_system_id: bool = False):
    nodes = 10
    edges = 6
    positions = torch.randn((nodes, 3))
    senders = torch.tensor([0, 1, 2, 4, 5, 3])
    receivers = torch.tensor([1, 0, 1, 5, 3, 5])
    vectors = torch.randn((edges, 3))
    lengths = vectors.norm(dim=1)
    atomic_numbers = torch.arange(0, nodes)
    noise_or_forces = torch.randn_like(positions)
    if use_system_id:
        system_id = torch.tensor([100, 30])
    else:
        system_id = None
    return AtomGraphs(
        senders=senders,
        receivers=receivers,
        n_node=torch.tensor([nodes - 4, 4]),
        n_edge=torch.tensor([edges]),
        node_features=dict(
            atomic_numbers=atomic_numbers,
            positions=positions,
            atomic_numbers_embedding=torch.nn.functional.one_hot(
                atomic_numbers, num_classes=118
            ).to(positions.dtype),
        ),
        edge_features=dict(
            vectors=vectors,
            r=lengths,
            unit_shifts=torch.zeros_like(vectors),
        ),
        system_features={"cell": torch.zeros((3, 3)).unsqueeze(0)},
        node_targets=dict(noise_target=noise_or_forces),
        edge_targets={},
        system_targets=dict(system_target=torch.tensor([[23.3, 11.3]])),
        system_id=system_id,
        fix_atoms=None,
        tags=None,
        radius=6.0,
        max_num_neighbors=torch.tensor([20]),
    )


@pytest.fixture()
def molecule_gns_model():
    """Instantiates the molecular model."""
    m = gns.MoleculeGNS(
        num_node_in_features=118,
        num_node_out_features=3,
        num_edge_in_features=13,
        latent_dim=8,
        num_message_passing_steps=1,
        num_mlp_layers=1,
        mlp_hidden_dim=8,
        edge_feature_names=["feat"],
        node_feature_names=["feat"],
        rbf_transform=ExpNormalSmearing(num_rbf=10),
    )
    return m.to(dtype=torch.get_default_dtype())


def get_regressor(model, latent_dim):
    """Instantiates a graph regression model."""
    node_head = ForceHead(
        latent_dim=latent_dim,
        num_mlp_layers=0,
        mlp_hidden_dim=0,
    )
    graph_head = EnergyHead(
        latent_dim=latent_dim,
        num_mlp_layers=0,
        mlp_hidden_dim=0,
    )
    return DirectForcefieldRegressor(
        heads={
            "forces": node_head,
            "energy": graph_head,
        },
        model=model,
        model_requires_grad=True,
    )


def test_predict_split(molecule_gns_model):
    graph_no_system_id = graph(use_system_id=False)
    regressor = get_regressor(molecule_gns_model, latent_dim=8)

    predictions = regressor.predict(graph_no_system_id, split=True)

    assert isinstance(predictions, dict)
    assert "forces" in predictions
    assert "energy" in predictions
    assert len(predictions["forces"]) == 2
    assert len(predictions["energy"]) == 2
