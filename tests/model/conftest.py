import copy
import pytest
import torch

from orb_models.forcefield import gns
from orb_models.forcefield.property_definitions import PropertyDefinition
from orb_models.forcefield.rbf import ExpNormalSmearing
from orb_models.forcefield import base
from orb_models.forcefield import graph_regressor


def one_hot(x):
    return torch.nn.functional.one_hot(x, num_classes=118).float()


@pytest.fixture()
def graph():
    nodes, edges = 10, 6
    positions = torch.randn((nodes, 3))
    atomic_numbers = torch.arange(0, nodes)
    vectors = torch.randn((edges, 3))
    lengths = vectors.norm(dim=1)
    return base.AtomGraphs(
        senders=torch.tensor([0, 1, 2, 1, 2, 0]),
        receivers=torch.tensor([1, 0, 1, 2, 0, 2]),
        n_node=torch.tensor([nodes]),
        n_edge=torch.tensor([edges]),
        node_features=dict(
            atomic_numbers=atomic_numbers,
            atomic_numbers_embedding=one_hot(atomic_numbers),
            positions=positions,
        ),
        edge_features=dict(
            vectors=vectors,
            r=lengths,
            unit_shifts=torch.zeros_like(vectors),
        ),
        system_features={
            "cell": torch.eye(3).unsqueeze(0),
            "prior_loss": torch.tensor([0.0]),
        },
        node_targets={"noise_target": torch.randn_like(positions)},
        edge_targets={},
        system_targets={"graph_target": torch.tensor([[23.3]])},
        fix_atoms=torch.tensor([1, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=torch.bool),
        system_id=None,
        tags=None,
        radius=6.0,
        max_num_neighbors=torch.tensor([20]),
    )


@pytest.fixture()
def graph_binary(graph):
    cloned_graph = copy.deepcopy(graph)
    node_target = torch.randint(
        0, 2, (len(cloned_graph.node_features["positions"]),)
    ).long()
    return cloned_graph._replace(
        node_targets={"binary_node_target": node_target},
        system_targets={"binary_graph_target": torch.tensor([[1]]).long()},
    )


@pytest.fixture()
def graph_categorical(graph):
    cloned_graph = copy.deepcopy(graph)
    node_target = torch.randint(
        0, 5, (len(cloned_graph.node_features["positions"]),)
    ).long()
    return cloned_graph._replace(
        node_targets={"cat_node_target": node_target},
        system_targets={"cat_graph_target": torch.tensor([[3]]).long()},
    )


@pytest.fixture()
def single_node_graph():
    nodes, edges = 1, 1
    positions = torch.randn((nodes, 3))
    atomic_numbers = torch.arange(0, nodes)
    vectors = torch.randn((edges, 3))
    lengths = vectors.norm(dim=1)
    return base.AtomGraphs(
        senders=torch.tensor([0]),
        receivers=torch.tensor([0]),
        n_node=torch.tensor([nodes]),
        n_edge=torch.tensor([edges]),
        node_features=dict(
            atomic_numbers=atomic_numbers,
            atomic_numbers_embedding=one_hot(atomic_numbers),
            positions=positions,
        ),
        edge_features=dict(
            vectors=vectors,
            r=lengths,
            unit_shifts=torch.zeros_like(vectors),
        ),
        system_features={"cell": torch.eye(3).unsqueeze(0)},
        node_targets={"noise_target": torch.randn_like(positions)},
        edge_targets={},
        system_targets={"graph_target": torch.tensor([[23.3]])},
        system_id=None,
        fix_atoms=None,
        tags=None,
        radius=6.0,
        max_num_neighbors=20,
    )


@pytest.fixture()
def single_node_graph_binary(single_node_graph):
    cloned_graph = copy.deepcopy(single_node_graph)
    node_target = torch.randint(
        0, 2, (len(cloned_graph.node_features["positions"]),)
    ).long()
    return cloned_graph._replace(
        node_targets={"binary_node_target": node_target},
        system_targets={"binary_graph_target": torch.tensor([[1]]).long()},
    )


@pytest.fixture()
def single_node_graph_categorical(single_node_graph):
    cloned_graph = copy.deepcopy(single_node_graph)
    node_target = torch.randint(
        0, 5, (len(cloned_graph.node_features["positions"]),)
    ).long()
    return cloned_graph._replace(
        node_targets={"cat_node_target": node_target},
        system_targets={"cat_graph_target": torch.tensor([[3]]).long()},
    )


@pytest.fixture
def gns_model():
    """Instantiates the molecular model."""
    return gns.MoleculeGNS(
        num_node_in_features=118,
        num_node_out_features=3,
        num_edge_in_features=13,
        latent_dim=8,
        num_message_passing_steps=2,
        num_mlp_layers=1,
        mlp_hidden_dim=8,
        edge_feature_names=["feat"],
        node_feature_names=["feat"],
        rbf_transform=ExpNormalSmearing(num_rbf=10),
    )


@pytest.fixture
def graph_head():
    def graph_head_(target=None, latent_dim=8):
        if target is None:
            target = PropertyDefinition(
                "graph_target", dim=1, domain="real", row_to_property_fn=lambda x: x
            )
        return graph_regressor.GraphHead(
            latent_dim=latent_dim,
            num_mlp_layers=0,
            mlp_hidden_dim=0,
            target=target,
            node_aggregation="sum",
        )

    return graph_head_


@pytest.fixture
def node_head():
    def node_head_(target=None):
        if target is None:
            target = PropertyDefinition(
                "noise_target", dim=3, domain="real", row_to_property_fn=lambda x: x
            )
        return graph_regressor.NodeHead(
            latent_dim=8,
            num_mlp_layers=0,
            mlp_hidden_dim=0,
            target=target,
        )

    return node_head_
