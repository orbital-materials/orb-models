import pytest
import torch

from orb_models.forcefield import gns
from orb_models.forcefield.rbf import ExpNormalSmearing


@pytest.fixture
def featurized_graph(graph):
    rbf = ExpNormalSmearing(num_rbf=10)
    node_features = {
        "feat": torch.nn.functional.one_hot(
            graph.node_features["atomic_numbers"], num_classes=118
        )
        .squeeze()
        .type(torch.float32)
    }
    lengths = graph.edge_features["vectors"].norm(dim=1)
    rbfs = rbf(lengths)
    unit_vectors = graph.edge_features["vectors"] / lengths.unsqueeze(1)

    graph = graph._replace(
        node_features=node_features,
        edge_features={
            **graph.edge_features,
            **{"feat": torch.cat([rbfs, unit_vectors], dim=1)},
        },
    )
    return graph


def test_encoder(featurized_graph):
    enc = gns.Encoder(
        num_node_in_features=118,
        num_edge_in_features=13,
        latent_dim=3,
        num_mlp_layers=1,
        mlp_hidden_dim=8,
    )

    node_features = torch.cat(
        [featurized_graph.node_features[k] for k in ["feat"]], dim=-1
    )
    edge_features = torch.cat(
        [featurized_graph.edge_features[k] for k in ["feat"]], dim=-1
    )
    nodes, edges = enc(node_features, edge_features)

    assert nodes.shape[-1] == 3
    assert edges.shape[-1] == 3


def test_decoder(featurized_graph):
    decoder = gns.Decoder(
        num_node_in=118,
        num_node_out=3,
        num_mlp_layers=1,
        mlp_hidden_dim=8,
    )

    node_features = torch.cat(
        [featurized_graph.node_features[k] for k in ["feat"]], dim=-1
    )
    pred = decoder(node_features)

    assert pred.shape[-1] == 3


def test_interaction_network(featurized_graph):
    enc = gns.Encoder(
        num_node_in_features=118,
        num_edge_in_features=13,
        latent_dim=3,
        num_mlp_layers=1,
        mlp_hidden_dim=8,
    )

    node_features = torch.cat(
        [featurized_graph.node_features[k] for k in ["feat"]], dim=-1
    )
    edge_features = torch.cat(
        [featurized_graph.edge_features[k] for k in ["feat"]], dim=-1
    )
    nodes, edges = enc(node_features, edge_features)

    net = gns.AttentionInteractionNetwork(
        latent_dim=3,
        num_mlp_layers=1,
        mlp_hidden_dim=8,
    )

    cutoff = torch.ones_like(featurized_graph.edge_features["vectors"][:, 0])
    nodes, edges = net(
        nodes, edges, featurized_graph.senders, featurized_graph.receivers, cutoff
    )

    assert nodes.shape[-1] == 3
    assert edges.shape[-1] == 3


def test_model_parameters_have_grad(graph, gns_model):
    pred = gns_model.eval()(graph)["pred"]
    loss = pred.sum()
    loss.backward()
    params = gns_model.state_dict(keep_vars=True)
    for key, param in params.items():
        # Can't assert non zero because of relu
        if param.grad is not None:
            assert not torch.all(~param.grad.bool()), f"{key} has no grad"
        else:
            print(f"{key} has no grad")
            raise ValueError(f"{key} has no grad")


def test_gns_can_torch_compile(gns_model, graph):
    """Tests if the MoleculeGNS.forward is compilable with torch.compile."""
    compiled = torch.compile(gns_model, dynamic=True, mode="default", fullgraph=True)

    compiled(graph)
