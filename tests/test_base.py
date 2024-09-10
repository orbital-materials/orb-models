# type: ignore
import pytest
import torch

from orb_models.forcefield import base
from orb_models.forcefield.base import refeaturize_atomgraphs


def random_graph(nodes, edges):
    positions = torch.randn((nodes, 3))
    senders = torch.randint(0, nodes, (edges,))
    receivers = torch.randint(0, nodes, (edges,))
    edge_features = torch.randn((edges, 12))
    atomic_numbers = torch.arange(0, nodes)
    noise_or_forces = torch.randn_like(positions)
    return base.AtomGraphs(
        senders=senders,
        receivers=receivers,
        n_node=torch.tensor([nodes]),
        n_edge=torch.tensor([edges]),
        node_features=dict(atomic_numbers=atomic_numbers, positions=positions),
        edge_features=dict(feat=edge_features),
        node_targets=dict(node_target=noise_or_forces),
        system_features={"cell": torch.eye(3).view(1, 3, 3)},
        system_targets=dict(sys_target=torch.tensor([[23.3]])),
        system_id=torch.randint(10, (1,)),
        radius=5.0,
        max_num_neighbors=10,
    )


def graph():
    nodes = 10
    edges = 6
    positions = torch.linspace(-nodes, nodes, nodes * 3).view(nodes, 3)
    senders = torch.tensor([0, 1, 2, 1, 2, 0])
    receivers = torch.tensor([1, 0, 1, 2, 0, 2])
    edge_features = torch.linspace(-edges, edges, edges * 12).view(edges, 12)
    atomic_numbers = torch.arange(0, nodes)
    noise_or_forces = positions.clone() * 10.0
    return base.AtomGraphs(
        senders=senders,
        receivers=receivers,
        n_node=torch.tensor([nodes]),
        n_edge=torch.tensor([edges]),
        node_features=dict(atomic_numbers=atomic_numbers, positions=positions),
        edge_features=dict(feat=edge_features),
        node_targets=dict(node_target=noise_or_forces),
        system_features={"cell": torch.eye(3).view(1, 3, 3)},
        system_targets=dict(sys_target=torch.tensor([[23.3]])),
        system_id=torch.tensor([3]),
        radius=5.0,
        max_num_neighbors=10,
    )


def test_equality():
    g1 = graph()
    g2 = graph()
    assert g1.equals(g2)
    assert g1.allclose(g2)

    # small changes break equality, but not closeness
    g2.positions = g2.positions + 1e-5
    assert not g1.equals(g2)
    assert g1.allclose(g2, atol=1e-4)

    # big changes break both
    g2.positions = g2.positions * 2.0
    assert not g1.equals(g2)
    assert not g1.allclose(g2)

    # repeat the above for edge vectors (just to be safe)
    g2 = graph()
    g2.edge_features["feat"] = g2.edge_features["feat"] + 1e-5
    assert not g1.equals(g2)
    assert g1.allclose(g2, atol=1e-4)

    g2.edge_features["feat"] = g2.edge_features["feat"] * 2.0
    assert not g1.equals(g2)
    assert not g1.allclose(g2)


@pytest.mark.parametrize("clone", [True, False])
def test_batching(clone):
    graphs = [graph(), graph()]
    batched = base.batch_graphs(graphs)
    unbatched = batched.split(clone)

    # check that unbatched matches original list
    for new, orig in zip(unbatched, graphs):
        assert new.equals(orig)

    # check that batched was unaffected by our call to split_graphs
    assert batched.equals(base.batch_graphs(graphs))


@pytest.mark.parametrize("clone", [True, False])
def test_split_graphs_cloning(clone):
    # check that if we call split_graphs with clone=False,
    # then the returned list of graphs are views not copies
    batched = base.batch_graphs([graph(), graph()])
    orig_positions = batched.positions.clone()

    unbatched = batched.split(clone=clone)
    unbatched[0].positions += 2.0

    if clone:
        assert torch.equal(batched.positions, orig_positions)
    else:
        assert not torch.equal(batched.positions, orig_positions)


def test_random_batching():
    graphs = [random_graph(i, i) for i in range(1, 5)]
    batched = base.batch_graphs(graphs)

    res = batched.split()
    for r, o in zip(res, graphs):
        assert r.equals(o)

    # check that batched was unaffected by our call to split_graphs
    assert batched.equals(base.batch_graphs(graphs))


def test_refeaturization_is_differentiable():
    # check that the refeaturization is differentiable
    # i.e. that the gradients of the refeaturized graph wrt the input positions
    # are non-zero
    datapoint = graph()
    atomic_number_embeddings = torch.randn(datapoint.positions.shape[0], 10)
    graph_ = refeaturize_atomgraphs(
        atoms=datapoint,
        positions=datapoint.positions,
        atomic_number_embeddings=atomic_number_embeddings,
        differentiable=True,
    )
    loss = (
        graph_.edge_features["vectors"].sum()
        + graph_.node_features["atomic_numbers_embedding"].sum()
    )
    loss.backward()
    assert graph_.positions.grad is not None
    assert graph_.node_features["atomic_numbers_embedding"].grad is not None
    assert not torch.all(graph_.positions.grad == 0.0)
