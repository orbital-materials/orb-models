# type: ignore
import pytest
import torch

from orb_models.forcefield import base
from orb_models.forcefield.atomic_system import atom_graphs_to_ase_atoms
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
        system_features={"cell": torch.eye(3).view(1, 3, 3)},
        node_targets=dict(node_target=noise_or_forces),
        edge_targets={},
        system_targets=dict(sys_target=torch.tensor([[23.3]])),
        system_id=torch.randint(10, (1,)),
        fix_atoms=None,
        tags=None,
        radius=6.0,
        max_num_neighbors=torch.tensor([20]),
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
        system_features={
            "cell": torch.eye(3).view(1, 3, 3),
            "pbc": torch.tensor([[True, True, True]]),
        },
        node_targets=dict(node_target=noise_or_forces),
        edge_targets={},
        system_targets=dict(sys_target=torch.tensor([[23.3]])),
        system_id=torch.tensor([3]),
        fix_atoms=None,
        tags=None,
        radius=6.0,
        max_num_neighbors=torch.tensor([20]),
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


def test_refeaturization_pos_substitution(dataset_and_loader):
    dataset = dataset_and_loader[0]
    datapoint = dataset[0]

    graph = refeaturize_atomgraphs(
        atoms=datapoint,
        positions=torch.zeros_like(datapoint.positions),
    )
    assert torch.all(graph.positions == torch.zeros_like(datapoint.positions)).item()


def test_refeaturization_cell_remapping(dataset_and_loader):
    dataset = dataset_and_loader[0]
    datapoint = dataset[0]
    giant_positions = torch.ones_like(datapoint.positions) * 100

    # PBC remapping should mean that the 100 coords will be mapped to something else
    graph = refeaturize_atomgraphs(atoms=datapoint, positions=giant_positions)
    assert torch.all(graph.positions != giant_positions)

    # No cell should mean positions are unchanged
    new_unit_cell = torch.zeros_like(datapoint.cell)
    graph = refeaturize_atomgraphs(
        atoms=datapoint, positions=giant_positions, cell=new_unit_cell
    )
    assert torch.all(graph.positions == giant_positions).item()

    # check refeaturized graph has the new unit cell
    assert (graph.cell == new_unit_cell).all()


def test_volume_atomgraphs(dataset_and_loader):
    dataloader = dataset_and_loader[1]
    batch = next(iter(dataloader))
    v1 = base.volume_atomgraphs(batch)
    ase_datapoints = atom_graphs_to_ase_atoms(batch)
    v2 = torch.tensor([a.get_volume() for a in ase_datapoints], dtype=torch.float32)
    assert torch.allclose(v1, v2, atol=1e-3)


def test_refeaturization_is_differentiable():
    # check that the refeaturization is differentiable i.e. that the gradients
    # of the refeaturized graph wrt the input positions are non-zero
    datapoint = graph()
    atomic_numbers_embedding = torch.randn(datapoint.positions.shape[0], 10)
    graph_ = refeaturize_atomgraphs(
        atoms=datapoint,
        positions=datapoint.positions,
        atomic_numbers_embedding=atomic_numbers_embedding,
        differentiable=True,
    )
    vectors, _, _ = graph_.compute_differentiable_edge_vectors()
    loss = vectors.sum() + graph_.atomic_numbers_embedding.sum()
    loss.backward()
    assert graph_.positions.grad is not None
    assert graph_.atomic_numbers_embedding.grad is not None
    assert not torch.all(graph_.positions.grad == 0.0)
