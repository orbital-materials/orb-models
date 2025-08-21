"""Unit tests for charge and spin conditioning functionality."""

import pytest
import torch
from copy import deepcopy

from orb_models.forcefield.nn_util import ChargeSpinEmbedding, ChargeSpinConditioner
from orb_models.forcefield import gns
from orb_models.forcefield.rbf import ExpNormalSmearing


@pytest.fixture
def graph_with_charge_spin_batch(graph):
    """Create a batch of graphs with charge and spin system features."""
    graph_with_features = deepcopy(graph)
    # Create 3 identical graphs
    graph_with_features = graph_with_features._replace(
        n_node=torch.tensor([10, 10, 10]),  # 3 graphs, 10 nodes each
        n_edge=torch.tensor([6, 6, 6]),     # 3 graphs, 6 edges each
        system_features={
            **graph_with_features.system_features,
            "total_charge": torch.tensor([-1.0, 0.0, 1.0]),  # 3 graphs
            "total_spin": torch.tensor([0.0, 1.0, 2.0]),     # 3 graphs
        }
    )
    return graph_with_features


@pytest.fixture
def gns_model_with_conditioner():
    """Create a GNS model with charge/spin conditioning."""
    conditioner = ChargeSpinConditioner(
        latent_dim=8,
        embedding_type="sin_emb",
        emits_node_embs=True,
        emits_edge_embs=False,
    )

    return gns.MoleculeGNS(
        latent_dim=8,
        num_message_passing_steps=2,
        num_mlp_layers=1,
        mlp_hidden_dim=8,
        rbf_transform=ExpNormalSmearing(num_rbf=10),
        conditioning=True,
        conditioner=conditioner,
    )


@pytest.fixture
def graph_with_charge_spin(graph):
    """Create a graph with charge and spin system features."""
    graph_with_features = deepcopy(graph)
    graph_with_features.system_features.update(
        {
            "total_charge": torch.tensor([0.0]),  # Single graph with charge 0
            "total_spin": torch.tensor([1.0]),  # Single graph with spin 1
        }
    )
    return graph_with_features


def test_spin_embedding_zero_handling():
    """Test that zero spin produces zero embedding."""
    embedding = ChargeSpinEmbedding(
        num_channels=64,
        embedding_target="spin",
        embedding_type="sin_emb",
    )
    values = torch.tensor([0.0, 1.0])
    output = embedding(values)

    # Zero spin should produce zero embedding
    assert torch.allclose(output[0], torch.zeros_like(output[0]))
    # Non-zero spin should produce non-zero embedding
    assert not torch.allclose(output[1], torch.zeros_like(output[1]))


def test_charge_spin_conditioner_forward(graph_with_charge_spin_batch):
    """Test forward pass of conditioner with batch."""
    conditioner = ChargeSpinConditioner(
        latent_dim=64,
        embedding_type="sin_emb",
        emits_node_embs=True,
        emits_edge_embs=False,
    )

    node_embs, edge_embs = conditioner(graph_with_charge_spin_batch)

    assert node_embs is not None
    assert edge_embs is None

    total_nodes = graph_with_charge_spin_batch.n_node.sum().item()
    assert node_embs.shape == (total_nodes, 64)


def test_conditioner_missing_features(graph):
    """Test that missing system features raises error."""
    conditioner = ChargeSpinConditioner(
        latent_dim=64,
        embedding_type="sin_emb",
        emits_node_embs=True,
        emits_edge_embs=False,
    )

    with pytest.raises(AssertionError):
        conditioner(graph)


def test_conditioning_affects_output(gns_model_with_conditioner, graph):
    """Test that conditioning affects the model output."""
    # Create two graphs with different charge/spin
    graph1 = deepcopy(graph)
    graph1.system_features.update(
        {
            "total_charge": torch.tensor([0.0]),
            "total_spin": torch.tensor([0.0]),
        }
    )

    graph2 = deepcopy(graph)
    graph2.system_features.update(
        {
            "total_charge": torch.tensor([1.0]),
            "total_spin": torch.tensor([1.0]),
        }
    )

    output1 = gns_model_with_conditioner(graph1)
    output2 = gns_model_with_conditioner(graph2)

    # Outputs should be different due to conditioning
    assert not torch.allclose(output1["pred"], output2["pred"])


def test_conditioning_gradients(gns_model_with_conditioner, graph_with_charge_spin):
    """Test that conditioning parameters have gradients."""
    output = gns_model_with_conditioner(graph_with_charge_spin)
    loss = output["pred"].sum()
    loss.backward()

    # Check that conditioner parameters have gradients
    # Some parameters might have zero gradients, which is normal
    has_gradients = False
    for name, param in gns_model_with_conditioner.conditioner.named_parameters():
        if param.grad is not None:
            if torch.any(param.grad != 0):
                has_gradients = True
                break

    # At least some parameters should have non-zero gradients
    assert has_gradients, "No conditioner parameters have non-zero gradients"
