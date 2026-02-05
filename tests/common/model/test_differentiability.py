import pytest
import torch

from orb_models.common.atoms.batch.graph_batch import AtomGraphs, refeaturize_atomgraphs
from orb_models.common.dataset.ase_sqlite_dataset import AseSqliteDataset
from orb_models.common.dataset.property_definitions import PropertyConfig
from orb_models.common.models import gns
from orb_models.common.models.rbf import ExpNormalSmearing
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter


@pytest.fixture()
def model():
    """Instantiates the molecular model."""
    m = gns.MoleculeGNS(
        num_node_in_features=118,
        num_node_out_features=3,
        num_edge_in_features=13,
        latent_dim=8,
        num_message_passing_steps=1,
        num_mlp_layers=1,
        mlp_hidden_dim=8,
        rbf_transform=ExpNormalSmearing(num_rbf=10),
    )
    m = m.to(dtype=torch.get_default_dtype())
    return m.eval()


@pytest.fixture
def dataset(shared_fixtures_path):
    dataset_config = ForcefieldAtomsAdapter(
        radius=6.0,
        max_num_neighbors=20,
    )
    dataset = AseSqliteDataset(
        name="test_dataset",
        path=str(shared_fixtures_path / "databases/test_dataset.db"),
        atoms_adapter=dataset_config,
        target_config=PropertyConfig(),
    )
    return dataset


def test_model_output_is_differentiable_wrt_positions(model, dataset):
    datapoint = dataset[0]
    datapoint.system_features["timesteps"] = torch.tensor([0.0])
    batch = AtomGraphs.batch([datapoint])
    batch = refeaturize_atomgraphs(atoms=batch, positions=batch.positions, differentiable=True)
    out = model(batch)
    grad = torch.autograd.grad(out["pred"].sum(), batch.positions)[0]
    assert grad is not None


def test_model_output_is_differentiable_wrt_positions_in_for_loop(model, dataset):
    datapoint = dataset[0]
    datapoint.system_features["timesteps"] = torch.tensor([0.0])
    batch = AtomGraphs.batch([datapoint])
    batch = refeaturize_atomgraphs(atoms=batch, positions=batch.positions, differentiable=True)
    for _ in range(3):
        out = model(batch)
        grad = torch.autograd.grad(out["pred"].sum(), batch.positions, retain_graph=True)[0]
        assert grad is not None
