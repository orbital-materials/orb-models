import ase
import pytest
import torch
import numpy as np

from orb_models.forcefield.direct_regressor import DirectForcefieldRegressor
from orb_models.forcefield.forcefield_heads import EnergyHead, ForceHead, StressHead
from orb_models.forcefield import base
from orb_models.forcefield import atomic_system
from orb_models.forcefield.gns import _KEY
from orb_models.forcefield.rbf import BesselBasis, ExpNormalSmearing
from orb_models.forcefield import gns, segment_ops
from orb_models.forcefield.conservative_regressor import ConservativeForcefieldRegressor


def one_hot(x):
    return torch.nn.functional.one_hot(x, num_classes=118).float()


def get_batch_from_ase_with_latents():
    latent_dim = 9
    atoms = ase.Atoms("H2O", positions=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]]))
    atoms.set_cell([10, 10, 10])
    atoms.set_pbc(True)

    system_config = atomic_system.SystemConfig(radius=6.0, max_num_neighbors=20)
    atom_graphs = atomic_system.ase_atoms_to_atom_graphs(
        atoms, system_config=system_config
    )

    nodes = len(atoms)
    n_edges = atom_graphs.n_edge[0]

    # add targets
    energy_target = torch.randn((1, 1))
    force_targets = torch.randn((nodes, 3))
    stress_target = torch.randn((1, 6))

    # Mix in some of the target in to the
    # feats, so there is learning signal
    features = 0.8 * torch.randn((nodes, latent_dim))
    features += 0.1 * force_targets.repeat_interleave(3, dim=1)
    features += 0.1 * energy_target[0]

    return atom_graphs._replace(
        node_features={
            **atom_graphs.node_features,
            _KEY: features,
        },
        edge_features={
            **atom_graphs.edge_features,
            _KEY: torch.randn((n_edges, latent_dim)),
        },
        system_features={
            **atom_graphs.system_features,
            "cell": torch.eye(3).unsqueeze(0),
        },
        node_targets={"forces": force_targets},
        edge_targets={},
        system_targets={"energy": energy_target, "stress": stress_target},
    )


class EuclideanNormModel(torch.nn.Module):

    def __init__(self, minimum=[-0.5, -2.0, -1.0]):
        super(EuclideanNormModel, self).__init__()
        self.minimum = torch.tensor(minimum)
        self.heads = torch.nn.ModuleDict(
            {
                "energy": torch.nn.Linear(3, 1),  # Dummy head}
                "forces": torch.nn.Linear(3, 3),  # Dummy head}
                "stress": torch.nn.Linear(3, 6),  # Dummy head}
            }
        )
        # unused dummy model
        self.model = gns.MoleculeGNS(
            latent_dim=9,
            num_message_passing_steps=2,
            num_mlp_layers=1,
            mlp_hidden_dim=16,
            rbf_transform=BesselBasis(6.0)
        )
        self.has_stress = True

    def forward(self, batch):
        positions = batch.positions
        sqnorm = (torch.norm(positions - self.minimum, dim=1)) ** 2
        energies = segment_ops.aggregate_nodes(sqnorm, batch.n_node, reduction="sum")
        neg_grad = 2 * (self.minimum - positions)  # analytical gradient
        stress = torch.zeros(
            energies.shape + (6,),
            dtype=positions.dtype,
            device=positions.device,
        )
        return energies, neg_grad, stress

    def predict(self, batch):
        energy, forces, stress = self.forward(batch)
        return {"energy": energy, "forces": forces, "stress": stress}

    @property
    def properties(self):
        return list(self.heads.keys())


@pytest.fixture
def batch():
    """Simulate a batch that's been processed by a GNS and so has _KEY vars."""
    graph = get_batch_from_ase_with_latents()
    return base.batch_graphs([graph, graph])


@pytest.fixture
def single_graph():
    return get_batch_from_ase_with_latents()


@pytest.fixture
def energy_head():
    return EnergyHead(
        latent_dim=9,
        num_mlp_layers=1,
        mlp_hidden_dim=16,
        predict_atom_avg=True,
        dropout=None,
        checkpoint=None,
        online_normalisation=True,
    )


@pytest.fixture
def force_head():
    return ForceHead(
        latent_dim=9,
        num_mlp_layers=1,
        mlp_hidden_dim=16,
        remove_mean=True,
        remove_torque_for_nonpbc_systems=True,
        loss_type="mae",
        dropout=None,
        checkpoint=None,
        output_size=3,
        online_normalisation=True,
    )


@pytest.fixture
def stress_head():

    return StressHead(
        latent_dim=9,
        num_mlp_layers=1,
        mlp_hidden_dim=16,
        node_aggregation="sum",
        dropout=None,
        checkpoint=None,
        online_normalisation=True,
    )


@pytest.fixture
def gns_model():
    """Instantiates the molecular model."""
    return gns.MoleculeGNS(
        num_node_in_features=118,
        num_node_out_features=3,
        num_edge_in_features=13,
        latent_dim=9,
        num_message_passing_steps=2,
        num_mlp_layers=1,
        mlp_hidden_dim=16,
        edge_feature_names=["feat"],
        node_feature_names=["feat"],
        rbf_transform=ExpNormalSmearing(num_rbf=10),
    )


@pytest.fixture
def conservative_regressor(gns_model, energy_head):
    return ConservativeForcefieldRegressor(
        heads={"energy": energy_head},
        model=gns_model,
        loss_weights={
            "energy": 1.0,
            "grad_forces": 1.0,
            "grad_stress": 1.0,
            "rotational_grad": 1.0,
        },
    )


@pytest.fixture
def graph_regressor(gns_model, energy_head, force_head, stress_head):
    return DirectForcefieldRegressor(
        heads={"energy": energy_head, "forces": force_head, "stress": stress_head},
        model=gns_model,
        loss_weights={
            "energy": 1.0,
            "forces": 1.0,
            "stress": 1.0,
        },
    )


@pytest.fixture
def euclidean_norm():
    minimum = [-0.5, -2.0, -1.0]
    return EuclideanNormModel(minimum)


@pytest.fixture
def euclidean_norm_invariant():
    minimum = [0.0, 0.0, 0.0]
    return EuclideanNormModel(minimum)
