import pytest
import numpy as np
import torch

from ase import Atoms
from ase.data import covalent_radii

from orb_models.forcefield.pair_repulsion import ZBLBasis
from orb_models.forcefield.forcefield_heads import EnergyHead, ForceHead
from orb_models.forcefield.forcefield_utils import (
    compute_gradient_forces_and_stress,
)
from orb_models.forcefield.conservative_regressor import ConservativeForcefieldRegressor
from orb_models.forcefield.graph_regressor import GraphHead, GraphRegressor
from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs, SystemConfig
from orb_models.forcefield import base
from orb_models.forcefield.property_definitions import PropertyDefinition


def test_pair_conservative(gns_model):
    """Test pair repulsion is decreasing and zero beyond covalent radii."""
    head = EnergyHead(
        latent_dim=9,
        num_mlp_layers=1,
        mlp_hidden_dim=16,
        predict_atom_avg=True,
        dropout=None,
        checkpoint=None,
        online_normalisation=True,
    )
    for parameter in head.parameters():
        parameter.data.fill_(0.0)
        parameter.data.fill_(0.0)

    forces_head = ForceHead(
        latent_dim=9,
        num_mlp_layers=1,
        mlp_hidden_dim=16,
    )
    stress_head = GraphHead(
        target=PropertyDefinition(
            "stress", dim=6, domain="real", row_to_property_fn=lambda x: x
        ),
        latent_dim=9,
        num_mlp_layers=1,
        mlp_hidden_dim=16,
    )

    regressor = ConservativeForcefieldRegressor(
        heads={"energy": head, "forces": forces_head, "stress": stress_head},
        model=gns_model,
        pair_repulsion=True,
        ensure_grad_loss_weights=False,
    )

    numbers = [1, 8, 13, 88]
    for i in numbers:
        for j in numbers:
            r_max = covalent_radii[i] + covalent_radii[j]
            distances = np.linspace(0.1, r_max, 10)
            atoms_list = [
                Atoms(
                    numbers=[i, j],
                    positions=[[0, 0, 0], [d, 0, 0]],
                    pbc=False,
                )
                for d in distances
            ]

            system_config = SystemConfig(
                radius=6.0,
                max_num_neighbors=10,
            )
            graphs = [
                ase_atoms_to_atom_graphs(a, system_config=system_config)
                for a in atoms_list
            ]
            batch = base.batch_graphs(graphs)
            out = regressor.predict(batch)

            # energy head has predict_atom_avg=True, so pair energy is normalized by
            # number of nodes in graph
            pair_energy = regressor.pair_repulsion_fn(batch)["energy"]
            np.testing.assert_allclose(
                out["energy"].detach().numpy(),
                pair_energy.detach().numpy(),
                atol=1e-5,
            )

            assert out["energy"][-1] == 0.0
            assert torch.all(torch.diff(out["energy"]) <= 0)  # decreasing


@pytest.mark.parametrize("pbc", [True, False])
def test_pair_grads(pbc):
    """Test analytical gradients of pair repulsion energy match autograd."""
    system_config = SystemConfig(radius=3.0, max_num_neighbors=120)
    pair_repulsion = ZBLBasis(p=6, node_aggregation="sum", compute_gradients=True)

    positions = 0.4 * np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    cell = np.diag([1, 1, 1.1])
    atoms = Atoms("H2O", positions=positions, cell=cell, pbc=pbc)

    graphs = [
        ase_atoms_to_atom_graphs(
            atoms,
            system_config=system_config,
            output_dtype=torch.float64,
        )
        for a in [atoms, atoms]
    ]
    batch = base.batch_graphs(graphs)
    vectors, displacement, generator = batch.compute_differentiable_edge_vectors()
    batch.edge_features["vectors"] = vectors
    out = pair_repulsion(batch)

    forces, stress, _ = compute_gradient_forces_and_stress(
        energy=out["energy"],
        positions=batch.node_features["positions"],
        displacement=displacement,
        cell=batch.system_features["cell"],
        training=False,
        compute_stress=True,
        generator=generator,
    )

    np.testing.assert_allclose(
        out["forces"].detach().numpy(),
        forces.detach().numpy(),
    )
    np.testing.assert_allclose(
        out["stress"].detach().numpy(),
        stress.detach().numpy(),
    )


def test_pair_direct(gns_model):
    """Test pair repulsion is decreasing and zero beyond covalent radii."""
    energy_head = EnergyHead(
        latent_dim=9,
        num_mlp_layers=1,
        mlp_hidden_dim=16,
        predict_atom_avg=True,
        dropout=None,
        checkpoint=None,
        online_normalisation=True,
    )
    forces_head = ForceHead(
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
    stress_head = GraphHead(
        latent_dim=9,
        num_mlp_layers=1,
        mlp_hidden_dim=16,
        target=PropertyDefinition(
            "stress", dim=6, domain="real", row_to_property_fn=lambda x: x
        ),
        node_aggregation="sum",
        dropout=None,
        checkpoint=None,
        online_normalisation=True,
    )
    for head in [energy_head, forces_head, stress_head]:
        for parameter in head.parameters():
            parameter.data.fill_(0.0)
            parameter.data.fill_(0.0)

    regressor = GraphRegressor(
        heads={
            "energy": energy_head,
            "forces": forces_head,
            "stress": stress_head,
        },
        pair_repulsion=True,
        model=gns_model,
        loss_weights={
            "energy": 1.0,
            "forces": 1.0,
            "stress": 1.0,
        },
    )

    numbers = [1, 8, 13, 88]
    for i in numbers:
        for j in numbers:
            r_max = covalent_radii[i] + covalent_radii[j]
            distances = np.linspace(0.1, r_max, 10)
            atoms_list = [
                Atoms(
                    numbers=[i, j],
                    positions=[[0, 0, 0], [d, 0, 0]],
                    pbc=False,
                )
                for d in distances
            ]

            system_config = SystemConfig(
                radius=6.0,
                max_num_neighbors=10,
            )
            graphs = [
                ase_atoms_to_atom_graphs(a, system_config=system_config)
                for a in atoms_list
            ]
            batch = base.batch_graphs(graphs)
            forward = regressor(batch)
            predict = regressor.predict(batch)

            pair_energy = regressor.pair_repulsion_fn(batch)["energy"]
            pair_forces = regressor.pair_repulsion_fn(batch)["forces"]
            pair_stress = regressor.pair_repulsion_fn(batch)["stress"]

            np.testing.assert_allclose(
                forward["energy"].detach().numpy(),
                energy_head.normalizer(pair_energy / batch.n_node, online=False)
                .unsqueeze(1)
                .detach()
                .numpy(),
                atol=1e-5,
            )
            np.testing.assert_allclose(
                predict["forces"].detach().numpy(),
                pair_forces.detach().numpy(),
                atol=1e-5,
            )
            np.testing.assert_allclose(
                predict["stress"].detach().numpy(),
                pair_stress.detach().numpy(),
                atol=1e-5,
            )

            assert forward["forces"].shape == predict["forces"].shape
            assert forward["stress"].shape == predict["stress"].shape
