import ase.db
import numpy
import pytest
import torch
from torch.utils.data import DataLoader

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.dataset.ase_sqlite_dataset import AseSqliteDataset
from orb_models.common.dataset.augmentations import geometric_augmentations
from orb_models.common.dataset.property_definitions import (
    PropertyConfig,
    PropertyDefinition,
    energy_row_fn,
    forces_row_fn,
    get_property_from_row,
)
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter

forcefield_adapter = ForcefieldAtomsAdapter(
    radius=6.0,
    max_num_neighbors=20,
)

adapter_maxnumneigh120 = ForcefieldAtomsAdapter(
    radius=6.0,
    max_num_neighbors=120,
)

energy = PropertyDefinition(
    name="energy",
    dim=1,
    domain="real",
    row_to_property_fn=lambda row, dataset: get_property_from_row("data.energy", row),
)
dummy_forces = PropertyDefinition(
    name="dummy_forces",
    dim=3,
    domain="real",
    row_to_property_fn=lambda row, dataset: torch.from_numpy(row.toatoms().positions),
)

target_cfg = PropertyConfig()
target_cfg.graph_properties = {"energy": energy}
target_cfg.node_properties = {"dummy_forces": dummy_forces}


def test_read_dataset(fixtures_path):
    dataset = AseSqliteDataset(
        name="test_dataset",
        path=str(fixtures_path / "test_dataset.db"),
        atoms_adapter=forcefield_adapter,
        target_config=PropertyConfig(),
    )
    assert len(dataset) == 6


def test_dataset_db_files_must_exist(fixtures_path):
    with pytest.raises(ValueError):
        _ = AseSqliteDataset(
            name="test_dataset",
            path=str(fixtures_path / "not_present_on_disk.db"),
            atoms_adapter=forcefield_adapter,
            target_config=PropertyConfig(),
        )


def test_read_dataset_with_targets(fixtures_path):
    dataset = AseSqliteDataset(
        name="test_dataset",
        path=str(fixtures_path / "test_with_target.db"),
        atoms_adapter=forcefield_adapter,
        target_config=target_cfg,
    )

    assert len(dataset) == 6

    item = dataset[0]
    assert item.system_targets is not None and "energy" in item.system_targets
    assert item.system_features is not None and "energy" not in item.system_features


def test_read_dataset_with_features(fixtures_path):
    forcefield_adapter.extra_features = {"graph": ["test-graph-fixture"]}
    dataset = AseSqliteDataset(
        name="test_dataset",
        path=str(fixtures_path / "test_with_target.db"),
        atoms_adapter=forcefield_adapter,
    )

    assert len(dataset) == 6

    item = dataset[0]
    assert item.system_features is not None and "test-graph-fixture" in item.system_features
    assert item.system_targets == {}


def test_default_energy_and_forces_funcs(fixtures_path):
    db = ase.db.connect(str(fixtures_path / "test_with_target.db"))
    row = db.get(1)

    row.energy = row.data.energy
    row.forces = numpy.random.randn(row.natoms, 3)
    energy = energy_row_fn(row, "not-a-dataset")
    forces = forces_row_fn(row, "not-a-dataset")

    assert energy == row.energy
    assert numpy.allclose(forces.numpy(), row.forces)


def test_read_dataset_with_rotation_augmentation(fixtures_path):
    # Check that the augmentation works, but don't add noise,
    # because if we add noise the norms of the edge difference vectors
    # will be different, so we can't test that the rotations preserve the norms.
    featurization_config = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=20)
    dataset = AseSqliteDataset(
        name="test_dataset",
        path=str(fixtures_path / "test_with_target.db"),
        atoms_adapter=featurization_config,
        target_config=target_cfg,
    )
    dataset_aug = AseSqliteDataset(
        name="test_dataset",
        path=str(fixtures_path / "test_with_target.db"),
        atoms_adapter=featurization_config,
        target_config=target_cfg,
        augmentations=[geometric_augmentations.rotate_randomly],  # type: ignore
    )

    item_aug = dataset_aug[0]
    item = dataset[0]

    # vectors differ
    assert (item_aug.edge_features["vectors"] != item.edge_features["vectors"]).all()
    # norm are the same
    assert torch.allclose(
        item_aug.edge_features["vectors"].norm(dim=1),
        item.edge_features["vectors"].norm(dim=1),
    )

    # check target features are also rotated
    assert item_aug.node_targets is not None
    assert item.node_targets is not None
    assert (item_aug.node_targets["dummy_forces"] != item.node_targets["dummy_forces"]).all()
    assert torch.allclose(
        item_aug.node_targets["dummy_forces"].norm(dim=1),
        item.node_targets["dummy_forces"].norm(dim=1),
    )


def test_dataset_can_batch_with_dataloader(fixtures_path):
    dataset = AseSqliteDataset(
        name="test_dataset",
        path=str(fixtures_path / "test_dataset.db"),
        atoms_adapter=forcefield_adapter,
        target_config=PropertyConfig(),
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=AtomGraphs.batch)
    for b in loader:
        assert b.n_node.shape == (2,)


def test_batch_splitting(fixtures_path):
    dataset = AseSqliteDataset(
        name="test_dataset",
        path=str(fixtures_path / "test_dataset.db"),
        atoms_adapter=forcefield_adapter,
        target_config=PropertyConfig(),
    )
    batch = AtomGraphs.batch([dataset[0], dataset[1]])
    individual = batch.split()
    batch_ = AtomGraphs.batch(individual)
    assert isinstance(batch_, AtomGraphs)
    assert batch_.equals(batch)


def test_finetune_dataset_can_batch_with_dataloader(fixtures_path):
    dataset = AseSqliteDataset(
        name="test_dataset",
        path=str(fixtures_path / "test_with_target.db"),
        atoms_adapter=forcefield_adapter,
        target_config=target_cfg,
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=AtomGraphs.batch)
    for b in loader:
        assert isinstance(b.system_targets["energy"], torch.Tensor)
        assert b.system_targets["energy"].shape == (2,)
        assert b.n_node.shape == (2,)
