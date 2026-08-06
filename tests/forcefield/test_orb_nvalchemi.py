"""Tests for OrbWrapper (nvalchemi BaseModelMixin integration)."""

import pytest
import torch

nvalchemi = pytest.importorskip("nvalchemi", reason="nvalchemi-toolkit not installed")

from nvalchemi.data import AtomicData, Batch  # noqa: E402
from nvalchemi.models.base import NeighborListFormat  # noqa: E402

from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter  # noqa: E402
from orb_models.forcefield.inference.orb_nvalchemi import OrbWrapper  # noqa: E402
from orb_models.forcefield.pretrained import (  # noqa: E402
    orb_v3_conservative_architecture,
    orb_v3_direct_architecture,
)

_CUTOFF = 6.0
_MAX_NUM_NEIGHBORS = 120
_LATENT_DIM = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_water(device: str = "cpu") -> AtomicData:
    """Single H2O molecule with a pre-computed full edge list (no PBC)."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    numbers = torch.tensor([8, 1, 1], dtype=torch.long, device=device)
    neighbor_list = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]],
        dtype=torch.long,
        device=device,
    )
    return AtomicData(
        positions=positions,
        atomic_numbers=numbers,
        neighbor_list=neighbor_list,
        charge=torch.tensor([[0.0]]),
        spin=torch.tensor([[1.0]]),
    )


def _make_pbc_water(device: str = "cpu") -> AtomicData:
    """H2O in a periodic cubic box with integer neighbor_list_shifts on edges."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    numbers = torch.tensor([8, 1, 1], dtype=torch.long, device=device)
    neighbor_list = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]],
        dtype=torch.long,
        device=device,
    )
    cell = (torch.eye(3, dtype=torch.float32, device=device) * 10.0).unsqueeze(0)
    neighbor_list_shifts = torch.zeros(6, 3, dtype=torch.float32, device=device)
    pbc = torch.tensor([[True, True, True]], device=device)
    return AtomicData(
        positions=positions,
        atomic_numbers=numbers,
        neighbor_list=neighbor_list,
        cell=cell,
        neighbor_list_shifts=neighbor_list_shifts,
        pbc=pbc,
        charge=torch.tensor([[0.0]]),
        spin=torch.tensor([[1.0]]),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter():
    return ForcefieldAtomsAdapter(radius=_CUTOFF, max_num_neighbors=_MAX_NUM_NEIGHBORS)


@pytest.fixture
def direct_wrapper(adapter):
    model = orb_v3_direct_architecture(latent_dim=_LATENT_DIM)
    return OrbWrapper(model, adapter)


@pytest.fixture
def conservative_wrapper(adapter):
    model = orb_v3_conservative_architecture(latent_dim=_LATENT_DIM)
    return OrbWrapper(model, adapter)


@pytest.fixture
def single_batch() -> Batch:
    return Batch.from_data_list([_make_water()])


@pytest.fixture
def multi_batch() -> Batch:
    return Batch.from_data_list([_make_water(), _make_water()])


@pytest.fixture
def pbc_batch() -> Batch:
    return Batch.from_data_list([_make_pbc_water()])


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_wraps_model(self, adapter):
        model = orb_v3_conservative_architecture(latent_dim=_LATENT_DIM)
        w = OrbWrapper(model, adapter)
        assert w.model is model

    def test_conservative_model_config(self, conservative_wrapper):
        cfg = conservative_wrapper.model_config
        assert "energy" in cfg.outputs
        assert "forces" in cfg.outputs
        assert "stress" in cfg.outputs

    def test_direct_model_config(self, direct_wrapper):
        cfg = direct_wrapper.model_config
        assert "energy" in cfg.outputs
        assert "forces" in cfg.outputs
        assert "stress" in cfg.outputs


class TestModelConfigCapabilities:
    def test_conservative_autograd(self, conservative_wrapper):
        cfg = conservative_wrapper.model_config
        assert cfg.autograd_outputs == frozenset({"forces", "stress"})
        assert "positions" in cfg.autograd_inputs

    def test_direct_no_autograd(self, direct_wrapper):
        assert not direct_wrapper.model_config.autograd_outputs

    def test_pbc_and_neighbor_config(self, conservative_wrapper):
        cfg = conservative_wrapper.model_config
        assert cfg.supports_pbc is True
        assert cfg.needs_pbc is False
        assert cfg.neighbor_config is not None
        assert cfg.neighbor_config.format == NeighborListFormat.COO
        assert cfg.neighbor_config.cutoff == pytest.approx(_CUTOFF)


class TestProperties:
    def test_cutoff(self, conservative_wrapper):
        assert conservative_wrapper.cutoff == pytest.approx(_CUTOFF)
        assert isinstance(conservative_wrapper.cutoff, float)

    def test_embedding_shapes(self, conservative_wrapper):
        shapes = conservative_wrapper.embedding_shapes
        assert shapes["node_embeddings"] == (_LATENT_DIM,)


class TestAdaptInput:
    def test_positions_present(self, conservative_wrapper, single_batch):
        result = conservative_wrapper.adapt_input(single_batch)
        assert "positions" in result.node_features
        assert result.node_features["positions"].shape == (3, 3)

    def test_atomic_numbers_present(self, conservative_wrapper, single_batch):
        result = conservative_wrapper.adapt_input(single_batch)
        assert "atomic_numbers" in result.node_features
        assert result.node_features["atomic_numbers"].shape == (3,)

    def test_edge_vectors_present(self, conservative_wrapper, single_batch):
        result = conservative_wrapper.adapt_input(single_batch)
        assert "vectors" in result.edge_features
        E = single_batch.neighbor_list.shape[0]
        assert result.edge_features["vectors"].shape == (E, 3)

    def test_atomic_data_promoted_to_batch(self, conservative_wrapper):
        data = _make_water()
        result = conservative_wrapper.adapt_input(data)
        assert result.n_node.shape[0] == 1
        assert result.n_node[0].item() == 3

    def test_no_pbc_identity_cell(self, conservative_wrapper, single_batch):
        result = conservative_wrapper.adapt_input(single_batch)
        B = single_batch.num_graphs
        assert result.system_features["cell"].shape == (B, 3, 3)
        expected = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
        assert torch.allclose(result.system_features["cell"], expected)

    def test_no_pbc_zero_shifts(self, conservative_wrapper, single_batch):
        result = conservative_wrapper.adapt_input(single_batch)
        assert result.edge_features["unit_shifts"].abs().max().item() == pytest.approx(0.0)

    def test_pbc_cell_passed_through(self, conservative_wrapper, pbc_batch):
        result = conservative_wrapper.adapt_input(pbc_batch)
        assert torch.allclose(result.system_features["cell"][0], torch.eye(3) * 10.0, atol=1e-5)

    def test_multi_batch_n_node(self, conservative_wrapper, multi_batch):
        result = conservative_wrapper.adapt_input(multi_batch)
        assert result.n_node.tolist() == [3, 3]

    def test_charge_and_spin_passed_through(self, conservative_wrapper):
        data = AtomicData(
            positions=torch.randn(3, 3),
            atomic_numbers=torch.tensor([6, 8, 1]),
            neighbor_list=torch.tensor([[0, 1], [1, 0]]),
            charge=torch.tensor([[2.0]]),
            spin=torch.tensor([[3.0]]),
        )
        result = conservative_wrapper.adapt_input(data)
        assert result.system_features["total_charge"].item() == pytest.approx(2.0)
        assert result.system_features["spin_multiplicity"].item() == pytest.approx(3.0)


class TestAdaptOutput:
    def test_energy_key_in_output(self, conservative_wrapper, single_batch):
        raw = {"energy": torch.randn(1)}
        out = conservative_wrapper.adapt_output(raw, single_batch)
        assert "energy" in out

    def test_energy_shape_1d_unsqueezed(self, conservative_wrapper, single_batch):
        raw = {"energy": torch.randn(1)}
        out = conservative_wrapper.adapt_output(raw, single_batch)
        assert out["energy"].shape == (1, 1)

    def test_energy_already_2d(self, conservative_wrapper, single_batch):
        raw = {"energy": torch.randn(1, 1)}
        out = conservative_wrapper.adapt_output(raw, single_batch)
        assert out["energy"].shape == (1, 1)

    def test_forces_passed_through(self, conservative_wrapper, single_batch):
        raw = {"energy": torch.randn(1), "forces": torch.randn(3, 3)}
        out = conservative_wrapper.adapt_output(raw, single_batch)
        assert "forces" in out
        assert out["forces"].shape == (3, 3)

    def test_stress_voigt_to_3x3(self, conservative_wrapper, single_batch):
        raw = {"energy": torch.randn(1), "stress": torch.randn(1, 6)}
        out = conservative_wrapper.adapt_output(raw, single_batch)
        assert "stress" in out
        assert out["stress"].shape == (1, 3, 3)

    def test_stress_voigt_symmetry(self, conservative_wrapper, single_batch):
        voigt = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        raw = {"energy": torch.randn(1), "stress": voigt}
        out = conservative_wrapper.adapt_output(raw, single_batch)
        s = out["stress"][0]
        assert torch.allclose(s, s.T)

    def test_missing_optional_outputs_absent(self, conservative_wrapper, single_batch):
        raw = {"energy": torch.randn(1)}
        out = conservative_wrapper.adapt_output(raw, single_batch)
        assert out.get("forces") is None
        assert out.get("stress") is None


class TestForwardDirect:
    def test_energy_shape_single(self, direct_wrapper, single_batch):
        out = direct_wrapper(single_batch)
        assert out["energy"].shape == (1, 1)

    def test_energy_shape_multi(self, direct_wrapper, multi_batch):
        out = direct_wrapper(multi_batch)
        assert out["energy"].shape == (2, 1)

    def test_forces_shape(self, direct_wrapper, single_batch):
        out = direct_wrapper(single_batch)
        assert out["forces"].shape == (3, 3)

    def test_forces_shape_multi(self, direct_wrapper, multi_batch):
        out = direct_wrapper(multi_batch)
        assert out["forces"].shape == (6, 3)

    def test_stress_shape(self, direct_wrapper, single_batch):
        out = direct_wrapper(single_batch)
        assert out["stress"].shape == (1, 3, 3)

    def test_stress_shape_multi(self, direct_wrapper, multi_batch):
        out = direct_wrapper(multi_batch)
        assert out["stress"].shape == (2, 3, 3)

    def test_no_forces_when_disabled(self, direct_wrapper, single_batch):
        direct_wrapper.model_config.active_outputs = {"energy"}
        out = direct_wrapper(single_batch)
        assert out.get("forces") is None

    def test_no_stress_when_disabled(self, direct_wrapper, single_batch):
        direct_wrapper.model_config.active_outputs = {"energy", "forces"}
        out = direct_wrapper(single_batch)
        assert out.get("stress") is None


class TestForwardConservative:
    def test_energy_shape_single(self, conservative_wrapper, single_batch):
        out = conservative_wrapper(single_batch)
        assert out["energy"].shape == (1, 1)

    def test_energy_shape_multi(self, conservative_wrapper, multi_batch):
        out = conservative_wrapper(multi_batch)
        assert out["energy"].shape == (2, 1)

    def test_forces_shape(self, conservative_wrapper, single_batch):
        out = conservative_wrapper(single_batch)
        assert out["forces"].shape == (3, 3)

    def test_forces_shape_multi(self, conservative_wrapper, multi_batch):
        out = conservative_wrapper(multi_batch)
        assert out["forces"].shape == (6, 3)

    def test_stress_shape(self, conservative_wrapper, single_batch):
        out = conservative_wrapper(single_batch)
        assert out["stress"].shape == (1, 3, 3)

    def test_no_forces_when_disabled(self, conservative_wrapper, single_batch):
        conservative_wrapper.model_config.active_outputs = {"energy"}
        out = conservative_wrapper(single_batch)
        assert out.get("forces") is None

    def test_no_stress_when_disabled(self, conservative_wrapper, single_batch):
        conservative_wrapper.model_config.active_outputs = {"energy", "forces"}
        out = conservative_wrapper(single_batch)
        assert out.get("stress") is None


class TestPipeline:
    def test_pipeline_returns_energy_only(self, conservative_wrapper, single_batch):
        conservative_wrapper.model_config.active_outputs = {"energy"}
        out = conservative_wrapper(single_batch)
        assert "energy" in out
        assert "forces" not in out
        assert "stress" not in out

    def test_pipeline_does_not_mutate_active_outputs(self, conservative_wrapper, single_batch):
        conservative_wrapper.model_config.active_outputs = {"energy"}
        conservative_wrapper(single_batch)
        assert conservative_wrapper.model_config.active_outputs == {"energy"}

    def test_direct_derivative_keys_conservative(self, conservative_wrapper):
        assert conservative_wrapper.direct_derivative_keys() == set()

    def test_direct_derivative_keys_direct(self, direct_wrapper):
        assert direct_wrapper.direct_derivative_keys() == set()


class TestComputeEmbeddings:
    def test_node_embeddings_shape(self, conservative_wrapper, single_batch):
        result = conservative_wrapper.compute_embeddings(single_batch)
        assert result.node_embeddings.shape == (3, _LATENT_DIM)

    def test_node_embeddings_shape_multi(self, conservative_wrapper, multi_batch):
        result = conservative_wrapper.compute_embeddings(multi_batch)
        assert result.node_embeddings.shape == (6, _LATENT_DIM)

    def test_atomic_data_input(self, conservative_wrapper):
        data = _make_water()
        result = conservative_wrapper.compute_embeddings(data)
        assert result.node_embeddings.shape == (3, _LATENT_DIM)

    def test_does_not_mutate_model_config(self, conservative_wrapper, single_batch):
        conservative_wrapper.model_config.active_outputs = {"energy", "forces", "stress"}
        conservative_wrapper.compute_embeddings(single_batch)
        assert "forces" in conservative_wrapper.model_config.active_outputs
        assert "stress" in conservative_wrapper.model_config.active_outputs
