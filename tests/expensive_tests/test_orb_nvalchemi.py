import numpy as np
import pytest
import torch
from ase.build import bulk, molecule

nvalchemi = pytest.importorskip("nvalchemi", reason="nvalchemi-toolkit not installed")

from nvalchemi.data import AtomicData, Batch  # noqa: E402
from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper  # noqa: E402

from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter  # noqa: E402
from orb_models.forcefield.inference.calculator import ORBCalculator  # noqa: E402
from orb_models.forcefield.inference.orb_nvalchemi import OrbWrapper  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CONSERVATIVE_MODELS = [
    "orbmol-v2",
    "orbmol-v1-conservative",
    "orb-v3-conservative-inf-omat",
]
_DIRECT_MODELS = [
    "orb-v3-direct-omol",
    "orbmol-v1-direct",
    "orb-v3-direct-inf-omat",
]


@pytest.fixture(scope="module", params=_CONSERVATIVE_MODELS)
def conservative_wrapper(request) -> OrbWrapper:
    return OrbWrapper.from_pretrained(request.param, compile=True)


@pytest.fixture(scope="module", params=_DIRECT_MODELS)
def direct_wrapper(request) -> OrbWrapper:
    return OrbWrapper.from_pretrained(request.param, compile=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ase_to_nvalchemi_batch(atoms, adapter: ForcefieldAtomsAdapter) -> Batch:
    """Build a nvalchemi Batch romrom ASE Atoms."""
    graph = adapter.from_ase_atoms(atoms)
    neighbor_list = torch.stack([graph.senders, graph.receivers], dim=-1)

    kwargs: dict = {
        "positions": graph.node_features["positions"],
        "atomic_numbers": graph.node_features["atomic_numbers"],
        "neighbor_list": neighbor_list,
    }

    cell = graph.system_features.get("cell")
    if cell is not None:
        kwargs["cell"] = cell
    unit_shifts = graph.edge_features.get("unit_shifts")
    if unit_shifts is not None:
        kwargs["neighbor_list_shifts"] = unit_shifts
    pbc = graph.system_features.get("pbc")
    if pbc is not None:
        kwargs["pbc"] = pbc

    charge = graph.system_features.get("total_charge")
    if charge is not None:
        kwargs["charge"] = charge.reshape(1, 1)
    spin = graph.system_features.get("spin_multiplicity")
    if spin is not None:
        kwargs["spin"] = spin.reshape(1, 1)

    return Batch.from_data_list([AtomicData(**kwargs)])


def _voigt_6_from_3x3(s: np.ndarray) -> np.ndarray:
    """Extract Voigt-6 [xx, yy, zz, yz, xz, xy] from a 3x3 symmetric tensor."""
    return np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])


def _nacl_bulk():
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    # Strain to produce non-trivial forces
    atoms.positions[0] += [0.1, -0.05, 0.08]
    return atoms


def _h2o():
    atoms = molecule("H2O")
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    return atoms


def _compare_standalone_predictions(wrapper: OrbWrapper, atoms, *, atol: float = 1e-5):
    """Run both OrbWrapper and ORBCalculator on the same atoms and assert matching predictions."""
    adapter = wrapper.atoms_adapter
    calc = ORBCalculator(wrapper.model, adapter, device="cpu")

    calc.calculate(atoms)
    ref_energy = calc.results["energy"]
    ref_forces = calc.results["forces"]
    ref_stress = calc.results.get("stress")

    batch = _ase_to_nvalchemi_batch(atoms, adapter)
    out = wrapper(batch)

    energy = out["energy"].detach().cpu().numpy().squeeze()
    np.testing.assert_allclose(energy, ref_energy, atol=atol, err_msg="energy mismatch")

    forces = out["forces"].detach().cpu().numpy()
    np.testing.assert_allclose(forces, ref_forces, atol=atol, err_msg="forces mismatch")

    if ref_stress is not None:
        stress_3x3 = out["stress"].detach().cpu().numpy().squeeze()
        stress_voigt = _voigt_6_from_3x3(stress_3x3)
        np.testing.assert_allclose(stress_voigt, ref_stress, atol=atol, err_msg="stress mismatch")


def _compare_pipeline_predictions(wrapper: OrbWrapper, atoms, *, atol: float = 1e-5):
    """Run standalone vs pipeline autograd and assert matching forces/stress."""
    adapter = wrapper.atoms_adapter

    ref = wrapper(_ase_to_nvalchemi_batch(atoms, adapter))
    ref_forces = ref["forces"].detach().cpu().numpy()

    pipe = PipelineModelWrapper(groups=[PipelineGroup(steps=[wrapper], use_autograd=True)])
    pipe.eval()
    out = pipe(_ase_to_nvalchemi_batch(atoms, adapter))

    np.testing.assert_allclose(
        out["forces"].detach().cpu().numpy(), ref_forces, atol=atol, err_msg="pipeline forces"
    )

    has_pbc = hasattr(atoms, "pbc") and np.any(atoms.pbc)
    ref_stress = ref.get("stress")
    if ref_stress is not None and has_pbc:
        ref_voigt = _voigt_6_from_3x3(ref_stress.detach().cpu().numpy().squeeze())
        out_voigt = _voigt_6_from_3x3(out["stress"].detach().cpu().numpy().squeeze())
        np.testing.assert_allclose(out_voigt, ref_voigt, atol=atol, err_msg="pipeline stress")


# ---------------------------------------------------------------------------
# Conservative models
# ---------------------------------------------------------------------------


class TestConservativeWrapper:
    def test_standalone_nacl_bulk(self, conservative_wrapper):
        _compare_standalone_predictions(conservative_wrapper, _nacl_bulk())

    def test_standalone_h2o(self, conservative_wrapper):
        _compare_standalone_predictions(conservative_wrapper, _h2o())

    def test_pipeline_autograd_nacl_bulk(self, conservative_wrapper):
        _compare_pipeline_predictions(conservative_wrapper, _nacl_bulk())

    def test_pipeline_autograd_h2o(self, conservative_wrapper):
        _compare_pipeline_predictions(conservative_wrapper, _h2o())

    def test_compile_with_training_raises(self):
        """compile (True or the default) + inference=False on a conservative model raises."""
        with pytest.raises(AssertionError, match="conservative model in training mode"):
            OrbWrapper.from_pretrained(
                "orb-v3-conservative-inf-omat", inference=False, compile=True
            )


# ---------------------------------------------------------------------------
# Direct models
# ---------------------------------------------------------------------------


class TestDirectWrapper:
    def test_standalone_nacl_bulk(self, direct_wrapper):
        _compare_standalone_predictions(direct_wrapper, _nacl_bulk())

    def test_standalone_h2o(self, direct_wrapper):
        _compare_standalone_predictions(direct_wrapper, _h2o())
