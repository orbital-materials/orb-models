"""Test tojax compatibility for all pretrained orb models."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from ase.build import bulk
from tojax import tojax
from torch.jit._state import disable as torch_jit_disable
from torch.jit._state import enable as torch_jit_enable

import orb_models.common.models.angular as _angular_mod
from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS


@pytest.fixture(autouse=True, scope="module")
def _ensure_plain_spherical_harmonics():
    """Replace JIT-compiled _spherical_harmonics with a plain Python function.

    If another test module imported angular.py with JIT enabled,
    _spherical_harmonics is a ScriptFunction that rejects TensorWrappers.
    We re-evaluate the module source into a scratch namespace with JIT
    disabled and patch only the function — no importlib.reload, so existing
    classes (and their super() chains) are untouched.
    """
    original_fn = _angular_mod._spherical_harmonics
    if isinstance(original_fn, torch.jit.ScriptFunction):
        torch_jit_disable()
        scratch = dict(vars(_angular_mod))
        with open(_angular_mod.__file__) as f:
            exec(compile(f.read(), _angular_mod.__file__, "exec"), scratch)
        _angular_mod._spherical_harmonics = scratch["_spherical_harmonics"]
        torch_jit_enable()
    yield
    _angular_mod._spherical_harmonics = original_fn


DEPRECATED_MODELS = {
    "orb-v1",
    "orb-d3-v1",
    "orb-d3-sm-v1",
    "orb-d3-xs-v1",
    "orb-v1-mptraj-only",
}

ACTIVE_MODELS = sorted(name for name in ORB_PRETRAINED_MODELS if name not in DEPRECATED_MODELS)
CONSERVATIVE_MODELS = {name for name in ACTIVE_MODELS if "conservative" in name}


def _make_atoms():

    atoms = bulk("Cu", "fcc", a=3.6)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    return atoms


# NOTE: This fails for conservative models because tojax does not support torch.autograd.grad
@pytest.mark.parametrize("model_name", ACTIVE_MODELS)
def test_tojax_outputs_match_pytorch(model_name):
    load_fn = ORB_PRETRAINED_MODELS[model_name]
    model, adapter = load_fn(device="cpu", compile=False)
    model.eval()

    atoms = _make_atoms()
    torch_batch = adapter.from_ase_atoms(atoms, device="cpu")
    jax_batch = adapter.from_ase_atoms(atoms, device="cpu")

    jax_model = tojax(model)

    with torch.enable_grad():
        torch_out = model(torch_batch)

    jax_out = jax_model(jax_batch)

    # Only compare model predictions, not internal intermediates like
    # node_features / edge_features where float32 rounding differences
    # accumulate across message-passing layers and many edges.
    INTERNAL_KEYS = {"node_features", "edge_features"}
    prediction_keys = set(torch_out) - INTERNAL_KEYS
    missing = prediction_keys - set(jax_out)
    assert not missing, f"Keys missing from JAX output: {missing}"

    for key in prediction_keys:
        torch_val = torch_out[key].detach().float().cpu().numpy()
        jax_val = np.asarray(jax_out[key], dtype=np.float32)
        np.testing.assert_allclose(
            jax_val,
            torch_val,
            atol=1e-5,
            rtol=1e-4,
            err_msg=f"Output mismatch for '{model_name}' key '{key}'",
        )


def _from_kups_atoms(adapter: ForcefieldAtomsAdapter, data: dict[str, torch.Tensor]) -> AtomGraphs:
    """Build AtomGraphs from a kups AtomGraphInput dict."""
    senders = data["edge_index"][0]
    receivers = data["edge_index"][1]
    n_systems = data["cell"].shape[0]
    batch = data["batch"]
    src_batch = batch[senders]

    n_node = torch.zeros(n_systems, dtype=torch.int64).scatter_add_(
        0, batch, torch.ones_like(batch)
    )
    nedges = torch.zeros(n_systems, dtype=torch.int64).scatter_add_(
        0, src_batch, torch.ones_like(src_batch)
    )

    atomic_numbers = data["atomic_numbers"]
    atomic_numbers_embedding = torch.nn.functional.one_hot(atomic_numbers.long(), 118).float()

    system_features: dict[str, torch.Tensor] = {
        "cell": data["cell"],
        "pbc": data["pbc"],
    }
    if "charge" in data:
        system_features["total_charge"] = data["charge"]
    if "spin" in data:
        system_features["spin_multiplicity"] = data["spin"]

    cells_per_edge = data["cell"][src_batch]
    shifts = torch.bmm(data["cell_offsets"].unsqueeze(1), cells_per_edge).squeeze(1)
    vectors = data["pos"][receivers] - data["pos"][senders] + shifts

    assert adapter.radius is not None, "Adapter radius must be set"
    return AtomGraphs(
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=nedges,
        node_features={
            "positions": data["pos"],
            "atomic_numbers": atomic_numbers,
            "atomic_numbers_embedding": atomic_numbers_embedding,
            "atom_identity": torch.arange(data["pos"].shape[0], dtype=torch.int64),
        },
        system_features=system_features,
        edge_features={"unit_shifts": data["cell_offsets"], "vectors": vectors},
        node_targets={},
        edge_targets={},
        system_targets={},
        system_id=None,
        fix_atoms=None,
        tags=None,
        radius=adapter.radius,
        max_num_neighbors=nedges,
        half_supercell=False,
    )


def _to_kups_atoms(atomgraph: AtomGraphs) -> dict[str, torch.Tensor]:
    """Convert to the flat AtomGraphInput dict used by kups/tojax."""
    n_atoms = int(atomgraph.n_node.sum())
    n_systems = atomgraph.n_node.shape[0]
    device = atomgraph.n_node.device
    return {
        "pos": atomgraph.node_features["positions"],
        "atomic_numbers": atomgraph.node_features["atomic_numbers"],
        "cell": atomgraph.system_features["cell"],
        "pbc": atomgraph.system_features["pbc"],
        "edge_index": torch.stack([atomgraph.senders, atomgraph.receivers]),
        "cell_offsets": atomgraph.edge_features["unit_shifts"],
        "batch": torch.arange(n_systems, device=device).repeat_interleave(
            atomgraph.n_node, output_size=n_atoms
        ),
        "charge": atomgraph.system_features.get(
            "total_charge", torch.zeros(n_systems, dtype=torch.long, device=device)
        ).view(-1),
        "spin": atomgraph.system_features.get(
            "spin_multiplicity", torch.zeros(n_systems, dtype=torch.long, device=device)
        ).view(-1),
    }


def _make_predict_fn(adapter, model):
    """Build a tojax-traceable predict function: kups AtomGraphInput -> {energy, forces, stress}."""

    def predict_fn(data):
        graph = _from_kups_atoms(adapter, data)
        result = model.predict(graph, split=False)
        out = {"energy": result["energy"]}
        if "forces" in result:
            out["forces"] = result["forces"]
        if "stress" in result:
            out["stress"] = result["stress"]
        return out

    return predict_fn


@pytest.mark.parametrize("model_name", ACTIVE_MODELS)
def test_kups_compatibility(model_name):
    """Test energy, forces, and stress using the kups AtomGraphInput format.

    Since tojax cannot translate torch.autograd.grad, this test follows the
    pattern from the tojax export example (export_orb.py) where model + graph
    construction are packaged into a single function that accepts a flat
    AtomGraphInput dict — the same format kups uses.

    Unlike test_tojax_outputs_match_pytorch which passes the model directly
    to tojax(model), this test tojax-traces a function that builds an
    AtomGraphs from raw tensors and calls model.predict(). For conservative
    models, forces and stress are computed via jax.grad of the energy
    (replacing torch.autograd.grad with JAX's own autodiff).
    """

    load_fn = ORB_PRETRAINED_MODELS[model_name]
    model, adapter = load_fn(device="cpu", compile=False)
    model.eval()
    is_conservative = model_name in CONSERVATIVE_MODELS

    atoms = _make_atoms()
    batch = adapter.from_ase_atoms(atoms, device="cpu")
    data = _to_kups_atoms(batch)

    # PyTorch reference
    ref_batch = adapter.from_ase_atoms(atoms, device="cpu")
    with torch.enable_grad():
        torch_out = model.predict(ref_batch)

    jax_data = tojax(data)

    if is_conservative:
        # Energy + forces/stress via jax.grad (torch.autograd.grad can't be translated)
        predict_fn = _make_predict_fn(adapter, model)
        jax_energy_fn = tojax(lambda data: predict_fn(data)["energy"])
        batch_indices = jax_data["batch"]

        def _energy_with_strain(pos, strain):
            sym_strain = 0.5 * (strain + jnp.swapaxes(strain, -1, -2))
            deformed_cell = jax_data["cell"] + jnp.einsum(
                "bij,bjk->bik", jax_data["cell"], sym_strain
            )
            deformed_pos = pos + jnp.einsum("ni,nij->nj", pos, sym_strain[batch_indices])
            return jax_energy_fn({**jax_data, "pos": deformed_pos, "cell": deformed_cell}).sum()

        energy_and_grads = jax.value_and_grad(_energy_with_strain, argnums=(0, 1))
        jax_energy, (neg_forces, virial) = energy_and_grads(
            jax_data["pos"], jnp.zeros_like(jax_data["cell"])
        )

        np.testing.assert_allclose(
            np.asarray(jax_energy),
            torch_out[model.energy_name].detach().float().numpy(),
            atol=1e-5,
            rtol=1e-4,
            err_msg=f"energy mismatch for '{model_name}'",
        )
        np.testing.assert_allclose(
            np.asarray(-neg_forces, dtype=np.float32),
            torch_out[model.grad_forces_name].detach().float().numpy(),
            atol=1e-4,
            rtol=1e-4,
            err_msg=f"force mismatch for '{model_name}'",
        )
        if model.has_stress:
            volume = jnp.abs(jnp.linalg.det(jax_data["cell"]))
            jax_stress_3x3 = np.asarray(virial / volume[:, None, None])
            jax_stress = np.stack(
                [
                    jax_stress_3x3[..., 0, 0],
                    jax_stress_3x3[..., 1, 1],
                    jax_stress_3x3[..., 2, 2],
                    (jax_stress_3x3[..., 1, 2] + jax_stress_3x3[..., 2, 1]) / 2,
                    (jax_stress_3x3[..., 0, 2] + jax_stress_3x3[..., 2, 0]) / 2,
                    (jax_stress_3x3[..., 0, 1] + jax_stress_3x3[..., 1, 0]) / 2,
                ],
                axis=-1,
            ).astype(np.float32)
            np.testing.assert_allclose(
                jax_stress.reshape(torch_out[model.grad_stress_name].shape),
                torch_out[model.grad_stress_name].detach().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
                err_msg=f"stress mismatch for '{model_name}'",
            )
    else:
        # Direct models: energy, forces, stress from tojax'd predict
        jax_predict_fn = tojax(_make_predict_fn(adapter, model))
        jax_out = jax_predict_fn(jax_data)

        np.testing.assert_allclose(
            np.asarray(jax_out["energy"]),
            torch_out["energy"].detach().float().numpy(),
            atol=1e-5,
            rtol=1e-4,
            err_msg=f"energy mismatch for '{model_name}'",
        )
        if "forces" in jax_out:
            np.testing.assert_allclose(
                np.asarray(jax_out["forces"], dtype=np.float32),
                torch_out["forces"].detach().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
                err_msg=f"force mismatch for '{model_name}'",
            )
        if "stress" in jax_out:
            np.testing.assert_allclose(
                np.asarray(jax_out["stress"], dtype=np.float32),
                torch_out["stress"].detach().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
                err_msg=f"stress mismatch for '{model_name}'",
            )
