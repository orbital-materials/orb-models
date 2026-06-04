from copy import deepcopy

import pytest
import torch
from ase import Atom, Atoms

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.models.conservative_regressor import ConservativeForcefieldRegressor


@pytest.mark.parametrize("graph_name", ["batch", "single_graph"])
def test_regressor_forward(request, conservative_regressor, graph_name):
    graph = request.getfixturevalue(graph_name)
    out = conservative_regressor(graph)
    assert "energy" in out
    assert "forces" in out
    assert "stress" in out


def test_regressor_loss(conservative_regressor, batch):
    out = conservative_regressor.loss(batch)
    out.loss.backward()

    assert any("energy" in k for k in out.log)
    assert any("forces" in k for k in out.log)
    assert any("stress" in k for k in out.log)
    assert any("rotational_grad" in k for k in out.log)


def test_regressor_head_config_raises_error(gns_model, energy_head):
    with pytest.raises(ValueError, match="Loss weights for unknown targets"):
        ConservativeForcefieldRegressor(
            heads={"energy": energy_head},
            model=gns_model,
            loss_weights={
                "energy": 1.0,
                "forces": 1.0,
                "stress": 1.0,
                "nonexistent_head": 1.0,
            },
        )


def test_forces_stress_heads_rejected(gns_model, energy_head, force_head, stress_head):
    with pytest.raises(AssertionError, match="collide with gradient-based prediction keys"):
        ConservativeForcefieldRegressor(
            heads={"energy": energy_head, "forces": force_head, "stress": stress_head},
            model=gns_model,
            loss_weights={"energy": 1.0, "forces": 1.0, "stress": 1.0},
        )


def test_conservative_forces_twice_differentiable(batch, conservative_regressor):
    # Make positions require grad
    batch.node_features["positions"].requires_grad_(True)

    # First forward pass
    out = conservative_regressor(batch)
    energy = out["energy"].sum()

    # First backward pass to get forces
    grad_forces = torch.autograd.grad(energy, batch.node_features["positions"], create_graph=True)[
        0
    ]

    # Second backward pass should work (important for training)
    grad_forces.sum().backward()

    # Check that gradients were computed
    assert batch.node_features["positions"].grad is not None


def test_regressor_predict(batch, conservative_regressor):
    conservative_regressor.eval()
    inference = conservative_regressor.predict(batch)
    assert "energy" in inference
    assert "forces" in inference
    assert "stress" in inference


def test_regressor_predict_preserves_kjmol_at_scale(batch, conservative_regressor):
    """End-to-end mirror of test_energy_head_absolute_energy_preserves_kjmol_at_scale."""
    conservative_regressor.eval()
    large_ref = 1e5

    energy_head = conservative_regressor.heads["energy"]
    n_atoms = int(batch.n_node[0].item())
    assert (batch.n_node == n_atoms).all(), "fixture assumption: uniform atom count"
    with torch.no_grad():
        energy_head.reference.linear.weight.fill_(large_ref / n_atoms)

    # The MLP-produced interaction energy is whatever gets added to reference inside predict
    interaction_energy = conservative_regressor(batch)["interaction_energy"].detach()

    # fp64 path preserves the interaction energy against the large reference.
    absolute = conservative_regressor.predict(batch, fp64_energy=True)["energy"]
    recovered = (absolute - large_ref).float()
    torch.testing.assert_close(recovered, interaction_energy, atol=1e-6, rtol=0)

    # fp32 path loses the interaction energy precision, demonstrating why fp64 is required
    absolute_fp32 = conservative_regressor.predict(batch, fp64_energy=False)["energy"]
    fp32_roundtrip = absolute_fp32 - large_ref
    assert not torch.allclose(fp32_roundtrip, interaction_energy, atol=1e-6, rtol=0)


def test_featurization_differentiability_with_conservative_regressor(
    conservative_regressor,
):
    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=10)
    atoms = Atoms([Atom("C", [0, 0, 0]), Atom("H", [1, 1, 1]), Atom("O", [2, 1, 1])])
    atoms2 = Atoms([Atom("C", [-1, 0, -1]), Atom("H", [1, 2, 0]), Atom("O", [1, 3, 1])])
    atom_graphs = AtomGraphs.batch(
        [
            adapter.from_ase_atoms(atoms),
            adapter.from_ase_atoms(atoms2),
        ]
    )
    out = conservative_regressor(atom_graphs)
    grad = torch.autograd.grad(out["pred"].sum(), atom_graphs.positions)[0]
    # assert grad exists and all its elements are distinct
    assert grad is not None
    assert len(torch.unique(grad)) == grad.numel()


def test_modules_have_float64_dtypes_for_float64_model(batch, conservative_regressor):
    conservative_regressor = deepcopy(conservative_regressor)

    def check_dtype_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            assert output.dtype == torch.float64, (
                f"Layer {module.__class__.__name__} output dtype is {output.dtype}, "
                "expected torch.float64"
            )
        elif isinstance(output, (tuple, list)):
            for o in output:
                if isinstance(o, torch.Tensor):
                    assert o.dtype == torch.float64, (
                        f"Layer {module.__class__.__name__} output dtype is {o.dtype}, "
                        "expected torch.float64"
                    )

    hooks = []
    for name, module in conservative_regressor.named_modules():
        hooks.append(module.register_forward_hook(check_dtype_hook))
        hooks.append(module.register_full_backward_hook(check_dtype_hook))

    conservative_regressor.to(torch.float64)

    non_float64_params = []
    for k, v in conservative_regressor.named_parameters():
        if v.dtype != torch.float64:
            non_float64_params.append(k)

    assert len(non_float64_params) == 0, f"Non-float64 parameters: {non_float64_params}"

    batch = batch.to(dtype=torch.float64)

    out = conservative_regressor(batch)

    for key, value in out.items():
        if isinstance(value, torch.Tensor):
            assert value.dtype == torch.float64, (
                f"Output {key} dtype is {value.dtype}, expected torch.float64"
            )

    for hook in hooks:
        hook.remove()


@pytest.mark.parametrize("graph_name", ["single_graph", "batch"])
def test_regressor_compile(conservative_regressor, graph_name, request):
    """Tests compile (backbone + heads + autograd, with graph breaks)."""
    graph = request.getfixturevalue(graph_name)
    conservative_regressor.eval()
    conservative_regressor.compile(mode="default", dynamic=True)
    conservative_regressor(graph)


@pytest.mark.parametrize("graph_name", ["single_graph", "batch"])
def test_regressor_compile_matches_eager(conservative_regressor, graph_name, request):
    """Compile produces the same outputs as eager."""
    graph = request.getfixturevalue(graph_name)
    conservative_regressor.eval()
    eager_out = conservative_regressor(graph)
    conservative_regressor.compile(mode="default", dynamic=True)
    compiled_out = conservative_regressor(graph)
    for key in eager_out:
        torch.testing.assert_close(
            compiled_out[key],
            eager_out[key],
            atol=1e-5,
            rtol=1e-5,
            msg=f"Mismatch for {key!r}",
        )
