import pytest
import torch
from ase import Atoms, Atom

from orb_models.forcefield.conservative_regressor import ConservativeForcefieldRegressor
from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
from orb_models.forcefield.atomic_system import SystemConfig
from orb_models.forcefield.base import batch_graphs
from copy import deepcopy


@pytest.mark.parametrize("graph_name", ["batch", "single_graph"])
def test_regressor_forward(request, conservative_regressor, graph_name):
    graph = request.getfixturevalue(graph_name)
    out = conservative_regressor(graph)
    assert "energy" in out
    assert "grad_forces" in out
    assert "grad_stress" in out


def test_regressor_loss(conservative_regressor, batch):
    out = conservative_regressor.loss(batch)
    out.loss.backward()

    for k in out.log.keys():
        print(k)

    # Check that metrics are computed for both direct and conservative predictions
    assert any("energy" in k for k in out.log)
    assert any("grad-forces" in k for k in out.log)
    assert any("grad-stress" in k for k in out.log)
    assert any("rotational_grad" in k for k in out.log)


def test_regressor_head_config_raises_error(
    gns_model, energy_head, force_head, stress_head
):
    with pytest.raises(ValueError, match="Loss weights for unknown targets"):
        ConservativeForcefieldRegressor(
            heads={"energy": energy_head, "forces": force_head},
            model=gns_model,
            loss_weights={
                "energy": 1.0,
                "forces": 1.0,
                "stress": 1.0,
                "grad_forces": 1.0,
                "grad_stress": 1.0,
            },
        )

    # check error is raised if grad_forces or grad_stress are not in loss_weights
    with pytest.raises(
        ValueError, match="grad_forces and grad_stress must be in loss_weights"
    ):
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
    grad_forces = torch.autograd.grad(
        energy, batch.node_features["positions"], create_graph=True
    )[0]

    # Second backward pass should work (important for training)
    grad_forces.sum().backward()

    # Check that gradients were computed
    assert batch.node_features["positions"].grad is not None


def test_regressor_predict(batch, conservative_regressor):
    conservative_regressor.eval()
    inference = conservative_regressor.predict(batch)
    assert "energy" in inference
    assert "grad_forces" in inference
    assert "grad_stress" in inference


def test_featurization_differentiability_with_conservative_regressor(
    conservative_regressor,
):

    system_config = SystemConfig(radius=6.0, max_num_neighbors=10)
    atoms = Atoms([Atom("C", [0, 0, 0]), Atom("H", [1, 1, 1]), Atom("O", [2, 1, 1])])
    atoms2 = Atoms([Atom("C", [-1, 0, -1]), Atom("H", [1, 2, 0]), Atom("O", [1, 3, 1])])
    atom_graphs = batch_graphs(
        [
            ase_atoms_to_atom_graphs(atoms, system_config=system_config),
            ase_atoms_to_atom_graphs(atoms2, system_config=system_config),
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
            assert (
                output.dtype == torch.float64
            ), f"Layer {module.__class__.__name__} output dtype is {output.dtype}, expected torch.float64"
        elif isinstance(output, (tuple, list)):
            for o in output:
                if isinstance(o, torch.Tensor):
                    assert (
                        o.dtype == torch.float64
                    ), f"Layer {module.__class__.__name__} output dtype is {o.dtype}, expected torch.float64"

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
            assert (
                value.dtype == torch.float64
            ), f"Output {key} dtype is {value.dtype}, expected torch.float64"

    for hook in hooks:
        hook.remove()


@pytest.mark.xfail(True, reason="The regressor forward is currently not compilable.")
def test_regressor_can_torch_compile(conservative_regressor, batch):
    """Tests if the ConservativeForcefieldRegressor.forward is compilable with torch.compile."""
    conservative_regressor.eval()
    compiled = torch.compile(
        conservative_regressor, mode="default", dynamic=True, fullgraph=True
    )
    compiled(batch)


def test_regressor_module_compiles(conservative_regressor, batch):
    """Tests if ConservativeForcefieldRegressor.forward (partially) compiles with Module.compile.

    This tests whether the override of ConservativeForcefieldRegressor.compile allows us to
    compile the model. If there is no override of ConservativeForcefieldRegressor.compile
    then this test should be effectively equivalent to _can_compile test above.
    """
    conservative_regressor.eval()
    conservative_regressor.compile(mode="default", dynamic=True, fullgraph=True)
    conservative_regressor(batch)
