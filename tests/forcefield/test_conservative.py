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
    assert "grad_forces" in out
    assert "grad_stress" in out


def test_regressor_loss(conservative_regressor, batch):
    out = conservative_regressor.loss(batch)
    out.loss.backward()

    # Check that metrics are computed for both direct and conservative predictions
    assert any("energy" in k for k in out.log)
    assert any("grad-forces" in k for k in out.log)
    assert any("grad-stress" in k for k in out.log)
    assert any("rotational_grad" in k for k in out.log)


def test_regressor_head_config_raises_error(gns_model, energy_head, force_head, stress_head):
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
    assert "grad_forces" in inference
    assert "grad_stress" in inference


def test_conservative_model_can_distill(batch, conservative_regressor):
    conservative_regressor.eval()
    conservative_regressor.distill_direct_heads = True
    distill_output = conservative_regressor.loss(batch)

    conservative_regressor.distill_direct_heads = False
    output = conservative_regressor.loss(batch)
    assert not torch.allclose(output.loss, distill_output.loss)


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


@pytest.mark.xfail(True, reason="The regressor forward is currently not compilable.")
def test_regressor_can_torch_compile(conservative_regressor, batch):
    """Tests if the ConservativeForcefieldRegressor.forward is compilable with torch.compile."""
    conservative_regressor.eval()
    compiled = torch.compile(conservative_regressor, mode="default", dynamic=True, fullgraph=True)
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


def test_pair_repulsion_default_aggregation_is_mean(gns_model, energy_head):
    """BC guard: changing this default would silently break every public orb-v3 conservative model.

    Pre-orbmol-v2 models (orb-v3-conservative-omol/omat/mpa) were trained with
    ZBLBasis(node_aggregation="mean"). The regressor's default must stay "mean"
    so those S3 weights continue to produce the same predictions on reload.
    """
    regressor = ConservativeForcefieldRegressor(
        heads={"energy": energy_head},
        model=gns_model,
        pair_repulsion=True,
    )
    assert regressor.pair_repulsion_fn.node_aggregation == "mean"


def test_pair_repulsion_sum_when_specified(gns_model, energy_head):
    """orbmol_v2_architecture trained with sum-aggregation; opt-in via the kwarg."""
    regressor = ConservativeForcefieldRegressor(
        heads={"energy": energy_head},
        model=gns_model,
        pair_repulsion=True,
        pair_repulsion_node_aggregation="sum",
    )
    assert regressor.pair_repulsion_fn.node_aggregation == "sum"


def test_energy_head_does_not_have_absolute_energy():
    """BC guard: absolute_energy lives only on ChargeConditionedEnergyHead.

    Adding it to the base EnergyHead would be a behavior change for all v3 models
    (since they all subclass or use EnergyHead). Keep it scoped to the orbmol-v2
    head where the fp64 promotion is needed for OMol-scale references.
    """
    from orb_models.forcefield.models.forcefield_heads import (
        ChargeConditionedEnergyHead,
        EnergyHead,
    )

    assert not hasattr(EnergyHead, "absolute_energy")
    assert hasattr(ChargeConditionedEnergyHead, "absolute_energy")


def test_orbmol_v2_architecture_uses_sum_zbl():
    """Integration check: orbmol_v2_architecture wires sum-aggregation ZBL."""
    from orb_models.forcefield.pretrained import orbmol_v2_architecture

    model = orbmol_v2_architecture(device="cpu")
    assert model.pair_repulsion is True
    assert model.pair_repulsion_fn.node_aggregation == "sum"
    assert model.coulomb_module is not None
    assert "latent_charges" in model.heads
    assert "latent_spins" in model.heads
