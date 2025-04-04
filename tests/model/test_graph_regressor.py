import pytest
import torch

from orb_models.forcefield import base
from orb_models.forcefield.direct_regressor import DirectForcefieldRegressor


def test_regressor_forward_on_single_node_graph(
    single_node_graph, gns_model, forces_head
):
    regressor = DirectForcefieldRegressor(
        heads={"forces": forces_head},
        model=gns_model,
    )
    _ = regressor(single_node_graph)


def test_regressor_can_predict(graph, gns_model, energy_head, forces_head):
    regressor = DirectForcefieldRegressor(
        heads={"energy": energy_head, "forces": forces_head},
        model=gns_model,
    )
    regressor.eval()
    inference = regressor.predict(graph)
    assert inference["energy"] is not None
    assert inference["forces"] is not None


def test_regressor_can_fine_tune_on_graph_target(graph, gns_model, forces_head):
    regressor = DirectForcefieldRegressor(
        heads={"forces": forces_head},
        model=gns_model,
    )
    regressor.eval()

    batch = base.batch_graphs([graph.clone(), graph.clone()])
    pred = regressor.loss(batch)
    pred.loss.sum().backward()

    assert "forces_loss" in pred.log
    assert "forces_mae_raw" in pred.log


def test_regressor_can_fine_tune_on_node_targets(graph, gns_model, energy_head):
    regressor = DirectForcefieldRegressor(
        heads={"energy": energy_head},
        model=gns_model,
    )
    out = regressor.loss(graph)
    out.loss.sum().backward()
    assert "energy_loss" in out.log


@pytest.mark.parametrize("requires_grad", [True, False])
def test_regressor_base_requires_grad(requires_grad, gns_model, forces_head):
    regressor_nograd = DirectForcefieldRegressor(
        {"forces": forces_head},
        gns_model,
        model_requires_grad=requires_grad,
    )

    # check base model does / doesn't require grad
    for param in regressor_nograd.model.parameters():
        if requires_grad:
            assert param.requires_grad
        else:
            assert not param.requires_grad

    # the heads always require grad
    head = regressor_nograd.heads["forces"]
    assert head is not None
    assert head.mlp is not None
    for param in head.mlp.parameters():
        assert param.requires_grad


def test_regressor_can_predict_scaled(graph, gns_model, energy_head, forces_head):
    regressor = DirectForcefieldRegressor(
        {"forces": forces_head},
        model=gns_model,
    )
    regressor.eval()
    # update normalizer
    regressor.loss(graph)
    forward_result = regressor(graph)

    inference = regressor.predict(graph)
    assert (
        inference["forces"] is not None
    ), "forces not found in inference results"

    forces_head = regressor.heads["forces"]
    assert torch.allclose(
        inference["forces"],
        forces_head.normalizer.inverse(forward_result["forces"]),
        atol=1e-5,
    )


@pytest.mark.xfail(True, reason="The regressor forward is currently not compilable.")
def test_regressor_can_torch_compile(graph, gns_model, energy_head, forces_head):
    """Tests if the GraphRegressor.forward is compilable with torch.compile."""
    regressor = DirectForcefieldRegressor(
        heads={
            "energy": energy_head,
            "forces": forces_head,
        },
        model=gns_model,
    )
    regressor.eval()
    compiled = torch.compile(regressor, mode="default", dynamic=True, fullgraph=True)
    compiled(graph)


def test_regressor_module_compiles(graph, gns_model, energy_head, forces_head):
    """Tests if GraphRegressor.forward (partially) compiles with Module.compile.

    This tests whether the override of GraphRegressor.compile allows us to
    compile the model. If there is no override of GraphRegressor.compile
    then this test should be effectively equivalent to _can_compile test above.
    """
    regressor = DirectForcefieldRegressor(
        heads={
            "energy": energy_head,
            "forces": forces_head,
        },
        model=gns_model,
    )
    regressor.eval()
    regressor.compile(mode="default", dynamic=True, fullgraph=True)
    regressor(graph)
