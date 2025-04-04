import pytest
import torch

from orb_models.forcefield import base
from orb_models.forcefield.direct_regressor import DirectForcefieldRegressor


def test_regressor_forward_on_single_node_graph(
    single_node_graph, gns_model, graph_head
):
    regressor = DirectForcefieldRegressor(
        heads={"graph_target": graph_head()},
        model=gns_model,
    )
    _ = regressor(single_node_graph)


def test_regressor_can_predict(graph, gns_model, node_head, graph_head):
    regressor = DirectForcefieldRegressor(
        heads={"noise_target": node_head(), "graph_target": graph_head()},
        model=gns_model,
    )
    regressor.eval()
    inference = regressor.predict(graph)
    assert inference["noise_target"] is not None
    assert inference["graph_target"] is not None


def test_regressor_can_fine_tune_on_graph_target(graph, gns_model, graph_head):
    regressor = DirectForcefieldRegressor(
        heads={"graph_target": graph_head()},
        model=gns_model,
    )
    regressor.eval()

    batch = base.batch_graphs([graph.clone(), graph.clone()])
    pred = regressor.loss(batch)
    pred.loss.sum().backward()

    assert "graph_target_loss" in pred.log
    assert "graph_target_mae_raw" in pred.log


def test_regressor_can_fine_tune_on_node_targets(graph, gns_model, node_head):
    regressor = DirectForcefieldRegressor(
        heads={"noise_target": node_head()},
        model=gns_model,
    )
    out = regressor.loss(graph)
    out.loss.sum().backward()
    assert "noise_target_loss" in out.log


@pytest.mark.parametrize("requires_grad", [True, False])
def test_regressor_base_requires_grad(requires_grad, gns_model, graph_head):
    head = graph_head()
    regressor_nograd = DirectForcefieldRegressor(
        {"graph_target": head},
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
    head = regressor_nograd.heads["graph_target"]
    assert head is not None
    assert head.mlp is not None
    for param in head.mlp.parameters():
        assert param.requires_grad


def test_regressor_can_predict_scaled(graph, gns_model, node_head, graph_head):
    regressor = DirectForcefieldRegressor(
        {"noise_target": node_head(), "graph_target": graph_head()},
        model=gns_model,
    )
    regressor.eval()
    # update normalizer
    regressor.loss(graph)
    forward_result = regressor(graph)

    inference = regressor.predict(graph)
    assert (
        inference["noise_target"] is not None
    ), "noise_target not found in inference results"
    assert (
        inference["graph_target"] is not None
    ), "graph_target not found in inference results"

    noise_head = regressor.heads["noise_target"]
    torch.testing.assert_close(
        inference["noise_target"],
        noise_head.normalizer.inverse(forward_result["noise_target"]),
    )
    graph_head = regressor.heads["graph_target"]
    assert inference["graph_target"] == graph_head.normalizer.inverse(
        forward_result["graph_target"]
    )


@pytest.mark.xfail(True, reason="The regressor forward is currently not compilable.")
def test_regressor_can_torch_compile(graph, gns_model, node_head, graph_head):
    """Tests if the GraphRegressor.forward is compilable with torch.compile."""
    regressor = DirectForcefieldRegressor(
        heads={
            "noise_target": node_head(),
            "graph_target": graph_head(),
        },
        model=gns_model,
    )
    regressor.eval()
    compiled = torch.compile(regressor, mode="default", dynamic=True, fullgraph=True)
    compiled(graph)


def test_regressor_module_compiles(graph, gns_model, node_head, graph_head):
    """Tests if GraphRegressor.forward (partially) compiles with Module.compile.

    This tests whether the override of GraphRegressor.compile allows us to
    compile the model. If there is no override of GraphRegressor.compile
    then this test should be effectively equivalent to _can_compile test above.
    """
    regressor = DirectForcefieldRegressor(
        heads={
            "noise_target": node_head(),
            "graph_target": graph_head(),
        },
        model=gns_model,
    )
    regressor.eval()
    regressor.compile(mode="default", dynamic=True, fullgraph=True)
    regressor(graph)
