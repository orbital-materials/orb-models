import pytest
import torch

from orb_models.forcefield.property_definitions import PropertyDefinition
from orb_models.forcefield import base
from orb_models.forcefield.graph_regressor import GraphRegressor


def test_regressor_forward_on_single_node_graph(
    single_node_graph, gns_model, graph_head
):
    regressor = GraphRegressor(heads=[graph_head()], model=gns_model)
    _ = regressor(single_node_graph)


def test_regressor_can_predict(graph, gns_model, node_head, graph_head):
    regressor = GraphRegressor([node_head(), graph_head()], model=gns_model)
    regressor.eval()
    inference = regressor.predict(graph)
    assert inference["noise_target"] is not None
    assert inference["graph_target"] is not None


def test_regressor_can_fine_tune_on_graph_target(graph, gns_model, graph_head):
    regressor = GraphRegressor(heads=[graph_head()], model=gns_model)
    regressor.eval()

    batch = base.batch_graphs([graph.clone(), graph.clone()])
    pred = regressor.loss(batch)
    pred.loss.sum().backward()

    assert "graph_target_loss" in pred.log
    assert "graph_target_mae_raw" in pred.log


def test_regressor_can_fine_tune_on_node_targets(graph, gns_model, node_head):
    regressor = GraphRegressor(heads=[node_head()], model=gns_model)
    out = regressor.loss(graph)
    out.loss.sum().backward()
    assert "noise_target_loss" in out.log


@pytest.mark.parametrize("requires_grad", [True, False])
def test_regressor_base_requires_grad(requires_grad, gns_model, graph_head):
    head = graph_head()
    regressor_nograd = GraphRegressor(
        [head], gns_model, model_requires_grad=requires_grad
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


def test_regressor_repeated_head_throws_error(gns_model, graph_head):
    head = graph_head()
    with pytest.raises(ValueError):
        GraphRegressor([head, head], model=gns_model)


@pytest.mark.parametrize("cutoff_layers", [1, 2])
def test_regressor_can_take_cutoff_layers(
    graph, gns_model, node_head, graph_head, cutoff_layers
):
    regressor = GraphRegressor(
        heads=[node_head(), graph_head()],
        model=gns_model,
        cutoff_layers=cutoff_layers,
    )
    regressor.eval()
    inference = regressor.predict(graph)
    assert len(regressor.model.gnn_stacks) == cutoff_layers
    assert inference["noise_target"] is not None
    assert inference["graph_target"] is not None


def test_regressor_can_predict_scaled(graph, gns_model, node_head, graph_head):
    regressor = GraphRegressor([node_head(), graph_head()], model=gns_model)
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


@pytest.mark.parametrize("graph_name", ["graph_binary", "single_node_graph_binary"])
def test_binary_classifier_can_predict(request, gns_model, graph_head, graph_name):
    graph = request.getfixturevalue(graph_name)
    target = PropertyDefinition(
        "binary_graph_target", dim=1, domain="binary", row_to_property_fn=lambda x: x
    )
    regressor = GraphRegressor(heads=[graph_head(target)], model=gns_model)
    inference = regressor.predict(graph)
    assert inference["binary_graph_target"] is not None


@pytest.mark.parametrize("graph_name", ["graph_binary", "single_node_graph_binary"])
def test_binary_classifier_can_fine_tune(request, gns_model, graph_head, graph_name):
    graph = request.getfixturevalue(graph_name)
    target = PropertyDefinition(
        "binary_graph_target", dim=1, domain="binary", row_to_property_fn=lambda x: x
    )
    regressor = GraphRegressor(heads=[graph_head(target)], model=gns_model)
    pred = regressor.loss(graph)
    pred.loss.sum().backward()
    assert "binary_graph_target_accuracy" in pred.log


@pytest.mark.parametrize(
    "graph_name", ["graph_categorical", "single_node_graph_categorical"]
)
def test_classifier_can_predict(request, gns_model, graph_head, graph_name):
    graph = request.getfixturevalue(graph_name)
    target = PropertyDefinition(
        "cat_graph_target", dim=1, domain="binary", row_to_property_fn=lambda x: x
    )
    regressor = GraphRegressor(heads=[graph_head(target)], model=gns_model)
    inference = regressor.predict(graph)
    assert inference["cat_graph_target"] is not None


@pytest.mark.parametrize(
    "graph_name", ["graph_categorical", "single_node_graph_categorical"]
)
def test_classifier_can_fine_tune(request, gns_model, graph_head, graph_name):
    graph = request.getfixturevalue(graph_name)
    graph_target = PropertyDefinition(
        "cat_graph_target", dim=5, domain="categorical", row_to_property_fn=lambda x: x
    )
    regressor = GraphRegressor(heads=[graph_head(graph_target)], model=gns_model)
    pred = regressor.loss(graph)
    pred.loss.sum().backward()
    assert "cat_graph_target_accuracy" in pred.log


@pytest.mark.xfail(True, reason="The regressor forward is currently not compilable.")
def test_regressor_can_torch_compile(graph, gns_model, node_head, graph_head):
    """Tests if the GraphRegressor.forward is compilable with torch.compile."""
    regressor = GraphRegressor(
        heads=[node_head(), graph_head()],
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
    regressor = GraphRegressor(
        heads=[node_head(), graph_head()],
        model=gns_model,
    )
    regressor.eval()
    regressor.compile(mode="default", dynamic=True, fullgraph=True)
    regressor(graph)
