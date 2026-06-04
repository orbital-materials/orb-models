import pytest
import torch


@pytest.mark.parametrize("graph_name", ["single_graph", "batch"])
def test_regressor_compile(direct_regressor, graph_name, request):
    """Tests compile (backbone + heads, with graph breaks)."""
    graph = request.getfixturevalue(graph_name)
    direct_regressor.eval()
    direct_regressor.compile(mode="default", dynamic=True)
    direct_regressor(graph)


def test_compile_engages_backbone(direct_regressor):
    """``.compile()`` must compile the GNS backbone, not just the regressor wrapper.

    ``predict()`` calls ``self.model(batch)`` directly and bypasses ``__call__``, so
    compiling only the regressor leaves inference fully eager. Guards against silently
    falling back to ``nn.Module.compile()`` semantics in a future refactor.
    """
    assert direct_regressor.model._compiled_call_impl is None
    direct_regressor.compile(mode="default", dynamic=True)
    assert direct_regressor.model._compiled_call_impl is not None


@pytest.mark.parametrize("graph_name", ["single_graph", "batch"])
def test_regressor_compile_matches_eager(direct_regressor, graph_name, request):
    """Compile produces the same outputs as eager."""
    graph = request.getfixturevalue(graph_name)
    direct_regressor.eval()
    eager_out = direct_regressor(graph)
    direct_regressor.compile(mode="default", dynamic=True)
    compiled_out = direct_regressor(graph)
    for key in eager_out:
        torch.testing.assert_close(
            compiled_out[key],
            eager_out[key],
            atol=1e-5,
            rtol=1e-5,
            msg=f"Mismatch for {key!r}",
        )
