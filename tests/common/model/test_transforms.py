import numpy as np
import pytest
import torch

from orb_models.common.models.graph_regressor import ScalarNormalizer


def test_normalizer_scales_to_unit():
    mean = 1000
    std = 100
    normalizer = ScalarNormalizer(init_mean=torch.Tensor([mean]), init_std=torch.Tensor([std]))

    random_data = torch.randn(100000) * 100 + 1000
    scaled = normalizer(random_data)
    assert np.isclose(scaled.mean(), 0, atol=0.01)
    assert np.isclose(scaled.std(), 1, atol=0.01)


def test_normalizer_scales_to_unit_2D():
    mean = 1000
    std = 100
    normalizer = ScalarNormalizer(init_mean=torch.Tensor([mean]), init_std=torch.Tensor([std]))

    random_data = torch.randn(1000, 1000) * 100 + 1000
    scaled = normalizer(random_data)
    assert np.isclose(scaled.mean(), 0, atol=0.01)
    assert np.isclose(scaled.std(), 1, atol=0.01)


def test_normalizer_invertible():
    normalizer = ScalarNormalizer(init_mean=torch.Tensor([10]), init_std=torch.Tensor([10]))
    random_data = torch.randn(1000)
    scaled = normalizer(random_data)
    inverted = normalizer.inverse(scaled)
    assert torch.allclose(random_data, inverted, atol=1e-5)


@pytest.mark.parametrize("online", [True, False])
def test_normalizer_online(online):
    normalizer = ScalarNormalizer(
        init_mean=torch.Tensor([10]),
        init_std=torch.Tensor([10]),
        init_num_batches=1,
        online=online,
    )
    random_data = torch.randn(100)
    inverted_data = normalizer.inverse(random_data)

    # if online, this for loop will change normalizer's behaviour
    for _ in range(3):
        reinverted_data = normalizer(inverted_data)

    if online:
        assert not torch.allclose(random_data, reinverted_data, atol=1e-5)
    else:
        assert torch.allclose(random_data, reinverted_data, atol=1e-5)
