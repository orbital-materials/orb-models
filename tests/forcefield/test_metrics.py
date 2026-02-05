import torch

from orb_models.forcefield.models.forcefield_utils import forces_within_threshold


def test_forces_within_threshold():
    pred = torch.randn(10, 3)
    nodes = torch.tensor([5, 5])
    assert forces_within_threshold(pred, pred, nodes) == 1.0
    pred = torch.randn(10, 3)
    pred[:6, :] = 2
    target = torch.ones(10, 3) * 2
    nodes = torch.tensor([6, 4])
    assert forces_within_threshold(pred, target, nodes) == 0.5
