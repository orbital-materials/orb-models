from pathlib import Path

import pytest
import torch

from orb_models.common import utils


@pytest.fixture(autouse=True, scope="function")
def default_test_setup():
    """
    Ensure all tests by default use float32, are deterministic and have the same seed.
    Deviations should explicitly be made within each test.
    """
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    utils.seed_everything(42, 0)
    yield


@pytest.fixture(scope="module")
def fixtures_path(request):
    """Return the file fixtures path for any script."""
    return Path(request.fspath).parent / "fixtures"


@pytest.fixture(scope="module")
def shared_fixtures_path(request):
    """Return the top-level fixtures path for any script."""
    return Path(request.fspath).parent.parent / "fixtures"
