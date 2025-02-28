import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def fixtures_path(request):
    """Return the file fixtures path for any script."""
    return Path(request.fspath).parent / "fixtures"


@pytest.fixture(scope="module")
def shared_fixtures_path(request):
    """Return the top-level fixtures path for any script."""
    return Path(request.fspath).parent.parent / "fixtures"
