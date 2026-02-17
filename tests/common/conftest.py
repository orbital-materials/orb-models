from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def fixtures_path(request):
    """Return the file fixtures path for any script."""
    return Path(request.fspath).parent / "fixtures"


# Synthetic test fixtures
@pytest.fixture()
def minimal_valid_cif(fixtures_path):
    """Minimal synthetic CIF with simple cubic carbon structure.

    Used for: Basic parsing tests, primitive vs conventional cell tests.
    Parsers: Works with both Pymatgen (strict) and ASE.
    """
    return fixtures_path / "minimal_valid.cif"


@pytest.fixture()
def pymatgen_fails_ase_succeeds_cif(fixtures_path):
    """Synthetic CIF that triggers Pymatgen failure but ASE can parse.

    Used for: Testing ASE fallback mechanism (CRITICAL test).
    Parsers: Fails with strict Pymatgen, succeeds with ASE.
    Trigger: Multiple atoms at very close positions with fractional occupancies.
    """
    return fixtures_path / "pymatgen_fails_ase_succeeds.cif"


@pytest.fixture()
def high_occupancy_cif(fixtures_path):
    """Synthetic CIF with disorder from symmetry operations (CSD MOF pattern).

    Used for: Testing pymatgen_occupancy_tolerance parameter.
    Parsers: Fails with strict Pymatgen (tolerance=1.0), succeeds with tolerance=100.0.
    Trigger: Multiple atoms at identical fractional coordinates with occupancies summing > 1.0.
    Use case: Mimics real CSD MOF structures that are "disordered by symmetry" where
    symmetry operations create overlapping atoms with partial occupancies.
    """
    return fixtures_path / "symmetry_disorder.cif"


@pytest.fixture()
def malformed_cif(fixtures_path):
    """Synthetic CIF with invalid syntax that no parser can handle.

    Used for: Testing PymatgenCifNoStructureError when CIF has no valid structure.
    Parsers: Pymatgen fails before ASE fallback is attempted.
    Trigger: Invalid CIF syntax (malformed cell parameters, missing data).
    """
    return fixtures_path / "malformed.cif"


# Real MOF structure fixtures
@pytest.fixture()
def good_cif_file(fixtures_path):
    """MOF structure processed by PACMAN (Cd-based, 12 Cd + 96 H + 152 C + 60 O atoms).
    Used for: Real-world MOF validation, CSD workflow integration tests.
    Parsers: Successfully parsed by Pymatgen with lenient settings.
    """
    return fixtures_path / "0000[Cd][nan]3[FSR]1.cif"


@pytest.fixture()
def wrong_num_tokens_cif_file(fixtures_path):
    """MOF structure processed by PACMAN (Zn-based, 6 Zn + 42 N + 114 C + 14 O atoms).
    Used for: Testing PymatgenCifNoStructureError with real data.
    Parsers: Triggers "Invalid CIF file with no structures!" error with strict Pymatgen.
    Note: Can be parsed with lenient settings or via ASE fallback.
    """
    return fixtures_path / "ENIXEH_FSR_pacman.cif"
