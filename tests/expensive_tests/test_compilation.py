import copy

import ase
import ase.build
import numpy as np
import pytest
import torch

from orb_models.common.atoms.batch.graph_batch import AtomGraphs


def _make_periodic_system(adapter) -> AtomGraphs:
    atoms = ase.Atoms(
        symbols="Si4",
        pbc=True,
        cell=np.diag([5.43, 5.43, 5.43]),
        positions=[
            [0.0, 0.0, 0.0],
            [1.36, 1.36, 1.36],
            [2.72, 2.72, 0.0],
            [4.08, 4.08, 1.36],
        ],
    )
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    return adapter.from_ase_atoms(atoms)


def _make_nacl_bulk(adapter) -> AtomGraphs:
    atoms = ase.build.bulk("NaCl", crystalstructure="rocksalt", a=5.64)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    return adapter.from_ase_atoms(atoms)


def _make_nonperiodic_system(adapter) -> AtomGraphs:
    atoms = ase.Atoms(
        symbols="H2O",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.0, 0.76, -0.58]],
    )
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    return adapter.from_ase_atoms(atoms)


def _make_ethanol(adapter) -> AtomGraphs:
    atoms = ase.build.molecule("CH3CH2OH")
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    return adapter.from_ase_atoms(atoms)


ALL_MODELS = [
    "orb_v2_and_config",
    "orb_v3_conservative_omat_and_config",
    "orb_v3_conservative_omol_and_config",
    "orb_v3_direct_omat_and_config",
    "orbmol_v2_and_config",
]


@pytest.mark.parametrize("model_fixture", ALL_MODELS)
@pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
def test_compile(model_fixture, batched, request):
    model, sys_cfg = request.getfixturevalue(model_fixture)
    model = copy.deepcopy(model)
    device = next(model.parameters()).device
    model.eval()
    model.compile(mode="default", dynamic=True)
    graph = _make_periodic_system(sys_cfg)
    if batched:
        graph = AtomGraphs.batch([graph, _make_periodic_system(sys_cfg)])
    graph = graph.to(device=device)
    model.predict(graph)


@pytest.mark.parametrize("model_fixture", ALL_MODELS)
@pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
@pytest.mark.parametrize(
    "make_system",
    [_make_periodic_system, _make_nacl_bulk, _make_nonperiodic_system, _make_ethanol],
    ids=["periodic", "nacl_bulk", "nonperiodic", "ethanol"],
)
def test_compile_matches_eager(model_fixture, batched, make_system, request):
    """Compiled predictions must match eager to float-precision.

    Includes non-periodic (H2O) coverage: a Si4 PBC system has all atoms
    equivalent, so mean-centered per-graph quantities (e.g. charges in
    LatentChargeHead) are trivially ~0 and would mask aggregation bugs that
    only manifest on asymmetric inputs.
    """
    model, sys_cfg = request.getfixturevalue(model_fixture)
    model = copy.deepcopy(model).double()
    device = next(model.parameters()).device
    model.eval()
    graph = make_system(sys_cfg)
    if batched:
        graph = AtomGraphs.batch([graph, make_system(sys_cfg)])
    graph = graph.to(device=device, dtype=torch.float64)
    eager_out = model.predict(graph)
    model.compile(mode="default", dynamic=True)
    compiled_out = model.predict(graph)
    for key in eager_out:
        torch.testing.assert_close(
            compiled_out[key],
            eager_out[key],
            atol=1e-10,
            rtol=1e-10,
        )


@pytest.mark.parametrize("model_fixture", ALL_MODELS)
def test_compile_nonperiodic(model_fixture, request):
    """Compile on a non-periodic (molecular) system."""
    model, sys_cfg = request.getfixturevalue(model_fixture)
    model = copy.deepcopy(model)
    device = next(model.parameters()).device
    model.eval()
    graph = _make_nonperiodic_system(sys_cfg)
    graph = graph.to(device=device)
    model.compile(mode="default", dynamic=True)
    model.predict(graph)
