"""Compare eager vs compiled predictions to float-precision across a suite of systems.

Useful when investigating torch.compile + autograd numerical issues. Loads
the public orbmol_v2 checkpoint, runs predict() under eager and again under
full-model compile, and reports the per-system max |ΔE| and |Δforce|.

Two systems are excluded from the eager-vs-compile baseline, Si4 and
water_box×8, because they happen to mask mean-aggregation bugs (Si4: all-
equivalent atoms make mean ~0; water_box×8: dynamic-shape specialization can
produce the right answer at that size). The H2O / ethanol / NaCl / water_box×2
systems are what reliably expose discrepancies.

Usage:
    python scripts/compile_numerical_check.py --device cuda
"""

import argparse
import copy

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk, molecule

from orb_models.forcefield import pretrained


def water_box(n: int = 8, density: float = 1.0) -> Atoms:
    mw = 18.015
    volume = n * mw / density / 6.022e23 * 1e24
    a = volume ** (1 / 3)
    rng = torch.Generator().manual_seed(0)
    positions = torch.rand(n * 3, 3, generator=rng).numpy() * a
    return Atoms(
        symbols=["O", "H", "H"] * n,
        positions=positions,
        cell=[a, a, a],
        pbc=True,
    )


def make_ion(symbol: str, charge: int) -> Atoms:
    atoms = Atoms(symbols=[symbol], positions=[[0, 0, 0]])
    atoms.info["charge"] = charge
    atoms.info["spin"] = 1 if abs(charge) % 2 == 0 else 2
    return atoms


def build_systems() -> list[tuple[str, Atoms, int, int]]:
    si4 = Atoms(
        symbols="Si4",
        pbc=True,
        cell=np.diag([5.43, 5.43, 5.43]),
        positions=[
            [0, 0, 0],
            [1.36, 1.36, 1.36],
            [2.72, 2.72, 0],
            [4.08, 4.08, 1.36],
        ],
    )
    systems = [
        ("Si4 (symmetric)", si4, 0, 1),
        ("h2o", molecule("H2O"), 0, 1),
        ("ethanol", molecule("CH3CH2OH"), 0, 1),
        ("water_box_2", water_box(2), 0, 1),
        ("water_box_8", water_box(8), 0, 1),
        ("NaCl_bulk", bulk("NaCl", crystalstructure="rocksalt", a=5.64), 0, 1),
        ("NaCl_dimer", Atoms(["Na", "Cl"], positions=[[0, 0, 0], [2.4, 0, 0]]), 0, 1),
        ("Na+ ion", make_ion("Na", 1), 1, 1),
        ("Cl- ion", make_ion("Cl", -1), -1, 1),
    ]
    for _, atoms, charge, spin in systems:
        atoms.info["charge"] = charge
        atoms.info["spin"] = spin
    return systems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--energy-tol", type=float, default=1e-7, help="Max |ΔE| in eV to PASS")
    args = parser.parse_args()

    device = torch.device(args.device)
    master, adapter = pretrained.orbmol_v2(device=device, compile=False)
    master.eval()
    systems = build_systems()

    print(f"\n{'system':>18s}  {'q':>3s}  {'n':>3s}  {'|ΔE| eV':>12s}  {'|Δf|':>12s}  verdict")
    fail_count = 0
    for name, atoms, charge, _ in systems:
        eager_m = copy.deepcopy(master).double().to(device)
        eager_m.eval()
        compiled_m = copy.deepcopy(master).double().to(device)
        compiled_m.eval()
        compiled_m.compile(mode="default", dynamic=True)
        b1 = adapter.from_ase_atoms(atoms=atoms, device=device)
        b1 = b1.to(device=device, dtype=torch.float64)
        b2 = adapter.from_ase_atoms(atoms=atoms, device=device)
        b2 = b2.to(device=device, dtype=torch.float64)
        eager_out = eager_m.predict(b1)  # type: ignore[operator]
        compiled_out = compiled_m.predict(b2)  # type: ignore[operator]
        force_key = "grad_forces" if "grad_forces" in eager_out else "forces"
        de = (eager_out["energy"] - compiled_out["energy"]).abs().max().item()
        df = (eager_out[force_key] - compiled_out[force_key]).abs().max().item()
        verdict = "PASS" if de < args.energy_tol else "FAIL"
        if verdict == "FAIL":
            fail_count += 1
        print(f"{name:>18s}  {charge:>+3d}  {len(atoms):>3d}  {de:>12.3e}  {df:>12.3e}  {verdict}")

    if fail_count:
        print(f"\n{fail_count} systems FAILED, compiled predictions differ from eager.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
