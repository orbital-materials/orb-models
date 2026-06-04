import ase
import pytest
import torch
from nvalchemiops.torch.interactions.electrostatics import estimate_pme_parameters

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.atoms.featurization import rotation_from_generator
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.models.coulomb_module import (
    CoulombModule,
    _fully_connected_senders_receivers,
)
from orb_models.forcefield.models.forcefield_utils import torch_full_3x3_to_voigt_6_stress


def _get_coulomb_module() -> CoulombModule:
    coulomb = CoulombModule(pme_accuracy=1e-6)
    coulomb.eval()
    return coulomb


def _make_batch(atoms_list: list[ase.Atoms]) -> AtomGraphs:
    adapter = ForcefieldAtomsAdapter(radius=6.0, max_num_neighbors=20)
    return AtomGraphs.batch([adapter.from_ase_atoms(a) for a in atoms_list])


def _water_molecules() -> list[ase.Atoms]:
    return [
        ase.Atoms(
            "H2O",
            positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]],
            cell=[10, 10, 10],
            pbc=True,
        ),
        ase.Atoms(
            "H2O",
            positions=[[1, 1, 1], [1.96, 1, 1], [1.24, 1.93, 1]],
            cell=[10, 10, 10],
            pbc=True,
        ),
    ]


def _nonperiodic_molecules() -> list[ase.Atoms]:
    return [
        ase.Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]]),
        ase.Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]),
    ]


def _make_charge_fn(n_atoms: int, seed: int = 42) -> torch.nn.Module:
    """Differentiable charge model that mixes all positions: q = W @ flatten(r).

    Each charge depends on all atom positions (not just its own), which better
    tests cross-atom gradient flow (dq_i/dr_j for i != j).
    """
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * n_atoms, n_atoms, bias=False),
        torch.nn.Unflatten(1, (n_atoms, 1)),
    )


def _total_forces(
    energy: torch.Tensor,
    explicit_forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Compute total Coulomb forces = explicit + autograd(-dE/dr)."""
    (autograd_forces,) = torch.autograd.grad(energy.sum(), positions, create_graph=True)
    return explicit_forces - autograd_forces


def _virial_to_stress(virial: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 virial to Voigt-6 stress: stress = -virial / volume."""
    volume = torch.linalg.det(cell.view(-1, 3, 3)).abs()
    stress_3x3 = -virial / volume.view(-1, 1, 1)
    return torch_full_3x3_to_voigt_6_stress(stress_3x3)


def _total_stress(
    energy: torch.Tensor,
    explicit_stress: torch.Tensor,
    strain: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    """Compute total Coulomb stress = explicit + autograd component."""
    (autograd_virial,) = torch.autograd.grad(energy.sum(), strain, create_graph=True)
    autograd_stress = _virial_to_stress(autograd_virial.unsqueeze(0), cell).squeeze(0)
    return explicit_stress.squeeze(0) - autograd_stress


def _total_forces_and_stress(
    energy: torch.Tensor,
    explicit_forces: torch.Tensor,
    explicit_stress: torch.Tensor,
    positions: torch.Tensor,
    strain: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    (autograd_forces, autograd_virial) = torch.autograd.grad(
        outputs=[energy.sum()],
        inputs=[positions, strain],
        create_graph=training,
        retain_graph=training,
    )
    autograd_stress = _virial_to_stress(autograd_virial.unsqueeze(0), cell).squeeze(0)
    return explicit_forces + autograd_forces, explicit_stress.squeeze(0) + autograd_stress


def _apply_strain(
    positions: torch.Tensor,
    cell: torch.Tensor,
    strain: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply strain deformation: pos' = pos @ (I+ε)^T, cell' = cell @ (I+ε)^T."""
    deformation = torch.eye(3, dtype=strain.dtype, device=strain.device) + strain
    strained_positions = positions @ deformation.T
    strained_cell = (cell.squeeze(0) @ deformation.T).unsqueeze(0)
    return strained_positions, strained_cell


def _fd_forces(
    energy_fn,
    positions: torch.Tensor,
    delta: float = 1e-6,
) -> torch.Tensor:
    """Central finite-difference forces: F_i = -(E(r+d) - E(r-d)) / 2d."""
    forces = torch.zeros_like(positions)
    pos_base = positions.detach().clone()
    for i in range(pos_base.shape[0]):
        for j in range(3):
            pos_p = pos_base.clone()
            pos_p[i, j] += delta
            e_p = energy_fn(pos_p)

            pos_m = pos_base.clone()
            pos_m[i, j] -= delta
            e_m = energy_fn(pos_m)

            forces[i, j] = -(e_p - e_m) / (2 * delta)
    return forces


def _fd_virial(
    energy_fn,
    positions: torch.Tensor,
    cell: torch.Tensor,
    delta: float = 1e-6,
) -> torch.Tensor:
    """Central finite-difference virial: V_ij = -dE/dε_ij.

    Applies a uniform strain ε to both cell and Cartesian positions
    (preserving fractional coordinates) and numerically differentiates.
    """
    pos_base = positions.detach().clone()
    cell_base = cell.detach().clone()
    virial = torch.zeros(3, 3, dtype=positions.dtype, device=positions.device)

    for i in range(3):
        for j in range(3):
            eps = torch.zeros(3, 3, dtype=positions.dtype, device=positions.device)
            eps[i, j] = delta
            pos_p, cell_p = _apply_strain(pos_base, cell_base, eps)
            e_p = energy_fn(pos_p, cell_p)

            pos_m, cell_m = _apply_strain(pos_base, cell_base, -eps)
            e_m = energy_fn(pos_m, cell_m)

            virial[i, j] = -(e_p - e_m) / (2 * delta)
    return virial.unsqueeze(0)  # (1, 3, 3)


def _fd_stress(
    energy_fn,
    positions: torch.Tensor,
    cell: torch.Tensor,
    delta: float = 1e-6,
) -> torch.Tensor:
    """Central finite-difference stress in Voigt-6 notation."""
    virial = _fd_virial(energy_fn, positions, cell, delta)
    return _virial_to_stress(virial, cell).squeeze(0)


class TestNonPeriodic:
    """Direct Coulomb sum for non-periodic systems."""

    def test_shapes_and_finiteness(self):
        batch = _make_batch(_nonperiodic_molecules())
        n_atoms = batch.node_features["positions"].shape[0]
        charges = torch.randn(n_atoms, 1)

        coulomb = CoulombModule(direct_coulomb_erf_damping_sigma=1.0)
        energy, explicit_forces, explicit_stress = coulomb(charges, batch)

        assert energy.shape == (2,)
        assert torch.isfinite(energy).all()
        assert energy.abs().sum() > 0
        assert explicit_forces.shape == (n_atoms, 3)
        assert explicit_stress.shape == (2, 6)

    def test_zero_charges_zero_energy(self):
        batch = _make_batch(_nonperiodic_molecules())
        n_atoms = batch.node_features["positions"].shape[0]
        charges = torch.zeros(n_atoms, 1)

        coulomb = CoulombModule(direct_coulomb_erf_damping_sigma=1.0)
        energy, _, _ = coulomb(charges, batch)

        assert (energy.abs() < 1e-10).all()

    def test_opposite_charges_attract(self):
        atoms = ase.Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
        batch = _make_batch([atoms])
        charges = torch.tensor([[1.0], [-1.0]])

        coulomb = CoulombModule(direct_coulomb_erf_damping_sigma=1.0)
        energy, _, _ = coulomb(charges, batch)

        assert energy.item() < 0, "Opposite charges should attract (negative energy)"

    def test_dEdq_nonzero(self):
        batch = _make_batch(_nonperiodic_molecules())
        n_atoms = batch.node_features["positions"].shape[0]
        charges = torch.randn(n_atoms, 1, requires_grad=True)

        coulomb = CoulombModule(direct_coulomb_erf_damping_sigma=1.0)
        energy, _, _ = coulomb(charges, batch)

        grad = torch.autograd.grad(energy.sum(), charges)[0]
        assert grad is not None
        assert grad.abs().sum() > 0

    def test_forces_differentiable_wrt_charges(self):
        """Total forces are differentiable w.r.t. charge model parameters."""
        atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]])
        batch = _make_batch([atoms])
        charge_fn = _make_charge_fn(n_atoms=3)
        coulomb = CoulombModule(direct_coulomb_erf_damping_sigma=1.0)

        positions = batch.node_features["positions"]
        positions.requires_grad_(True)
        charges = charge_fn(positions.unsqueeze(0)).squeeze(0)
        energy, explicit_forces, _ = coulomb(charges, batch)
        total_f = _total_forces(energy, explicit_forces, positions)

        # Backprop through total forces to charge model weights
        total_f.sum().backward()
        linear = charge_fn[1]  # Linear layer inside Sequential
        assert linear.weight.grad is not None
        assert linear.weight.grad.abs().sum() > 0

    def test_total_forces(self):
        """Total force (explicit + autograd) with q=q(r) matches finite difference.

        Use double precision to avoid numerical instability in finite difference.
        """
        atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]])
        batch = _make_batch([atoms])
        charge_fn = _make_charge_fn(n_atoms=3).double()
        coulomb = CoulombModule(direct_coulomb_erf_damping_sigma=1.0)

        # Autograd total forces
        positions = batch.node_features["positions"].double()
        batch.node_features["positions"] = positions
        positions.requires_grad_(True)
        charges = charge_fn(positions.unsqueeze(0)).squeeze(0)
        energy, explicit_forces, _ = coulomb(charges, batch)
        forces_auto = _total_forces(energy, explicit_forces, positions)

        def energy_fn(pos):
            batch.node_features["positions"] = pos
            q = charge_fn(pos.unsqueeze(0)).squeeze(0)
            e, _, _ = coulomb(q, batch)
            return e.sum()

        forces_fd = _fd_forces(energy_fn, positions)
        torch.testing.assert_close(forces_auto.detach(), forces_fd, atol=1e-5, rtol=1e-5)


class TestPeriodic:
    """PME for periodic systems — energy, forces, stress, gradient flow."""

    def test_shapes_and_finiteness(self):
        batch = _make_batch(_water_molecules())
        n_atoms = batch.node_features["positions"].shape[0]
        charges = torch.randn(n_atoms, 1)

        coulomb = _get_coulomb_module()
        energy, explicit_forces, explicit_stress = coulomb(charges, batch)

        assert energy.shape == (2,)
        assert torch.isfinite(energy).all()
        assert explicit_forces.shape == (n_atoms, 3)
        assert explicit_stress.shape == (2, 6)
        assert explicit_forces.abs().sum() > 0
        assert explicit_stress.abs().sum() > 0

    def test_opposite_charges_attract(self):
        atoms = ase.Atoms(
            "NaCl",
            positions=[[0, 0, 0], [2, 0, 0]],
            cell=[20, 20, 20],
            pbc=True,
        )
        batch = _make_batch([atoms])
        charges = torch.tensor([[1.0], [-1.0]])

        coulomb = CoulombModule(pme_accuracy=1e-6)
        energy, _, _ = coulomb(charges, batch)

        assert energy.item() < 0, "Opposite charges should attract (negative energy)"

    def test_vacuum_gap_convergence(self):
        """Periodic Ewald in large box converges to non-periodic direct sum."""
        torch.manual_seed(42)
        positions = [[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]]
        charges_val = torch.randn(3, 1)

        # Non-periodic reference
        atoms_np = ase.Atoms("H2O", positions=positions)
        batch_np = _make_batch([atoms_np])
        coulomb = CoulombModule(direct_coulomb_erf_damping_sigma=0.5)
        e_nonperiodic, _, _ = coulomb(charges_val, batch_np)

        # Periodic with increasing box size
        energies = []
        for box_size in [15.0, 25.0, 40.0, 100.0]:
            atoms_p = ase.Atoms("H2O", positions=positions, cell=[box_size] * 3, pbc=True)
            batch_p = _make_batch([atoms_p])
            e, _, _ = CoulombModule(direct_coulomb_erf_damping_sigma=0.5, pme_accuracy=1e-6)(
                charges_val, batch_p
            )
            energies.append(e.item())

        errors = [abs(e - e_nonperiodic.item()) for e in energies]
        assert errors == sorted(errors, reverse=True), (
            f"Ewald should monotonically converge to direct sum: errors={errors}"
        )

    def test_dEdq_nonzero(self):
        batch = _make_batch(_water_molecules())
        n_atoms = batch.node_features["positions"].shape[0]
        charges = torch.randn(n_atoms, 1, requires_grad=True)

        coulomb = _get_coulomb_module()
        energy, _, _ = coulomb(charges, batch)

        grad = torch.autograd.grad(energy.sum(), charges)[0]
        assert grad is not None
        assert grad.abs().sum() > 0

    def test_explicit_stress_nonzero(self):
        batch = _make_batch(_water_molecules())
        n_atoms = batch.node_features["positions"].shape[0]
        torch.manual_seed(42)
        charges = torch.randn(n_atoms, 1)

        coulomb = _get_coulomb_module()
        _, _, explicit_stress = coulomb(charges, batch)

        assert explicit_stress.abs().sum() > 0

    def test_spatial_forces(self):
        """Spatial forces (fixed charges) match finite difference.

        With fixed charges, positions are detached inside PME, so the spatial
        component lives entirely in explicit_forces (no autograd component).

        Use double precision to avoid numerical instability in finite difference.
        """
        atoms = ase.Atoms(
            "NaCl",
            positions=[[2, 5, 5], [8, 5, 5]],
            cell=[10, 10, 10],
            pbc=True,
        )

        batch = _make_batch([atoms])
        batch.node_features["positions"] = batch.node_features["positions"].double()
        batch.system_features["cell"] = batch.system_features["cell"].double()
        charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
        coulomb = CoulombModule(pme_accuracy=1e-6)

        _, explicit_forces, _ = coulomb(charges, batch)

        pos_base = batch.node_features["positions"]

        def force_energy_fn(pos):
            batch.node_features["positions"] = pos
            e, _, _ = coulomb(charges, batch)
            return e.sum()

        forces_fd = _fd_forces(force_energy_fn, pos_base)
        torch.testing.assert_close(explicit_forces, forces_fd, atol=1e-5, rtol=1e-5)

    def test_spatial_stress(self):
        """Spatial stress (fixed charges) matches finite difference.

        With fixed charges, positions are detached inside PME, so the spatial
        component lives entirely in explicit_stress (no autograd component).

        Use double precision to avoid numerical instability in finite difference.
        """
        atoms = ase.Atoms(
            "NaCl",
            positions=[[2, 5, 5], [8, 5, 5]],
            cell=[10, 10, 10],
            pbc=True,
        )

        batch = _make_batch([atoms])
        batch.node_features["positions"] = batch.node_features["positions"].double()
        batch.system_features["cell"] = batch.system_features["cell"].double()
        charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
        coulomb = CoulombModule(pme_accuracy=1e-6)

        _, _, explicit_stress = coulomb(charges, batch)

        pos_base = batch.node_features["positions"]
        cell_base = batch.system_features["cell"]

        def virial_energy_fn(pos, cell):
            batch.node_features["positions"] = pos
            batch.system_features["cell"] = cell
            e, _, _ = coulomb(charges, batch)
            return e.sum()

        stress_fd = _fd_stress(virial_energy_fn, pos_base, cell_base)
        torch.testing.assert_close(explicit_stress.squeeze(0), stress_fd, atol=1e-5, rtol=1e-5)

    def test_spatial_forces_not_differentiable_wrt_charges_model(self):
        """Explicit forces are *not* differentiable w.r.t. charge model."""
        batch = _make_batch(_water_molecules()[:1])
        n_atoms = batch.node_features["positions"].shape[0]
        charge_fn = _make_charge_fn(n_atoms=n_atoms)
        charge_fn.train()
        coulomb = _get_coulomb_module()
        coulomb.train()

        positions = batch.node_features["positions"].clone().requires_grad_(True)
        batch.node_features["positions"] = positions
        charges = charge_fn(positions.unsqueeze(0)).squeeze(0)
        energy, explicit_forces, explicit_stress = coulomb(charges, batch)

        assert explicit_forces.requires_grad is False, (
            "explicit_forces should not be differentiable"
        )

    def test_spatial_stress_not_differentiable_wrt_charges_model(self):
        """Explicit stress is *not* differentiable w.r.t. charge model."""
        batch = _make_batch(_water_molecules()[:1])
        n_atoms = batch.node_features["positions"].shape[0]
        charge_fn = _make_charge_fn(n_atoms=n_atoms)
        charge_fn.train()
        coulomb = _get_coulomb_module()
        coulomb.train()

        positions = batch.node_features["positions"].clone().requires_grad_(True)
        batch.node_features["positions"] = positions
        charges = charge_fn(positions.unsqueeze(0)).squeeze(0)
        energy, explicit_forces, explicit_stress = coulomb(charges, batch)

        assert explicit_stress.requires_grad is False, (
            "explicit_stress should not be differentiable"
        )

    def test_total_forces_differentiable_wrt_charges_model(self):
        """Total forces are differentiable w.r.t. charge model."""
        batch = _make_batch(_water_molecules()[:1])
        n_atoms = batch.node_features["positions"].shape[0]
        charge_fn = _make_charge_fn(n_atoms=n_atoms)
        charge_fn.train()
        coulomb = _get_coulomb_module()
        coulomb.train()

        positions = batch.node_features["positions"].clone().requires_grad_(True)
        batch.node_features["positions"] = positions
        charges = charge_fn(positions.unsqueeze(0)).squeeze(0)
        energy, explicit_forces, explicit_stress = coulomb(charges, batch)

        total_forces = _total_forces(energy, explicit_forces, positions)

        linear = charge_fn[1]
        total_forces.sum().backward()
        assert linear.weight.grad is not None, (
            "total_forces should be differentiable w.r.t. charge model"
        )
        assert linear.weight.grad.abs().sum() > 0

    def test_total_stress_differentiable_wrt_charges_model(self):
        """Total stress is differentiable w.r.t. charge model."""
        batch = _make_batch(_water_molecules()[:1])
        n_atoms = batch.node_features["positions"].shape[0]
        charge_fn = _make_charge_fn(n_atoms=n_atoms)
        charge_fn.train()
        coulomb = _get_coulomb_module()
        coulomb.train()

        pos_base = batch.node_features["positions"].clone()
        cell_base = batch.system_features["cell"]

        strain = torch.zeros(3, 3, dtype=pos_base.dtype, requires_grad=True)
        strained_pos, strained_cell = _apply_strain(pos_base, cell_base, strain)
        batch.node_features["positions"] = strained_pos
        batch.system_features["cell"] = strained_cell

        charges = charge_fn(strained_pos.unsqueeze(0)).squeeze(0)
        energy, explicit_forces, explicit_stress = coulomb(charges, batch)

        total_stress = _total_stress(energy, explicit_stress, strain, strained_cell)

        linear = charge_fn[1]
        total_stress.sum().backward()
        assert linear.weight.grad is not None, (
            "total_stress should be differentiable w.r.t. charge model"
        )
        assert linear.weight.grad.abs().sum() > 0

    def test_combined_loss_differentiable_wrt_charges_model(self):
        """Combined loss on energy, total forces, and total virial is differentiable w.r.t. charge model.

        This is mostly just a smoke test to ensure that this works.
        """
        batch = _make_batch(_water_molecules()[:1])
        n_atoms = batch.node_features["positions"].shape[0]
        charge_fn = _make_charge_fn(n_atoms=n_atoms)
        charge_fn.train()
        coulomb = _get_coulomb_module()
        coulomb.train()

        pos_base = batch.node_features["positions"].clone()
        cell_base = batch.system_features["cell"]

        # Apply differentiable strain for virial
        strain = torch.zeros(3, 3, dtype=pos_base.dtype, requires_grad=True)
        strained_pos, strained_cell = _apply_strain(pos_base, cell_base, strain)
        positions = strained_pos.requires_grad_(True)
        batch.node_features["positions"] = positions
        batch.system_features["cell"] = strained_cell

        charges = charge_fn(positions.unsqueeze(0)).squeeze(0)
        energy, explicit_forces, explicit_stress = coulomb(charges, batch)

        total_forces, total_stress = _total_forces_and_stress(
            energy,
            explicit_forces,
            explicit_stress,
            positions,
            strain,
            strained_cell,
            training=True,
        )

        linear = charge_fn[1]
        loss = energy.sum() + total_forces.sum() + total_stress.sum()
        loss.backward()

        assert linear.weight.grad is not None, (
            "combined loss should be differentiable w.r.t. charge model"
        )
        assert linear.weight.grad.abs().sum() > 0

    def test_charge_equilibration_gradient_flows(self):
        """dE/dq and dE/dr through q(r) are nonzero in eval mode."""
        atoms = ase.Atoms(
            "NaCl",
            positions=[[2, 5, 5], [8, 5, 5]],
            cell=[10, 10, 10],
            pbc=True,
        )
        batch = _make_batch([atoms])
        charge_fn = _make_charge_fn(n_atoms=2)
        coulomb = _get_coulomb_module()

        positions = batch.node_features["positions"]
        positions.requires_grad_(True)
        charges = charge_fn(positions.unsqueeze(0)).squeeze(0)
        energy, _, _ = coulomb(charges, batch)

        # dE/dq should be nonzero
        dEdq = torch.autograd.grad(energy.sum(), charges, retain_graph=True)[0]
        assert dEdq.abs().sum() > 0, "dE/dq should be nonzero"

        # dE/dr through q(r) chain rule should be nonzero
        dEdr = torch.autograd.grad(energy.sum(), positions)[0]
        assert dEdr.abs().sum() > 0, "dE/dr through q(r) should be nonzero"

    def test_total_forces(self):
        """Total force with q=q(r) matches finite difference.

        Use double precision to avoid numerical instability in finite difference.
        """
        atoms = ase.Atoms(
            "NaCl",
            positions=[[2, 5, 5], [8, 5, 5]],
            cell=[10, 10, 10],
            pbc=True,
        )
        batch = _make_batch([atoms])
        batch.node_features["positions"] = batch.node_features["positions"].double()
        batch.system_features["cell"] = batch.system_features["cell"].double()
        charge_fn = _make_charge_fn(n_atoms=2).double()
        coulomb = _get_coulomb_module()

        positions = batch.node_features["positions"]
        positions.requires_grad_(True)
        charges = charge_fn(positions.unsqueeze(0)).squeeze(0)
        energy, explicit_forces, _ = coulomb(charges, batch)
        forces_auto = _total_forces(energy, explicit_forces, positions)

        def energy_fn(pos):
            batch.node_features["positions"] = pos
            q = charge_fn(pos.unsqueeze(0)).squeeze(0)
            e, _, _ = coulomb(q, batch)
            return e.sum()

        forces_fd = _fd_forces(energy_fn, positions)
        torch.testing.assert_close(forces_auto.detach(), forces_fd, atol=1e-5, rtol=1e-5)

    def test_total_stress(self):
        """Total stress with q=q(r) matches finite difference.

        Use double precision to avoid numerical instability in finite difference.
        Pin PME parameters so FD perturbations don't re-estimate them (discrete
        jumps in mesh_dimensions would make the energy non-smooth w.r.t. strain).
        """

        atoms = ase.Atoms(
            "NaCl",
            positions=[[8, 9, 10], [12, 11, 10.5]],
            cell=[20, 20, 20],
            pbc=True,
        )
        batch = _make_batch([atoms])
        batch.node_features["positions"] = batch.node_features["positions"].double()
        batch.system_features["cell"] = batch.system_features["cell"].double()
        charge_fn = _make_charge_fn(n_atoms=2).double()
        coulomb = _get_coulomb_module()

        pos_base = batch.node_features["positions"]
        cell_base = batch.system_features["cell"]

        # Pin all PME parameters so FD perturbations don't re-estimate them.
        params = estimate_pme_parameters(
            pos_base,
            cell_base,
            batch_idx=torch.zeros(pos_base.shape[0], dtype=torch.int32),
            accuracy=coulomb.pme_accuracy,
        )
        fixed_pme = {
            "pme_alpha": params.alpha.item(),
            "pme_cutoff": params.real_space_cutoff.max().item(),
            "pme_mesh_dimensions": tuple(params.mesh_dimensions),
        }

        # Apply a differentiable strain so autograd can compute dE/dε
        strain = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
        strained_pos, strained_cell = _apply_strain(pos_base, cell_base, strain)
        batch.node_features["positions"] = strained_pos
        batch.system_features["cell"] = strained_cell

        charges = charge_fn(strained_pos.unsqueeze(0)).squeeze(0)
        energy, _, explicit_stress = coulomb(charges, batch, kwargs=fixed_pme)
        stress_auto = _total_stress(energy, explicit_stress, strain, strained_cell)

        def virial_energy_fn(pos, cell):
            batch.node_features["positions"] = pos
            batch.system_features["cell"] = cell
            q = charge_fn(pos.unsqueeze(0)).squeeze(0)
            e, _, _ = coulomb(q, batch, kwargs=fixed_pme)
            return e.sum()

        stress_fd = _fd_stress(virial_energy_fn, pos_base, cell_base)
        torch.testing.assert_close(stress_auto.detach(), stress_fd, atol=1e-5, rtol=1e-5)

    def test_positions_detached_both_modes(self):
        """Positions are detached inside PME in both train/eval modes.

        Spatial forces come only from explicit_forces, not through autograd on energy.
        """
        batch = _make_batch(_water_molecules()[:1])
        n_atoms = batch.node_features["positions"].shape[0]
        for mode in ["train", "eval"]:
            coulomb = _get_coulomb_module()
            getattr(coulomb, mode)()

            positions = batch.node_features["positions"].clone().requires_grad_(True)
            batch.node_features["positions"] = positions
            # Charges require grad so the energy stays in the compute graph
            charges = torch.randn(n_atoms, 1, requires_grad=True)
            energy, explicit_forces, _ = coulomb(charges, batch)

            # dE/dr through the energy should be None (positions detached inside PME)
            grad = torch.autograd.grad(
                energy.sum(), positions, allow_unused=True, retain_graph=True
            )[0]
            assert grad is None or (grad.abs() < 1e-10).all(), (
                f"In {mode} mode, positions should be detached inside PME"
            )

            # But explicit_forces should be nonzero
            assert explicit_forces.abs().sum() > 0, (
                f"In {mode} mode, explicit_forces should carry the spatial component"
            )

    def test_cell_detached_both_modes(self):
        """Cell is detached inside PME in both train/eval modes.

        Stress contributions come only from explicit_stress, not through autograd
        on energy w.r.t. strain.
        """
        batch = _make_batch(_water_molecules()[:1])
        n_atoms = batch.node_features["positions"].shape[0]
        for mode in ["train", "eval"]:
            coulomb = _get_coulomb_module()
            getattr(coulomb, mode)()

            pos = batch.node_features["positions"].clone()
            cell = batch.system_features["cell"].clone()
            strain = torch.zeros(3, 3, requires_grad=True)
            strained_pos, strained_cell = _apply_strain(pos, cell, strain)
            batch.node_features["positions"] = strained_pos
            batch.system_features["cell"] = strained_cell

            charges = torch.randn(n_atoms, 1, requires_grad=True)
            energy, _, explicit_stress = coulomb(charges, batch)

            # dE/dε through the energy should be None (cell detached inside PME)
            grad = torch.autograd.grad(energy.sum(), strain, allow_unused=True, retain_graph=True)[
                0
            ]
            assert grad is None or (grad.abs() < 1e-10).all(), (
                f"In {mode} mode, cell should be detached inside PME"
            )

            # But explicit_stress should be nonzero
            assert explicit_stress.abs().sum() > 0, (
                f"In {mode} mode, explicit_stress should carry the stress component"
            )


class TestBatching:
    """Batched computation matches individual system computation."""

    def test_nonperiodic_batched_vs_individual(self):
        molecules = _nonperiodic_molecules()
        coulomb = _get_coulomb_module()

        individual_energies = []
        all_charges = []
        for mol in molecules:
            n = len(mol)
            q = torch.randn(n, 1)
            all_charges.append(q)
            e, _, _ = coulomb(q, _make_batch([mol]))
            individual_energies.append(e.item())

        batch = _make_batch(molecules)
        charges = torch.cat(all_charges, dim=0)
        batched_energies, _, _ = coulomb(charges, batch)

        for i, (batched, individual) in enumerate(
            zip(batched_energies.tolist(), individual_energies, strict=True)
        ):
            assert batched == pytest.approx(individual, rel=1e-5), (
                f"System {i}: batched={batched}, individual={individual}"
            )

    def test_periodic_batched_vs_individual(self):
        """Energy, explicit_forces, and stress match between batched and individual."""
        waters = _water_molecules()
        coulomb = _get_coulomb_module()

        individual_energies = []
        individual_forces = []
        individual_stresses = []
        all_charges = []
        for mol in waters:
            n = len(mol)
            q = torch.randn(n, 1)
            all_charges.append(q)
            e, f, v = coulomb(q, _make_batch([mol]))
            individual_energies.append(e.item())
            individual_forces.append(f)
            individual_stresses.append(v)

        batch = _make_batch(waters)
        charges = torch.cat(all_charges, dim=0)
        batched_energies, batched_forces, batched_stress = coulomb(charges, batch)

        # Energy
        for i, (batched, individual) in enumerate(
            zip(batched_energies.tolist(), individual_energies, strict=True)
        ):
            assert batched == pytest.approx(individual, rel=1e-5), (
                f"Energy system {i}: batched={batched}, individual={individual}"
            )

        # Forces (slice by n_node)
        n_node = batch.n_node
        offset = 0
        for i, n in enumerate(n_node):
            torch.testing.assert_close(
                batched_forces[offset : offset + n],
                individual_forces[i],
                atol=1e-3,
                rtol=1e-3,
            )
            offset += n

        # Stress
        for i in range(len(waters)):
            torch.testing.assert_close(
                batched_stress[i : i + 1],
                individual_stresses[i],
                atol=1e-3,
                rtol=1e-3,
            )

    def test_mixed_periodic_nonperiodic(self):
        """Mixed batch: each system matches its individual computation."""
        periodic = ase.Atoms(
            "H2O",
            positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]],
            cell=[10, 10, 10],
            pbc=True,
        )
        nonperiodic = ase.Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        coulomb = _get_coulomb_module()

        q_periodic = torch.randn(3, 1)
        q_nonperiodic = torch.randn(2, 1)

        e_periodic, f_periodic, _ = coulomb(q_periodic, _make_batch([periodic]))
        e_nonperiodic, f_nonperiodic, _ = coulomb(q_nonperiodic, _make_batch([nonperiodic]))

        # Mixed batch
        mixed_batch = _make_batch([periodic, nonperiodic])
        mixed_charges = torch.cat([q_periodic, q_nonperiodic], dim=0)
        mixed_energy, mixed_forces, _ = coulomb(mixed_charges, mixed_batch)

        assert mixed_energy.shape == (2,)
        torch.testing.assert_close(mixed_energy[0], e_periodic[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(mixed_energy[1], e_nonperiodic[0], atol=1e-5, rtol=1e-5)

        # Non-periodic explicit forces should be zero, periodic should be nonzero
        assert mixed_forces[:3].abs().sum() > 0, "Periodic explicit forces should be nonzero"
        assert (mixed_forces[3:].abs() < 1e-10).all(), "Non-periodic explicit forces should be zero"


class TestRotationalGradients:
    """Rotational gradients from CoulombModule energy."""

    @pytest.mark.parametrize(
        "atoms_fn",
        [lambda: _water_molecules()[:1], lambda: _nonperiodic_molecules()[:1]],
        ids=["periodic", "nonperiodic"],
    )
    def test_rotational_grad_zero_with_fixed_charges(self, atoms_fn):
        """dE/d(generator) is zero when charges are independent of geometry.

        Sanity check to ensure that we're not missing a contribution to equigrad gradients,
        since nvalchemiops detaches positions and cell from the graph.

        Coulomb energy is rotationally invariant, so rotating positions (and cell)
        via the generator should not change energy — gradient must vanish.
        """
        batch = _make_batch(atoms_fn())
        n_atoms = batch.node_features["positions"].shape[0]
        coulomb = _get_coulomb_module()

        generator = torch.zeros(1, 3, 3, requires_grad=True)
        rotation = rotation_from_generator(generator)

        pos = batch.node_features["positions"].clone()
        batch.node_features["positions"] = pos @ rotation.squeeze(0)

        cell = batch.system_features["cell"]
        if cell.abs().sum() > 0:
            batch.system_features["cell"] = cell @ rotation

        charges = torch.randn(n_atoms, 1, requires_grad=True)
        # For this test, we disable hybrid forces, so that nvalchemiops does not detach positions and cell from the graph.
        # This allows us to check that the rotational gradient is zero when charges are fixed.
        # We do this because our normal computation graph with hybrid_forces=True would detach the positions and cell
        # from the graph _and also_ compute_differentiable_edge_vectors does not set the rotated positions/cell in the AtomGraphs object,
        # hence their contributions to equigrad regularization would be lost.
        #
        # This check tests that we are not missing anything (rotational gradients are 0)
        energy, _, _ = coulomb(charges, batch, kwargs={"pme_hybrid_forces": False})

        grad = torch.autograd.grad(energy.sum(), generator, allow_unused=True, retain_graph=True)[0]
        assert (grad.abs() < 1e-7).all(), (
            f"Rotational gradient should be zero with fixed charges, got {grad}"
        )

    @pytest.mark.parametrize(
        "atoms_fn",
        [lambda: _water_molecules()[:1], lambda: _nonperiodic_molecules()[:1]],
        ids=["periodic", "nonperiodic"],
    )
    def test_rotational_grad_nonzero_with_position_dependent_charges(self, atoms_fn):
        """dE/d(generator) is nonzero when charges depend on rotated positions.

        Gradient flows through energy -> charges -> generator, breaking rotational invariance.
        """
        batch = _make_batch(atoms_fn())
        n_atoms = batch.node_features["positions"].shape[0]
        coulomb = _get_coulomb_module()
        charge_fn = _make_charge_fn(n_atoms)

        generator = torch.zeros(1, 3, 3, requires_grad=True)
        rotation = rotation_from_generator(generator)

        pos = batch.node_features["positions"].clone()
        rotated_pos = pos @ rotation.squeeze(0)
        batch.node_features["positions"] = rotated_pos

        cell = batch.system_features["cell"]
        if cell.abs().sum() > 0:
            batch.system_features["cell"] = cell @ rotation

        charges = charge_fn(rotated_pos.unsqueeze(0)).squeeze(0)
        energy, _, _ = coulomb(charges, batch)

        grad = torch.autograd.grad(energy.sum(), generator)[0]
        assert grad.abs().sum() > 0.5, (
            "Rotational gradient should be nonzero with position-dependent charges"
        )


class TestFullyConnectedSendersReceivers:
    """Tests for _fully_connected_senders_receivers helper."""

    def test_single_system(self):
        n_node = torch.tensor([3])
        senders, receivers = _fully_connected_senders_receivers(n_node, torch.device("cpu"))

        # 3 atoms -> 3*2 = 6 directed pairs (no self-loops)
        assert senders.shape[0] == 6
        assert receivers.shape[0] == 6

        pairs = set(zip(senders.tolist(), receivers.tolist(), strict=True))
        expected = {(i, j) for i in range(3) for j in range(3) if i != j}
        assert pairs == expected

    def test_no_self_loops(self):
        n_node = torch.tensor([4])
        senders, receivers = _fully_connected_senders_receivers(n_node, torch.device("cpu"))
        assert (senders != receivers).all()

    def test_batched_systems(self):
        n_node = torch.tensor([2, 3])
        senders, receivers = _fully_connected_senders_receivers(n_node, torch.device("cpu"))

        # System 0: atoms [0,1] -> 2 pairs; System 1: atoms [2,3,4] -> 6 pairs
        assert senders.shape[0] == 2 + 6

        pairs = set(zip(senders.tolist(), receivers.tolist(), strict=True))
        expected_sys0 = {(0, 1), (1, 0)}
        expected_sys1 = {(i, j) for i in range(2, 5) for j in range(2, 5) if i != j}
        assert pairs == expected_sys0 | expected_sys1

    def test_no_cross_system_pairs(self):
        n_node = torch.tensor([2, 2])
        senders, receivers = _fully_connected_senders_receivers(n_node, torch.device("cpu"))

        # Atoms 0,1 in system 0; atoms 2,3 in system 1 — no cross-system pairs
        for s, r in zip(senders.tolist(), receivers.tolist(), strict=True):
            assert (s < 2 and r < 2) or (s >= 2 and r >= 2)

    def test_single_atom_system(self):
        n_node = torch.tensor([1])
        senders, receivers = _fully_connected_senders_receivers(n_node, torch.device("cpu"))
        assert senders.shape[0] == 0
        assert receivers.shape[0] == 0


class TestEdgeCases:
    def test_single_atom(self):
        # Non-periodic: zero energy (no pairs)
        atoms_np = ase.Atoms("H", positions=[[0, 0, 0]])
        batch_np = _make_batch([atoms_np])
        charges = torch.tensor([[1.0]])
        coulomb = _get_coulomb_module()
        energy, _, _ = coulomb(charges, batch_np)
        assert energy.item() == pytest.approx(0.0, abs=1e-6)

        # Periodic: finite energy
        atoms_p = ase.Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        batch_p = _make_batch([atoms_p])
        coulomb_p = _get_coulomb_module()
        energy_p, _, _ = coulomb_p(charges, batch_p)
        assert torch.isfinite(energy_p).all()

    def test_mixed_system_sizes(self):
        atoms_list = [
            ase.Atoms("H", positions=[[0, 0, 0]]),
            ase.Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]]),
            ase.Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]]),
        ]
        batch = _make_batch(atoms_list)
        n_atoms = batch.node_features["positions"].shape[0]
        charges = torch.randn(n_atoms, 1)

        coulomb = _get_coulomb_module()
        energy, explicit_forces, explicit_stress = coulomb(charges, batch)

        assert energy.shape == (3,)
        assert explicit_forces.shape == (n_atoms, 3)
        assert explicit_stress.shape == (3, 6)
        assert torch.isfinite(energy).all()

    def test_partial_pbc_raises(self):
        # Build a valid periodic batch, then override pbc to partial
        atoms = ase.Atoms(
            "H2",
            positions=[[0, 0, 0], [1, 0, 0]],
            cell=[10, 10, 10],
            pbc=True,
        )
        batch = _make_batch([atoms])
        batch.system_features["pbc"] = torch.tensor([[True, False, False]])
        charges = torch.tensor([[1.0], [-1.0]])

        coulomb = _get_coulomb_module()
        with pytest.raises(NotImplementedError, match="1D and 2D PBC"):
            coulomb(charges, batch)
