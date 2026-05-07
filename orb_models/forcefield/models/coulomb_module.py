import math
from typing import Any

import torch
from nvalchemiops.torch.interactions.electrostatics import (
    estimate_pme_parameters,
    particle_mesh_ewald,
)

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.atoms.graph_featurization import _compute_neighbor_list_with_fallback
from orb_models.common.models.segment_ops import aggregate_nodes, segment_sum

COULOMB_CONSTANT = 14.3996  # eV*A/e^2


class CoulombModule(torch.nn.Module):
    """Computes long-range electrostatic energy, forces, and virial from predicted charges.

    Returns per-system energy (E), spatial forces (−∂E/∂r), and spatial virial (−Σ_i r_i ⊗ f_i).
    Charge-equilibration forces and virial are obtained via autograd through the energy.
    Spatial forces and virial are zero for non-periodic systems (autograd handles them).

    Non-periodic — erf-damped direct Coulomb sum (fully differentiable):

        E = k/2 Σ_{i≠j} q_i q_j erf(r_ij / σ√2) / r_ij

    Periodic — Particle Mesh Ewald via nvalchemiops (explicit forces/virial,
    surrogate energy for charge gradients):

        E = E_real + E_reciprocal − E_self − E_background
        E_real        = 1/2 Σ_{i≠j} q_i q_j erfc(α r_ij / √2) / r_ij
        E_reciprocal  = FFT-based approximation
        E_self        = Σ_i (α / √(2π)) q_i²
        E_background  = (π / 2α²V) Q_total²

    where k = COULOMB_CONSTANT = 14.3996 eV·Å/e² (absorbed into scaled charges
    for the nvalchemiops call, which computes unitless q_i q_j / r).
    """

    coulomb_constant: torch.Tensor

    def __init__(
        self,
        direct_coulomb_erf_damping_sigma: float | None = None,
        pme_accuracy: float = 1e-6,
        pme_spline_order: int = 4,
    ):
        super().__init__()

        self.direct_coulomb_erf_damping_sigma = direct_coulomb_erf_damping_sigma
        self.pme_accuracy = pme_accuracy
        self.pme_spline_order = pme_spline_order

        self.register_buffer("coulomb_constant", torch.tensor(COULOMB_CONSTANT), persistent=True)

    def forward(
        self,
        latent_charges: torch.Tensor,
        batch: AtomGraphs,
        *,
        kwargs: dict[str, Any] = {},
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-system electrostatic energy, forces, and virial.

        For non-periodic systems we use the direct Coulomb sum; for periodic systems we use Particle Mesh Ewald.

        Args:
            latent_charges: (n_atoms, 1) predicted charges (must require grad).
            batch: AtomGraphs batch (positions, cell, pbc read from here).
            kwargs: Additional keyword arguments to _direct_coulomb and _particle_mesh_ewald. (Useful for testing.)
        """
        # Validate kwargs against accepted PME parameters.
        _valid_pme_kwargs = {"pme_cutoff", "pme_alpha", "pme_mesh_dimensions", "pme_hybrid_forces"}
        if kwargs:
            unknown = set(kwargs) - _valid_pme_kwargs
            if unknown:
                raise ValueError(
                    f"Unknown kwargs: {unknown}, expected a subset of {sorted(_valid_pme_kwargs)}"
                )
        pme_kwargs = {k: v for k, v in kwargs.items() if k in _valid_pme_kwargs}

        positions = batch.node_features["positions"]
        cell = batch.system_features["cell"]
        pbc = batch.system_features["pbc"]
        latent_charges = latent_charges.squeeze(-1)

        is_periodic = pbc.all(dim=-1)
        is_nonperiodic = (~pbc).all(dim=-1)
        if not (is_periodic | is_nonperiodic).all():
            raise NotImplementedError("1D and 2D PBC systems are not supported.")
        batch_idx = batch.node_batch_index
        n_node = batch.n_node
        n_systems = n_node.shape[0]
        n_atoms = positions.shape[0]

        energy = torch.zeros(n_systems, device=positions.device, dtype=positions.dtype)
        explicit_forces = torch.zeros(n_atoms, 3, device=positions.device, dtype=positions.dtype)
        explicit_virial = torch.zeros(
            n_systems, 3, 3, device=positions.device, dtype=positions.dtype
        )

        # Non-periodic systems
        np_sys_idx = is_nonperiodic.nonzero(as_tuple=True)[0]
        if np_sys_idx.shape[0] > 0:
            np_atom_mask = is_nonperiodic[batch_idx]
            np_charges = latent_charges[np_atom_mask]
            np_positions = positions[np_atom_mask]
            np_n_node = n_node[np_sys_idx]
            np_batch_idx = torch.repeat_interleave(
                torch.arange(np_sys_idx.shape[0], device=batch_idx.device), np_n_node
            )
            energy[np_sys_idx] = self._direct_coulomb(
                np_charges, np_positions, np_batch_idx, np_n_node, np_sys_idx.shape[0]
            )

        # Periodic systems
        p_sys_idx = is_periodic.nonzero(as_tuple=True)[0]
        if p_sys_idx.shape[0] > 0:
            p_atom_mask = is_periodic[batch_idx]
            p_charges = latent_charges[p_atom_mask]
            p_positions = positions[p_atom_mask]
            p_cell = cell[p_sys_idx]
            p_n_node = n_node[p_sys_idx]
            p_pbc = batch.pbc[p_sys_idx]
            p_batch_idx = torch.repeat_interleave(
                torch.arange(p_sys_idx.shape[0], device=batch_idx.device), p_n_node
            )
            p_energy, p_forces, p_virial = self._particle_mesh_ewald(
                p_charges, p_positions, p_cell, p_batch_idx, p_sys_idx.shape[0], p_pbc, **pme_kwargs
            )
            energy[p_sys_idx] = p_energy
            # Map periodic atom forces and virial back to full batch
            p_atom_indices = p_atom_mask.nonzero(as_tuple=True)[0]
            explicit_forces[p_atom_indices] = p_forces
            explicit_virial[p_sys_idx] = p_virial

        return energy, explicit_forces, explicit_virial

    def _direct_coulomb(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        batch_idx: torch.Tensor,
        n_node: torch.Tensor,
        n_systems: int,
    ) -> torch.Tensor:
        """Vectorized direct Coulomb sum (fully connected, optionally erf-damped) for non-periodic systems.

        Scales as O(N²) in total atoms N (all N(N−1) pairs are computed and stored).
        """
        # Create fully connected senders and receivers (no self-loops)
        fc_senders, fc_receivers = _fully_connected_senders_receivers(n_node, positions.device)

        # Pairwise displacement vectors
        r_ij = positions[fc_senders] - positions[fc_receivers]
        dist = torch.norm(r_ij, dim=-1)

        if self.direct_coulomb_erf_damping_sigma is None:
            pair_energy = charges[fc_senders] * charges[fc_receivers] / dist
        else:
            convergence = torch.special.erf(
                dist / (self.direct_coulomb_erf_damping_sigma * math.sqrt(2.0))
            )
            pair_energy = charges[fc_senders] * charges[fc_receivers] * convergence / dist

        # Sum per-atom pair energies, then aggregate to per-system
        per_atom_energy = aggregate_nodes(
            pair_energy, (n_node - 1).repeat_interleave(n_node), reduction="sum"
        )
        per_system = segment_sum(per_atom_energy, batch_idx, n_systems)

        # Factor 0.5 for double counting, apply Coulomb constant
        return 0.5 * self.coulomb_constant * per_system

    def _particle_mesh_ewald(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        batch_idx: torch.Tensor,
        n_systems: int,
        pbc: torch.Tensor,
        *,
        pme_cutoff: float | None = None,
        pme_alpha: float | None = None,
        pme_mesh_dimensions: tuple[int, ...] | None = None,
        pme_hybrid_forces: bool = True,  # Used for testing only.
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Periodic PME with explicit forces, charge gradients, and virial.

        The total force is F_i = −∂E/∂r_i − Σ_j (∂E/∂q_j)(dq_j/dr_i).
        The returned forces/virial are only the first term (∂E/∂r, direct spatial dependence).
        The second term (charge equilibration force) is obtained via autograd through the energy
        and the charge-prediction graph (dq/dr).

        Scales as O(N log N) in total atoms N (real-space is O(N) due to
        erfc cutoff; reciprocal-space is O(N log N) due to FFT on the mesh).
        """
        cell_3d = cell.view(-1, 3, 3) if cell.dim() == 2 else cell

        # Estimate PME parameters and construct neighbor list.
        n_total = positions.shape[0]
        with torch.no_grad():
            params = estimate_pme_parameters(
                positions,
                cell,
                batch_idx=batch_idx.to(torch.int32),
                accuracy=self.pme_accuracy,
            )
            # NOTE: Because we're using max cutoff over the batch, this will be batch-dependent,
            # which can cause batch non-determinism.
            cutoff = params.real_space_cutoff.max().item()
            if pme_cutoff is not None:
                cutoff = pme_cutoff
            alpha = params.alpha
            if pme_alpha is not None:
                alpha = pme_alpha
            mesh_dimensions = tuple(params.mesh_dimensions)
            if pme_mesh_dimensions is not None:
                mesh_dimensions = pme_mesh_dimensions

            neighbor_matrix, num_neighbors, neighbor_shift_matrix = (
                _compute_neighbor_list_with_fallback(
                    positions=positions,
                    cell=cell_3d,
                    pbc=pbc,
                    cutoff=cutoff,
                    batch_idx=batch_idx.to(torch.int32),
                    fill_value=n_total,
                )
            )
        max_nn = max(int(num_neighbors.max().item()), 1)
        neighbor_matrix = neighbor_matrix[:, :max_nn]
        neighbor_shift_matrix = neighbor_shift_matrix[:, :max_nn, :]

        # nvalchemiops computes pure q_i*q_j/r (no Coulomb constant).
        # We absorb our constant by scaling charges: q_scaled = q * sqrt(k),
        # so E = k * sum(q_i*q_j/r) = sum(q_scaled_i * q_scaled_j / r).
        scaled_charges = charges * torch.sqrt(self.coulomb_constant)

        per_atom_energies, explicit_forces, explicit_virial = particle_mesh_ewald(
            positions=positions,
            charges=scaled_charges,
            cell=cell,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            spline_order=self.pme_spline_order,
            batch_idx=batch_idx.to(torch.int32),
            neighbor_matrix=neighbor_matrix.to(torch.int32),
            neighbor_matrix_shifts=neighbor_shift_matrix.to(torch.int32),
            mask_value=n_total,
            accuracy=self.pme_accuracy,
            compute_forces=True,
            compute_charge_gradients=False,
            compute_virial=True,
            hybrid_forces=pme_hybrid_forces,
        )
        per_atom_energies = per_atom_energies.to(positions.dtype)
        # Explicit forces/virial should not be differentiated (the contribution from the charges
        # is included in the energy gradient with hybrid_forces=True using the straight-through trick).
        # There's currently a bug in nvalchemiops where virials remain connected to the graph,
        # so we detach them here for now.
        explicit_forces = explicit_forces.detach().to(positions.dtype)
        explicit_virial = explicit_virial.detach().to(positions.dtype)

        # Per-system energy
        energies = segment_sum(per_atom_energies, batch_idx, n_systems)
        surrogate_energies = energies

        return surrogate_energies, explicit_forces, explicit_virial


def _fully_connected_senders_receivers(
    n_node: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create fully connected senders and receivers for a batch of non-periodic systems."""
    offsets = torch.zeros(n_node.shape[0], device=device, dtype=torch.long)
    offsets[1:] = n_node[:-1].cumsum(0)

    # For each system: senders = each atom repeated n times, receivers = tiled n times
    senders = torch.cat(
        [
            torch.arange(n, device=device).repeat_interleave(n) + off
            for n, off in zip(n_node, offsets, strict=True)
        ]
    )
    receivers = torch.cat(
        [
            torch.arange(n, device=device).repeat(n) + off
            for n, off in zip(n_node, offsets, strict=True)
        ]
    )
    # Remove self-loops
    mask = senders != receivers
    senders, receivers = senders[mask], receivers[mask]
    return senders, receivers
