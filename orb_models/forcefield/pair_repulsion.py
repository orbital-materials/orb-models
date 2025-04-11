import ase
import torch

from orb_models.forcefield.forcefield_utils import torch_full_3x3_to_voigt_6_stress
from orb_models.forcefield import base, segment_ops


class ZBLBasis(torch.nn.Module):
    """Implementation of the Ziegler-Biersack-Littmark (ZBL) potential.

    with a polynomial cutoff envelope.
    Includes direct calculation of forces and virial stress.
    """

    p: torch.Tensor
    c: torch.Tensor
    d: torch.Tensor
    a_exp: torch.Tensor
    a_prefactor: torch.Tensor
    covalent_radii: torch.Tensor

    def __init__(
        self,
        p: int = 6,
        node_aggregation: str = "mean",
        compute_gradients: bool = False,
    ):
        super().__init__()
        self.node_aggregation = node_aggregation
        self.compute_gradients = compute_gradients

        # Pre-calculate the p coefficients for the ZBL potential
        self.register_buffer(
            "c",
            torch.tensor(
                [0.1818, 0.5099, 0.2802, 0.02817], dtype=torch.get_default_dtype()
            ).unsqueeze(1),
            persistent=False,
        )
        # Exponential factors for ZBL
        self.register_buffer(
            "d",
            torch.tensor(
                [3.2, 0.9423, 0.4028, 0.2016], dtype=torch.get_default_dtype()
            ).unsqueeze(1),
            persistent=False,
        )
        self.register_buffer("p", torch.tensor(p, dtype=torch.int), persistent=False)
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
            persistent=False,
        )
        self.register_buffer("a_exp", torch.tensor(0.300), persistent=False)
        self.register_buffer("a_prefactor", torch.tensor(0.4543), persistent=False)

    def forward(
        self,
        batch: base.AtomGraphs,
    ) -> dict:
        """Forward pass with energy, forces, and stress calculation.

        Args:
            batch: The input atom graphs.

        Returns:
            dict: Dictionary containing energy, forces, and stress tensor.
        """
        senders = batch.senders
        receivers = batch.receivers
        one_hot = batch.node_features["atomic_numbers_embedding"]
        atomic_numbers = torch.argmax(one_hot, dim=1)
        Z_u = atomic_numbers[senders] + 1
        Z_v = atomic_numbers[receivers] + 1

        # Calculate ZBL screening distance
        a = (
            self.a_prefactor  # type: ignore
            * 0.529
            / (torch.pow(Z_u, self.a_exp) + torch.pow(Z_v, self.a_exp))  # type: ignore
        )

        # Get pairwise distances and unit vectors
        vectors = batch.edge_features["vectors"]
        x = vectors.norm(dim=1)  # distances between atoms

        # Unit vectors from atom i to atom j
        r_hat = vectors / x.unsqueeze(1)

        # ZBL potential calculation
        r_over_a = x / a

        # Calculate phi and its derivative
        phi = torch.zeros_like(x)
        dphi_dr_over_a = torch.zeros_like(x)

        exp_term = torch.exp(-self.d * r_over_a.unsqueeze(0))
        phi = torch.sum(self.c * exp_term, dim=0)

        # Potential without envelope
        coulomb_term = 14.3996 * Z_u * Z_v / x
        v_edges_raw = coulomb_term * phi

        # Cutoff envelope
        r_max = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        envelope, denvelope_dr = self._polynomial_cutoff_with_derivative(
            x, r_max, self.p
        )

        # Apply envelope to potential and its derivative (product rule)
        v_edges = 0.5 * v_edges_raw * envelope

        V_ZBL = segment_ops.segment_sum(
            v_edges,
            senders,
            one_hot.shape[0],
        )
        energy = segment_ops.aggregate_nodes(
            V_ZBL,
            batch.n_node,
            reduction=self.node_aggregation,
        )

        output = {"energy": energy}

        if self.compute_gradients:
            # Derivative of potential without envelope
            # d/dr(1/r * phi) = -1/r^2 * phi + 1/r * dphi/dr
            # dphi/dr = dphi/d(r/a) * d(r/a)/dr = dphi/d(r/a) * 1/a
            dphi_dr_over_a = torch.sum(-self.d * self.c * exp_term, dim=0)
            dphi_dr = dphi_dr_over_a / a

            dv_dr_raw = -coulomb_term / x * phi + coulomb_term * dphi_dr
            dv_dr = 0.5 * (dv_dr_raw * envelope + v_edges_raw * denvelope_dr)

            # Calculate per-edge forces (magnitude of force along the bond)
            force_magnitudes = -dv_dr  # Force = -∇V

            # Calculate force vectors (F_ij is force on atom i due to atom j)
            force_vectors = force_magnitudes.unsqueeze(1) * r_hat

            # Accumulate forces for each atom (F_i = sum_j F_ij)
            # Need to handle both sender and receiver contributions
            n_nodes = one_hot.shape[0]
            forces = torch.zeros(
                (n_nodes, 3), device=force_vectors.device, dtype=force_vectors.dtype
            )

            # Force on sender from receiver
            forces.index_add_(0, senders, -force_vectors)
            # Force on receiver from sender (equal and opposite)
            forces.index_add_(0, receivers, force_vectors)

            # Calculate virial stress tensor
            # σ = -1/V ∑_{i<j} r_ij ⊗ F_ij
            # Get unit cell and compute volume
            cell = batch.system_features["cell"]
            volume = torch.linalg.det(cell)

            # Compute outer products for stress tensor
            # Create stress tensor of shape [batch_size, 3, 3]
            stress_per_edge = -torch.einsum("bi,bj->bij", vectors, force_vectors)

            # Sum stress over all edges and divide by volume
            stress = segment_ops.aggregate_nodes(  # actually aggregates over edges
                stress_per_edge,
                batch.n_edge,
                reduction="sum",
            )
            stress = torch_full_3x3_to_voigt_6_stress(stress) / volume.unsqueeze(1)

            output["forces"] = forces
            output["stress"] = stress

        return output

    def _polynomial_cutoff_with_derivative(self, r, r_max, p):
        """Polynomial cutoff function with its derivative.

        Implements the specific polynomial cutoff:
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * (r / r_max)^p
            + p * (p + 2.0) * (r / r_max)^(p + 1)
            - (p * (p + 1.0) / 2) * (r / r_max)^(p + 2)
        )

        Args:
            r: Distances
            r_max: Cutoff radius
            p: Polynomial power

        Returns:
            tuple: (envelope, derivative)
        """
        # Convert p to float for calculations
        p_float = float(p)

        # Mask for r < r_max
        mask = (r < r_max).float()
        r_ratio = r / r_max

        # Calculate envelope according to the specified formula
        envelope = (
            1.0
            - ((p_float + 1.0) * (p_float + 2.0) / 2.0) * torch.pow(r_ratio, p_float)
            + p_float * (p_float + 2.0) * torch.pow(r_ratio, p_float + 1.0)
            - (p_float * (p_float + 1.0) / 2.0) * torch.pow(r_ratio, p_float + 2.0)
        ) * mask

        # Calculate derivative of envelope with respect to r
        term1_deriv = (
            -((p_float + 1.0) * (p_float + 2.0) * p_float / 2.0)
            * torch.pow(r_ratio, p_float - 1.0)
            * (1.0 / r_max)
        )

        term2_deriv = (
            p_float
            * (p_float + 2.0)
            * (p_float + 1.0)
            * torch.pow(r_ratio, p_float)
            * (1.0 / r_max)
        )

        term3_deriv = (
            -(p_float * (p_float + 1.0) * (p_float + 2.0) / 2.0)
            * torch.pow(r_ratio, p_float + 1.0)
            * (1.0 / r_max)
        )

        denvelope_dr = (term1_deriv + term2_deriv + term3_deriv) * mask

        return envelope, denvelope_dr

    def __repr__(self):
        """Text representation of module."""
        return f"{self.__class__.__name__}(c={self.c})"
