import math

import torch
import numpy as np


class CosineCutoff(torch.nn.Module):
    """Cosine cutoff function."""

    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        """Compute the cutoff function."""
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).to(distances.dtype)
            cutoffs = cutoffs * (distances > self.cutoff_lower).to(distances.dtype)
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).to(distances.dtype)
            return cutoffs


class ExpNormalSmearing(torch.nn.Module):
    """Exponential normal smearing function."""

    def __init__(self, cutoff_lower=0.0, cutoff_upper=10.0, num_rbf=50, trainable=True):
        """Exponential normal smearing function.

        Distances are expanded into exponential radial basis functions.
        Basis function parameters are initialised as proposed by Unke & Mewly 2019 Physnet,
        https://arxiv.org/pdf/1902.08408.pdf.
        A cosine cutoff function is used to ensure smooth transition to 0.

        Args:
            cutoff_lower (float): Lower cutoff radius.
            cutoff_upper (float): Upper cutoff radius.
            num_rbf (int): Number of radial basis functions.
            trainable (bool): Whether the parameters are trainable.
        """
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.num_bases = num_rbf  # Compatability
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = cutoff_upper / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", torch.nn.Parameter(means))
            self.register_parameter("betas", torch.nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(float(start_value), 1.0, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        """Reset the parameters to their default values.""" ""
        means, betas = self._initial_params()
        self.means.data.copy_(means)  # type: ignore
        self.betas.data.copy_(betas)  # type: ignore

    def forward(self, dist):
        """Expand incoming distances into basis functions."""
        dist = dist.unsqueeze(-1)
        assert isinstance(self.betas, torch.Tensor)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2  # type: ignore
        )


class BesselBasis(torch.nn.Module):
    """Bessel basis functions.

    NOTE: whilst similar to SphericalBesselBasis, this class
    has different bessel_weights and prefactor.

    Args:
        r_max: Maximum distance for the basis functions.
        trainable: Whether the basis functions are trainable.
    """

    def __init__(self, r_max: float, num_bases=8, trainable=False):
        super().__init__()
        self.num_bases = num_bases
        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_bases,
                steps=num_bases,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        """Compute the basis function."""
        numerator = torch.sin(self.bessel_weights * x[:, None])  # [..., num_basis]
        return self.prefactor * (numerator / x[:, None])  # type: ignore

    def __repr__(self):
        """Return a string representation of the basis function."""
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )
