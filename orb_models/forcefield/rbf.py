import math

import torch


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
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
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
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )
