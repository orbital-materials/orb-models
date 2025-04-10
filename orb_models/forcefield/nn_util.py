"""Shared neural net utility functions."""

from collections import OrderedDict
from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn.functional as F
from torch import nn

from torch.utils.checkpoint import checkpoint_sequential


class SSP(nn.Softplus):
    """Shifted Softplus activation function.

    This activation is twice differentiable so can be used when regressing
    gradients for conservative force fields.
    """

    def __init__(self, beta: int = 1, threshold: int = 20):
        """Initialised SSP activation."""
        super(SSP, self).__init__(beta, threshold)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of SSP."""
        sp0 = F.softplus(torch.zeros(1, device=input.device), self.beta, self.threshold)
        return F.softplus(input, self.beta, self.threshold) - sp0


def get_activation(activation: Optional[str]):
    """Build activation function."""
    if not activation:
        return nn.Identity()
    return {
        "ssp": SSP,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }[activation]


def get_mlp_norm(mlp_norm: str) -> Callable:
    """Get the MLP norm."""
    if mlp_norm == "rms_norm":
        return nn.RMSNorm  # type: ignore[attr-defined]
    elif mlp_norm == "layer_norm":
        return nn.LayerNorm
    else:
        raise ValueError(f"Unknown MLP norm: {mlp_norm}")


def build_mlp(
    input_size: int,
    hidden_layer_sizes: List[int],
    output_size: Optional[int] = None,
    output_activation: Type[nn.Module] = nn.Identity,
    activation: str = "ssp",
    dropout: Optional[float] = None,
    checkpoint: Optional[Union[str, bool]] = None,
) -> nn.Module:
    """Build a MultiLayer Perceptron.

    Args:
        input_size: Size of input layer.
        layer_sizes: An array of input size for each hidden layer.
        output_size: Size of the output layer.
        output_activation: Activation function for the output layer.
        activation: Activation function for the hidden layers.
        dropout: Dropout rate.
        checkpoint: Whether to use checkpointing and what type.
            None (no checkpointing), 'reentrant' or 'non-reentrant'.

    Returns:
        mlp: An MLP sequential container.
    """
    # Size of each layer
    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size:
        layer_sizes.append(output_size)

    # Number of layers
    nlayers = len(layer_sizes) - 1

    # Create a list of activation functions and
    # set the last element to output activation function
    act = [get_activation(activation) for i in range(nlayers)]
    act[-1] = output_activation

    # Create a torch sequential container
    if (
        checkpoint is not None and checkpoint is not False
    ):  # Wrangle for bool backwards compatibility
        if isinstance(checkpoint, bool):
            checkpoint = "reentrant" if checkpoint else None
        assert checkpoint in ["reentrant", "non-reentrant"]
        mlp = CheckpointedSequential(
            n_layers=nlayers, reentrant=checkpoint == "reentrant"
        )
    else:
        mlp = nn.Sequential()  # type: ignore
    for i in range(nlayers):
        if dropout is not None:
            mlp.add_module("Dropout" + str(i), nn.Dropout(dropout))
        mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        mlp.add_module("Act-" + str(i), act[i]())
    return mlp


def mlp_and_layer_norm(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    n_layers: int,
    checkpoint: Optional[str] = None,
    activation: str = "ssp",
    mlp_norm: str = "layer_norm",
) -> nn.Sequential:
    """Create an MLP followed by layer norm."""
    return nn.Sequential(
        OrderedDict(
            {
                "mlp": build_mlp(
                    in_dim,
                    [hidden_dim for _ in range(n_layers)],
                    out_dim,
                    activation=activation,
                    checkpoint=checkpoint,
                ),
                "layer_norm": get_mlp_norm(mlp_norm)(out_dim),
            }
        )
    )


class CheckpointedSequential(nn.Sequential):
    """Sequential container with checkpointing."""

    def __init__(self, *args, n_layers: int = 1, reentrant: bool = True):
        super().__init__(*args)
        self.n_layers = n_layers
        self.reentrant = reentrant

    def forward(self, input):
        """Forward pass with checkpointing enabled in training mode."""
        if self.training:
            return checkpoint_sequential(
                self, self.n_layers, input, use_reentrant=self.reentrant
            )
        else:
            return super().forward(input)


class ScalarNormalizer(torch.nn.Module):
    """Scalar normalizer that learns mean and std from data.

    NOTE: Multi-dimensional tensors are flattened before updating
    the running mean/std. This is desired behaviour for force targets.
    """

    def __init__(
        self,
        init_mean: Optional[torch.Tensor | float] = None,
        init_std: Optional[torch.Tensor | float] = None,
        init_num_batches: Optional[int] = 1000,
        online: bool = True,
    ) -> None:
        """Initializes the ScalarNormalizer.

        To enhance training stability, consider setting an init mean + std.
        """
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(1, affine=False, momentum=None)  # type: ignore
        if init_mean is not None:
            assert init_std is not None
            self.bn.running_mean = torch.tensor([init_mean])
            self.bn.running_var = torch.tensor([init_std**2])
            self.bn.num_batches_tracked = torch.tensor([init_num_batches])
        assert isinstance(online, bool)
        self.online = online

    def forward(self, x: torch.Tensor, online: Optional[bool] = None) -> torch.Tensor:
        """Normalize by running mean and std."""
        online = online if online is not None else self.online
        x_reshaped = x.reshape(-1, 1)
        if self.training and online and x_reshaped.shape[0] > 1:
            # hack: call batch norm, but only to update a running mean/std
            self.bn(x_reshaped)

        mu = self.bn.running_mean  # type: ignore
        sigma = torch.sqrt(self.bn.running_var)  # type: ignore
        if sigma < 1e-6:
            raise ValueError("ScalarNormalizer has ~zero std.")

        return (x - mu) / sigma  # type: ignore

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse the forward normalization."""
        return x * torch.sqrt(self.bn.running_var) + self.bn.running_mean  # type: ignore


def get_cutoff(r: torch.Tensor) -> torch.Tensor:
    """Get a hardcoded cutoff function for attention."""
    p = 4  # polynomial order
    r_max = 6.0  # value (in Å) for which cutoff returns zero
    envelope = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(r / r_max, p)  # type: ignore
        + p * (p + 2.0) * torch.pow(r / r_max, p + 1)  # type: ignore
        - (p * (p + 1.0) / 2) * torch.pow(r / r_max, p + 2)  # type: ignore
    )
    cutoff = (envelope * (r < r_max)).unsqueeze(-1)
    return cutoff
