"""Shared neural net utility functions."""

from typing import List, Optional, Type

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


def build_mlp(
    input_size: int,
    hidden_layer_sizes: List[int],
    output_size: Optional[int] = None,
    output_activation: Type[nn.Module] = nn.Identity,
    activation: Type[nn.Module] = SSP,
    dropout: Optional[float] = None,
    checkpoint: bool = True,
) -> nn.Module:
    """Build a MultiLayer Perceptron.

    Args:
      input_size: Size of input layer.
      layer_sizes: An array of input size for each hidden layer.
      output_size: Size of the output layer.
      output_activation: Activation function for the output layer.
      activation: Activation function for the hidden layers.

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
    act = [activation for i in range(nlayers)]
    act[-1] = output_activation

    # Create a torch sequential container
    if checkpoint:
        mlp = CheckpointedSequential(n_layers=nlayers)
    else:
        mlp = nn.Sequential()  # type: ignore
    for i in range(nlayers):
        if dropout is not None:
            mlp.add_module("Dropout" + str(i), nn.Dropout(dropout))
        mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        mlp.add_module("Act-" + str(i), act[i]())
    return mlp


class CheckpointedSequential(nn.Sequential):
    """Sequential container with checkpointing."""

    def __init__(self, *args, n_layers: int = 1):
        super().__init__(*args)
        self.n_layers = n_layers

    def forward(self, input):
        """Forward pass with checkpointing enabled in training mode."""
        if self.training:
            return checkpoint_sequential(self, self.n_layers, input, use_reentrant=True)
        else:
            return super().forward(input)
