"""Shared neural net utility functions."""

from collections import OrderedDict
from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn.functional as F
from torch import nn

from torch.utils.checkpoint import checkpoint_sequential


class ChargeSpinEmbedding(nn.Module):
    """Embedding module for charge and spin values with configurable types.

    Different embedding strategies:
    - sin_emb: Sinusoidal positional embeddings that encode continuous values
               using learned frequency components. Good for smooth interpolation
               and capturing periodic relationships in charge/spin values.
               pos_emb in older models now got renamed to sin_emb.
    - lin_emb: Linear transformation that learns a direct mapping from scalar
               values to embeddings. Simple and effective for continuous values.
    - rand_emb: Discrete lookup table with separate embeddings for each integer
               value. Good for categorical-like behavior and fixed value ranges.

    Args:
        num_channels: total latent dim shared between charge and spin
        embedding_target: "charge" or "spin"
        embedding_type: "sin_emb", "lin_emb", or "rand_emb"
        scale: scaling factor for positional weights
        requires_grad: whether weights are trainable
    """

    def __init__(
        self,
        num_channels: int,
        embedding_target: str = "charge",
        embedding_type: str = "sin_emb",
        scale: float = 1.0,
        requires_grad: bool = True,
    ):
        super().__init__()
        assert embedding_target in ["charge", "spin"]
        assert embedding_type in ["sin_emb", "pos_emb", "lin_emb", "rand_emb"]

        self.embedding_target = embedding_target
        self.embedding_type = embedding_type
        dim = num_channels // 2  # half for charge, half for spin

        if self.embedding_type == "sin_emb" or self.embedding_type == "pos_emb":
            self.W = nn.Parameter(
                torch.randn(dim // 2) * scale, requires_grad=requires_grad
            )

        elif embedding_type == "lin_emb":
            self.lin_emb = nn.Linear(1, dim)
            if not requires_grad:
                for p in self.lin_emb.parameters():
                    p.requires_grad = False

        elif embedding_type == "rand_emb":
            if embedding_target == "charge":
                self.min_val, self.max_val = -100, 100
                self.offset = 100
                self.rand_emb = nn.Embedding(201, dim)  # -100 → 100
            else:  # spin
                self.min_val, self.max_val = 0, 100
                self.offset = 0
                self.rand_emb = nn.Embedding(101, dim)  # 0 → 100

            if not requires_grad:
                for p in self.rand_emb.parameters():
                    p.requires_grad = False

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Forward pass to embed charge or spin values."""
        assert len(values.shape) == 1, "Expected 1D tensor"
        values = values.float()

        if self.embedding_type == "sin_emb" or self.embedding_type == "pos_emb":
            x_proj = values.unsqueeze(-1) * self.W.unsqueeze(0) * 2 * torch.pi
            emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            if self.embedding_target == "spin":
                emb[values == 0] = 0
            return emb

        elif self.embedding_type == "lin_emb":
            values_ = values.clone()
            if self.embedding_target == "spin":
                values_[values_ == 0] = -100  # optional null spin handling
            return self.lin_emb(values_.unsqueeze(-1))

        elif self.embedding_type == "rand_emb":
            values_rounded = values.round().long()
            values_clamped = values_rounded.clamp(min=self.min_val, max=self.max_val)
            indices = values_clamped + self.offset
            return self.rand_emb(indices)

        raise ValueError(f"Unsupported embedding type: {self.embedding_type}")


class ChargeSpinConditioner(nn.Module):
    """Handles charge and spin conditioning with multiple embedding types.

    Args:
        latent_dim: total latent dimension (split equally between charge and spin)
        embedding_type: "sin_emb", "lin_emb", or "rand_emb"
        emits_node_embs: if True, return node-wise embeddings
        emits_edge_embs: if True, return edge-wise embeddings
    """

    def __init__(
        self,
        latent_dim: int,
        embedding_type: str = "sin_emb",
        emits_node_embs: bool = True,
        emits_edge_embs: bool = False,
    ):
        super().__init__()
        self.charge_embedding = ChargeSpinEmbedding(
            num_channels=latent_dim,
            embedding_target="charge",
            embedding_type=embedding_type,
            requires_grad=True,
        )
        self.spin_embedding = ChargeSpinEmbedding(
            num_channels=latent_dim,
            embedding_target="spin",
            embedding_type=embedding_type,
            requires_grad=True,
        )
        self.emits_node_embs = emits_node_embs
        self.emits_edge_embs = emits_edge_embs

    def forward(self, batch) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Creates charge and spin embeddings for nodes and edges."""
        assert (
            batch.system_features is not None
            and "total_charge" in batch.system_features
            and "total_spin" in batch.system_features
        ), "Batch is missing required system_features."

        charges = batch.system_features["total_charge"]
        spins = batch.system_features["total_spin"]

        charge_emb = self.charge_embedding(charges)
        spin_emb = self.spin_embedding(spins)
        combined_emb = torch.cat([charge_emb, spin_emb], dim=-1)

        node_embs, edge_embs = None, None
        if self.emits_node_embs:
            node_embs = combined_emb.repeat_interleave(batch.n_node, dim=0)
        if self.emits_edge_embs:
            edge_embs = combined_emb.repeat_interleave(batch.n_edge, dim=0)

        return node_embs, edge_embs

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
    dropout: Optional[float] = None,
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
                    dropout=dropout,
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


def get_cutoff(r: torch.Tensor, r_max: float = 6.0) -> torch.Tensor:
    """Get a hardcoded cutoff function for attention. Default cutoff is 6 angstrom."""
    p = 4  # polynomial order
    envelope = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(r / r_max, p)  # type: ignore
        + p * (p + 2.0) * torch.pow(r / r_max, p + 1)  # type: ignore
        - (p * (p + 1.0) / 2) * torch.pow(r / r_max, p + 2)  # type: ignore
    )
    cutoff = (envelope * (r < r_max)).unsqueeze(-1)
    return cutoff
