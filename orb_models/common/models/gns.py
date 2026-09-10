"""Pyg implementation of Graph Net Simulator."""

import typing
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from orb_models.common.atoms.batch.abstract_batch import AbstractAtomBatch
from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.models import base, segment_ops
from orb_models.common.models.angular import UnitVector
from orb_models.common.models.embedding import AtomEmbedding, AtomEmbeddingBag
from orb_models.common.models.nn_util import (
    build_mlp,
    chunked_apply,
    get_cutoff,
    mlp_and_layer_norm,
)

ConditioningType = Literal["additive", "concatenative", "none"]


class Encoder(nn.Module):
    r"""Graph network encoder. Encode nodes and edges states to an MLP.

    The Encode: :math: `\mathcal{X} \rightarrow \mathcal{G}` embeds the
    particle-based state representation, :math: `\mathcal{X}`, as a latent graph, :math:
    `G^0 = encoder(\mathcal{X})`, where :math: `G = (V, E, u), v_i \in V`, and
    :math: `e_{i,j} in E`.
    """

    def __init__(
        self,
        num_node_in_features: int,
        num_edge_in_features: int,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        checkpoint: str | None = None,
        activation: str = "ssp",
        mlp_norm: str = "layer_norm",
    ):
        """Graph Network Simulator Encoder.

        Args:
            num_node_in_features (int): Number of node input features.
            num_edge_in_features (int): Number of edge input featuers.
            latent_dim (int): Latent size for encoder
            num_mlp_layers (int): Number of mlp layers.
            mlp_hidden_dim (int): MLP hidden dimension size.
            checkpoint (Optional[str]): Whether or not to use recomputation checkpoint.
                None (no checkpointing), 'reentrant' or 'non-reentrant'.
            activation (str): Activation function to use.
            layer_norm (str): Normalization layer to use in the MLP.
        """
        super().__init__()

        # Encode node features with MLP
        self._node_fn = mlp_and_layer_norm(
            num_node_in_features,
            latent_dim,
            mlp_hidden_dim,
            num_mlp_layers,
            checkpoint=checkpoint,
            activation=activation,
            mlp_norm=mlp_norm,
        )
        # Encode edge features with MLP
        self._edge_fn = mlp_and_layer_norm(
            num_edge_in_features,
            latent_dim,
            mlp_hidden_dim,
            num_mlp_layers,
            checkpoint=checkpoint,
            activation=activation,
            mlp_norm=mlp_norm,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        chunk_size: int | None = None,
        checkpoint_edge_block: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to encode node and edge features.

        Args:
            node_features: Input node features tensor
            edge_features: Input edge features tensor
            chunk_size: Max edges per chunk in the edge MLP, which is by far the largest
                activation here. The node MLP is num_nodes-shaped and left whole.
            checkpoint_edge_block: Recompute the edge MLP in the backward pass instead of
                storing it. None means "checkpoint iff chunking".

        Returns:
            Tuple of (encoded_nodes, encoded_edges)
        """
        encoded_nodes = self._node_fn(node_features)
        encoded_edges = chunked_apply(
            self._edge_fn, edge_features, chunk_size, checkpoint_chunks=checkpoint_edge_block
        )
        return encoded_nodes, encoded_edges


class AttentionInteractionNetwork(nn.Module):
    """Attention Interaction Network."""

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        attention_gate: Literal["sigmoid", "softmax"] = "sigmoid",
        conditioning: ConditioningType | tuple[ConditioningType, ConditioningType] = "none",
        distance_cutoff: bool = False,
        checkpoint: str | None = None,
        activation: str = "ssp",
        mlp_norm: str = "layer_norm",
        dropout: float | None = None,
    ):
        """Interaction network, similar to an MPNN.

        This version uses attention to aggregate features from neighbors.
        Additionally, it uses both the sent and recieved features to update
        the node features, as opposed to just the received features.

        Args:
            latent_dim (int): The size of the input and output features.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden dimension size.
            attention_gate (Literal["sigmoid", "softmax"]): Which attention gate to use.
            conditioning (Tuple[ConditioningType, ConditioningType]):
                What type of conditioning to use for (nodes, edges).
            distance_cutoff (bool): Whether or not to use a distance cutoff for attention
                to smooth the distribution.
            checkpoint (bool): Whether or not to use recomputation checkpoint.
                None (no checkpointing), 'reentrant' or 'non-reentrant'.
            activation (str): Activation function to use.
            mlp_norm (str): Normalization layer to use in the MLP.
            dropout (float): Dropout rate to use in the MLP.
        """
        super().__init__()
        if isinstance(conditioning, tuple):
            self._node_cond, self._edge_cond = conditioning
        else:
            self._node_cond, self._edge_cond = conditioning, conditioning

        assert self._node_cond in typing.get_args(ConditioningType)
        assert self._edge_cond in typing.get_args(ConditioningType)

        # Get the dimension of the conditioning features for the MLPs
        node_mlp_cond_dim = latent_dim if self._node_cond == "concatenative" else 0
        edge_mlp_cond_dim = latent_dim if self._edge_cond == "concatenative" else 0

        self._node_mlp = mlp_and_layer_norm(
            latent_dim * 3 + node_mlp_cond_dim,
            latent_dim,
            mlp_hidden_dim,
            num_mlp_layers,
            checkpoint=checkpoint,
            activation=activation,
            mlp_norm=mlp_norm,
            dropout=dropout,
        )
        self._edge_mlp = mlp_and_layer_norm(
            latent_dim * 3 + edge_mlp_cond_dim + 2 * node_mlp_cond_dim,
            latent_dim,
            mlp_hidden_dim,
            num_mlp_layers,
            checkpoint=checkpoint,
            activation=activation,
            mlp_norm=mlp_norm,
            dropout=dropout,
        )

        self._receive_attn = nn.Linear(latent_dim + edge_mlp_cond_dim, 1)
        self._send_attn = nn.Linear(latent_dim + edge_mlp_cond_dim, 1)

        if self._node_cond != "none":
            self._cond_node_proj = nn.Linear(latent_dim, latent_dim)
        if self._edge_cond != "none":
            self._cond_edge_proj = nn.Linear(latent_dim, latent_dim)

        self._distance_cutoff = distance_cutoff
        self._attention_gate = attention_gate
        self.latent_dim = latent_dim

    @property
    def conditioning_type(self) -> tuple[ConditioningType, ConditioningType]:
        """The type of conditioning used by the interaction network."""
        return self._node_cond, self._edge_cond

    def _edge_block(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        receive_attn: torch.Tensor,
        send_attn: torch.Tensor,
        segment_sum_impl: Callable,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The message-passing step for one chunk of edges.

        The gathers and their [num_edges, 3 * latent_dim] concatenation belong in here
        alongside the MLP: they are its input, so leaving them outside would materialise
        them at full width and cost more than half of what chunking can save. The
        attention weights, by contrast, are [num_edges, 1] and stay outside.

        Wrapped in a checkpoint, only `updated_edges` and the two
        [num_nodes, latent_dim] partial sums survive into the backward pass.
        """
        edge_features = torch.cat([edges, nodes[senders], nodes[receivers]], dim=1)
        updated_edges = self._edge_mlp(edge_features)

        num_segments = nodes.shape[0]
        return (
            updated_edges,
            segment_sum_impl(updated_edges * send_attn, senders, num_segments),
            segment_sum_impl(updated_edges * receive_attn, receivers, num_segments),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        cutoff: torch.Tensor,
        cond_nodes: torch.Tensor | None = None,
        cond_edges: torch.Tensor | None = None,
        segment_sum_impl: Callable | None = segment_ops.segment_sum,
        segment_softmax_impl: Callable | None = segment_ops.segment_softmax,
        chunk_size: int | None = None,
        checkpoint_edge_block: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run interaction network forward pass.

        Args:
            nodes: Node features tensor [num_nodes, hidden_dim]
            edges: Edge features tensor [num_edges, hidden_dim]
            senders: Sender node indices [num_edges]
            receivers: Receiver node indices [num_edges]
            cutoff: Edge cutoff values [num_edges, 1]
            cond_nodes: Optional conditioning for nodes
            cond_edges: Optional conditioning for edges
            segment_sum_impl: Whether to use default or mesh-aware segment sum implementation
            segment_softmax_impl: Whether to use default or mesh-aware segment softmax implementation
            chunk_size: Max edges per chunk in the edge block. Bounds the transient that the
                backward recompute holds; on its own it saves almost nothing.
            checkpoint_edge_block: Recompute the edge block in the backward pass instead of
                storing it. None means "checkpoint iff chunking".
        Returns:
            Tuple of (updated_nodes, updated_edges)
        """
        # Condition on nodes
        if self._node_cond == "additive":
            if cond_nodes is not None:
                nodes = nodes + self._cond_node_proj(cond_nodes)
        elif self._node_cond == "concatenative" and cond_nodes is not None:
            nodes = torch.cat([nodes, self._cond_node_proj(cond_nodes)], dim=-1)

        # Condition on edges
        if self._edge_cond == "additive":
            if cond_edges is not None:
                edges = edges + self._cond_edge_proj(cond_edges)
        elif self._edge_cond == "concatenative" and cond_edges is not None:
            edges = torch.cat([edges, self._cond_edge_proj(cond_edges)], dim=-1)

        if self._attention_gate == "softmax":
            num_segments = nodes.shape[0]
            receive_attn = segment_softmax_impl(
                self._receive_attn(edges),
                receivers,
                num_segments,
                weights=cutoff if self._distance_cutoff else None,
            )
            send_attn = segment_softmax_impl(
                self._send_attn(edges),
                senders,
                num_segments,
                weights=cutoff if self._distance_cutoff else None,
            )
        else:
            receive_attn = F.sigmoid(self._receive_attn(edges))
            send_attn = F.sigmoid(self._send_attn(edges))

        if self._distance_cutoff:
            receive_attn = receive_attn * cutoff
            send_attn = send_attn * cutoff

        num_edges = edges.shape[0]
        chunk_size = min(chunk_size or num_edges, num_edges)
        if checkpoint_edge_block is None:
            checkpoint_edge_block = chunk_size < num_edges
        use_checkpoint = checkpoint_edge_block and torch.is_grad_enabled()
        # An edgeless graph (e.g. a padding graph) still runs the block once, on empties.
        starts = range(0, num_edges, chunk_size) if num_edges else [0]

        # The aggregations accumulate as we go, so each chunk's partial sums are freed
        # once added; the updated edges are collected, as they are all needed at the end.
        edge_chunks: list[torch.Tensor] = []
        sent_attributes = edges.new_zeros(nodes.shape[0], self.latent_dim)
        received_attributes = edges.new_zeros(nodes.shape[0], self.latent_dim)
        for start in starts:
            chunk = slice(start, start + chunk_size)
            block_args = (
                nodes,
                edges[chunk],
                senders[chunk],
                receivers[chunk],
                receive_attn[chunk],
                send_attn[chunk],
                segment_sum_impl,
            )
            if use_checkpoint:
                chunk_edges, sent, received = checkpoint(
                    self._edge_block, *block_args, use_reentrant=False
                )
            else:
                chunk_edges, sent, received = self._edge_block(*block_args)
            edge_chunks.append(chunk_edges)
            sent_attributes = sent_attributes + sent
            received_attributes = received_attributes + received

        # torch.cat always copies, so skip it on the single-chunk (default) path.
        updated_edges = edge_chunks[0] if len(edge_chunks) == 1 else torch.cat(edge_chunks, dim=0)

        node_features = torch.cat([nodes, received_attributes, sent_attributes], dim=1)
        updated_nodes = self._node_mlp(node_features)

        # Remove the conditioning features, if using concatenation
        if self._node_cond == "concatenative":
            nodes = nodes[:, : self.latent_dim]
        if self._edge_cond == "concatenative":
            edges = edges[:, : self.latent_dim]

        nodes = nodes + updated_nodes
        edges = edges + updated_edges

        return nodes, edges


class Decoder(nn.Module):
    r"""The Decoder.

    :math: `\mathcal{G} \rightarrow \mathcal{Y}` extracts the
    dynamics information from the nodes of the final latent graph,
    :math: `y_i = \delta v (v_i^M)`
    """

    def __init__(
        self,
        num_node_in: int,
        num_node_out: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        checkpoint: str | None = None,
        activation: str = "ssp",
    ):
        """The decoder of the GNS.

        Args:
            num_node_in (int): Number of input nodes features.
            num_node_out (int): Number of output node features.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden dimension.
            checkpoint (Optional[str]): Whether or not to use recomputation checkpoint.
                None (no checkpointing), 'reentrant' or 'non-reentrant'.
            activation (str): Activation function to use.
        """
        super().__init__()
        seq = OrderedDict(
            {
                "mlp": build_mlp(
                    num_node_in,
                    [mlp_hidden_dim for _ in range(num_mlp_layers)],
                    num_node_out,
                    activation=activation,
                    checkpoint=checkpoint,
                )
            }
        )
        self.node_fn = nn.Sequential(seq)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """Forward pass to decode node features."""
        return self.node_fn(nodes)


class MoleculeGNS(base.ModelMixin):
    """GNS that works on molecular data."""

    _deprecated_args = [
        "noise_scale",
        "add_virtual_node",
        "self_cond",
        "interactions",
        "num_node_in_features",
        "num_edge_in_features",
        "conditioning_encoder",
        "conditioning",
    ]

    def __init__(
        self,
        latent_dim: int,
        num_message_passing_steps: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        rbf_transform: Callable,
        angular_transform: Callable | None = None,
        outer_product_with_cutoff: bool = False,
        use_embedding: bool = False,  # atom type embedding
        expects_atom_type_embedding: bool = False,
        interaction_params: dict[str, Any] | None = None,
        num_node_out_features: int = 3,
        extra_embed_dims: int | tuple[int, int] = 0,
        node_feature_names: list[str] | None = None,
        edge_feature_names: list[str] | None = None,
        conditioner: Callable | None = None,
        conditioning_type: ConditioningType = "additive",
        checkpoint: str | None = None,
        activation="ssp",
        mlp_norm: str = "layer_norm",
        **kwargs,
    ) -> None:
        """Initializes the molecular GNS.

        Args:
            latent_dim (int): Latent dimension of processor.
            num_message_passing_steps (int): Number of message passing steps.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden dimension.
            rbf_transform (Callable): A function that takes in edge lengths and returns
                a tensor of RBF features.
            angular_transform (Callable): A function that takes in edge vectors and
                returns a tensor of angular features.
            outer_product_with_cutoff (bool): Create initial edge embeddings via
                an outer product of rbf and angular embeddings and a envelope cutoff.
            use_embedding: Whether to embed atom types using an embedding table or embedding bag.
            expects_atom_type_embedding (bool): Whether or not the model expects the input
                to be pre-embedded. This is used for atom type models, because the one-hot
                embedding is noised, rather than being explicitly one-hot.
            interaction_params (Optional[Dict[str, Any]]): Additional parameters
                to pass to the interaction network.
            num_node_out_features (int): Number output nodes features.
            extra_embed_dims (Union[int, Tuple[int, int]]): Number of extra embedding dimensions to use.
                If an int, both the node and edge embeddings will have this number of extra dims.
                If a tuple, then it is interpreted as [extra_node_embed_dim, extra_edge_embed_dim].
            node_feature_names (List[str]): Which tensors from batch.node_features to
                concatenate to form the initial node latents. Note: These are "extra"
                features - we assume the base atomic number representation is already included.
            edge_feature_names (List[str]): Which tensors from batch.edge_features to
                concatenate to form the initial edge latents. Note: These are "extra"
                features - we assume the base edge vector features are already included.
            conditioner (Optional[Callable]): A conditioner module that takes in a batch and returns
                conditioning features for nodes and edges. The conditioner should have attributes
                `emits_node_embs` and `emits_edge_embs` that indicate what it returns.
            conditioning_type (ConditioningType): 'Additive' / 'Concatenative' / 'None'
            checkpoint (bool): Whether or not to use checkpointing.
            activation (str): Activation function to use.
            mlp_norm (str): Normalization layer to use in the MLP.
        """
        super().__init__()

        kwargs = {k: v for k, v in kwargs.items() if k not in self._deprecated_args}
        if kwargs:
            raise ValueError(
                f"The following kwargs are not arguments to MoleculeGNS: {kwargs.keys()}"
            )

        self.latent_dim = latent_dim
        self.node_feature_names = node_feature_names or []
        self.edge_feature_names = edge_feature_names or []
        self.num_message_passing_steps = num_message_passing_steps

        # Edge embedding
        self.outer_product_with_cutoff = outer_product_with_cutoff
        self.rbf_transform = rbf_transform
        if angular_transform is None:
            angular_transform = UnitVector()
        self.angular_transform = angular_transform
        if self.outer_product_with_cutoff:
            self.edge_embed_size = rbf_transform.num_bases * angular_transform.dim  # type: ignore
        else:
            if hasattr(rbf_transform, "num_bases"):
                num_bases = rbf_transform.num_bases
            else:
                num_bases = rbf_transform.keywords["num_bases"]  # type: ignore
            self.edge_embed_size = num_bases + angular_transform.dim  # type: ignore

        # Node embedding
        self.expects_atom_type_embedding = expects_atom_type_embedding
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.node_embed_size = latent_dim
            if self.expects_atom_type_embedding:
                self.atom_emb = AtomEmbeddingBag(self.node_embed_size, 118)
            else:
                self.atom_emb = AtomEmbedding(self.node_embed_size, 118)  # type: ignore
        else:
            self.node_embed_size = 118
        if isinstance(extra_embed_dims, int):
            extra_embed_dims = (extra_embed_dims, extra_embed_dims)  # type: ignore

        # Conditioning
        if conditioner is not None:
            node_conditioning = conditioning_type if conditioner.emits_node_embs else "none"  # type: ignore
            edge_conditioning = conditioning_type if conditioner.emits_edge_embs else "none"  # type: ignore
            self.conditioner: Callable | None = conditioner
        else:
            node_conditioning, edge_conditioning = "none", "none"
            self.conditioner = None

        self._encoder = Encoder(
            num_node_in_features=self.node_embed_size + extra_embed_dims[0],
            num_edge_in_features=self.edge_embed_size + extra_embed_dims[1],
            latent_dim=latent_dim,
            num_mlp_layers=num_mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            checkpoint=checkpoint,
            activation=activation,
            mlp_norm=mlp_norm,
        )
        self.gnn_stacks = nn.ModuleList(
            [
                AttentionInteractionNetwork(
                    latent_dim=latent_dim,
                    num_mlp_layers=num_mlp_layers,
                    mlp_hidden_dim=mlp_hidden_dim,
                    conditioning=(node_conditioning, edge_conditioning),
                    **(interaction_params or {}),
                    checkpoint=checkpoint,
                    activation=activation,
                    mlp_norm=mlp_norm,
                )
                for _ in range(self.num_message_passing_steps)
            ]
        )
        self._decoder = Decoder(
            num_node_in=latent_dim,
            num_node_out=num_node_out_features,
            num_mlp_layers=num_mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            checkpoint=checkpoint,
            activation=activation,
        )

    def forward(
        self,
        batch: AtomGraphs,
        segment_sum_impl: Callable = segment_ops.segment_sum,
        segment_softmax_impl: Callable = segment_ops.segment_softmax,
        chunk_size: int | None = None,
        checkpoint_edge_block: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode a graph using molecular GNS.

        Args:
            batch: Input molecular graph
            segment_sum_impl: Whether to use default or mesh-aware segment sum implementation.
            chunk_size: Max edges per chunk in the encoder's and the interaction networks'
                edge paths. None disables chunking.
            checkpoint_edge_block: Recompute those edge paths in the backward pass instead of
                storing them. None means "checkpoint iff chunking"

        Returns:
            Dictionary containing node_features, edge_features, and predictions
        """
        # Featurize inputs
        edge_features = self.featurize_edges(batch)
        node_features = self.featurize_nodes(batch)
        if self.conditioner is not None:
            cond_nodes, cond_edges = self.conditioner(batch)
        else:
            cond_nodes, cond_edges = None, None

        # Encode
        nodes, edges = self._encoder(
            node_features,
            edge_features,
            chunk_size=chunk_size,
            checkpoint_edge_block=checkpoint_edge_block,
        )

        # Process through interaction networks
        cutoff = get_cutoff(batch.edge_features["vectors"].norm(dim=-1))
        for gnn in self.gnn_stacks:
            nodes, edges = gnn(
                nodes,
                edges,
                batch.senders,
                batch.receivers,
                cutoff,
                cond_nodes=cond_nodes,
                cond_edges=cond_edges,
                segment_sum_impl=segment_sum_impl,
                segment_softmax_impl=segment_softmax_impl,
                chunk_size=chunk_size,
                checkpoint_edge_block=checkpoint_edge_block,
            )

        # Decode
        pred = self._decoder(nodes)

        return {
            "node_features": nodes,
            "edge_features": edges,
            "pred": pred,
        }

    def featurize_nodes(self, batch: AtomGraphs) -> torch.Tensor:
        """Featurize the nodes of a graph."""
        if self.use_embedding:
            atomic_embedding = self.atom_emb(batch)
        else:
            # Avoid getters because torch.compile can't handle them.
            atomic_embedding = batch.node_features["atomic_numbers_embedding"]

        # Exclusion of 'feat' is for backward compatibility with old code
        feature_names = [k for k in self.node_feature_names if k != "feat"]
        return torch.cat(
            [atomic_embedding, *[batch.node_features[k] for k in feature_names]], dim=-1
        )

    def featurize_edges(self, batch: AtomGraphs) -> torch.Tensor:
        """Featurize the edges of a graph."""
        vectors = batch.edge_features["vectors"]
        lengths = vectors.norm(dim=1)

        angular_embedding = self.angular_transform(vectors)  # (nedges, x)
        rbfs = self.rbf_transform(lengths)  # (nedges, y)

        if self.outer_product_with_cutoff:
            cutoff = get_cutoff(lengths)
            # (nedges, x, y)
            outer_product = rbfs[:, :, None] * angular_embedding[:, None, :]
            # (nedges, x * y)
            edge_features = cutoff * outer_product.view(vectors.shape[0], self.edge_embed_size)
        else:
            edge_features = torch.cat([rbfs, angular_embedding], dim=1)

        # For backwards compatibility, exclude 'feat'
        feature_names = [k for k in self.edge_feature_names if k != "feat"]
        return torch.cat([edge_features, *[batch.edge_features[k] for k in feature_names]], dim=-1)

    def loss(self, batch: AbstractAtomBatch) -> base.ModelOutput:
        """Loss function for molecular GNS. NOTE: this is rarely used directly."""
        assert isinstance(batch, AtomGraphs), f"Expected AtomGraphs, got {type(batch)}"
        out = self(batch)
        if batch.node_targets is not None:
            assert "noise_target" in batch.node_targets
            noise_target = batch.node_targets["noise_target"]
            position_loss = torch.mean(
                (out["pred"] - noise_target) ** 2,
            )
            loss = torch.tensor(0).type_as(position_loss)
            loss += position_loss
            metric_kwargs = {"position_loss": position_loss}
        else:
            raise ValueError("Noise scale is None - loss not supported.")

        return base.ModelOutput(loss=loss, log=dict(loss=loss, **metric_kwargs))
