"""Pyg implementation of Graph Net Simulator."""

from collections import OrderedDict
from typing import List, Literal, Optional, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from orb_models.forcefield import base, segment_ops
from orb_models.forcefield.nn_util import build_mlp

_KEY = "feat"


def mlp_and_layer_norm(
    in_dim: int, out_dim: int, hidden_dim: int, n_layers: int
) -> nn.Sequential:
    """Create an MLP followed by layer norm."""
    return nn.Sequential(
        OrderedDict(
            {
                "mlp": build_mlp(
                    in_dim,
                    [hidden_dim for _ in range(n_layers)],
                    out_dim,
                ),
                "layer_norm": nn.LayerNorm(out_dim),
            }
        )
    )


def get_cutoff(p: int, r: torch.Tensor, r_max: float) -> torch.Tensor:
    """Get the cutoff function for attention."""
    envelope = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(r / r_max, p)  # type: ignore
        + p * (p + 2.0) * torch.pow(r / r_max, p + 1)  # type: ignore
        - (p * (p + 1.0) / 2) * torch.pow(r / r_max, p + 2)  # type: ignore
    )
    cutoff = (envelope * (r < r_max)).unsqueeze(-1)
    return cutoff


class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type.

    Arguments
    ---------
    emb_size: int
        Atom embeddings size
    """

    def __init__(self, emb_size, num_elements, sparse=False):
        super().__init__()
        self.emb_size = emb_size
        self.embeddings = torch.nn.Embedding(num_elements + 1, emb_size, sparse=sparse)
        # init by uniform distribution
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, Z):
        """
        Forward pass of the atom embedding layer.

        Returns
        -------
        h: torch.Tensor, shape=(nAtoms, emb_size)
            Atom embeddings.
        """
        h = self.embeddings(Z)
        return h


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
        num_node_out_features: int,
        num_edge_in_features: int,
        num_edge_out_features: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        node_feature_names: List[str],
        edge_feature_names: List[str],
    ):
        """Graph Network Simulator Encoder.

        Args:
            num_node_in_features (int): Number of node input features.
            num_node_out_features (int): Number of node output features.
            num_edge_in_features (int): Number of edge input featuers.
            num_edge_out_features (int): Number of edge output features.
            num_mlp_layers (int): Number of mlp layers.
            mlp_hidden_dim (int): MLP hidden dimension size.
            node_feature_names (List[str]): Which tensors from ndata to encode
            edge_feature_names (List[str]): Which tensors from edata to encode
        """
        super(Encoder, self).__init__()
        self.node_feature_names = node_feature_names
        self.edge_feature_names = edge_feature_names

        # Encode node features with MLP
        self._node_fn = mlp_and_layer_norm(
            num_node_in_features, num_node_out_features, mlp_hidden_dim, num_mlp_layers
        )
        # Encode edge features with MLP
        self._edge_fn = mlp_and_layer_norm(
            num_edge_in_features, num_edge_out_features, mlp_hidden_dim, num_mlp_layers
        )

    def forward(self, graph: base.AtomGraphs) -> base.AtomGraphs:
        """Forward.

        Args:
          graph: The molecular graph.

        Returns:
            An encoded molecular graph.
        """
        edges = graph.edge_features
        nodes = graph.node_features
        edge_features = torch.cat([edges[k] for k in self.edge_feature_names], dim=-1)
        node_features = torch.cat([nodes[k] for k in self.node_feature_names], dim=-1)

        edges = {**edges, _KEY: self._edge_fn(edge_features)}
        nodes = {**nodes, _KEY: self._node_fn(node_features)}
        return graph._replace(edge_features=edges, node_features=nodes)


class InteractionNetwork(nn.Module):
    """Interaction Network."""

    def __init__(
        self,
        num_node_in: int,
        num_node_out: int,
        num_edge_in: int,
        num_edge_out: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
    ):
        """Interaction network, similar to an MPNN.

        Args:
            num_node_in (int): Number of input node features.
            num_node_out (int): Number of output node features.
            num_edge_in (int): Number of input edge features.
            num_edge_out (int): Number of output edge features.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden dimension size.
        """
        # Aggregate features from neighbors
        super(InteractionNetwork, self).__init__()
        self._num_node_in = num_node_in
        self._num_node_out = num_node_out
        self._num_edge_in = num_edge_in
        self._num_edge_out = num_edge_out
        self._num_mlp_layers = num_mlp_layers
        self._mlp_hidden_dim = mlp_hidden_dim

        self._node_mlp = mlp_and_layer_norm(
            num_node_in + num_edge_out, num_node_out, mlp_hidden_dim, num_mlp_layers
        )
        self._edge_mlp = mlp_and_layer_norm(
            num_node_in + num_node_in + num_edge_in,
            num_edge_out,
            mlp_hidden_dim,
            num_mlp_layers,
        )

    def forward(self, graph: base.AtomGraphs) -> base.AtomGraphs:
        """Run the interaction network forward."""
        nodes = graph.node_features[_KEY]
        edges = graph.edge_features[_KEY]
        senders = graph.senders
        receivers = graph.receivers

        sent_attributes = nodes[senders]
        received_attributes = nodes[receivers]

        edge_features = torch.cat([edges, sent_attributes, received_attributes], dim=1)
        updated_edges = self._edge_mlp(edge_features)

        received_attributes = segment_ops.segment_sum(
            updated_edges, receivers, nodes.shape[0]
        )

        node_features = torch.cat([nodes, received_attributes], dim=1)
        updated_nodes = self._node_mlp(node_features)

        nodes = graph.node_features[_KEY] + updated_nodes
        edges = graph.edge_features[_KEY] + updated_edges

        return graph._replace(
            node_features={**graph.node_features, _KEY: nodes},
            edge_features={**graph.edge_features, _KEY: edges},
        )


class AttentionInteractionNetwork(nn.Module):
    """Attention Interaction Network."""

    def __init__(
        self,
        num_node_in: int,
        num_node_out: int,
        num_edge_in: int,
        num_edge_out: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        attention_gate: Literal["sigmoid", "softmax"] = "sigmoid",
        distance_cutoff: bool = True,
        polynomial_order: Optional[int] = 4,
        cutoff_rmax: Optional[float] = 6.0,
    ):
        """Interaction network, similar to an MPNN.

        This version uses attention to aggregate features from neighbors.
        Additionally, it uses both the sent and recieved features to update
        the node features, as opposed to just the received features.

        Args:
            num_node_in (int): Number of input node features.
            num_node_out (int): Number of output node features.
            num_edge_in (int): Number of input edge features.
            num_edge_out (int): Number of output edge features.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden dimension size.
            attention_gate (Literal["sigmoid", "softmax"]): Which attention gate to use.
            distance_cutoff (bool): Whether or not to use a distance cutoff for attention
                to smooth the distribution.
            polynomial_order (Optional[int]): The order of the polynomial for cutoff envelope.
            cutoff_rmax (Optional[float]): The maximum cutoff radius that enforces the
                high radius state to be disconnected.
        """
        super(AttentionInteractionNetwork, self).__init__()
        self._num_node_in = num_node_in
        self._num_node_out = num_node_out
        self._num_edge_in = num_edge_in
        self._num_edge_out = num_edge_out
        self._num_mlp_layers = num_mlp_layers
        self._mlp_hidden_dim = mlp_hidden_dim

        self._node_mlp = mlp_and_layer_norm(
            num_node_in + num_edge_out + num_edge_out,
            num_node_out,
            mlp_hidden_dim,
            num_mlp_layers,
        )
        self._edge_mlp = mlp_and_layer_norm(
            num_node_in + num_node_in + num_edge_in,
            num_edge_out,
            mlp_hidden_dim,
            num_mlp_layers,
        )

        self._receive_attn = nn.Linear(num_edge_in, 1)
        self._send_attn = nn.Linear(num_edge_in, 1)

        self._distance_cutoff = distance_cutoff
        self._r_max = cutoff_rmax
        self._polynomial_order = polynomial_order
        self._attention_gate = attention_gate

    def forward(self, graph: base.AtomGraphs) -> base.AtomGraphs:
        """Run the interaction network forward."""
        nodes = graph.node_features[_KEY]
        edges = graph.edge_features[_KEY]
        senders = graph.senders
        receivers = graph.receivers

        p = self._polynomial_order
        r_max = self._r_max
        r = graph.edge_features["r"]
        cutoff = get_cutoff(p, r, r_max)  # type: ignore

        sent_attributes = nodes[senders]
        received_attributes = nodes[receivers]

        if self._attention_gate == "softmax":
            num_segments = int(graph.n_node.sum())
            receive_attn = segment_ops.segment_softmax(
                self._receive_attn(edges),
                receivers,
                num_segments,
                weights=cutoff if self._distance_cutoff else None,
            )
            send_attn = segment_ops.segment_softmax(
                self._send_attn(edges),
                senders,
                num_segments,
                weights=cutoff if self._distance_cutoff else None,
            )
        else:
            receive_attn = F.sigmoid(self._receive_attn(edges))
            send_attn = F.sigmoid(self._send_attn(edges))
        # We need to apply this cutoff here even though it is already applied
        # in the attention (if softmax attention is used).
        if self._distance_cutoff:
            receive_attn = receive_attn * cutoff
            send_attn = send_attn * cutoff

        edge_features = torch.cat([edges, sent_attributes, received_attributes], dim=1)
        updated_edges = self._edge_mlp(edge_features)

        sent_attributes = segment_ops.segment_sum(
            updated_edges * send_attn, senders, nodes.shape[0]
        )

        received_attributes = segment_ops.segment_sum(
            updated_edges * receive_attn, receivers, nodes.shape[0]
        )

        node_features = torch.cat([nodes, received_attributes, sent_attributes], dim=1)
        updated_nodes = self._node_mlp(node_features)

        nodes = graph.node_features[_KEY] + updated_nodes
        edges = graph.edge_features[_KEY] + updated_edges

        return graph._replace(
            node_features={**graph.node_features, _KEY: nodes},
            edge_features={**graph.edge_features, _KEY: edges},
        )


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
        batch_norm: bool = False,
    ):
        """The decoder of the GNS.

        Args:
            num_node_in (int): Number of input nodes features.
            num_node_out (int): Number of output node features.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden dimension.
            batch_norm (bool): Whether or not to have batch norm
                at the end of predictions. WARNING this is likely
                to be harmful unless you make sure your targets
                are normalised to std 1 and mean 0.
        """
        super(Decoder, self).__init__()
        seq = OrderedDict(
            {
                "mlp": build_mlp(
                    num_node_in,
                    [mlp_hidden_dim for _ in range(num_mlp_layers)],
                    num_node_out,
                )
            }
        )
        if batch_norm:
            seq["batch_norm"] = nn.BatchNorm1d(num_node_out)
        self.node_fn = nn.Sequential(seq)

    def forward(self, graph: base.AtomGraphs) -> base.AtomGraphs:
        """Forward."""
        nodes = graph.node_features[_KEY]
        updated = self.node_fn(nodes)
        return graph._replace(
            node_features={**graph.node_features, "pred": updated},
        )


class MoleculeGNS(nn.Module):
    """GNS that works on molecular data."""

    def __init__(
        self,
        num_node_in_features: int,
        num_node_out_features: int,
        num_edge_in_features: int,
        latent_dim: int,
        num_message_passing_steps: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        node_feature_names: List[str],
        edge_feature_names: List[str],
        rbf_transform: nn.Module,
        use_embedding: bool = True,
        interactions: Literal["default", "simple_attention"] = "simple_attention",
        interaction_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes the molecular GNS.

        Args:
            num_node_in_features (int): Number input nodes features.
            num_node_out_features (int): Number output nodes features.
            num_edge_in_features (int): Number input edge features.
            latent_dim (int): Latent dimension of processor.
            num_message_passing_steps (int): Number of message passing steps.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden dimension.
            node_feature_names (List[str]): Which tensors from batch.node_features to
                concatenate to form the initial node latents.
            edge_feature_names (List[str]): Which tensors from batch.edge_features to
                concatenate to form the initial edge latents.
            rbf_transform: An optional RBF transform to use for the edge features.
            use_embedding: Whether to embed atom types using an embedding table or embedding bag.
            interactions: The type of interaction network (default message passing or simple attention) to use.
            interaction_params: Additional parameters for the interaction network.
        """
        super().__init__()

        self._encoder = Encoder(
            num_node_in_features=num_node_in_features,
            num_node_out_features=latent_dim,
            num_edge_in_features=num_edge_in_features,
            num_edge_out_features=latent_dim,
            num_mlp_layers=num_mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            node_feature_names=node_feature_names,
            edge_feature_names=edge_feature_names,
        )

        if interactions == "default":
            InteractionNetworkClass = InteractionNetwork
        elif interactions == "simple_attention":
            InteractionNetworkClass = AttentionInteractionNetwork  # type: ignore
        self.num_message_passing_steps = num_message_passing_steps
        if interaction_params is None:
            interaction_params = {}
        self.gnn_stacks = nn.ModuleList(
            [
                InteractionNetworkClass(
                    num_node_in=latent_dim,
                    num_node_out=latent_dim,
                    num_edge_in=latent_dim,
                    num_edge_out=latent_dim,
                    num_mlp_layers=num_mlp_layers,
                    mlp_hidden_dim=mlp_hidden_dim,
                    **interaction_params,
                )
                for _ in range(self.num_message_passing_steps)
            ]
        )

        self._decoder = Decoder(
            num_node_in=latent_dim,
            num_node_out=num_node_out_features,
            num_mlp_layers=num_mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self.rbf = rbf_transform
        self.use_embedding = use_embedding

        if self.use_embedding:
            self.atom_emb = AtomEmbedding(latent_dim, 118)  # type: ignore

    def forward(self, batch: base.AtomGraphs) -> base.AtomGraphs:
        """Encode a graph using molecular GNS."""
        batch = self.featurize_edges(batch)
        batch = self.featurize_nodes(batch)

        batch = self._encoder(batch)

        for gnn in self.gnn_stacks:
            batch = gnn(batch)

        batch = self._decoder(batch)
        return batch

    def featurize_nodes(self, batch: base.AtomGraphs) -> base.AtomGraphs:
        """Featurize the nodes of a graph."""
        one_hot_atomic = torch.nn.functional.one_hot(
            batch.node_features["atomic_numbers"], num_classes=118
        ).type(torch.float32)

        if self.use_embedding:
            # The AtomicEmbedding is expecting indices with type Long
            atomic_number_rep = batch.node_features["atomic_numbers"].long()
            atomic_embedding = self.atom_emb(atomic_number_rep)
        else:
            atomic_embedding = one_hot_atomic

        return batch._replace(
            node_features={**batch.node_features, **{_KEY: atomic_embedding}},
        )

    def featurize_edges(self, batch: base.AtomGraphs) -> base.AtomGraphs:
        """Featurize the edges of a graph."""
        lengths = batch.edge_features["vectors"].norm(dim=1)

        # replace 0s with 1s to avoid division by zero
        non_zero_divisor = torch.where(lengths == 0, torch.ones_like(lengths), lengths)
        unit_vectors = batch.edge_features["vectors"] / non_zero_divisor.unsqueeze(1)
        rbfs = self.rbf(lengths)
        edge_features = torch.cat([rbfs, unit_vectors], dim=1)

        return batch._replace(
            edge_features={
                **batch.edge_features,
                **{_KEY: edge_features},
            },
        )
