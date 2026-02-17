import abc

import numpy as np
import torch

from orb_models.common.atoms.batch.graph_batch import AtomGraphs


class EmbeddingModule(torch.nn.Module, abc.ABC):
    """Abstract base class for embedding modules."""

    @property
    @abc.abstractmethod
    def out_dim(self) -> int:
        """Size of the embedding."""
        pass


class AtomEmbedding(EmbeddingModule):
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

    @property
    def out_dim(self):
        """Size of the embedding."""
        return self.emb_size

    def forward(self, batch: AtomGraphs):
        """
        Forward pass of the atom embedding layer.

        Returns
        -------
        h: torch.Tensor, shape=(nAtoms, emb_size)
            Atom embeddings.
        """
        # NOTE: We can't use getters or setters here because torch.compile can't handle them.
        atomic_number_rep = batch.node_features["atomic_numbers"].long()
        h = self.embeddings(atomic_number_rep)
        return h


class AtomEmbeddingBag(EmbeddingModule):
    """
    Initial atom embeddings based on a weighted avg of the atom types.

    Arguments
    ---------
    emb_size: int
        Atom embeddings size
    """

    def __init__(self, emb_size, num_elements):
        super().__init__()
        self.emb_size = emb_size

        self.embeddings = torch.nn.EmbeddingBag(num_elements + 1, emb_size, mode="sum")
        # init by uniform distribution
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    @property
    def out_dim(self):
        """Size of the embedding."""
        return self.emb_size

    def forward(self, batch: AtomGraphs):
        """
        Forward pass of the atom embedding layer.

        Returns
        -------
        h: torch.Tensor, shape=(nAtoms, emb_size)
            Atom embeddings.
        """
        # NOTE: We can't use getters or setters here because torch.compile can't handle them.
        one_hot_atomic = batch.node_features["atomic_numbers_embedding"]
        indices = torch.arange(one_hot_atomic.shape[1], device=one_hot_atomic.device)
        indices = indices.expand_as(one_hot_atomic)
        h = self.embeddings(indices, per_sample_weights=one_hot_atomic)
        return h
