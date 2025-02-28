import numpy as np
import torch


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


class AtomEmbeddingBag(torch.nn.Module):
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

    def forward(self, Z):
        """
        Forward pass of the atom embedding layer.

        Returns
        -------
        h: torch.Tensor, shape=(nAtoms, emb_size)
            Atom embeddings.
        """
        indices = torch.arange(Z.shape[1], device=Z.device)
        indices = indices.expand_as(Z)
        h = self.embeddings(indices, per_sample_weights=Z)
        return h
