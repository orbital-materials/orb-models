from collections import defaultdict


def _get_edge_sets(atom_graphs, with_vectors=False, precision=4):
    """Extract edge sets from atom graphs with optional vector information.

    Args:
        atom_graphs: AtomGraphs object containing sender/receiver indices and edge vectors.
        with_vectors: If True, include vector components in edge tuples. Defaults to False.
        precision: Number of decimal places to round vector components and norms. Defaults to 4.

    Returns:
        dict: Dictionary mapping edge tuples to their counts. Edge tuples contain:
            - sender index
            - receiver index
            - vector norm (rounded to precision)
            - vector components (if with_vectors=True, rounded to precision)
    """
    edges_with_counts = defaultdict(int)
    for s, r, vec in zip(
        atom_graphs.senders,
        atom_graphs.receivers,
        atom_graphs.edge_features["vectors"],
        strict=False,
    ):
        vec_tuple = tuple([round(x.item(), precision) for x in vec])
        norm = round(vec.norm(dim=-1).item(), precision)
        if with_vectors:
            edges_with_counts[(s.item(), r.item(), norm, vec_tuple)] += 1
        else:
            edges_with_counts[(s.item(), r.item(), norm)] += 1
    return dict(edges_with_counts)
