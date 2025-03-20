"""Base Model class."""

from collections import defaultdict
from copy import deepcopy
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Union,
    Tuple,
)

import torch
import tree

from orb_models.forcefield import featurization_utilities
from orb_models.forcefield.featurization_utilities import TORCH_FLOAT_DTYPES

Metric = Union[torch.Tensor, int, float]
TensorDict = Mapping[str, Optional[torch.Tensor]]


class ModelOutput(NamedTuple):
    """A model's output."""

    loss: torch.Tensor
    log: Mapping[str, Metric]


class AtomGraphs(NamedTuple):
    """A class representing the input to a model for a graph.

    Args:
        senders (torch.Tensor): The integer source nodes for each edge.
        receivers (torch.Tensor): The integer destination nodes for each edge.
        n_node (torch.Tensor): A (batch_size, ) shaped tensor containing the number of nodes per graph.
        n_edge (torch.Tensor): A (batch_size, ) shaped tensor containing the number of edges per graph.
        node_features (Dict[str, torch.Tensor]): A dictionary containing node feature tensors.
            It will always contain "atomic_numbers" and "positions" keys, representing the
            atomic numbers of each node, and the 3d cartesian positions of them respectively.
        edge_features (Dict[str, torch.Tensor]): A dictionary containing edge feature tensors.
        system_features (Dict[str, torch.Tensor]): A dictionary containing system-level features.
        node_targets (Optional[Dict[str, torch.Tensor]]): An optional dict of tensors containing targets
            for individual nodes. This tensor is commonly expected to have shape (num_nodes, *).
        edge_targets (Optional[Dict[str, torch.Tensor]]): An optional dict of tensors containing targets for
            individual edges. This tensor is commonly expected to have (num_edges, *).
        system_targets (Optional[Dict[str, torch.Tensor]]): An optional dict of tensors containing targets for the
            entire system.
        system_id (Optional[torch.Tensor]): An optional tensor containing a dataset-specific index for a datapoint.
        fix_atoms (Optional[torch.Tensor]): An optional tensor containing information on fixed atoms in the system.
        tags (Optional[torch.Tensor]): An optional tensor containing ase tags for each node.
        radius (float): The radius used for neighbor calculation.
        max_num_neighbors (int): The maximum number of neighbors used for the neighbor calculation.
        half_supercell (bool): Whether to use half the supercell for graph construction, and then symmetrize.
    """

    senders: torch.Tensor
    receivers: torch.Tensor
    n_node: torch.Tensor
    n_edge: torch.Tensor
    node_features: Dict[str, torch.Tensor]
    edge_features: Dict[str, torch.Tensor]
    system_features: Dict[str, torch.Tensor]
    node_targets: Optional[Dict[str, torch.Tensor]]
    edge_targets: Optional[Dict[str, torch.Tensor]]
    system_targets: Optional[Dict[str, torch.Tensor]]
    system_id: Optional[torch.Tensor]
    fix_atoms: Optional[torch.Tensor]
    tags: Optional[torch.Tensor]
    radius: float
    max_num_neighbors: torch.Tensor
    half_supercell: bool = False

    @property
    def positions(self):
        """Get positions of atoms."""
        return self.node_features["positions"]

    @positions.setter
    def positions(self, val: torch.Tensor):
        self.node_features["positions"] = val

    @property
    def atomic_numbers(self):
        """Get integer atomic numbers."""
        return self.node_features["atomic_numbers"]

    @atomic_numbers.setter
    def atomic_numbers(self, val: torch.Tensor):
        self.node_features["atomic_numbers"] = val

    @property
    def atomic_numbers_embedding(self):
        """Get atom type embedding."""
        return self.node_features["atomic_numbers_embedding"]

    @atomic_numbers_embedding.setter
    def atomic_numbers_embedding(self, val: torch.Tensor):
        self.node_features["atomic_numbers_embedding"] = val

    @property
    def cell(self):
        """Get unit cells."""
        assert self.system_features
        return self.system_features.get("cell")

    @cell.setter
    def cell(self, val: torch.Tensor):
        assert self.system_features
        self.system_features["cell"] = val

    @property
    def pbc(self):
        """Get pbc."""
        assert self.system_features
        return self.system_features.get("pbc")

    def compute_differentiable_edge_vectors(
        self,
        use_stress_displacement: bool = True,
        use_rotation: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute pbc-aware edge vectors such that gradients flow back to positions.

        Args:
            stress_displacement (bool): If True, a zero 'displacement' tensor is created
            and matrix-multiplied with positions and cell, such that:
                    stress = grad(E)_{displacement} / volume
        """
        positions = self.node_features["positions"]  # (natoms, 3)
        positions.requires_grad_(True)
        unit_shifts = self.edge_features["unit_shifts"].unsqueeze(1)  # (nedges, 1, 3)
        cell = self.system_features["cell"]  # (ngraphs, 3, 3)

        displacement = None
        if use_stress_displacement:
            per_node_graph_indices = self._get_per_node_graph_indices()
            positions, cell, displacement = create_and_apply_stress_displacement(
                positions, cell, per_node_graph_indices
            )

        generator = None
        if use_rotation:
            per_node_graph_indices = self._get_per_node_graph_indices()
            generator = torch.zeros_like(cell, requires_grad=True)
            rotation = featurization_utilities.rotation_from_generator(
                generator,
            )
            positions = torch.bmm(
                positions.unsqueeze(1),
                rotation[per_node_graph_indices],
            ).squeeze(1)
            cell = torch.bmm(cell, rotation)

        # This is a compilable equivalent of cells.repeat_interleave(self.n_edge, dim=0)
        per_edge_graph_indices = self._get_per_edge_graph_indices()
        cells_repeat = cell[per_edge_graph_indices]  # (nedges, 3, 3)

        shifts = torch.bmm(unit_shifts, cells_repeat).squeeze(1)  # (nedges, 3)
        vectors = positions[self.receivers] - positions[self.senders] + shifts

        return vectors, displacement, generator

    def _get_per_edge_graph_indices(self):
        """Get the graph index for each edge in the system."""
        graph_indices = torch.zeros_like(self.senders)  # (nedges,)
        cumsums = torch.cumsum(self.n_edge, dim=0)  # (ngraphs,)
        graph_indices[:] = torch.searchsorted(
            cumsums,
            torch.arange(len(self.senders), device=self.senders.device),
            right=True,
        )
        return graph_indices  # (nedges,)

    def _get_per_node_graph_indices(self):
        """Get the graph index for each node in the system."""
        positions = self.node_features["positions"]
        graph_indices = torch.zeros(
            len(positions), device=positions.device, dtype=torch.int
        )
        cumsums = torch.cumsum(self.n_node, dim=0)
        graph_indices[:] = torch.searchsorted(
            cumsums,
            torch.arange(len(positions), device=positions.device),
            right=True,
        )
        return graph_indices  # (natoms,)

    def clone(self) -> "AtomGraphs":
        """Clone the AtomGraphs object.

        Note: this differs from deepcopy() because it preserves gradients.
        """

        def _clone(x):
            if isinstance(x, torch.Tensor):
                return x.clone()
            else:
                return x

        return tree.map_structure(_clone, self)

    def to(
        self,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "AtomGraphs":
        """Move AtomGraphs child tensors to a device and/or dtype.

        NOTE: only floating point tensors are cast to the specified dtype.
        """
        if dtype is not None and dtype not in TORCH_FLOAT_DTYPES:
            raise ValueError(f"dtype must be a floating point type, got {dtype}")

        if isinstance(device, str):
            device = torch.device(device)

        def _to(x):
            if hasattr(x, "to"):
                # Only cast if x is a floating-point tensor
                if torch.is_floating_point(x):
                    return x.to(device=device, dtype=dtype)
                else:
                    return x.to(device=device)
            else:
                return x

        return tree.map_structure(_to, self)

    def detach(self) -> "AtomGraphs":
        """Detach all child tensors."""

        def _detach(x):
            if hasattr(x, "detach"):
                return x.detach()
            else:
                return x

        return tree.map_structure(_detach, self)

    def equals(self, graphs: "AtomGraphs") -> bool:
        """Check two atomgraphs are equal."""

        def _is_equal(x, y):
            if isinstance(x, torch.Tensor):
                return torch.equal(x, y)
            else:
                return x == y

        flat_results = tree.flatten(tree.map_structure(_is_equal, self, graphs))
        return all(flat_results)

    def allclose(self, graphs: "AtomGraphs", rtol=1e-5, atol=1e-8) -> bool:
        """Check all tensors/scalars of two atomgraphs are close."""

        def _is_close(x, y):
            if isinstance(x, torch.Tensor):
                return torch.allclose(x, y, rtol=rtol, atol=atol)
            elif isinstance(x, (float, int)):
                return torch.allclose(
                    torch.tensor(x), torch.tensor(y), rtol=rtol, atol=atol
                )
            else:
                return x == y

        flat_results = tree.flatten(tree.map_structure(_is_close, self, graphs))
        return all(flat_results)

    def to_dict(self):
        """Return a dictionary mapping each AtomGraph property to a corresponding tensor/scalar.

        Any nested attributes of the AtomGraphs are unpacked so the
        returned dict has keys like "positions" and "atomic_numbers".

        Any None attributes are not included in the dictionary.

        Returns:
            dict: A dictionary mapping attribute_name -> tensor/scalar
        """
        ret = {}
        for key, val in self._asdict().items():
            if val is None:
                continue
            if isinstance(val, dict):
                for k, v in val.items():
                    ret[k] = v
            else:
                ret[key] = val

        return ret

    def to_batch_dict(self) -> Dict[str, Any]:
        """Return a single dictionary mapping each AtomGraph property to a corresponding list of tensors/scalars.

        Returns:
            dict: A dict mapping attribute_name -> list of length batch_size containing tensors/scalars.
        """
        batch_dict = defaultdict(list)
        for graph in self.split(self):
            for key, value in graph.to_dict().items():
                batch_dict[key].append(value)
        return batch_dict

    def split(self, clone=True) -> List["AtomGraphs"]:
        """Splits batched AtomGraphs into constituent system AtomGraphs.

        Args:
            graphs (AtomGraphs): A batched AtomGraphs object.
            clone (bool): Whether to clone the graphs before splitting.
                Cloning removes risk of side-effects, but uses more memory.
        """
        graphs = self.clone() if clone else self

        batch_nodes = graphs.n_node.tolist()
        batch_edges = graphs.n_edge.tolist()

        if len(batch_nodes) == 0:
            raise ValueError("Cannot split empty batch")
        if len(batch_nodes) == 1:
            return [graphs]

        batch_systems = torch.ones(len(batch_nodes), dtype=torch.int).tolist()
        node_features = _split_features(graphs.node_features, batch_nodes)
        node_targets = _split_features(graphs.node_targets, batch_nodes)
        edge_features = _split_features(graphs.edge_features, batch_edges)
        edge_targets = _split_features(graphs.edge_targets, batch_edges)
        system_features = _split_features(graphs.system_features, batch_systems)
        system_targets = _split_features(graphs.system_targets, batch_systems)
        system_ids = _split_tensors(graphs.system_id, batch_systems)
        fix_atoms = _split_tensors(graphs.fix_atoms, batch_nodes)
        tags = _split_tensors(graphs.tags, batch_nodes)
        batch_nodes = [torch.tensor([n]) for n in batch_nodes]
        batch_edges = [torch.tensor([e]) for e in batch_edges]
        max_num_neighbors = [torch.tensor([m]) for m in graphs.max_num_neighbors]

        # calculate the new senders and receivers
        senders = list(_split_tensors(graphs.senders, batch_edges))
        receivers = list(_split_tensors(graphs.receivers, batch_edges))
        n_graphs = graphs.n_node.shape[0]
        offsets = torch.cumsum(graphs.n_node[:-1], 0)
        offsets = torch.cat([torch.tensor([0], device=offsets.device), offsets])
        unbatched_senders = []
        unbatched_recievers = []
        for graph_index in range(n_graphs):
            s = senders[graph_index] - offsets[graph_index]
            r = receivers[graph_index] - offsets[graph_index]
            unbatched_senders.append(s)
            unbatched_recievers.append(r)

        return [
            AtomGraphs(*args)
            for args in zip(
                unbatched_senders,
                unbatched_recievers,
                batch_nodes,
                batch_edges,
                node_features,
                edge_features,
                system_features,
                node_targets,
                edge_targets,
                system_targets,
                system_ids,
                fix_atoms,
                tags,
                [graphs.radius for _ in range(len(batch_nodes))],
                max_num_neighbors,
                [graphs.half_supercell for _ in range(len(batch_nodes))],
                strict=True,
            )
        ]


def batch_graphs(graphs: List[AtomGraphs]) -> AtomGraphs:
    """Batch graphs together by concatenating their nodes, edges, and features.

    Args:
        graphs (List[AtomGraphs]): A list of AtomGraphs to be batched together.

    Returns:
        AtomGraphs: A new AtomGraphs object with the concatenated nodes,
        edges, and features from the input graphs, along with concatenated target,
        system ID, and other information.
    """
    # Calculates offsets for sender and receiver arrays, caused by concatenating
    # the nodes arrays.
    offsets = torch.cumsum(
        torch.tensor([0] + [torch.sum(g.n_node) for g in graphs[:-1]]), 0
    )
    radius = graphs[0].radius
    assert set([graph.radius for graph in graphs]) == {radius}

    return AtomGraphs(
        n_node=torch.concatenate([g.n_node for g in graphs]).long(),
        n_edge=torch.concatenate([g.n_edge for g in graphs]).long(),
        senders=torch.concatenate(
            [g.senders + o for g, o in zip(graphs, offsets)]
        ).long(),
        receivers=torch.concatenate(
            [g.receivers + o for g, o in zip(graphs, offsets)]
        ).long(),
        node_features=_map_concat([g.node_features for g in graphs]),
        edge_features=_map_concat([g.edge_features for g in graphs]),
        system_features=_map_concat([g.system_features for g in graphs]),
        node_targets=_map_concat([g.node_targets for g in graphs]),
        edge_targets=_map_concat([g.edge_targets for g in graphs]),
        system_targets=_map_concat([g.system_targets for g in graphs]),
        system_id=_concat([g.system_id for g in graphs]),
        fix_atoms=_concat([g.fix_atoms for g in graphs]),
        tags=_concat([g.tags for g in graphs]),
        radius=radius,
        max_num_neighbors=torch.concatenate(
            [g.max_num_neighbors for g in graphs]
        ).long(),
        half_supercell=graphs[0].half_supercell,
    )


def refeaturize_atomgraphs(
    atoms: AtomGraphs,
    positions: torch.Tensor,
    atomic_numbers_embedding: Optional[torch.Tensor] = None,
    cell: Optional[torch.Tensor] = None,
    recompute_neighbors=True,
    updates: Optional[torch.Tensor] = None,
    differentiable: bool = False,
) -> AtomGraphs:
    """Return a graph updated according to the new positions, and (if given) atomic numbers and unit cells.

    NOTE:
        - if 'cell' is specified, positions will be remapped using it.
        - if atoms.fix_atoms is not None, then atoms at those indices have
          *both* their positions and atomic embeddings held fixed.

    Args:
        atoms (AtomGraphs): The original AtomGraphs object.
        positions (torch.Tensor): The new positions of the atoms.
        atomic_numbers_embedding (Optional[torch.Tensor]): The new atomic number embeddings.
        cell (Optional[torch.Tensor]): The new unit cell.
        recompute_neighbors (bool): Whether to recompute the neighbor list.
        updates (Optional[torch.Tensor]): The updates to the positions.
        differentiable (bool): Whether to make the graph inputs require_grad. This includes
            the positions and atomic number embeddings, if passed.

    Returns:
        AtomGraphs: A refeaturized AtomGraphs object.
    """
    original_device = atoms.positions.device
    original_dtype = atoms.positions.dtype

    if cell is None:
        cell = atoms.cell

    if atoms.fix_atoms is not None:
        positions[atoms.fix_atoms] = atoms.positions[atoms.fix_atoms]
        if atomic_numbers_embedding is not None:
            atomic_numbers_embedding[atoms.fix_atoms] = atoms.atomic_numbers_embedding[
                atoms.fix_atoms
            ]

    num_atoms = atoms.n_node
    positions = featurization_utilities.batch_map_to_pbc_cell(
        positions, cell, num_atoms
    )

    if differentiable:
        positions.requires_grad = True
        if atomic_numbers_embedding is not None:
            atomic_numbers_embedding.requires_grad = True

    if recompute_neighbors:
        assert atoms.radius is not None and atoms.max_num_neighbors is not None

        (
            edge_index,
            edge_vectors,
            unit_shifts,
            batch_num_edges,
        ) = featurization_utilities.batch_compute_pbc_radius_graph(
            positions=positions,
            cells=cell,
            pbc=atoms.pbc,
            radius=atoms.radius,
            n_node=num_atoms,
            max_number_neighbors=atoms.max_num_neighbors,
            half_supercell=atoms.half_supercell,
        )
        new_senders = edge_index[0]
        new_receivers = edge_index[1]
    else:
        assert updates is not None
        new_senders = atoms.senders
        new_receivers = atoms.receivers
        edge_vectors = recompute_edge_vectors(atoms, updates)
        batch_num_edges = atoms.n_edge

    edge_features = {"vectors": edge_vectors, "unit_shifts": unit_shifts}

    new_node_features = {}
    if atoms.node_features is not None:
        new_node_features = deepcopy(atoms.node_features)
    new_node_features["positions"] = positions
    if atomic_numbers_embedding is not None:
        new_node_features["atomic_numbers_embedding"] = atomic_numbers_embedding

    new_system_features = {}
    if atoms.system_features is not None:
        new_system_features = deepcopy(atoms.system_features)
    new_system_features["cell"] = cell

    new_atoms = AtomGraphs(
        senders=new_senders,
        receivers=new_receivers,
        n_node=atoms.n_node,
        n_edge=batch_num_edges,
        node_features=new_node_features,
        edge_features=edge_features,
        system_features=new_system_features,
        node_targets=atoms.node_targets,
        edge_targets=atoms.edge_targets,
        system_targets=atoms.system_targets,
        system_id=atoms.system_id,
        fix_atoms=atoms.fix_atoms,
        tags=atoms.tags,
        radius=atoms.radius,
        max_num_neighbors=atoms.max_num_neighbors,
        half_supercell=atoms.half_supercell,
    ).to(device=original_device, dtype=original_dtype)

    return new_atoms


def recompute_edge_vectors(atoms, updates):
    """Recomputes edge vectors with per node updates."""
    updates = -updates
    senders = atoms.senders
    receivers = atoms.receivers
    edge_translation = updates[senders] - updates[receivers]
    return atoms.edge_features["vectors"].cpu() + edge_translation.cpu()


def volume_atomgraphs(atoms: AtomGraphs):
    """Returns the volume of the unit cell."""
    cell = atoms.cell
    return (cell[:, 0] * torch.linalg.cross(cell[:, 1], cell[:, 2])).sum(-1)


def create_and_apply_stress_displacement(
    positions: torch.Tensor, cell: torch.Tensor, per_node_graph_indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create and apply a displacement s.t. stress = grad(E)_{displacement} / volume."""
    displacement = torch.zeros_like(cell)
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
    positions = positions + torch.bmm(
        positions.unsqueeze(1), symmetric_displacement[per_node_graph_indices]
    ).squeeze(1)
    cell = cell + torch.bmm(cell, symmetric_displacement)
    return positions, cell, displacement


def _map_concat(nests):
    concat = lambda *args: _concat(args)
    return tree.map_structure(concat, *nests)


def _concat(
    tensors: List[Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    """Concatenate tensors."""
    if any([x is None for x in tensors]):
        return None
    return torch.concat(tensors, dim=0)  # type: ignore


def _split_tensors(
    features: Optional[torch.Tensor],
    split_sizes: List[int],
) -> Sequence[Optional[torch.Tensor]]:
    """Splits tensors based on the intended split sizes."""
    if features is None:
        return [None] * len(split_sizes)

    return torch.split(features, split_sizes)


def _split_features(
    features: Optional[TensorDict],
    split_sizes: List[int],
) -> Sequence[Optional[TensorDict]]:
    """Splits features based on the intended split sizes."""
    if features is None:
        return [None] * len(split_sizes)
    elif features == {}:
        return [{}] * len(split_sizes)

    split_dict = {
        k: torch.split(v, split_sizes) if v is not None else [None] * len(split_sizes)  # type: ignore
        for k, v in features.items()
    }
    individual_tuples = zip(*[v for v in split_dict.values()])
    individual_dicts: List[Optional[TensorDict]] = list(
        map(lambda k: dict(zip(split_dict.keys(), k)), individual_tuples)
    )
    return individual_dicts
