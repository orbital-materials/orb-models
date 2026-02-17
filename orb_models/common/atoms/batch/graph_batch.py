from collections.abc import Sequence
from copy import deepcopy
from typing import NamedTuple, TypeVar, overload

import torch
import tree

import orb_models.common.atoms.featurization
from orb_models.common.atoms import graph_featurization
from orb_models.common.atoms.batch.abstract_batch import AbstractAtomBatch, TensorDict
from orb_models.common.torch_utils import torch_lexsort

_T = TypeVar("_T", bound="AtomGraphs")


class AtomGraphs(AbstractAtomBatch):
    """A batch of atomic systems represented as graphs with edge vectors and indices.

    Args:
        senders (torch.Tensor): The integer source nodes for each edge.
        receivers (torch.Tensor): The integer destination nodes for each edge.
        n_node (torch.Tensor): the number of atoms in each system.
        n_edge (torch.Tensor): A (batch_size, ) shaped tensor containing the number of edges per graph.
        node_features (Dict[str, torch.Tensor]): A dictionary of node feature tensors.
        edge_features (Dict[str, torch.Tensor]): A dictionary containing edge feature tensors.
        system_features (Dict[str, torch.Tensor]): A dictionary of system feature tensors of shape (batch_size,)
        node_targets (Dict[str, torch.Tensor]): A dictionary of node target tensors.
        edge_targets (Dict[str, torch.Tensor]): A dict of tensors containing targets for
            individual edges. This tensor is commonly expected to have (num_edges, *).
        system_targets (Dict[str, torch.Tensor]): A dictionary of system target tensors of shape (batch_size,)
        system_id (Optional[torch.Tensor]): tensor of shape (batch_size,) containing dataset indices.
        fix_atoms (Optional[torch.Tensor]): tensor of shape (batch_size,) indicating whether each atom is fixed.
        tags (Optional[torch.Tensor]): An tensor of shape (batch_size,) conaining per-atom tags (like ase).
        radius (float): The radius used for neighbor calculation.
        max_num_neighbors (int): The maximum number of neighbors used for the neighbor calculation.
        half_supercell (bool): Whether to use half the supercell for graph construction, and then symmetrize.
    """

    def __init__(
        self,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        n_node: torch.Tensor,
        n_edge: torch.Tensor,
        node_features: dict[str, torch.Tensor],
        edge_features: dict[str, torch.Tensor],
        system_features: dict[str, torch.Tensor],
        node_targets: dict[str, torch.Tensor],
        edge_targets: dict[str, torch.Tensor],
        system_targets: dict[str, torch.Tensor],
        system_id: torch.Tensor | None,
        fix_atoms: torch.Tensor | None,
        tags: torch.Tensor | None,
        radius: float,
        max_num_neighbors: torch.Tensor,
        half_supercell: bool = False,  # FTODO (BEN): remove
    ):
        """Initialize the AtomGraphs instance."""
        super().__init__(
            n_node=n_node,
            node_features=node_features,
            system_features=system_features,
            node_targets=node_targets,
            system_targets=system_targets,
            system_id=system_id,
            fix_atoms=fix_atoms,
            tags=tags,
        )
        self.senders = senders
        self.receivers = receivers
        self.n_edge = n_edge
        self.edge_features = edge_features
        self.edge_targets = edge_targets
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        self.half_supercell = half_supercell

        # Create an index from n_node, so we can use it instead of
        # repeat_interleave, e.g. tensor[node_batch_index] instead of tensor.repeat_interleave(n_node)
        self.node_batch_index = torch.arange(
            self.n_node.shape[0], dtype=torch.int64, device=self.n_node.device
        ).repeat_interleave(self.n_node)

    def split(self: _T, clone=True) -> list["AtomGraphs"]:
        """Splits batched AtomGraphs into constituent system AtomGraphs.

        Args:
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
        unbatched_receivers = []
        for graph_index in range(n_graphs):
            s = senders[graph_index] - offsets[graph_index]
            r = receivers[graph_index] - offsets[graph_index]
            unbatched_senders.append(s)
            unbatched_receivers.append(r)

        return [
            AtomGraphs(*args)
            for args in zip(
                unbatched_senders,
                unbatched_receivers,
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

    @classmethod
    def batch(cls: type[_T], graphs: list[_T]) -> _T:
        """Batch graphs together by concatenating their nodes, edges, and features.

        Args:
            graphs (List[AtomGraphs]): A list of AtomGraphs to be batched together.

        Returns:
            AtomGraphs: A new AtomGraphs object with the concatenated nodes,
                edges, and features from the input graphs, along with concatenated
                target, system ID, and other information.
        """
        if not graphs:
            raise ValueError("Cannot batch an empty list of graphs.")

        # Calculates offsets for sender and receiver arrays,
        # caused by concatenating the nodes arrays.
        offsets = torch.cumsum(
            torch.cat(
                [torch.tensor([0], device=graphs[0].n_node.device)]
                + [g.n_node for g in graphs[:-1]]
            ),
            0,
        )

        radius = graphs[0].radius
        if not all(graph.radius == radius for graph in graphs):
            raise ValueError("All graphs in the batch must have the same radius.")
        half_supercell = graphs[0].half_supercell
        if not all(graph.half_supercell == half_supercell for graph in graphs):
            raise ValueError("All graphs in the batch must have the same half_supercell flag.")

        n_edge = torch.cat([g.n_edge for g in graphs]).long()
        senders = torch.cat([g.senders + o for g, o in zip(graphs, offsets, strict=False)]).long()
        receivers = torch.cat(
            [g.receivers + o for g, o in zip(graphs, offsets, strict=False)]
        ).long()
        edge_features = _map_concat([g.edge_features for g in graphs])
        edge_targets = _map_concat([g.edge_targets for g in graphs])

        max_num_neighbors = torch.cat(
            [g.max_num_neighbors for g in graphs]
        ).long()  # Should already be tensor per graph

        return cls(
            n_node=_concat([g.n_node for g in graphs]).long(),  # type: ignore
            node_features=_map_concat([g.node_features for g in graphs]),
            system_features=_map_concat([g.system_features for g in graphs]),
            node_targets=_map_concat([g.node_targets for g in graphs]),
            system_targets=_map_concat([g.system_targets for g in graphs]),
            system_id=_concat([g.system_id for g in graphs]),
            fix_atoms=_concat([g.fix_atoms for g in graphs]),
            tags=_concat([g.tags for g in graphs]),
            senders=senders,
            receivers=receivers,
            n_edge=n_edge,
            edge_features=edge_features,
            edge_targets=edge_targets,
            radius=radius,
            max_num_neighbors=max_num_neighbors,
            half_supercell=half_supercell,
        )

    def _get_per_node_graph_indices(self):
        """Get the graph index for each node in the system.

        TODO: we could cache these indices.
        """
        positions = self.node_features["positions"]
        graph_indices = torch.zeros(len(positions), device=positions.device, dtype=torch.int)
        cumsums = torch.cumsum(self.n_node, dim=0)
        graph_indices[:] = torch.searchsorted(
            cumsums,
            torch.arange(len(positions), device=positions.device),
            right=True,
        )
        return graph_indices  # (natoms,)

    def _get_per_edge_graph_indices(self):
        """Get the graph index for each edge in the system.

        TODO: we could cache these indices, like senders & receivers.
        """
        graph_indices = torch.zeros_like(self.senders)  # (nedges,)
        cumsums = torch.cumsum(self.n_edge, dim=0)  # (ngraphs,)
        graph_indices[:] = torch.searchsorted(
            cumsums,
            torch.arange(len(self.senders), device=self.senders.device),
            right=True,
        )
        return graph_indices  # (nedges,)

    # FTODO (BEN): should not be a member function. Move to forcefield
    def compute_differentiable_edge_vectors(
        self,
        use_stress_displacement: bool = True,
        use_rotation: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Compute pbc-aware edge vectors such that gradients flow back to positions.

        Args:
            use_stress_displacement (bool): If True, a zero 'displacement' tensor is created
                and matrix-multiplied with positions and cell, such that:
                stress = grad(E)_{displacement} / volume
            use_rotation (bool): If True, apply a differentiable rotation.
        """
        positions = self.node_features["positions"]  # (natoms, 3)
        positions.requires_grad_(True)
        unit_shifts = self.edge_features["unit_shifts"].unsqueeze(1)  # (nedges, 1, 3)
        cell = self.system_features.get("cell") if self.system_features else None  # (ngraphs, 3, 3)
        if cell is None:
            raise ValueError(
                "Cell must be present in system_features for differentiable edge vectors."
            )

        stress_displacement = None
        if use_stress_displacement:
            per_node_graph_indices = self._get_per_node_graph_indices()
            positions, cell, stress_displacement = create_and_apply_stress_displacement(
                positions, cell, per_node_graph_indices
            )

        equigrad_generator = None
        if use_rotation:
            per_node_graph_indices = self._get_per_node_graph_indices()
            equigrad_generator = torch.zeros_like(cell, requires_grad=True)
            rotation = orb_models.common.atoms.featurization.rotation_from_generator(
                equigrad_generator,
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

        return vectors, stress_displacement, equigrad_generator

    def check_edges_allclose(self, other: _T, msg: str = "", tol: float = 1e-6):
        """Assert that the edges of two atomgraphs are equal."""
        # Ensure edge_features and vectors exist
        if "vectors" not in self.edge_features or "vectors" not in other.edge_features:
            raise ValueError("Cannot check edge closeness without 'vectors' in edge_features.")

        edges1 = Edges(
            self.senders,
            self.receivers,
            self.edge_features["vectors"].norm(dim=-1),
            self.edge_features["vectors"],
        )
        edges2 = Edges(
            other.senders,
            other.receivers,
            other.edge_features["vectors"].norm(dim=-1),
            other.edge_features["vectors"],
        )
        edges1.check_allclose(edges2, msg, tol=tol)


@overload
def _split_tensors(features: torch.Tensor, split_sizes: list[int]) -> Sequence[torch.Tensor]: ...
@overload
def _split_tensors(features: None, split_sizes: list[int]) -> Sequence[None]: ...
def _split_tensors(
    features: torch.Tensor | None, split_sizes: list[int]
) -> Sequence[torch.Tensor | None]:
    """Splits tensors based on the intended split sizes."""
    if features is None:
        return [None] * len(split_sizes)

    return torch.split(features, split_sizes)


@overload
def _split_features(features: TensorDict, split_sizes: list[int]) -> Sequence[TensorDict]: ...
@overload
def _split_features(features: None, split_sizes: list[int]) -> Sequence[None]: ...
def _split_features(
    features: TensorDict | None, split_sizes: list[int]
) -> Sequence[TensorDict | None]:
    """Splits features based on the intended split sizes."""
    if features is None:
        return [None] * len(split_sizes)
    elif features == {}:
        return [{}] * len(split_sizes)

    split_dict = {
        k: torch.split(v, split_sizes) if v is not None else [None] * len(split_sizes)  # type: ignore
        for k, v in features.items()
    }
    individual_tuples = zip(*[v for v in split_dict.values()], strict=False)
    individual_dicts: list[TensorDict | None] = list(
        map(lambda k: dict(zip(split_dict.keys(), k, strict=False)), individual_tuples)
    )
    return individual_dicts


def _concat(tensors: list[torch.Tensor | None]) -> torch.Tensor | None:
    """Concatenate tensors."""
    if any([x is None for x in tensors]):
        return None
    return torch.concat(tensors, dim=0)  # type: ignore


def _map_concat(nests):
    """Map concat over a nested structure."""
    concat = lambda *args: _concat(args)
    return tree.map_structure(concat, *nests)


def refeaturize_atomgraphs(
    atoms: AtomGraphs,
    positions: torch.Tensor,
    atomic_numbers_embedding: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
    recompute_neighbors=True,
    updates: torch.Tensor | None = None,
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
    positions = orb_models.common.atoms.featurization.batch_map_to_pbc_cell(
        positions, cell, atoms.pbc, num_atoms
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
        ) = graph_featurization.batch_compute_pbc_radius_graph(
            positions=positions,
            cells=cell,
            pbcs=atoms.pbc,
            radius=atoms.radius,
            n_node=num_atoms,
            node_batch_index=atoms.node_batch_index,
            max_number_neighbors=atoms.max_num_neighbors,
            half_supercell=atoms.half_supercell,
        )
        new_senders = edge_index[0]
        new_receivers = edge_index[1]
    else:
        assert updates is not None
        new_senders = atoms.senders
        new_receivers = atoms.receivers
        edge_vectors = _recompute_edge_vectors(atoms, updates)
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


def _recompute_edge_vectors(atoms: AtomGraphs, updates: torch.Tensor) -> torch.Tensor:
    """Recomputes edge vectors with per node updates."""
    updates = -updates
    senders = atoms.senders
    receivers = atoms.receivers
    edge_translation = updates[senders] - updates[receivers]
    return atoms.edge_features["vectors"].cpu() + edge_translation.cpu()


# FTODO (BEN): relocate to forcefield
def create_and_apply_stress_displacement(
    positions: torch.Tensor, cell: torch.Tensor, per_node_graph_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create and apply a displacement s.t. stress = grad(E)_{displacement} / volume."""
    displacement = torch.zeros_like(cell)
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
    positions = positions + torch.bmm(
        positions.unsqueeze(1), symmetric_displacement[per_node_graph_indices]
    ).squeeze(1)
    cell = cell + torch.bmm(cell, symmetric_displacement)
    return positions, cell, displacement


class Edges(NamedTuple):
    """A minimal representation of edges that can be sorted and compared.

    Args:
        senders: Tensor of source node indices for each edge
        receivers: Tensor of destination node indices for each edge
        edge_lengths: Tensor of edge lengths
        edge_vectors: Tensor of edge vectors (displacement vectors between nodes)
    """

    senders: torch.Tensor
    receivers: torch.Tensor
    edge_lengths: torch.Tensor
    edge_vectors: torch.Tensor

    def sort(self) -> "Edges":
        """Sort the graph by senders, then receivers, then lengths, then edge-vector components."""
        # The last key in the sequence is used for the primary sort order, the
        # second-to-last key for the secondary sort order, and so on
        # Round edge vectors to 3 decimal places before sorting
        rounded_edge_vectors = torch.round(self.edge_vectors, decimals=3)
        sort_order = torch_lexsort(
            [
                rounded_edge_vectors[:, 2],  # z-component
                rounded_edge_vectors[:, 1],  # y-component
                rounded_edge_vectors[:, 0],  # x-component
                self.receivers,  # receiver index
                self.senders,  # sender index
            ]
        )
        return Edges(
            senders=self.senders[sort_order],
            receivers=self.receivers[sort_order],
            edge_vectors=self.edge_vectors[sort_order],
            edge_lengths=self.edge_lengths[sort_order],
        )

    def check_allclose(self, other: "Edges", msg: str = "", tol: float = 1e-6):
        """Check that two graphs are equal, raising errors with detailed differences if not."""
        # first check that the shapes are the same
        for field_name in self._fields:
            a = getattr(self, field_name)
            b = getattr(other, field_name)
            if a.shape != b.shape:
                raise ValueError(f"{field_name} have different shapes: {a.shape} != {b.shape}")

        # now check that the sorted values are the same
        graph1 = self.sort()
        graph2 = other.sort()
        for field_name in self._fields:
            a = getattr(graph1, field_name)
            b = getattr(graph2, field_name)
            diff = (a - b).abs() > tol
            if len(a.shape) == 1:
                mismatched_indices = torch.where(diff)[0]
            else:
                mismatched_indices = torch.where(diff.any(dim=1))[0]

            if mismatched_indices.numel() > 0:
                raise ValueError(
                    f"{field_name} do not match: \n {self._detailed_error_message(msg, graph1, graph2, mismatched_indices)}"  # noqa
                )

    def _detailed_error_message(self, msg, graph1, graph2, mismatched_indices):
        mismatch_msg = f"{msg}\n" if msg else ""
        mismatch_msg += (
            "\n" + "-" * 30 + "mismatched indices" + "-" * 30 + f"\n{mismatched_indices}\n"
        )
        for name in self._fields:
            v1 = getattr(graph1, name)[mismatched_indices]
            v2 = getattr(graph2, name)[mismatched_indices]
            mismatch_msg += "-" * 30 + f"Difference in {name}" + "-" * 30 + f"\n {v1 - v2}\n"
            mismatch_msg += "-" * 30 + name + "-" * 30 + f"\n Graph 1: {v1} \n Graph 2: {v2}\n"
        return mismatch_msg
