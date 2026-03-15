from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, cast, override

import ase
import torch

if TYPE_CHECKING:
    import torch_sim as ts

try:
    import torch_sim as ts

    _TORCH_SIM_AVAILABLE = True
except ImportError:
    ts = None  # type: ignore[assignment]
    _TORCH_SIM_AVAILABLE = False

from orb_models.common.atoms import featurization as feat_utils
from orb_models.common.atoms import graph_featurization as graph_feat
from orb_models.common.atoms.abstract_atoms_adapter import AbstractAtomsAdapter
from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.torch_utils import get_device


class ForcefieldAtomsAdapter(AbstractAtomsAdapter):
    """Adapter for converting external-library atomic representations to AtomGraphs for a forcefield model."""

    output_cls = AtomGraphs
    _deprecated_args = set()

    def __init__(
        self,
        radius: float,
        max_num_neighbors: int,
        min_num_neighbors: int | None = None,
        max_num_neighbors_alpha: float | None = None,
        extra_features: dict | None = None,
        **kwargs,
    ) -> None:
        """Initialize the ForcefieldAtomsAdapter.

        Args:
            radius: radius for edge construction.
            max_num_neighbors: maximum number of neighbours each node can send messages to.
                NOTE: if max_num_neighbors_alpha is set, then during training, this
                maximum will defines the upper limit of a distribution over number of neighbours.
            min_num_neighbors: min number of neighbours each node can send messages to.
                NOTE: if min_num_neighbors_alpha is set, then during training, this
                minimum will defines the upper limit of a distribution over number of neighbours.
                If not, min_num_neighbors would not be used.
            max_num_neighbors_alpha: parameter controlling how heavily-weighted our
                distribtution over number of neighbours is towards the maximum.
                - alpha = None means no distribution, always max_num_neighbors
                - alpha = 0 means uniform distribution
                - alpha = inf means all probability on the maximum
                - If neighbours are in the range [3, 20], then:
                    - alpha=1 places 5x more probability on 20 than 3
                    - alpha=2 places 30x more probability on 20 than 3
            extra_features: Dictionary specifying which extra features to use.
                Format: {'node': [...], 'edge': [...], 'graph': [...]}
        """
        super().__init__(
            radius=radius,
            max_num_neighbors=max_num_neighbors,
            extra_features=extra_features,
            **kwargs,
        )
        self.min_num_neighbors = min_num_neighbors
        self.max_num_neighbors_alpha = max_num_neighbors_alpha

    @override
    def from_ase_atoms(
        self,
        atoms: ase.Atoms,
        *,
        max_num_neighbors: int | None = None,
        edge_method: graph_feat.EdgeCreationMethod | None = None,
        half_supercell: bool | None = None,
        wrap: bool = True,
        device: torch.device | str | None = None,
        output_dtype: torch.dtype | None = None,
        graph_construction_dtype: torch.dtype | None = None,
        system_id: int | None = None,
    ) -> AtomGraphs:
        """Convert an ase.Atoms object into our internal AtomGraphs format, ready for use in a model.

        Args:
            atoms: ase.Atoms object
            max_num_neighbors: Maximum number of neighbors each node can send messages to.
                If None, will use self.max_num_neighbors.
            edge_method (EdgeCreationMethod, optional): The method to use for graph edge construction. If None, defaults to knn_alchemi.
            half_supercell (bool): Whether to use half the supercell for graph construction, and then symmetrize.
                This flag does not affect the resulting graph; it is purely an optimization that can double
                throughput and half memory for very large cells (e.g. 5k+ atoms). For smaller systems, it can harm
                performance due to additional computation to enforce max_num_neighbors.
            wrap: whether to wrap atomic positions into the central unit cell (if there is one).
            device: The device to put the tensors on.
            output_dtype: The dtype to use for all floating point tensors stored on the AtomGraphs object.
            graph_construction_dtype: The dtype to use for floating point tensors in the graph construction.
            system_id: Optional index that is relative to a particular dataset.
        """
        output_dtype = torch.get_default_dtype() if output_dtype is None else output_dtype
        graph_construction_dtype = (
            torch.get_default_dtype()
            if graph_construction_dtype is None
            else graph_construction_dtype
        )
        self._validate_inputs(atoms, output_dtype)

        positions = torch.from_numpy(atoms.positions)
        cell = torch.from_numpy(atoms.cell.array)
        pbc = torch.from_numpy(atoms.pbc)
        if wrap:
            positions = feat_utils.map_to_pbc_cell(positions, cell, pbc)

        max_num_neighbors = max_num_neighbors or self.max_num_neighbors
        assert self.radius is not None, "radius must be set"
        assert max_num_neighbors is not None, "max_num_neighbors must be set"
        edge_index, edge_vectors, unit_shifts = graph_feat.compute_pbc_radius_graph(
            positions=positions,
            cell=cell,
            pbc=pbc,
            radius=self.radius,
            max_number_neighbors=max_num_neighbors,
            edge_method=edge_method,
            half_supercell=half_supercell,
            float_dtype=graph_construction_dtype,
            device=device,
        )
        senders, receivers = edge_index[0], edge_index[1]

        node_feats = {
            **atoms.info.get("node_features", {}),
            "positions": positions,
            "atomic_numbers": torch.from_numpy(atoms.numbers).to(torch.long),
            "atomic_numbers_embedding": feat_utils.get_atom_embedding(atoms),
            "atom_identity": torch.arange(len(atoms)).to(torch.long),
        }
        edge_feats = {
            **atoms.info.get("edge_features", {}),
            "vectors": edge_vectors,
            "unit_shifts": unit_shifts,
        }
        graph_feats = {
            **atoms.info.get("graph_features", {}),
            "cell": cell,
            "pbc": pbc,
            **_get_charge_and_spin(atoms),
        }

        # Add a batch dimension to non-scalar graph features/targets
        graph_feats = {k: v.unsqueeze(0) if v.numel() > 1 else v for k, v in graph_feats.items()}
        graph_targets = {
            k: v.unsqueeze(0) if v.numel() > 1 else v
            for k, v in atoms.info.get("graph_targets", {}).items()
        }

        return AtomGraphs(
            senders=senders,
            receivers=receivers,
            n_node=torch.tensor([len(positions)]),
            n_edge=torch.tensor([len(senders)]),
            node_features=node_feats,
            edge_features=edge_feats,
            system_features=graph_feats,
            node_targets=deepcopy(atoms.info.get("node_targets", {})),
            edge_targets=deepcopy(atoms.info.get("edge_targets", {})),
            system_targets=deepcopy(graph_targets),
            fix_atoms=feat_utils.ase_fix_atoms_to_tensor(atoms),
            tags=feat_utils.get_ase_tags(atoms),
            radius=self.radius,
            max_num_neighbors=torch.tensor([max_num_neighbors]),
            system_id=(torch.LongTensor([system_id]) if system_id is not None else system_id),
        ).to(device=device, dtype=output_dtype)

    @override
    def from_ase_atoms_list(
        self,
        atoms: list[ase.Atoms],
        *,
        max_num_neighbors: int | None = None,
        edge_method: graph_feat.EdgeCreationMethod | None = None,
        wrap: bool = True,
        device: torch.device | str | None = None,
        output_dtype: torch.dtype | None = None,
        graph_construction_dtype: torch.dtype | None = None,
    ) -> AtomGraphs:
        """Convert a list of ase.Atoms into a single batched AtomGraphs using parallel graph construction.

        This method leverages the batched Alchemi-based graph construction for better performance
        compared to processing atoms one by one.

        Args:
            atoms: List of ase.Atoms objects.
            max_num_neighbors: Maximum number of neighbors each node can send messages to.
                If None, will use self.max_num_neighbors.
            edge_method: The method to use for graph edge construction. If None, defaults to knn_alchemi.
            wrap: Whether to wrap atomic positions into the central unit cell.
            device: The device to put the tensors on.
            output_dtype: The dtype to use for all floating point tensors on the AtomGraphs.
            graph_construction_dtype: The dtype to use for floating point tensors in graph construction.
        """
        if len(atoms) == 0:
            raise ValueError("atoms list must not be empty")

        # Fall back to sequential processing for single atom
        if len(atoms) == 1:
            return self.from_ase_atoms(
                atoms[0],
                edge_method=edge_method,
                wrap=wrap,
                device=device,
                output_dtype=output_dtype,
                graph_construction_dtype=graph_construction_dtype,
            )

        output_dtype = torch.get_default_dtype() if output_dtype is None else output_dtype
        graph_construction_dtype = (
            torch.get_default_dtype()
            if graph_construction_dtype is None
            else graph_construction_dtype
        )

        # Resolve device early so all tensors are on the same device
        resolved_device = get_device(device)

        for a in atoms:
            self._validate_inputs(a, output_dtype)

        # Extract per-system data
        all_positions = []
        all_cells = []
        all_pbcs = []
        all_atomic_numbers = []
        all_atomic_numbers_embedding = []
        all_fix_atoms = []
        all_tags = []
        n_atoms = []

        for a in atoms:
            positions_i = torch.from_numpy(a.positions)
            cell_i = torch.from_numpy(a.cell.array)
            pbc_i = torch.from_numpy(a.pbc)

            all_positions.append(positions_i)
            all_cells.append(cell_i)
            all_pbcs.append(pbc_i)
            all_atomic_numbers.append(torch.from_numpy(a.numbers).to(torch.long))
            all_atomic_numbers_embedding.append(feat_utils.get_atom_embedding(a))
            all_fix_atoms.append(feat_utils.ase_fix_atoms_to_tensor(a))
            all_tags.append(feat_utils.get_ase_tags(a))
            n_atoms.append(len(a))

        # Build batched tensors and move to resolved device
        positions = torch.cat(all_positions, dim=0).to(device=resolved_device)
        cells = torch.stack(all_cells, dim=0).to(device=resolved_device)
        pbcs = torch.stack(all_pbcs, dim=0).to(device=resolved_device)
        n_node = torch.tensor(n_atoms, dtype=torch.long, device=resolved_device)
        node_batch_index = torch.arange(
            len(atoms), dtype=torch.int64, device=resolved_device
        ).repeat_interleave(n_node)

        if wrap:
            positions = feat_utils.batch_map_to_pbc_cell(
                positions=positions, cell=cells, pbc=pbcs, n_node=n_node
            )

        max_num_neighbors = max_num_neighbors or self.max_num_neighbors
        assert max_num_neighbors is not None, "max_num_neighbors must be set"
        assert self.radius is not None, "radius must be set"
        max_num_neighbors_tensor = torch.full_like(n_node, fill_value=max_num_neighbors)
        (
            edge_index,
            edge_vectors,
            unit_shifts,
            batch_num_edges,
        ) = graph_feat.batch_compute_pbc_radius_graph(
            positions=positions.contiguous(),
            cells=cells,
            pbcs=pbcs,
            radius=torch.tensor([self.radius], device=resolved_device),
            max_number_neighbors=max_num_neighbors_tensor,
            n_node=n_node,
            node_batch_index=node_batch_index,
            edge_method=edge_method,
            device=resolved_device,
        )
        senders = edge_index[0].long()
        receivers = edge_index[1].long()

        atomic_numbers = torch.cat(all_atomic_numbers, dim=0)
        atomic_numbers_embedding = torch.cat(all_atomic_numbers_embedding, dim=0)

        # Concatenate fix_atoms: None if no system has constraints
        if any(f is not None for f in all_fix_atoms):
            fix_atoms = torch.cat(
                [
                    f if f is not None else torch.zeros(n, dtype=torch.bool)
                    for f, n in zip(all_fix_atoms, n_atoms, strict=True)
                ],
                dim=0,
            )
        else:
            fix_atoms = None

        tags = torch.cat(all_tags, dim=0)

        node_feats = {
            "positions": positions,
            "atomic_numbers": atomic_numbers,
            "atomic_numbers_embedding": atomic_numbers_embedding,
        }
        edge_feats = {
            "vectors": edge_vectors,
            "unit_shifts": unit_shifts.to(dtype=output_dtype),
        }
        graph_feats: dict[str, torch.Tensor] = {
            "cell": cells,
            "pbc": pbcs,
        }
        # Collect charge and spin: all-or-nothing semantics
        charge_spin_list = [_get_charge_and_spin(a) for a in atoms]
        has_charge_spin = [bool(cs) for cs in charge_spin_list]
        if any(has_charge_spin):
            if not all(has_charge_spin):
                raise ValueError("Either all atoms must have charge and spin, or none of them.")
            graph_feats["total_charge"] = torch.cat(
                [cs["total_charge"] for cs in charge_spin_list], dim=0
            )
            graph_feats["spin_multiplicity"] = torch.cat(
                [cs["spin_multiplicity"] for cs in charge_spin_list], dim=0
            )

        return AtomGraphs(
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=batch_num_edges,
            node_features=node_feats,
            edge_features=edge_feats,
            system_features=graph_feats,
            node_targets={},
            edge_targets={},
            system_targets={},
            system_id=None,
            fix_atoms=fix_atoms,
            tags=tags,
            radius=self.radius,
            max_num_neighbors=max_num_neighbors_tensor,
        ).to(device=resolved_device, dtype=output_dtype)

    def from_torchsim_state(
        self,
        state: ts.SimState,
        *,
        max_num_neighbors: int | None = None,
        edge_method: graph_feat.EdgeCreationMethod | None = None,
        wrap: bool = True,
        device: torch.device | str | None = None,
        output_dtype: torch.dtype | None = None,
        graph_construction_dtype: torch.dtype | None = None,
    ) -> AtomGraphs:
        """Convert a SimState object into AtomGraphs format, ready for use in an ORB model.

        Requires torch_sim to be installed. Install with: pip install torch-sim-atomistic

        Args:
            state: SimState object containing atomic positions, cell, and atomic numbers.
            max_num_neighbors: Maximum number of neighbors each node can send messages to.
                If None, will use self.max_num_neighbors.
            edge_method (EdgeCreationMethod, optional): The method to use for graph edge
                construction. If None, the edge method is chosen automatically.
            wrap: Whether to wrap atomic positions into the central unit cell.
            device: The device to put the tensors on.
            output_dtype: The dtype to use for all floating point tensors on the AtomGraphs.
            graph_construction_dtype: The dtype to use for floating point tensors in the
                graph construction.
        """
        if not _TORCH_SIM_AVAILABLE:
            raise ImportError(
                "torch_sim is required for from_torchsim_state(). "
                "Install it with: pip install orb-models[torchsim]"
            )
        output_dtype = torch.get_default_dtype() if output_dtype is None else output_dtype
        graph_construction_dtype = (
            torch.get_default_dtype()
            if graph_construction_dtype is None
            else graph_construction_dtype
        )

        n_node = state.n_atoms_per_system
        if state.system_idx is not None:
            node_batch_index = state.system_idx.contiguous()
        else:
            node_batch_index = torch.arange(
                n_node.shape[0], dtype=torch.int64, device=device
            ).repeat_interleave(n_node)

        positions = state.positions
        cell = state.row_vector_cell.contiguous()
        pbc = torch.repeat_interleave(
            cast(torch.Tensor, state.pbc).view(-1, 3), n_node.shape[0], dim=0
        )
        if wrap:
            positions = feat_utils.batch_map_to_pbc_cell(
                positions=positions, cell=cell, pbc=pbc, n_node=n_node
            )

        max_num_neighbors = max_num_neighbors or self.max_num_neighbors
        assert self.radius is not None, "radius must be set"
        assert max_num_neighbors is not None, "max_num_neighbors must be set"
        max_num_neighbors_tensor = torch.full_like(n_node, fill_value=max_num_neighbors)
        (
            edge_index,
            edge_vectors,
            unit_shifts,
            batch_num_edges,
        ) = graph_feat.batch_compute_pbc_radius_graph(
            positions=positions.contiguous(),
            cells=cell,
            pbcs=pbc,
            radius=torch.tensor([self.radius], device=device),
            max_number_neighbors=max_num_neighbors_tensor,
            n_node=n_node,
            node_batch_index=node_batch_index,
            edge_method=edge_method,
            device=device,
        )
        senders = edge_index[0].long()
        receivers = edge_index[1].long()

        atomic_numbers = state.atomic_numbers.long()
        atom_type_embedding = torch.nn.functional.one_hot(atomic_numbers, num_classes=118)
        atomic_numbers_embedding = atom_type_embedding.to(output_dtype)

        # Create features dictionaries
        node_feats = {
            "positions": positions,
            "atomic_numbers": atomic_numbers,
            "atomic_numbers_embedding": atomic_numbers_embedding,
        }
        edge_feats = {
            "vectors": edge_vectors,
            "unit_shifts": unit_shifts.to(dtype=output_dtype),
        }
        graph_feats = {
            "cell": cell,
            "pbc": pbc,
            **_get_charge_and_spin(state),
        }
        return AtomGraphs(
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=batch_num_edges,
            node_features=node_feats,
            edge_features=edge_feats,
            system_features=graph_feats,
            node_targets={},
            edge_targets={},
            system_targets={},
            system_id=None,
            fix_atoms=None,
            tags=None,
            radius=self.radius,
            max_num_neighbors=max_num_neighbors_tensor,
        ).to(device=device, dtype=output_dtype)

    def is_compatible_with(self, other: AbstractAtomsAdapter):
        """Check if this AtomsConstructor is compatible with another and print incompatibilities."""
        if not isinstance(other, ForcefieldAtomsAdapter):
            raise ValueError(f"Incompatible AtomsConstructor: {type(self)} != {type(other)}")
        errors = []
        if self.radius != other.radius:
            errors.append(f"Radius: {self.radius} != {other.radius}")
        if errors:
            error_message = "SystemConfig Incompatibilities found:\n" + "\n".join(errors)
            raise ValueError(error_message)

        return True


def _get_charge_and_spin(atoms: ase.Atoms | ts.SimState) -> dict[str, torch.Tensor]:
    out = {}
    if isinstance(atoms, ase.Atoms) and ("charge" in atoms.info or "spin" in atoms.info):
        assert "charge" in atoms.info and "spin" in atoms.info, (
            "Charge and spin must be present together"
        )

        chg, spin = atoms.info["charge"], atoms.info["spin"]
        assert isinstance(chg, (float, int)), "Charge must be a float or int"
        assert isinstance(spin, (float, int)), "Spin must be a float or int"
        out["total_charge"] = torch.tensor([chg], dtype=torch.get_default_dtype())
        out["spin_multiplicity"] = torch.tensor([spin], dtype=torch.get_default_dtype())
    elif (
        _TORCH_SIM_AVAILABLE
        and isinstance(atoms, ts.SimState)
        and (atoms.charge is not None or atoms.spin is not None)
    ):
        assert atoms.charge is not None and atoms.spin is not None, (
            "Charge and spin must be present together"
        )
        out["total_charge"] = atoms.charge
        out["spin_multiplicity"] = atoms.spin

    return out
