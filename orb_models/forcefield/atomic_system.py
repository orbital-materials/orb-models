from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict
from copy import deepcopy

import ase
import torch
from ase import constraints
from ase.geometry.cell import cell_to_cellpar
from ase.calculators.singlepoint import SinglePointCalculator

from orb_models.forcefield import featurization_utilities as feat_util
from orb_models.forcefield.base import AtomGraphs
from orb_models.forcefield.featurization_utilities import EdgeCreationMethod


@dataclass
class SystemConfig:
    """Config controlling how to featurize a system of atoms.

    Args:
        radius: radius for edge construction
        max_num_neighbors: maximum number of neighbours each node can send messages to.
    """

    radius: float
    max_num_neighbors: int

def atom_graphs_to_ase_atoms(
    graphs: AtomGraphs,
    energy: Optional[torch.Tensor] = None,
    forces: Optional[torch.Tensor] = None,
    stress: Optional[torch.Tensor] = None,
) -> List[ase.Atoms]:
    """Converts a list of graphs to a list of ase.Atoms."""
    graphs = graphs.to("cpu")

    atomic_numbers = torch.argmax(graphs.atomic_numbers_embedding, dim=-1)
    atomic_numbers_split = torch.split(atomic_numbers, graphs.n_node.tolist())
    positions_split = torch.split(graphs.positions, graphs.n_node.tolist())
    assert graphs.tags is not None and graphs.system_features is not None
    tags = torch.split(graphs.tags, graphs.n_node.tolist())

    calculations = {}
    if energy is not None:
        energy_list = torch.unbind(energy.cpu().detach())
        assert len(energy_list) == len(atomic_numbers_split)
        calculations["energy"] = energy_list
    if forces is not None:
        forces_list = torch.split(forces.cpu().detach(), graphs.n_node.tolist())
        assert len(forces_list) == len(atomic_numbers_split)
        calculations["forces"] = forces_list  # type: ignore
    if stress is not None:
        stress_list = torch.unbind(stress.cpu().detach())
        assert len(stress_list) == len(atomic_numbers_split)
        calculations["stress"] = stress_list

    atoms_list = []
    for index, (n, p, c, t) in enumerate(
        zip(atomic_numbers_split, positions_split, graphs.cell, tags)
    ):
        atoms = ase.Atoms(
            numbers=n.detach(),
            positions=p.detach(),
            cell=c.detach(),
            tags=t.detach(),
            pbc=torch.any(c != 0),
        )
        if calculations != {}:
            # note: important to save scalar energy as a float not array
            spc = SinglePointCalculator(
                atoms=atoms,
                **{
                    key: (
                        val[index].item()
                        if val[index].nelement() == 1
                        else val[index].numpy()
                    )
                    for key, val in calculations.items()
                },
            )
            atoms.calc = spc
        atoms_list.append(atoms)

    return atoms_list

def ase_atoms_to_atom_graphs(
    atoms: ase.Atoms,
    system_config: SystemConfig,
    *,
    wrap: bool = True,
    edge_method: Optional[EdgeCreationMethod] = None,
    max_num_neighbors: Optional[int] = None,
    system_id: Optional[int] = None,
    half_supercell: bool = False,
    device: Optional[torch.device] = None,
    output_dtype: Optional[torch.dtype] = None,
    graph_construction_dtype: Optional[torch.dtype] = None,
) -> AtomGraphs:
    """Convert an ase.Atoms object into AtomGraphs format, ready for use in a model.

    Args:
        atoms: ase.Atoms object
        wrap: whether to wrap atomic positions into the central unit cell (if there is one).
        edge_method (EdgeCreationMethod, optional): The method to use for graph edge construction.
            If None, the edge method is chosen as follows:
            * knn_brute_force: If device is not CPU, and cuML is not installed or num_atoms is < 5000 (PBC)
                or < 30000 (non-PBC).
            * knn_cuml_rbc: If device is not CPU, and cuML is installed, and num_atoms is >= 5000 (PBC) or
                >= 30000 (non-PBC).
            * knn_scipy (default): If device is CPU.
            On GPU, for num_atoms ≲ 5000 (PBC) or ≲ 30000 (non-PBC), knn_brute_force is faster than knn_cuml_*,
            but uses more memory. For num_atoms ≳ 5000 (PBC) or ≳ 30000 (non-PBC), knn_cuml_* is faster and uses
            less memory, but requires cuML to be installed. knn_scipy is typically fastest on the CPU.
        system_config: The system configuration to use for graph construction.
        max_num_neighbors: Maximum number of neighbors each node can send messages to.
            If None, will use system_config.max_num_neighbors.
        system_id: Optional index that is relative to a particular dataset.
        half_supercell (bool): Whether to use half the supercell for graph construction, and then symmetrize.
            This flag does not affect the resulting graph; it is purely an optimization that can double
            throughput and half memory for very large cells (e.g. 10k+ atoms). For smaller systems, it can harm
            performance due to additional computation to enforce max_num_neighbors.
        device: The device to put the tensors on.
        output_dtype: The dtype to use for all floating point tensors stored on the AtomGraphs object.
        graph_construction_dtype: The dtype to use for floating point tensors in the graph construction.
    Returns:
        AtomGraphs object
    """
    if isinstance(atoms.pbc, Iterable) and any(atoms.pbc) and not all(atoms.pbc):
        raise NotImplementedError(
            "We do not support periodicity along a subset of axes. Please ensure atoms.pbc is "
            "True/False for all axes and you have padded your systems with sufficient vacuum if necessary."
        )
    output_dtype = torch.get_default_dtype() if output_dtype is None else output_dtype
    graph_construction_dtype = (
        torch.get_default_dtype()
        if graph_construction_dtype is None
        else graph_construction_dtype
    )
    if output_dtype == torch.float64:
        # when using fp64 precision, we must ensure all features + targets
        # stored in the atoms.info dict are already fp64
        _check_floating_point_tensors_are_fp64(atoms.info)

    max_num_neighbors = max_num_neighbors or system_config.max_num_neighbors
    atomic_numbers = torch.from_numpy(atoms.numbers).to(torch.long)
    atomic_numbers_embedding = atoms.info.get("node_features", {}).get(
        "atomic_numbers_embedding",
        feat_util.get_atom_embedding(atoms, k_hot=False),
    )
    positions = torch.from_numpy(atoms.positions)
    cell = torch.from_numpy(atoms.cell.array)
    pbc = torch.from_numpy(atoms.pbc)
    lattice = torch.from_numpy(cell_to_cellpar(cell))
    if wrap and (torch.any(cell != 0) and torch.any(pbc)):
        positions = feat_util.map_to_pbc_cell(positions, cell)

    edge_index, edge_vectors, unit_shifts = feat_util.compute_pbc_radius_graph(
        positions=positions,
        cell=cell,
        pbc=pbc,
        radius=system_config.radius,
        max_number_neighbors=max_num_neighbors,
        edge_method=edge_method,
        half_supercell=half_supercell,
        float_dtype=graph_construction_dtype,
        device=device,
    )
    senders, receivers = edge_index[0], edge_index[1]

    node_feats = {
        **atoms.info.get("node_features", {}),
        # NOTE: positions are stored as features on the AtomGraphs,
        # but not actually used as input features to the model.
        "positions": positions,
        "atomic_numbers": atomic_numbers.to(torch.long),
        "atomic_numbers_embedding": atomic_numbers_embedding,
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
        "lattice": lattice,
    }

    # Add a batch dimension to non-scalar graph features/targets
    graph_feats = {
        k: v.unsqueeze(0) if v.numel() > 1 else v for k, v in graph_feats.items()
    }
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
        fix_atoms=ase_fix_atoms_to_tensor(atoms),
        tags=_get_ase_tags(atoms),
        radius=system_config.radius,
        max_num_neighbors=torch.tensor([max_num_neighbors]),
        system_id=torch.LongTensor([system_id]) if system_id is not None else system_id,
    ).to(device=device, dtype=output_dtype)


def _get_ase_tags(atoms: ase.Atoms) -> torch.Tensor:
    """Get tags from ase.Atoms object."""
    tags = atoms.get_tags()
    if tags is not None:
        tags = torch.Tensor(tags)
    else:
        tags = torch.zeros(len(atoms))
    return tags


def ase_fix_atoms_to_tensor(atoms: ase.Atoms) -> Optional[torch.Tensor]:
    """Get fixed atoms from ase.Atoms object."""
    fixed_atoms = None
    if atoms.constraints is not None and len(atoms.constraints) > 0:
        constraint = atoms.constraints[0]
        if isinstance(constraint, constraints.FixAtoms):
            fixed_atoms = torch.zeros((len(atoms)), dtype=torch.bool)
            fixed_atoms[constraint.index] = True
    return fixed_atoms


def _check_floating_point_tensors_are_fp64(obj):
    """Recursively check that all floating point tensors are fp64."""
    if isinstance(obj, torch.Tensor) and torch.is_floating_point(obj):
        if obj.dtype != torch.float64:
            raise ValueError("All torch tensors stored in atoms.info must be fp64")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _check_floating_point_tensors_are_fp64(v)
