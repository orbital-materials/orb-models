from typing import Optional, List, Dict
from dataclasses import dataclass

import ase
from ase import constraints
from ase.calculators.singlepoint import SinglePointCalculator

from orb_models.forcefield.property_definitions import PROPERTIES, PropertyDefinition
from orb_models.forcefield import featurization_utilities
from orb_models.forcefield.base import AtomGraphs
import torch


@dataclass
class SystemConfig:
    """Config controlling how to featurize a system of atoms.

    Args:
        radius: radius for edge construction
        max_num_neighbors: maximum number of neighbours each node can send messages to.
        use_timestep_0: (unused - purely for compatibility with internal models)
    """

    radius: float
    max_num_neighbors: int
    use_timestep_0: bool = True


@dataclass
class PropertyConfig:
    """Defines which properties should be calculated and stored on the AtomGraphs batch.

    These are numerical physical properties that can be used as features/targets for a model.
    """

    def __init__(
        self,
        node_names: Optional[List[str]] = None,
        edge_names: Optional[List[str]] = None,
        graph_names: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Initialize PropertyConfig.

        Args:
            node_names: List of node property names in PROPERTIES
            edge_names: List of edge property names in PROPERTIES
            graph_names: List of graph property names in PROPERTIES
            **kwargs: Additional keyword arguments
        """
        if node_names is not None:
            self.node_properties: Optional[Dict[str, PropertyDefinition]] = {
                name: PROPERTIES[name] for name in node_names
            }
        else:
            self.node_properties = None
        if edge_names is not None:
            self.edge_properties: Optional[Dict[str, PropertyDefinition]] = {
                name: PROPERTIES[name] for name in edge_names
            }
        else:
            self.edge_properties = None
        if graph_names is not None:
            self.graph_properties: Optional[Dict[str, PropertyDefinition]] = {
                name: PROPERTIES[name] for name in graph_names
            }
        else:
            self.graph_properties = None


def atom_graphs_to_ase_atoms(
    graphs: AtomGraphs,
    energy: Optional[torch.Tensor] = None,
    forces: Optional[torch.Tensor] = None,
    stress: Optional[torch.Tensor] = None,
) -> List[ase.Atoms]:
    """Converts a list of graphs to a list of ase.Atoms."""
    graphs = graphs.to("cpu")
    if "atomic_numbers_embedding" in graphs.node_features:
        atomic_numbers = torch.argmax(
            graphs.node_features["atomic_numbers_embedding"], dim=-1
        )
    else:
        atomic_numbers = graphs.node_features["atomic_numbers"]
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
    system_config: SystemConfig = SystemConfig(
        radius=10.0, max_num_neighbors=20, use_timestep_0=True
    ),
    system_id: Optional[int] = None,
    brute_force_knn: Optional[bool] = None,
) -> AtomGraphs:
    """Generate AtomGraphs from an ase.Atoms object.

    Args:
        atoms: ase.Atoms object
        system_config: SystemConfig object
        system_id: Optional system_id
        brute_force_knn: whether to use a 'brute force' knn approach with torch.cdist for kdtree construction.
            Defaults to None, in which case brute_force is used if we a GPU is avaiable (2-6x faster),
            but not on CPU (1.5x faster - 4x slower). For very large systems, brute_force may OOM on GPU,
            so it is recommended to set to False in that case.

    Returns:
        AtomGraphs object
    """
    atomic_numbers = torch.from_numpy(atoms.numbers).to(torch.long)
    atom_type_embedding = torch.nn.functional.one_hot(
        atomic_numbers, num_classes=118
    ).type(torch.float32)

    node_feats = {
        "atomic_numbers": atomic_numbers.to(torch.int64),
        "atomic_numbers_embedding": atom_type_embedding.to(torch.float32),
        # NOTE: positions are stored as features on the AtomGraphs,
        # but not actually used as input features to the model.
        "positions": torch.from_numpy(atoms.positions).to(torch.float32),
    }
    system_feats = {"cell": torch.Tensor(atoms.cell.array[None, ...]).to(torch.float)}
    edge_feats, senders, receivers = _get_edge_feats(
        node_feats["positions"],  # type: ignore
        system_feats["cell"][0],
        system_config.radius,
        system_config.max_num_neighbors,
        brute_force=brute_force_knn,
    )

    num_atoms = len(node_feats["positions"])  # type: ignore
    return AtomGraphs(
        senders=senders,
        receivers=receivers,
        n_node=torch.tensor([num_atoms]),
        n_edge=torch.tensor([len(senders)]),
        node_features=node_feats,
        edge_features=edge_feats,
        system_features=system_feats,
        system_id=torch.LongTensor([system_id]) if system_id is not None else system_id,
        fix_atoms=ase_fix_atoms_to_tensor(atoms),
        tags=_get_ase_tags(atoms),
        radius=system_config.radius,
        max_num_neighbors=system_config.max_num_neighbors,
    )


def _get_edge_feats(
    positions: torch.Tensor,
    cell: torch.Tensor,
    radius: float,
    max_num_neighbours: int,
    brute_force: Optional[bool] = None,
):
    """Get edge features.

    Args:
        positions: (n_nodes, 3) positions tensor
        cell: 3x3 tensor unit cell for a system
        radius: radius for edge construction
        max_num_neighbours: maximum number of neighbours each node can send messages to.
        n_kdtree_workers: number of workers to use for kdtree construction.
        brute_force: whether to use brute force for kdtree construction.
    """
    # Construct a graph from a 3x3 supercell (as opposed to an infinite supercell).
    # This could be innaccurate for thin unit cells, but we have yet to encounter a
    # major issue and this approach is faster.
    (
        edge_index,
        edge_vectors,
    ) = featurization_utilities.compute_pbc_radius_graph(
        positions=positions,
        periodic_boundaries=cell,
        radius=radius,
        max_number_neighbors=max_num_neighbours,
        brute_force=brute_force,
    )
    edge_feats = {
        "vectors": edge_vectors.to(torch.float32),
    }
    senders, receivers = edge_index[0], edge_index[1]
    return edge_feats, senders, receivers


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


def make_property_definitions_from_config(
    config: Optional[Dict] = None,
) -> PropertyConfig:
    """Get PropertyConfig object from config."""
    if config is None:
        return PropertyConfig()
    assert all(
        key in ["node", "edge", "graph"] for key in config
    ), "Only node, edge and graph properties are supported."

    node_properties = edge_properties = graph_properties = None
    if config.get("node"):
        node_properties = [name for name in config["node"]]
    if config.get("edge"):
        edge_properties = [name for name in config["edge"]]
    if config.get("graph"):
        graph_properties = [name for name in config["graph"]]
    return PropertyConfig(
        node_names=node_properties,
        edge_names=edge_properties,
        graph_names=graph_properties,
    )
