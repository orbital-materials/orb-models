"""Classes that define prediction targets."""

from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Tuple,
    Union,
    List,
    Optional,
    MutableMapping,
    MutableSequence,
)
from dataclasses import dataclass

import ase.data
import ase.db
import ase.db.row
import ase.db.sqlite
import numpy as np
import torch


def recursive_getattr(obj: object, attr: str) -> Any:
    """Recursively access an object property using dot notation."""
    for sub_attr in attr.split("."):
        obj = getattr(obj, sub_attr)

    return obj


def get_property_from_row(
    name: Union[str, List[str]],
    row: ase.db.row.AtomsRow,
    conversion_factor: float = 1.0,
    conversion_shift: float = 0.0,
) -> torch.Tensor:
    """Retrieve arbitary values from ase db data dict."""
    if isinstance(name, str):
        names = [name]
    else:
        names = name
    values = []
    for name in names:
        attribute = recursive_getattr(row, name)
        target = np.array(attribute)
        values.append(target)

    property_tensor = torch.from_numpy(np.hstack(values))

    if property_tensor.shape == torch.Size([3, 3]) and name == "data.d3.stress":
        # Some of our D3 data has (3, 3) shaped stress tensors.
        # This block of code reshapes the stress tensor to (6,) voigt notation.
        property_tensor = torch.tensor(
            [
                property_tensor[0, 0],
                property_tensor[1, 1],
                property_tensor[2, 2],
                property_tensor[1, 2],
                property_tensor[0, 2],
                property_tensor[0, 1],
            ],
            dtype=torch.float64,
        )

    valid_graph_shape = len(property_tensor.shape) == 1
    valid_node_shape = (
        len(property_tensor.shape) == 2 and property_tensor.shape[0] == row.natoms
    )
    if not (valid_graph_shape or valid_node_shape):
        raise ValueError(
            f"Property {name} has invalid shape {property_tensor.shape} for row {row.id}"
        )
    assert (
        property_tensor.dtype != torch.float32
    ), "All properties should be highest precision i.e. float64, not float32"

    return (property_tensor * conversion_factor) + conversion_shift


@dataclass
class PropertyDefinition:
    """Defines how to extract and transform a quantative property from an ase db.

    Such properties have two primary use-cases:
        - as features for the model to use / condition on.
        - as target variables for regression tasks.

    Args:
        name: The name of the property.
        dim: The dimensionality of the property variable.
        domain: Whether the variable is real, binary or categorical. If using
            this variable as a regression target, then var_type determines
            the loss function used e.g. MSE, BCE or cross-entropy loss.
        row_to_property_fn: A function defining how a target can be
            retrieved from an ase database row and dataset name.
        means: The mean to transform this by in the model.
        stds: The std to scale this by in the model.
        level_of_theory: The level of DFT theory used to compute this property.
    """

    name: str
    dim: int
    domain: Literal["real", "binary", "categorical"]
    row_to_property_fn: Callable[[ase.db.row.AtomsRow, str], torch.Tensor]
    means: Optional[torch.Tensor] = None
    stds: Optional[torch.Tensor] = None
    level_of_theory: Optional[str] = None

    @property
    def fullname(self) -> str:
        """Return the <name>-<level_of_theory> if level_of_theory is defined."""
        return (
            f"{self.name}-{self.level_of_theory}" if self.level_of_theory else self.name
        )


@dataclass
class Extractor:
    """Defines which property to extract from an ase db row and how to transform it."""

    name: str
    mult: float = 1.0
    bias: float = 0.0


def energy_row_fn(row: ase.db.row.AtomsRow, dataset: str):
    """Energy data in eV.

    - Some datasets use sums of energy values e.g. PBE + D3.
    - For external datasets, we should explicitly register how
      to extract the energy property by adding it to `extract_info'.
    - Unregistered datasets default to using the `energy` attribute
      and a conversion factor of 1, which is always correct for our
      internally generated datasets.
    """
    extract_info: Dict[str, Extractor] = {}

    if dataset not in extract_info:
        if not hasattr(row, "energy"):
            raise ValueError(
                f"db row {row.id} of {dataset} doesn't have an energy attribute "
                "and we haven't defined an alternative method to extract energy info."
            )
        return get_property_from_row("energy", row, 1)  # type: ignore

    extractor = extract_info[dataset]
    energy = get_property_from_row(  # type: ignore
        extractor.name, row, extractor.mult, extractor.bias
    )
    return energy


def forces_row_fn(row: ase.db.row.AtomsRow, dataset: str):
    """Force data in eV / Angstrom.

    - For certain external datasets, we need to specify how to extract and
      convert a row's forces inside the `extract_info' dictionary below.
    - Otherwise, the default behaviour is to use a row's `forces` attribute
      and use a conversion factor of 1, which is always correct for our
      internally generated datasets.
    """
    extract_info: Dict[str, Tuple] = {}
    if dataset not in extract_info:
        if not hasattr(row, "forces"):
            raise ValueError(
                f"db row {row.id} of {dataset} doesn't have a forces attribute, "
                "and we haven't defined an alternative method to extract forces."
            )
        return get_property_from_row("forces", row, 1)  # type: ignore

    row_attribute, conversion_factor = extract_info[dataset]
    forces = get_property_from_row(row_attribute, row, conversion_factor)  # type: ignore
    return forces


def stress_row_fn(row: ase.db.row.AtomsRow, dataset: str):
    """Extract stress data.

    - For certain external datasets, we need to specify how to extract and
      convert a row's stress inside the `extract_info' dictionary below.
    - Otherwise, the default behaviour is to use a row's `stress` attribute
      and use a conversion factor of 1, which is always correct for our
      internally generated datasets.
    """
    extract_info: Dict[str, List[Tuple]] = {}
    if dataset not in extract_info:
        if not hasattr(row, "stress"):
            raise ValueError(
                f"db row {row.id} of {dataset} doesn't have an stress attribute "
                "and we haven't defined an alternative method to extract stress info."
            )
        return get_property_from_row("stress", row, 1)  # type: ignore

    row_attribute, conversion_factor = extract_info[dataset]
    stress = get_property_from_row(row_attribute, row, conversion_factor)  # type: ignore
    return stress


def test_fixture_node_row_fn(row: ase.db.row.AtomsRow, dataset: str):
    """Just return random noise."""
    pos = torch.from_numpy(row.toatoms().positions)
    return torch.rand_like(pos)


def test_fixture_graph_row_fn(row: ase.db.row.AtomsRow, dataset: str):
    """Just return random noise."""
    return torch.randn((1, 1))


energy = PropertyDefinition(
    name="energy",
    dim=1,
    domain="real",
    row_to_property_fn=energy_row_fn,
    # means + stds are inited by reference energy class
)

forces = PropertyDefinition(
    name="forces",
    dim=3,
    domain="real",
    row_to_property_fn=forces_row_fn,
    # means + stds are learned from scratch
)

stress = PropertyDefinition(
    name="stress",
    dim=6,
    domain="real",
    row_to_property_fn=stress_row_fn,
    # means + stds are learned from scratch
)

energy_d3_zero = PropertyDefinition(
    name="energy",
    dim=1,
    domain="real",
    row_to_property_fn=lambda row, dataset: get_property_from_row(
        "data.d3.energy", row
    ),
    level_of_theory="d3-zero",
)

forces_d3_zero = PropertyDefinition(
    name="forces",
    dim=3,
    domain="real",
    row_to_property_fn=lambda row, dataset: get_property_from_row(
        "data.d3.forces", row
    ),
    level_of_theory="d3-zero",
)

stress_d3_zero = PropertyDefinition(
    name="stress",
    dim=6,
    domain="real",
    row_to_property_fn=lambda row, dataset: get_property_from_row(
        "data.d3.stress", row
    ),
    level_of_theory="d3-zero",
)

test_fixture = PropertyDefinition(
    name="test-fixture",
    dim=3,
    domain="real",
    row_to_property_fn=test_fixture_node_row_fn,
)

test_graph_fixture = PropertyDefinition(
    name="test-graph-fixture",
    dim=1,
    domain="real",
    row_to_property_fn=test_fixture_graph_row_fn,
)


PROPERTIES = {
    energy.fullname: energy,
    forces.fullname: forces,
    stress.fullname: stress,
    energy_d3_zero.fullname: energy_d3_zero,
    forces_d3_zero.fullname: forces_d3_zero,
    stress_d3_zero.fullname: stress_d3_zero,
    test_fixture.fullname: test_fixture,
    test_graph_fixture.fullname: test_graph_fixture,
}


@dataclass
class PropertyConfig:
    """Defines which properties should be extracted and stored on the AtomGraphs batch.

    These are numerical physical properties that can be used as features/targets for a model.
    """

    def __init__(
        self,
        node_names: Optional[List[str]] = None,
        edge_names: Optional[List[str]] = None,
        graph_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize PropertyConfig.

        Args:
            node_names: List of node property names in PROPERTIES
            edge_names: List of edge property names in PROPERTIES
            graph_names: List of graph property names in PROPERTIES
        """
        self.node_properties = {name: PROPERTIES[name] for name in (node_names or [])}
        self.edge_properties = {name: PROPERTIES[name] for name in (edge_names or [])}
        self.graph_properties = {name: PROPERTIES[name] for name in (graph_names or [])}

    def extract(
        self, row: ase.db.row.AtomsRow, dataset_name: str, suffix: Optional[str] = None
    ) -> Dict:
        """Extract properties from a row in an ase db."""
        all_properties = {}
        for type in ["node", "edge", "graph"]:
            props = getattr(self, f"{type}_properties")
            key = type if suffix is None else f"{type}_{suffix}"
            all_properties[key] = {
                name: p.row_to_property_fn(row=row, dataset=dataset_name)
                for name, p in props.items()
            }
        return all_properties


def instantiate_property_config(
    config: Optional[MutableMapping[Any, Any]] = None,
) -> PropertyConfig:
    """Get PropertyConfig object from Dict config."""
    if config is None:
        return PropertyConfig()
    assert all(
        key in ["node", "edge", "graph"] for key in config
    ), "Only node, edge and graph properties are supported."

    node_properties = edge_properties = graph_properties = None
    if config.get("node"):
        assert isinstance(config["node"], MutableSequence)
        node_properties = [name for name in config["node"]]
    if config.get("edge"):
        assert isinstance(config["edge"], MutableSequence)
        edge_properties = [name for name in config["edge"]]
    if config.get("graph"):
        assert isinstance(config["graph"], MutableSequence)
        graph_properties = [name for name in config["graph"]]
    return PropertyConfig(
        node_names=node_properties,
        edge_names=edge_properties,
        graph_names=graph_properties,
    )
