import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import (
    Any,
    TypeVar,
)

import ase
import torch
import tree
from ase.calculators.singlepoint import SinglePointCalculator

from orb_models.common import TORCH_FLOAT_DTYPES

_T = TypeVar("_T", bound="AbstractAtomBatch")
TensorDict = Mapping[str, torch.Tensor | None]


class AbstractAtomBatch(ABC):
    """An abstract base class representing atomic systems as a collection of torch tensors.

    This class provides a common interface for storing and manipulating atomic systems,
    including properties like positions, atomic numbers, cell, and targets. It also provides
    a convenient API for standard tensor operations including batching, unbatching, cloning,
    device/dtype casting, and equality checking.

    Subclasses must implement two methods:
        - split: split the batch into a list of individual batch instances
        - batch: combine a list of individual batch instances into a single batch instance

    Splitting and batching are defined per-subclass because tensor shape logic is unique to
    each subclass and depends on the modelling approach. For instance, a GNN model may operate
    on flat tensors of concatenated node features, while a transformer may use an explicit
    batch dimension and pad each system to a maximum number of atoms.

    Args:
        n_node (torch.Tensor): A tensor of shape (batch_size,) representing the number of atoms in each system.
        node_features (Dict[str, torch.Tensor]): A dictionary of node feature tensors.
        system_features (Dict[str, torch.Tensor]): A dictionary of system feature tensors of shape (batch_size,)
        node_targets (Dict[str, torch.Tensor]): A dictionary of node target tensors.
        system_targets (Dict[str, torch.Tensor]): A dictionary of system target tensors of shape (batch_size,)
        system_id (Optional[torch.Tensor]): tensor of shape (batch_size,) containing dataset indices.
        fix_atoms (Optional[torch.Tensor]): tensor of shape (batch_size,) indicating whether each atom is fixed.
        tags (Optional[torch.Tensor]): An tensor of shape (batch_size,) conaining per-atom tags (like ase).
    """

    def __init__(
        self,
        n_node: torch.Tensor,
        node_features: dict[str, torch.Tensor],
        system_features: dict[str, torch.Tensor],
        node_targets: dict[str, torch.Tensor],
        system_targets: dict[str, torch.Tensor],
        system_id: torch.Tensor | None,
        fix_atoms: torch.Tensor | None,
        tags: torch.Tensor | None,
    ):
        """Initialize the AbstractAtomBatch instance."""
        self.n_node = n_node
        self.node_features = node_features
        self.system_features = system_features
        self.node_targets = node_targets
        self.system_targets = system_targets
        self.system_id = system_id
        self.fix_atoms = fix_atoms
        self.tags = tags

        # Total number of nodes in the batch
        self.batch_total_n_node = n_node.sum().item()

    @property
    def positions(self):
        """Get positions of atoms."""
        return self.node_features["positions"]

    @positions.setter
    def positions(self, val: torch.Tensor):
        self.node_features["positions"] = val

    @property
    def fractional_positions(self):
        """Get fractional positions of atoms."""
        return self.node_features["fractional_positions"]

    @fractional_positions.setter
    def fractional_positions(self, val: torch.Tensor):
        self.node_features["fractional_positions"] = val

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
        if self.system_features is None:
            return None
        return self.system_features.get("cell")

    @cell.setter
    def cell(self, val: torch.Tensor):
        if self.system_features is None:
            self.system_features = {}
        self.system_features["cell"] = val

    @property
    def pbc(self):
        """Get pbc."""
        if self.system_features is None:
            return None
        return self.system_features.get("pbc")

    @pbc.setter
    def pbc(self, val: torch.Tensor):
        if self.system_features is None:
            self.system_features = {}
        self.system_features["pbc"] = val

    @abstractmethod
    def split(self: _T, clone=True) -> list[_T]:
        """Splits batched AbstractAtomBatch into constituent system AbstractAtomBatchs.

        Args:
            clone (bool): Whether to clone the batch before splitting.
                Cloning removes risk of side-effects, but uses more memory.
        """
        pass

    @classmethod
    @abstractmethod
    def batch(cls: type[_T], items: list[_T]) -> _T:
        """Combine AbstractAtomBatchs together.

        Args:
            items (List[_T]): A list of AbstractAtomBatchs (or subclasses) to be batched together.

        Returns:
            _T: A new batched instance of the same class.
        """
        pass

    def clone(self: _T) -> _T:
        """Clone the AbstractAtomBatch object.

        WARNING: if two tensors stored on the batch are linked in the
        computational graph, then cloning both of them will break that link.
        """

        def _clone(x):
            if isinstance(x, torch.Tensor):
                return x.clone()
            else:
                return x

        return map_structure(_clone, self)

    def to(
        self: _T,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> _T:
        """Move AbstractAtomBatch child tensors to a device and/or dtype.

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

        return map_structure(_to, self)

    def detach(self: _T) -> _T:
        """Detach all child tensors."""

        def _detach(x):
            if hasattr(x, "detach"):
                return x.detach()
            else:
                return x

        return map_structure(_detach, self)

    def equals(self: _T, other: _T) -> bool:
        """Check two AbstractAtomBatchs are equal."""
        if self.__class__ != other.__class__:
            return False

        def _is_equal(x, y):
            if isinstance(x, torch.Tensor):
                # Handle potential device mismatch
                if x.device != y.device:
                    y = y.to(x.device)
                return torch.equal(x, y)
            else:
                return x == y

        flat_results = tree.flatten(tree.map_structure(_is_equal, vars(self), vars(other)))
        return all(flat_results)

    def allclose(self: _T, other: _T, rtol=1e-5, atol=1e-8) -> bool:
        """Check all tensors/scalars of two AbstractAtomBatchs are close."""
        if self.__class__ != other.__class__:
            return False

        def _is_close(x, y):
            if isinstance(x, torch.Tensor):
                # Handle potential device mismatch
                if x.device != y.device:
                    y = y.to(x.device)
                # Handle potential dtype mismatch for comparison
                if x.dtype != y.dtype:
                    if torch.is_floating_point(x) and torch.is_floating_point(y):
                        # Promote to common float type if both are float
                        common_dtype = torch.promote_types(x.dtype, y.dtype)
                        x = x.to(common_dtype)
                        y = y.to(common_dtype)
                    elif (
                        x.dtype == torch.int64
                        and y.dtype == torch.int32
                        or x.dtype == torch.int32
                        and y.dtype == torch.int64
                    ):
                        # Allow int32/int64 comparison
                        pass
                    else:
                        # Dtypes mismatch in a way we don't automatically handle
                        return False
                return torch.allclose(x, y, rtol=rtol, atol=atol)
            elif isinstance(x, (float, int)):
                return torch.allclose(torch.tensor(x), torch.tensor(y), rtol=rtol, atol=atol)
            else:
                # Fallback for other types (like bool, None)
                return x == y

        mapped = tree.flatten(tree.map_structure(_is_close, vars(self), vars(other)))
        return all(mapped)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary mapping each AbstractAtomBatch property to a corresponding tensor/scalar.

        Any nested attributes of the AbstractAtomBatch are unpacked so the
        returned dict has keys like "positions" and "atomic_numbers".

        Any None attributes are not included in the dictionary.

        Returns:
            dict: A dictionary mapping attribute_name -> tensor/scalar
        """
        ret = {}
        for key, val in vars(self).items():
            if val is None:
                continue
            if isinstance(val, dict):
                # Unpack dictionaries like node_features, system_features etc.
                for k, v in val.items():
                    if v is not None:  # Don't add None values from unpacked dicts
                        ret[k] = v
            else:
                ret[key] = val
        return ret

    def to_batch_dict(self) -> dict[str, list[Any]]:
        """Return a single dictionary mapping each AbstractAtomBatch property to a list of tensors/scalars.

        Returns:
            dict: A dict mapping attribute_name -> list of length batch_size containing tensors/scalars.
        """
        batch_dict = defaultdict(list)
        for item in type(self).split(self):
            for key, value in item.to_dict().items():
                batch_dict[key].append(value)
        # Convert defaultdict back to dict for consistent return type
        return dict(batch_dict)

    def to_ase_atoms(
        self,
        energy: torch.Tensor | None = None,
        forces: torch.Tensor | None = None,
        stress: torch.Tensor | None = None,
    ) -> list[ase.Atoms]:
        """Converts a list of graphs to a list of ase.Atoms.

        Args:
            graphs: List of AbstractAtomBatch objects
            **kwargs: Additional keyword arguments.

        Returns:
            List of ase.Atoms objects
        """
        graphs = self.to("cpu")

        atomic_numbers = torch.argmax(graphs.atomic_numbers_embedding.detach(), dim=-1)
        atomic_numbers_split = torch.split(atomic_numbers, graphs.n_node.tolist())
        positions_split = torch.split(graphs.positions.detach(), graphs.n_node.tolist())

        if graphs.cell is None:
            cells = [None] * len(atomic_numbers_split)
        else:
            cells = graphs.cell.detach()
        if graphs.pbc is None:
            pbcs = [None] * len(atomic_numbers_split)
        else:
            pbcs = graphs.pbc.detach()
        if graphs.tags is None:
            tags: Iterable[Any] = [None] * len(atomic_numbers_split)
        else:
            tags = torch.split(graphs.tags.detach(), graphs.n_node.tolist())
        if graphs.system_id is None:
            system_ids: Iterable[Any] = [None] * len(atomic_numbers_split)
        else:
            system_ids = graphs.system_id

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
        for index, (n, p, c, t, pbc, system_id) in enumerate(
            zip(
                atomic_numbers_split,
                positions_split,
                cells,
                tags,
                pbcs,
                system_ids,
                strict=True,
            )
        ):
            info = {}
            if system_id is not None:
                info["system_id"] = (
                    system_id.item() if isinstance(system_id, torch.Tensor) else system_id
                )
            atoms = ase.Atoms(numbers=n, positions=p, cell=c, tags=t, pbc=pbc, info=info)
            if calculations != {}:
                # note: important to save scalar energy as a float not array
                spc = SinglePointCalculator(
                    atoms=atoms,
                    **{
                        key: (
                            val[index].item() if val[index].nelement() == 1 else val[index].numpy()
                        )
                        for key, val in calculations.items()
                    },
                )
                atoms.calc = spc
            atoms_list.append(atoms)

        return atoms_list

    def volume(self):
        """Returns the volume of the unit cell."""
        cell = self.cell
        if cell is None:
            raise ValueError("Cannot compute volume without a unit cell.")

        # Handle batch dimension: cell might be (3, 3) or (batch, 3, 3)
        if cell.ndim == 2:
            cell = cell.unsqueeze(0)  # Add batch dimension if missing
        elif cell.ndim != 3:
            raise ValueError(f"Expected cell to have 2 or 3 dimensions, got {cell.ndim}")

        # Compute volume using the scalar triple product for each cell in the batch
        # Ensure cross product operands are correct: cell[:, 1, :] and cell[:, 2, :]
        cross_product = torch.linalg.cross(cell[:, 1, :], cell[:, 2, :], dim=1)
        # Dot product with the first vector: cell[:, 0, :]
        # Use einsum for batched dot product: 'bi,bi->b'
        volume = torch.einsum("bi,bi->b", cell[:, 0, :], cross_product)

        # Return scalar if input was single cell, else return tensor
        return volume.squeeze(0) if self.cell.ndim == 2 else volume


def map_structure(fn: Callable, batch: AbstractAtomBatch):
    """Apply a function to each element of a nested structure.

    HACK: apply tree map_structure to vars(batch) and then construct a new instance of batch.
    Ideally, we would just directly call map_structure(fn, batch). This would work if batch was a
    NamedTuple, but we explictly avoid this because NamedTuples do not support inheritance.
    And we cannot use dataclasses either, since they aren't supported by tree or torch DDP.
    """
    mapped = tree.map_structure(fn, vars(batch))
    # Only provide properties as arguments to the constructor, if the class accepts them.
    # Other properties must be derived from the constructor arguments.
    constructor_args = inspect.signature(type(batch)).parameters
    mapped = {k: v for k, v in mapped.items() if k in constructor_args}
    return type(batch)(**mapped)
