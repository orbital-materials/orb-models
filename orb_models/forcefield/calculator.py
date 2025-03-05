from typing import Optional, Union

import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.data.vdw_alvarez import vdw_radii

from orb_models.forcefield.atomic_system import SystemConfig, ase_atoms_to_atom_graphs
from orb_models.forcefield.graph_regressor import GraphRegressor
from orb_models.forcefield.conservative_regressor import ConservativeForcefieldRegressor
from orb_models.forcefield.featurization_utilities import EdgeCreationMethod
from orb_models.utils import to_numpy


class ORBCalculator(Calculator):
    """ORB ASE Calculator."""

    def __init__(
        self,

    def __init__(
        self,
        model: Union[GraphRegressor, ConservativeForcefieldRegressor],
        *,
        conservative: Optional[bool] = None,
        edge_method: Optional[EdgeCreationMethod] = "knn_scipy",
        system_config: SystemConfig = SystemConfig(radius=6.0, max_num_neighbors=20),
        max_num_neighbors: Optional[int] = None,
        half_supercell: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
        directory: str = ".",
        return_bonding_graph: bool = False,  # Retained from local changes
    ):
        """Initializes the calculator.

        Args:
            model: The finetuned model to use for predictions.
            conservative (bool, optional): Enables conservative forcefield mode.
            edge_method (Optional[EdgeCreationMethod]): Method for edge creation (default: "knn_scipy").
            system_config (SystemConfig): The config defining how an atomic system is featurized.
            max_num_neighbors (Optional[int]): Maximum number of neighbors per atom.
            half_supercell (Optional[bool]): If True, considers only half of the periodic supercell.
            device (Optional[Union[torch.device, str]]): The device to use for the model.
            directory (str): The working directory for calculations.
            return_bonding_graph (bool): If True, includes a bonding graph in results.
        """
        model: Union[GraphRegressor, ConservativeForcefieldRegressor],
        *,
        conservative: Optional[bool] = None,
        edge_method: Optional[EdgeCreationMethod] = "knn_scipy",
        system_config: SystemConfig = SystemConfig(radius=6.0, max_num_neighbors=20),
        max_num_neighbors: Optional[int] = None,
        half_supercell: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
        directory: str = ".",

    def __init__(
        self,
        model: Union[GraphRegressor, ConservativeForcefieldRegressor],
        *,
        conservative: Optional[bool] = None,
        edge_method: Optional[EdgeCreationMethod] = "knn_scipy",
        system_config: SystemConfig = SystemConfig(radius=6.0, max_num_neighbors=20),
        max_num_neighbors: Optional[int] = None,
        half_supercell: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
        directory: str = ".",
        return_bonding_graph: bool = False,  # Retained from local changes
    ):
        """Initializes the calculator.

        Args:
            model: The finetuned model to use for predictions.
            conservative (bool, optional): Enables conservative forcefield mode.
            edge_method (Optional[EdgeCreationMethod]): Method for edge creation (default: "knn_scipy").
            system_config (SystemConfig): The config defining how an atomic system is featurized.
            max_num_neighbors (Optional[int]): Maximum number of neighbors per atom.
            half_supercell (Optional[bool]): If True, considers only half of the periodic supercell.
            device (Optional[Union[torch.device, str]]): The device to use for the model.
            directory (str): The working directory for calculations.
            return_bonding_graph (bool): If True, includes a bonding graph in results.
        """
            device (Optional[torch.device]): The device to use for the model.
            return_bonding_graph (bool): If True, includes a bonding graph in results.
            vdw_multiplier (float): Multiplier for the sum of VDW radii to determine bond cutoff (default: 0.5).
            **kwargs: Additional keyword arguments for the parent Calculator class.

    def __init__(
        self,
        model: Union[GraphRegressor, ConservativeForcefieldRegressor],
        *,
        conservative: Optional[bool] = None,
        edge_method: Optional[EdgeCreationMethod] = "knn_scipy",
        system_config: SystemConfig = SystemConfig(radius=6.0, max_num_neighbors=20),
        max_num_neighbors: Optional[int] = None,
        half_supercell: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
        directory: str = ".",
        return_bonding_graph: bool = False,  # Retained from local changes
    ):
        """Initializes the calculator.

        Args:
            model: The finetuned model to use for predictions.
            conservative (bool, optional): Enables conservative forcefield mode.
            edge_method (Optional[EdgeCreationMethod]): Method for edge creation (default: "knn_scipy").
            system_config (SystemConfig): The config defining how an atomic system is featurized.
            max_num_neighbors (Optional[int]): Maximum number of neighbors per atom.
            half_supercell (Optional[bool]): If True, considers only half of the periodic supercell.
            device (Optional[Union[torch.device, str]]): The device to use for the model.
            directory (str): The working directory for calculations.
            return_bonding_graph (bool): If True, includes a bonding graph in results.
        """
        """
        Calculator.__init__(self, directory=directory)
        self.results = {}  # type: ignore
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)  # type: ignore
        self.system_config = system_config

    def __init__(
        self,
        model: Union[GraphRegressor, ConservativeForcefieldRegressor],
        *,
        conservative: Optional[bool] = None,
        edge_method: Optional[EdgeCreationMethod] = "knn_scipy",
        system_config: SystemConfig = SystemConfig(radius=6.0, max_num_neighbors=20),
        max_num_neighbors: Optional[int] = None,
        half_supercell: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
        directory: str = ".",
        return_bonding_graph: bool = False,  # Retained from local changes
    ):
        """Initializes the calculator.

        Args:
            model: The finetuned model to use for predictions.
            conservative (bool, optional): Enables conservative forcefield mode.
            edge_method (Optional[EdgeCreationMethod]): Method for edge creation (default: "knn_scipy").
            system_config (SystemConfig): The config defining how an atomic system is featurized.
            max_num_neighbors (Optional[int]): Maximum number of neighbors per atom.
            half_supercell (Optional[bool]): If True, considers only half of the periodic supercell.
            device (Optional[Union[torch.device, str]]): The device to use for the model.
            directory (str): The working directory for calculations.
            return_bonding_graph (bool): If True, includes a bonding graph in results.
        """
        self.max_num_neighbors = max_num_neighbors
        self.edge_method = edge_method
        self.half_supercell = half_supercell
        self.conservative = conservative

        model_is_conservative = hasattr(self.model, "grad_forces_name")
        if self.conservative is None:
            self.conservative = model_is_conservative

        if self.conservative and not model_is_conservative:
            raise ValueError(
                "Conservative mode requested, but model is not a ConservativeForcefieldRegressor."
            )

        self.implemented_properties = model.properties  # type: ignore

    def __init__(
        self,
        model: Union[GraphRegressor, ConservativeForcefieldRegressor],
        *,
        conservative: Optional[bool] = None,
        edge_method: Optional[EdgeCreationMethod] = "knn_scipy",
        system_config: SystemConfig = SystemConfig(radius=6.0, max_num_neighbors=20),
        max_num_neighbors: Optional[int] = None,
        half_supercell: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
        directory: str = ".",
        return_bonding_graph: bool = False,  # Retained from local changes
    ):
        """Initializes the calculator.

        Args:
            model: The finetuned model to use for predictions.
            conservative (bool, optional): Enables conservative forcefield mode.
            edge_method (Optional[EdgeCreationMethod]): Method for edge creation (default: "knn_scipy").
            system_config (SystemConfig): The config defining how an atomic system is featurized.
            max_num_neighbors (Optional[int]): Maximum number of neighbors per atom.
            half_supercell (Optional[bool]): If True, considers only half of the periodic supercell.
            device (Optional[Union[torch.device, str]]): The device to use for the model.
            directory (str): The working directory for calculations.
            return_bonding_graph (bool): If True, includes a bonding graph in results.
        """
        if "forces" in self.implemented_properties:
            self.results["forces"] = out["node_pred"].detach().cpu().numpy()

        if "stress" in self.implemented_properties:
            raw_stress = out["stress_pred"].detach().cpu().numpy()
            # reshape from (1, 6) to (6,) if necessary
            self.results["stress"] = (
                raw_stress[0] if len(raw_stress.shape) > 1 else raw_stress
            )

        if self.return_bonding_graph:
            # Keep tensors on device
            atomic_numbers = batch.atomic_numbers
            senders = batch.senders
            receivers = batch.receivers
            vectors = batch.edge_features["vectors"]
            
            # Create VDW radii tensor on device with float32
            vdw_radii_tensor = torch.tensor(vdw_radii, device=self.device, dtype=torch.float32)
            
            # Calculate actual distances
            bond_lengths = torch.norm(vectors, dim=1)
            
            # Get VDW radii for each atom in the pair using device tensors
            sender_vdw = vdw_radii_tensor[atomic_numbers[senders]]
            receiver_vdw = vdw_radii_tensor[atomic_numbers[receivers]]
            
            # Use a fraction of the sum of VDW radii as the cutoff
            vdw_cutoff = self.vdw_multiplier * (sender_vdw + receiver_vdw)
            
            # Never bond H-H
            is_h_h = (atomic_numbers[senders] == 1) & (atomic_numbers[receivers] == 1)
            vdw_cutoff[is_h_h] = 0.0

            # **New: Save actual calculated values in the results dictionary**
            self.results["pair_bond_lengths"] = bond_lengths.cpu().numpy()
            self.results["pair_vdw_cutoffs"] = vdw_cutoff.cpu().numpy()
            self.results["pair_senders"] = senders.cpu().numpy()
            self.results["pair_receivers"] = receivers.cpu().numpy()
            
            # Create bonding matrix on device
            n_atoms = len(atoms)
            bonding_graph = torch.zeros((n_atoms, n_atoms), device=self.device)
            is_bonded = bond_lengths < vdw_cutoff
            
            # Fill the bonding matrix (symmetric)
            bonding_graph[senders[is_bonded], receivers[is_bonded]] = 1
            bonding_graph[receivers[is_bonded], senders[is_bonded]] = 1
            
            # Only convert to numpy at the very end
            self.results["bonding_graph"] = bonding_graph.cpu().numpy()

    def __init__(
        self,
        model: Union[GraphRegressor, ConservativeForcefieldRegressor],
        *,
        conservative: Optional[bool] = None,
        edge_method: Optional[EdgeCreationMethod] = "knn_scipy",
        system_config: SystemConfig = SystemConfig(radius=6.0, max_num_neighbors=20),
        max_num_neighbors: Optional[int] = None,
        half_supercell: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
        directory: str = ".",
        return_bonding_graph: bool = False,  # Retained from local changes
    ):
        """Initializes the calculator.

        Args:
            model: The finetuned model to use for predictions.
            conservative (bool, optional): Enables conservative forcefield mode.
            edge_method (Optional[EdgeCreationMethod]): Method for edge creation (default: "knn_scipy").
            system_config (SystemConfig): The config defining how an atomic system is featurized.
            max_num_neighbors (Optional[int]): Maximum number of neighbors per atom.
            half_supercell (Optional[bool]): If True, considers only half of the periodic supercell.
            device (Optional[Union[torch.device, str]]): The device to use for the model.
            directory (str): The working directory for calculations.
            return_bonding_graph (bool): If True, includes a bonding graph in results.
        """
