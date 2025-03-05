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
        model: Union[GraphRegressor, ConservativeForcefieldRegressor],
        *,
        conservative: Optional[bool] = None,
        edge_method: Optional[EdgeCreationMethod] = "knn_scipy",
        system_config: SystemConfig = SystemConfig(radius=6.0, max_num_neighbors=20),
        max_num_neighbors: Optional[int] = None,
        half_supercell: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
        return_bonding_graph: bool = False,
        vdw_multiplier: float = 0.5,  # Multiplier for the sum of VDW radii used as cutoff for bonding
        directory: str = ".",
    ):
        """Initializes the calculator.

        Args:
            model: The finetuned model to use for predictions.
            conservative (bool, optional):
                - Defaults to True if the model is a ConservativeForcefieldRegressor, otherwise False.
                - If True, conservative forces and stresses are computed as the gradient of the energy.
                  An error is raised if the model is not a ConservativeForcefieldRegressor.
                - If False, direct force and stress predictions are used, not gradient-based ones.
            edge_method (EdgeCreationMethod, optional): The method to use for graph edge construction.
                If None then knn_brute_force is used if tensors are on GPU (2-6x faster),
                otherwise defaults to knn_scipy. For very large systems, knn_brute_force may OOM on GPU.
            system_config (SystemConfig): The config defining how an atomic system is featurized.
            max_num_neighbors (int): The maximum number of neighbors for each atom.
                Larger values should generally increase performace, but the gains may be marginal,
                whilst the increse in latency could be significant (depending on num atoms).
                    - Defaults to system_config.max_num_neighbors.
                    - 120 is sufficient to capture all edges under 6A across all systems in mp-traj validation set.
            half_supercell (bool): Whether to use half the supercell for graph construction, and then symmetrize.
                Defaults to None, in which case half_supercells are used when num atoms > 5k.
                This flag does not affect the resulting graph; it is purely an optimization that can double
                throughput and half memory for very large cells (e.g. 5k+ atoms). For smaller systems, it can harm
                performance due to additional computation to enforce max_num_neighbors.
            device (Optional[torch.device], optional): The device to use for the model.
            return_bonding_graph (bool): If True, includes a bonding graph in results.
            vdw_multiplier (float): Multiplier for the sum of VDW radii to determine bond cutoff (default: 0.5).
            directory (Optional[str], optional): Working directory in which to read and write files and
                perform calculations.
        """
        Calculator.__init__(self, directory=directory)
        self.results = {}  # type: ignore
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)  # type: ignore
        self.system_config = system_config
        self.max_num_neighbors = max_num_neighbors
        self.edge_method = edge_method
        self.half_supercell = half_supercell
        self.conservative = conservative
        self.return_bonding_graph = return_bonding_graph
        self.vdw_multiplier = vdw_multiplier

        model_is_conservative = hasattr(self.model, "grad_forces_name")
        if self.conservative is None:
            self.conservative = model_is_conservative

        if self.conservative and not model_is_conservative:
            raise ValueError(
                "Conservative mode requested, but model is not a ConservativeForcefieldRegressor."
            )

        self.implemented_properties = model.properties  # type: ignore

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """Calculate properties.

        Args:
            atoms (ase.Atoms): ASE Atoms object.
            properties (list of str): Properties to be computed, used by ASE internally.
            system_changes (list of str): System changes since last calculation, used by ASE internally.

        Returns:
            None. Results are stored in self.results.
        """
        Calculator.calculate(self, atoms)

        half_supercell = (
            len(atoms.positions) >= 5_000
            if self.half_supercell is None
            else self.half_supercell
        )
        batch = ase_atoms_to_atom_graphs(
            atoms,
            system_config=self.system_config,
            max_num_neighbors=self.max_num_neighbors,
            edge_method=self.edge_method,
            half_supercell=half_supercell,
            device=self.device,
        )
        batch = batch.to(self.device)  # type: ignore
        out = self.model.predict(batch)  # type: ignore
        self.results = {}
        for property in self.implemented_properties:
            _property = "energy" if property == "free_energy" else property
            self.results[property] = to_numpy(out[_property].squeeze())

        if self.conservative:
            self.results["direct_forces"] = self.results["forces"]
            self.results["direct_stress"] = self.results["stress"]
            self.results["forces"] = self.results[self.model.grad_forces_name]
            self.results["stress"] = self.results[self.model.grad_stress_name]

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

            # Save actual calculated values in the results dictionary
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
