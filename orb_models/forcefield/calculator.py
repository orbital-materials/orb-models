from typing import Optional

import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.data.vdw_alvarez import vdw_radii

from orb_models.forcefield.atomic_system import SystemConfig, ase_atoms_to_atom_graphs
from orb_models.forcefield.graph_regressor import GraphRegressor


class ORBCalculator(Calculator):
    """ORB ASE Calculator.

    Args:
        model: torch.nn.Module finetuned Graph regressor
    """

    def __init__(
        self,
        model: GraphRegressor,
        brute_force_knn: Optional[bool] = None,
        system_config: SystemConfig = SystemConfig(radius=10.0, max_num_neighbors=20),
        device: Optional[torch.device] = None,
        return_bonding_graph: bool = False,
        vdw_multiplier: float = 0.5,  # Multiplier for the sum of VDW radii used as cutoff for bonding (default changed to 0.5)
        **kwargs,
    ):
        """Initializes the calculator.

        Args:
            model (GraphRegressor): The finetuned model to use for predictions.
            brute_force_knn: whether to use a 'brute force' k-nearest neighbors method for graph construction.
                Defaults to None, in which case brute_force is used if a GPU is available (2-6x faster),
                but not on CPU (1.5x faster - 4x slower). For very large systems (>10k atoms),
                brute_force may OOM on GPU, so it is recommended to set to False in that case.
            system_config (SystemConfig): The config defining how an atomic system is featurized.
            device (Optional[torch.device]): The device to use for the model.
            return_bonding_graph (bool): If True, includes a bonding graph in results.
            vdw_multiplier (float): Multiplier for the sum of VDW radii to determine bond cutoff (default: 0.5).
            **kwargs: Additional keyword arguments for the parent Calculator class.
        """
        Calculator.__init__(self, **kwargs)
        self.results = {}  # type: ignore
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.system_config = system_config
        self.brute_force_knn = brute_force_knn
        self.return_bonding_graph = return_bonding_graph
        self.vdw_multiplier = vdw_multiplier

        properties = []
        if model.node_head is not None:
            properties += ["energy", "free_energy"]
        if model.graph_head is not None:
            properties += ["forces"]
        if model.stress_head is not None:
            properties += ["stress"]
        if return_bonding_graph:
            properties += ["bonding_graph"]
        assert len(properties) > 0, "Model must have at least one output head."
        self.implemented_properties = properties

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """Calculate properties.

        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        batch = ase_atoms_to_atom_graphs(
            atoms,
            system_config=self.system_config,
            brute_force_knn=self.brute_force_knn,
            device=self.device,
        )
        self.model = self.model.to(self.device)  # type: ignore

        self.results = {}
        out = self.model.predict(batch)
        if "energy" in self.implemented_properties:
            self.results["energy"] = float(out["graph_pred"].detach().cpu().item())
            self.results["free_energy"] = self.results["energy"]

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
