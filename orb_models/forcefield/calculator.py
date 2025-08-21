from typing import Dict, Optional, Union

import torch
from ase.calculators.calculator import Calculator, all_changes

from orb_models.forcefield.atomic_system import SystemConfig, ase_atoms_to_atom_graphs
from orb_models.forcefield.direct_regressor import DirectForcefieldRegressor
from orb_models.forcefield.conservative_regressor import ConservativeForcefieldRegressor
from orb_models.forcefield.featurization_utilities import EdgeCreationMethod
from orb_models.forcefield.nn_util import ChargeSpinConditioner
from orb_models.utils import to_numpy


class ORBCalculator(Calculator):
    """ORB ASE Calculator."""

    def __init__(
        self,
        model: Union[DirectForcefieldRegressor, ConservativeForcefieldRegressor],
        *,
        system_config: Optional[SystemConfig] = None,
        conservative: Optional[bool] = None,
        edge_method: Optional[EdgeCreationMethod] = None,
        max_num_neighbors: Optional[int] = None,
        half_supercell: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
        directory: str = ".",
    ):
        """Initializes the calculator.

        Args:
            model: The finetuned model to use for predictions.
            system_config (SystemConfig): The config defining how an atomic system is featurized.
                If None, the system config from the model is used.
            conservative (bool, optional):
                - Defaults to True if the model is a ConservativeForcefieldRegressor, otherwise False.
                - If True, conservative forces and stresses are computed as the gradient of the energy.
                  An error is raised if the model is not a ConservativeForcefieldRegressor.
                - If False, direct force and stress predictions are used, not gradient-based ones.
            max_number_neighbors (int): The maximum number of neighbors for each atom.
                Larger values should generally increase performace, but the gains may be marginal,
                whilst the increse in latency could be significant (depending on num atoms).
                    - Defaults to system_config.max_num_neighbors.
                    - 120 is sufficient to capture all edges under 6A across all systems in mp-traj validation set.
            edge_method (EdgeCreationMethod, optional): The method to use for graph edge construction.
                If None, the edge method is chosen dynamically based on the device and system size.
            half_supercell (bool): Whether to use half the supercell for graph construction, and then symmetrize.
                Defaults to None, in which case half_supercells are used when num_atoms > 5k.
                This flag does not affect the resulting graph; it is purely an optimization that can double
                throughput and half memory for very large cells (e.g. 5k+ atoms). For smaller systems, it can harm
                performance due to additional computation to enforce max_num_neighbors.
            device (Optional[torch.device], optional): The device to use for the model.
            directory (Optional[str], optional): Working directory in which to read and write files and
                perform calculations.
        """
        Calculator.__init__(self, directory=directory)
        self.results = {}  # type: ignore
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)  # type: ignore
        self.system_config = system_config or model.system_config
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
        
        conditioner = model.model.conditioner
        self.expects_charge_and_spin = (
            (conditioner is not None) and isinstance(conditioner, ChargeSpinConditioner)
        )

        self.implemented_properties = model.properties  # type: ignore
        if self.conservative:
            self.implemented_properties.append("forces")
            if self.model.has_stress:
                self.implemented_properties.append("stress")

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

        if self.expects_charge_and_spin:
            if ("charge" not in atoms.info) or ("spin" not in atoms.info):
                raise ValueError("atoms.info must contain both 'charge' and 'spin'")

        batch = ase_atoms_to_atom_graphs(
            atoms,
            system_config=self.system_config,
            max_num_neighbors=self.max_num_neighbors,
            edge_method=self.edge_method,
            half_supercell=self.half_supercell,
            device=self.device,
        )
        batch = batch.to(self.device)  # type: ignore
        out = self.model.predict(batch)  # type: ignore
        self._update_results(out)

    def _update_results(self, out: Dict[str, torch.Tensor]):
        """Updates the results dictionary with the computed properties."""
        self.results = {}
        no_direct_energy_head = "energy" not in self.model.heads  # type: ignore
        no_direct_force_head = "forces" not in self.model.heads  # type: ignore
        no_direct_stress_head = "stress" not in self.model.heads  # type: ignore
        for property in self.implemented_properties:
            if property == "free_energy" and no_direct_energy_head:
                continue
            if property == "forces" and no_direct_force_head:
                continue
            if property == "stress" and no_direct_stress_head:
                continue
            _property = "energy" if property == "free_energy" else property

            # ASE expects:
            #  - stresses to be squeezed to a 1D array of shape (6,)
            #  - forces to never be squeezed i.e. single-atom systems should be (1, 3)
            if property == "stress" or property == "grad_stress":
                self.results[property] = to_numpy(out[_property].squeeze())
            else:
                self.results[property] = to_numpy(out[_property])

        if self.conservative:
            if self.model.forces_name in self.results:
                self.results["direct_forces"] = self.results[self.model.forces_name]
            self.results["forces"] = self.results[self.model.grad_forces_name]

            if self.model.has_stress:
                if self.model.stress_name in self.results:
                    self.results["direct_stress"] = self.results[self.model.stress_name]
                self.results["stress"] = self.results[self.model.grad_stress_name]
