import torch
from ase.calculators.calculator import Calculator, all_changes

from orb_models.common.atoms.abstract_atoms_adapter import AbstractAtomsAdapter
from orb_models.common.atoms.graph_featurization import EdgeCreationMethod
from orb_models.common.models.nn_util import ChargeSpinConditioner
from orb_models.common.torch_utils import to_numpy
from orb_models.forcefield.inference.d3_model import D3SumModel
from orb_models.forcefield.models.conservative_regressor import (
    ConservativeForcefieldRegressor,
)
from orb_models.forcefield.models.direct_regressor import DirectForcefieldRegressor


def _is_conservative(
    model: DirectForcefieldRegressor | ConservativeForcefieldRegressor | D3SumModel,
) -> bool:
    if isinstance(model, D3SumModel):
        model = model.xc_model
    return isinstance(model, ConservativeForcefieldRegressor)


class ORBCalculator(Calculator):
    """ORB ASE Calculator."""

    def __init__(
        self,
        model: DirectForcefieldRegressor | ConservativeForcefieldRegressor | D3SumModel,
        atoms_adapter: AbstractAtomsAdapter,
        *,
        edge_method: EdgeCreationMethod | None = None,
        max_num_neighbors: int | None = None,
        half_supercell: bool | None = None,
        device: torch.device | str | None = None,
        directory: str = ".",
    ):
        """Initializes the calculator.

        Args:
            model: The Orb forcefield model to use for predictions.
            atoms_adapter: The adapter to convert between ASE Atoms and AtomGraphs.
            edge_method (EdgeCreationMethod, optional): The method to use for graph edge construction. Defaults to knn_alchemi.
            max_number_neighbors (int): The maximum number of neighbors for each atom.
                Larger values should generally increase performace, but the gains may be marginal,
                whilst the increse in latency could be significant (depending on num atoms).
                    - Defaults to atoms_adapter.max_num_neighbors.
                    - 120 is sufficient to capture all edges under 6A across all systems in mp-traj validation set.
            half_supercell (bool): Whether to use half the supercell for graph construction, and then symmetrize.
                Defaults to None, in which case half_supercells are used when num_atoms > 5k.
                This flag does not affect the resulting graph; it is purely an optimization that can double
                throughput and half memory for very large cells (e.g. 5k+ atoms). For smaller systems, it can hurt
                performance due to additional computation to enforce max_num_neighbors.
            device (torch.device, optional): The device to use for the model.
            directory (str, optional): Working directory in which to read and write files and perform calculations.
        """
        Calculator.__init__(self, directory=directory)
        self.results = {}  # type: ignore
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)  # type: ignore
        self.adapter = atoms_adapter
        self.max_num_neighbors = max_num_neighbors
        self.edge_method = edge_method
        self.half_supercell = half_supercell
        self.conservative = _is_conservative(model)

        conditioner = (
            model.xc_model.model.conditioner
            if isinstance(model, D3SumModel)
            else model.model.conditioner  # type: ignore
        )
        self.expects_charge_and_spin = (conditioner is not None) and isinstance(
            conditioner, ChargeSpinConditioner
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

        if self.expects_charge_and_spin and (
            ("charge" not in atoms.info) or ("spin" not in atoms.info)
        ):
            raise ValueError("atoms.info must contain both 'charge' and 'spin'")

        batch = self.adapter.from_ase_atoms(
            atoms=atoms,
            max_num_neighbors=self.max_num_neighbors,
            edge_method=self.edge_method,
            half_supercell=self.half_supercell,
            device=self.device,  # type: ignore
        )
        batch = batch.to(self.device)  # type: ignore
        out = self.model.predict(batch)  # type: ignore
        self._update_results(out)

    def _update_results(self, out: dict[str, torch.Tensor]):
        """Updates the results dictionary with the computed properties."""
        self.results = {}
        model = self.model.xc_model if isinstance(self.model, D3SumModel) else self.model
        no_direct_energy_head = "energy" not in model.heads  # type: ignore
        no_direct_force_head = "forces" not in model.heads  # type: ignore
        no_direct_stress_head = "stress" not in model.heads  # type: ignore
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
            if model.forces_name in self.results:
                self.results["direct_forces"] = self.results[model.forces_name]
            self.results["forces"] = self.results[model.grad_forces_name]

            if model.has_stress:
                if model.stress_name in self.results:
                    self.results["direct_stress"] = self.results[model.stress_name]
                self.results["stress"] = self.results[model.grad_stress_name]
