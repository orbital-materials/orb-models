import torch
from ase.calculators.calculator import Calculator, all_changes

from orb_models.common.atoms.abstract_atoms_adapter import AbstractAtomsAdapter
from orb_models.common.atoms.graph_featurization import EdgeCreationMethod
from orb_models.common.models.nn_util import ChargeSpinConditioner
from orb_models.common.torch_utils import to_numpy
from orb_models.forcefield.inference.d3_model import D3SumModel
from orb_models.forcefield.models.conservative_regressor import ConservativeForcefieldRegressor
from orb_models.forcefield.models.direct_regressor import DirectForcefieldRegressor


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
            max_num_neighbors (int): The maximum number of neighbors for each atom.
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

        conditioner = (
            model.xc_model.model.conditioner
            if isinstance(model, D3SumModel)
            else model.model.conditioner  # type: ignore
        )
        self.expects_charge_and_spin = (conditioner is not None) and isinstance(
            conditioner, ChargeSpinConditioner
        )

        properties = list(model.properties)  # type: ignore
        if "charges" in properties:
            properties.append("dipole")
        self.implemented_properties = properties

    def check_state(self, atoms, tol=1e-15):
        """Check if calculation is needed.

        Extends ASE's default check to also detect changes in charge/spin,
        which are stored in atoms.info and not tracked by ASE's default
        change detection (positions, numbers, cell, pbc).
        """
        system_changes = Calculator.check_state(self, atoms, tol)
        if self.expects_charge_and_spin and self.atoms is not None:
            old_charge = self.atoms.info.get("charge")
            old_spin = self.atoms.info.get("spin")
            new_charge = atoms.info.get("charge")
            new_spin = atoms.info.get("spin")
            if (
                old_charge != new_charge or old_spin != new_spin
            ) and "positions" not in system_changes:
                system_changes.append("positions")
        return system_changes

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
        for prop in self.implemented_properties:
            out_key = "energy" if prop == "free_energy" else prop
            if out_key not in out:
                continue
            # ASE expects:
            #  - stresses to be squeezed to a 1D array of shape (6,)
            #  - forces to never be squeezed i.e. single-atom systems should be (1, 3)
            #  - charges to be (n_atoms,)
            if prop == "stress":
                self.results[prop] = to_numpy(out[out_key].squeeze())
            elif prop == "charges":
                self.results[prop] = out[out_key].detach().reshape(-1).cpu().numpy()
            else:
                self.results[prop] = to_numpy(out[out_key])

        if "dipole" in self.implemented_properties and "charges" in self.results:
            atoms = self.atoms
            # Point-charge dipole sum_i q_i r_i, in e*A (ASE's convention). Only
            # defined for non-periodic systems, and origin-dependent unless neutral.
            if atoms is not None and not atoms.pbc.any():
                self.results["dipole"] = self.results["charges"] @ atoms.get_positions()
