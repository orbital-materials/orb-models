from typing import Optional

import torch
from ase.calculators.calculator import Calculator, all_changes

from orb_models.forcefield.atomic_system import SystemConfig, ase_atoms_to_atom_graphs
from orb_models.forcefield.graph_regressor import GraphRegressor


class ORBCalculator(Calculator):
    """ORB ASE Calculator.

    args:
        model: torch.nn.Module finetuned Graph regressor
    """

    def __init__(
        self,
        model: GraphRegressor,
        brute_force_knn: Optional[bool] = None,
        system_config: SystemConfig = SystemConfig(radius=10.0, max_num_neighbors=20),
        device: Optional[torch.device] = None,
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
            device (Optional[torch.device], optional): The device to use for the model.
            **kwargs: Additional keyword arguments for parent Calculator class.
        """
        Calculator.__init__(self, **kwargs)
        self.results = {}  # type: ignore
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.system_config = system_config
        self.brute_force_knn = brute_force_knn

        # NOTE: we currently do not predict stress, but when we do,
        # we should add it here and also update calculate() below.
        properties = []
        if model.node_head is not None:
            properties += ["energy", "free_energy"]
        if model.graph_head is not None:
            properties += ["forces"]
        if model.stress_head is not None:
            properties += ["stress"]
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
        )
        batch = batch.to(self.device)  # type: ignore
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
