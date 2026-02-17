import os
from pathlib import Path

import scipy.constants
import torch
from nvalchemiops.interactions.dispersion.dftd3 import D3Parameters, dftd3

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.atoms.graph_featurization import _compute_neighbor_list_with_fallback
from orb_models.common.models.base import RegressorModelMixin
from orb_models.common.models.segment_ops import split_prediction
from orb_models.forcefield.models.conservative_regressor import ConservativeForcefieldRegressor
from orb_models.forcefield.models.direct_regressor import DirectForcefieldRegressor
from orb_models.forcefield.models.forcefield_utils import torch_full_3x3_to_voigt_6_stress

HARTREE_TO_EV = scipy.constants.value("Hartree energy in eV")
BOHR_TO_ANGSTROM = scipy.constants.value("Bohr radius") / 1e-10
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

dftd3_compiled = torch.compile(dftd3, dynamic=True)


class AlchemiDFTD3(torch.nn.Module):
    """Alchemiops DFT-D3(Becke-Johnson) dispersion corrections wrapper module.

    Internal computations use atomic units (Bohr, Hartree). The input/output units are conventional (Å, eV).

    Adapted from https://github.com/NVIDIA/nvalchemi-toolkit-ops/blob/main/examples/dispersion/utils.py
    """

    def __init__(
        self,
        *,
        functional: str = "PBE",
        damping: str = "BJ",
        cutoff: float = 22.0,
        k1: float = 16.0,
        k3: float = -4.0,
        s5_smoothing_on: float = 1e10,
        s5_smoothing_off: float = 1e10,
        has_stress: bool = True,
        compile: bool = False,
    ):
        """
        Initializes the DFT-D3(BJ) dispersion corrections module.

        Args:
            functional: The functional to use. See `DFTD3.get_d3_coefficients` for available functionals.
            damping: The damping to use. See `DFTD3.get_d3_coefficients` for available dampings.
            cutoff: Cutoff radius for the dispersion correction (Å).
            k1: CN counting function steepness parameter (dimensionless).
            k3: CN interpolation Gaussian width parameter (dimensionless).
            s5_smoothing_on: Distance where S5 switching begins (Å). Default: 1e10 (disabled)
            s5_smoothing_off: Distance where S5 switching completes (Å). Default: 1e10 (disabled)
            has_stress: Whether the model should compute stress correction.
            compile: Whether to compile the dftd3 function.
        """
        super().__init__()

        self.cutoff = cutoff * ANGSTROM_TO_BOHR  # Bohr
        self.has_stress = has_stress
        if compile:
            self.dtfd3 = dftd3_compiled
        else:
            self.dtfd3 = dftd3

        # Performance tuning: we estimate the number of maximum neighbors given a radius based on
        # assumed atomic density and a safety factor. We override the default parameters from alchemi,
        # which are *extremely* conservative.
        # With the safety_factor, a radius of 22.0 Å will allow for up to 8928 neighbors per atom,
        # which is ~9x lower than the default 78064 neighbors.
        # See https://nvidia.github.io/nvalchemi-toolkit-ops/userguide/components/neighborlist.html#performance-tuning
        # NOTE: If the system is very dense, underestimating max_num_neighbors_alchemi will cause
        # alchemiops to *silently* truncate a number of *random* neighbors.
        # To avoid silent errors we use a fallback safety factor, and if that fails raise an error in
        # _compute_neighbor_list_with_fallback.
        self.atomic_density = 0.2 / ANGSTROM_TO_BOHR**3  # Å^3 -> Bohr^3
        self.safety_factor = 1.0
        self.fallback_safety_factor = 5.0

        # Store D3 parameters
        d3_coefficients = AlchemiDFTD3.get_d3_coefficients(functional, damping)
        self.a1 = d3_coefficients["a1"]
        self.a2 = d3_coefficients["a2"]  # Bohr
        self.s6 = d3_coefficients["s6"]
        self.s8 = d3_coefficients["s8"]
        self.k1 = k1
        self.k3 = k3

        self.s5_smoothing_on = s5_smoothing_on * ANGSTROM_TO_BOHR  # Bohr
        self.s5_smoothing_off = s5_smoothing_off * ANGSTROM_TO_BOHR  # Bohr

        # All parameters are in atomic units
        d3_params = AlchemiDFTD3.load_d3_parameters()
        self.register_buffer("covalent_radii", d3_params.rcov.float(), persistent=True)
        self.register_buffer("r4r2", d3_params.r4r2.float(), persistent=True)
        self.register_buffer("c6_reference", d3_params.c6ab.float(), persistent=True)
        self.register_buffer("coord_num_ref", d3_params.cn_ref.float(), persistent=True)

    def predict(self, batch: AtomGraphs, split: bool = False) -> dict[str, torch.Tensor]:
        """Predict by summing outputs from both XC and D3 models."""
        out = self(
            batch.positions, batch.cell, batch.atomic_numbers, batch.pbc, batch.node_batch_index
        )

        if split:
            for name, pred in out.items():
                out[name] = split_prediction(pred, batch.n_node)
        return out

    def forward(
        self, positions, cell, atomic_numbers, pbc, node_batch_index
    ) -> dict[str, torch.Tensor]:
        """Compute DFT-D3 dispersion energy and forces."""
        with torch.inference_mode():
            # Convert inputs to atomic units
            cell_angstrom = cell.contiguous()
            positions = positions.contiguous() * ANGSTROM_TO_BOHR
            cell = cell_angstrom * ANGSTROM_TO_BOHR
            node_batch_index = node_batch_index.to(torch.int32)

            neighbor_matrix, num_neighbors, neighbor_shift_matrix = (
                _compute_neighbor_list_with_fallback(
                    positions=positions,
                    cell=cell,
                    pbc=pbc,
                    cutoff=self.cutoff,
                    batch_idx=node_batch_index,
                    atomic_density=self.atomic_density,
                    initial_safety_factor=self.safety_factor,
                    fallback_safety_factor=self.fallback_safety_factor,
                )
            )
            # Truncate the neighbor matrix to save computation/memory
            max_num_neighbors_alchemi = num_neighbors.max()
            neighbor_matrix = neighbor_matrix[:, :max_num_neighbors_alchemi]
            neighbor_shift_matrix = neighbor_shift_matrix[:, :max_num_neighbors_alchemi, :]

            d3_out = self.dtfd3(
                positions=positions,
                numbers=atomic_numbers.to(torch.int32),
                cell=cell,
                neighbor_matrix=neighbor_matrix,
                batch_idx=node_batch_index,
                neighbor_matrix_shifts=neighbor_shift_matrix,
                covalent_radii=self.covalent_radii,
                r4r2=self.r4r2,
                c6_reference=self.c6_reference,
                coord_num_ref=self.coord_num_ref,
                a1=self.a1,
                a2=self.a2,
                s6=self.s6,
                s8=self.s8,
                k1=self.k1,
                k3=self.k3,
                s5_smoothing_on=self.s5_smoothing_on,
                s5_smoothing_off=self.s5_smoothing_off,
                compute_virial=self.has_stress,
            )
            if self.has_stress:
                energy, forces, coord_num, virial = d3_out
            else:
                energy, forces, coord_num = d3_out

            # Convert outputs back to conventional units
            # Energy: Hartree -> eV
            energy = energy * HARTREE_TO_EV
            # Forces: Hartree/Bohr -> eV/Angstrom
            forces = forces * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)
            if self.has_stress:
                # Virial is in Hartree, convert to eV
                virial = virial * HARTREE_TO_EV
                # Convert to stress tensor (units: eV/Å³)
                volume = torch.det(cell_angstrom).abs()
                stress = -virial / volume.view(-1, 1, 1)
                # Convert full stress matrix [N, 3, 3] to Voigt notation [N, 6]
                stress_voigt = torch_full_3x3_to_voigt_6_stress(stress)

            out = {
                "energy": energy,
                "forces": forces,
            }
            if self.has_stress:
                out["stress"] = stress_voigt
        return out

    def extra_repr(self) -> str:
        """Return a string representation of module parameters."""
        return (
            f"a1={self.a1:.4f}, a2={self.a2:.4f} Bohr, s8={self.s8:.4f}, cutoff={self.cutoff} Bohr"
        )

    @staticmethod
    def load_d3_parameters() -> D3Parameters:
        d3_filepath = str(Path(os.path.abspath(__file__)).parent / "dftd3_parameters.pt")

        state_dict = torch.load(d3_filepath, map_location="cpu", weights_only=True)

        return D3Parameters(
            rcov=state_dict["rcov"],
            r4r2=state_dict["r4r2"],
            c6ab=state_dict["c6ab"],
            cn_ref=state_dict["cn_ref"],
        )

    @staticmethod
    def get_d3_coefficients(functional: str, damping: str) -> dict[str, float]:
        """
        Get the D3 coefficients for a given functional and damping.

        Adapted from https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/bj_damping
        and https://github.com/pfnet-research/torch-dftd/blob/master/torch_dftd/dftd3_xc_params.py

        Args:
            functional: The functional to use.
            damping: The damping to use.

        Returns:
            A dictionary of the D3 coefficients:
            - a1: the first parameter (dimensionless)
            - a2: the second parameter (Bohr)
            - s6: the sixth parameter (dimensionless)
            - s8: the eighth parameter (dimensionless)
        """
        d3_xc_parameters = {
            "BJ": {
                "BP": {"a1": 0.3946, "a2": 4.8516, "s6": 1.0, "s8": 3.2822},
                "BP86": {"a1": 0.3946, "a2": 4.8516, "s6": 1.0, "s8": 3.2822},
                "BLYP": {"a1": 0.4298, "a2": 4.2359, "s6": 1.0, "s8": 2.6996},
                "revPBE": {"a1": 0.5238, "a2": 3.5016, "s6": 1.0, "s8": 2.3550},
                "RPBE": {"a1": 0.1820, "a2": 4.0094, "s6": 1.0, "s8": 0.8318},
                "B97D": {"a1": 0.5545, "a2": 3.2297, "s6": 1.0, "s8": 2.2609},
                "PBE": {"a1": 0.4289, "a2": 4.4407, "s6": 1.0, "s8": 0.7875},
                "rPW86PBE": {"a1": 0.4613, "a2": 4.5062, "s6": 1.0, "s8": 1.3845},
                "B3LYP": {"a1": 0.3981, "a2": 4.4211, "s6": 1.0, "s8": 1.9889},
                "TPSS": {"a1": 0.4535, "a2": 4.4752, "s6": 1.0, "s8": 1.9435},
                "HF": {"a1": 0.3385, "a2": 2.8830, "s6": 1.0, "s8": 0.9171},
                "TPSS0": {"a1": 0.3768, "a2": 4.5865, "s6": 1.0, "s8": 1.2576},
                "PBE0": {"a1": 0.4145, "a2": 4.8593, "s6": 1.0, "s8": 1.2177},
                "HSE06": {"a1": 0.383, "a2": 5.685, "s6": 1.0, "s8": 2.310},
                "PBE38": {"a1": 0.3995, "a2": 5.1405, "s6": 1.000, "s8": 1.4623},
                "revPBE38": {"a1": 0.4309, "a2": 3.9446, "s6": 1.0, "s8": 1.4760},
                "PW6B95": {"a1": 0.2076, "a2": 6.3750, "s6": 1.0, "s8": 0.7257},
                "B2PLYP": {"a1": 0.3065, "a2": 5.0570, "s6": 0.64, "s8": 0.9147},
                "DSDBLYP": {"a1": 0.0000, "a2": 6.0519, "s6": 0.50, "s8": 0.2130},
                "DSDBLYPFC": {"a1": 0.0009, "a2": 5.9807, "s6": 0.50, "s8": 0.2112},
                "BOP": {"a1": 0.4870, "a2": 3.5043, "s6": 1.0, "s8": 3.2950},
                "mPWLYP": {"a1": 0.4831, "a2": 4.5323, "s6": 1.0, "s8": 2.0077},
                "OLYP": {"a1": 0.5299, "a2": 2.8065, "s6": 1.0, "s8": 2.6205},
                "PBESOL": {"a1": 0.4466, "a2": 6.1742, "s6": 1.0, "s8": 2.9491},
                "BPBE": {"a1": 0.4567, "a2": 4.3908, "s6": 1.0, "s8": 4.0728},
                "OPBE": {"a1": 0.5512, "a2": 2.9444, "s6": 1.0, "s8": 3.3816},
                "SSB": {"a1": -0.0952, "a2": 5.2170, "s6": 1.0, "s8": -0.1744},
                "revSSB": {"a1": 0.4720, "a2": 4.0986, "s6": 1.0, "s8": 0.4389},
                "oTPSS": {"a1": 0.4634, "a2": 4.3153, "s6": 1.0, "s8": 2.7495},
                "B3PW91": {"a1": 0.4312, "a2": 4.4693, "s6": 1.0, "s8": 2.8524},
                "BHLYP": {"a1": 0.2793, "a2": 4.9615, "s6": 1.0, "s8": 1.0354},
                "revPBE0": {"a1": 0.4679, "a2": 3.7619, "s6": 1.0, "s8": 1.7588},
                "TPSSh": {"a1": 0.4529, "a2": 4.6550, "s6": 1.0, "s8": 2.2382},
                "MPW1B95": {"a1": 0.1955, "a2": 6.4177, "s6": 1.0, "s8": 1.0508},
                "PWB6K": {"a1": 0.1805, "a2": 7.7627, "s6": 1.0, "s8": 0.9383},
                "B1B95": {"a1": 0.2092, "a2": 5.5545, "s6": 1.0, "s8": 1.4507},
                "BMK": {"a1": 0.1940, "a2": 5.9197, "s6": 1.0, "s8": 2.0860},
                "CAMB3LYP": {"a1": 0.3708, "a2": 5.4743, "s6": 1.0, "s8": 2.0674},
                "LCWPBE": {"a1": 0.3919, "a2": 5.0897, "s6": 1.0, "s8": 1.8541},
                "B2GPPLYP": {"a1": 0.0000, "a2": 6.3332, "s6": 0.560, "s8": 0.2597},
                "PTPSS": {"a1": 0.0000, "a2": 6.5745, "s6": 0.750, "s8": 0.2804},
                "PWPB95": {"a1": 0.0000, "a2": 7.3141, "s6": 0.820, "s8": 0.2904},
                "HFMIXED": {"a1": 0.5607, "a2": 4.5622, "s6": 1.0, "s8": 3.9027},
                "HFSV": {"a1": 0.4249, "a2": 4.2783, "s6": 1.0, "s8": 2.1849},
                "HFMINIS": {"a1": 0.1702, "a2": 3.8506, "s6": 1.0, "s8": 0.9841},
                "B3LYP631GD": {"a1": 0.5014, "a2": 4.8409, "s6": 1.0, "s8": 4.0672},
                "HCTH120": {"a1": 0.3563, "a2": 4.3359, "s6": 1.0, "s8": 1.0821},
                "DFTB3": {"a1": 0.5719, "a2": 3.6017, "s6": 1.0, "s8": 0.5883},
                "PW1PW": {"a1": 0.3807, "a2": 5.8844, "s6": 1.0, "s8": 2.3363},
                "PWGGA": {"a1": 0.2211, "a2": 6.7278, "s6": 1.0, "s8": 2.6910},
                "HSESOL": {"a1": 0.4650, "a2": 6.2003, "s6": 1.0, "s8": 2.9215},
                "HF3C": {"a1": 0.4171, "a2": 2.9149, "s6": 1.0, "s8": 0.8777},
                "HF3CV": {"a1": 0.3063, "a2": 3.9856, "s6": 1.0, "s8": 0.5022},
                "PBEH3C": {"a1": 0.4860, "a2": 4.5000, "s6": 1.0, "s8": 0.0000},
                "MPWB1K": {"a1": 0.1474, "a2": 6.6223, "s6": 1.000, "s8": 0.9499},
            },
            "BJM": {
                "B2PLYP": {"a1": 0.486434, "a2": 3.656466, "s6": 0.640000, "s8": 0.672820},
                "B3LYP": {"a1": 0.278672, "a2": 4.606311, "s6": 1.0, "s8": 1.466677},
                "B97D": {"a1": 0.240184, "a2": 3.864426, "s6": 1.0, "s8": 1.206988},
                "BLYP": {"a1": 0.448486, "a2": 3.610679, "s6": 1.0, "s8": 1.875007},
                "BP": {"a1": 0.821850, "a2": 2.728151, "s6": 1.0, "s8": 3.140281},
                "PBE": {"a1": 0.012092, "a2": 5.938951, "s6": 1.0, "s8": 0.358940},
                "PBE0": {"a1": 0.007912, "a2": 6.162326, "s6": 1.0, "s8": 0.528823},
                "LCWPBE": {"a1": 0.563761, "a2": 3.593680, "s6": 1.0, "s8": 0.906564},
            },
        }
        return d3_xc_parameters[damping][functional]


class D3SumModel(RegressorModelMixin):
    """Wrapper that sums predictions from an XC model and a D3 dispersion correction model.

    This class combines predictions from a main exchange-correlation (XC) model
    (either DirectForcefieldRegressor or ConservativeForcefieldRegressor) with
    predictions from a D3 dispersion correction model.
    """

    def __init__(
        self,
        model: DirectForcefieldRegressor | ConservativeForcefieldRegressor,
        d3_model: DirectForcefieldRegressor | AlchemiDFTD3,
    ):
        super().__init__()
        self.xc_model = model
        self.d3_model = d3_model

    @property
    def properties(self) -> list[str]:
        """Return the properties of the model."""
        return self.xc_model.properties

    @property
    def has_stress(self) -> bool:
        """Check if the model has stress prediction."""
        return self.xc_model.has_stress

    def predict(self, batch: AtomGraphs, split: bool = False) -> dict[str, torch.Tensor]:
        """Predict by summing outputs from both XC and D3 models.

        Handles the special naming conventions for conservative models where
        forces/stress may have grad_ prefixes or level-of-theory suffixes.

        Args:
            batch: Input batch of atomic structures
            split: Whether to split predictions by system

        Returns:
            Dictionary with summed predictions, including energy, forces, and stress
        """
        # Get predictions from both models
        xc_out = self.xc_model.predict(batch, split=split)
        d3_out = self.d3_model.predict(batch, split=split)

        # Start with XC model output
        out: dict[str, torch.Tensor] = xc_out.copy()

        # Sum energy predictions
        out["energy"] = out["energy"] + d3_out["energy"]

        # Sum force predictions (handling conservative naming)
        if "forces" in d3_out:
            if isinstance(self.xc_model, ConservativeForcefieldRegressor):
                # For conservative models, add to the gradient-based forces
                grad_forces_key: str = self.xc_model.grad_forces_name
                if grad_forces_key in out:
                    out[grad_forces_key] = out[grad_forces_key] + d3_out["forces"]

                # Also add to direct forces if they exist
                forces_key: str = self.xc_model.forces_name
                if forces_key in out:
                    out[forces_key] = out[forces_key] + d3_out["forces"]
            else:
                # For direct models, simple addition
                if "forces" in out:
                    out["forces"] = out["forces"] + d3_out["forces"]

        # Sum stress predictions (handling conservative naming)
        if "stress" in d3_out and self.xc_model.has_stress:
            if isinstance(self.xc_model, ConservativeForcefieldRegressor):
                # For conservative models, add to the gradient-based stress
                grad_stress_key: str = self.xc_model.grad_stress_name  # type: ignore
                if grad_stress_key in out:
                    out[grad_stress_key] = out[grad_stress_key] + d3_out["stress"]

                # Also add to direct stress if it exists
                stress_key: str = self.xc_model.stress_name  # type: ignore
                if stress_key in out:
                    out[stress_key] = out[stress_key] + d3_out["stress"]
            else:
                # For direct models, simple addition
                if "stress" in out:
                    out["stress"] = out["stress"] + d3_out["stress"]

        return out
