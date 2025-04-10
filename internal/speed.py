from typing import Callable, Dict, List, Optional
from copy import deepcopy
from dataclasses import dataclass, field
from ase.build import bulk
from ase.data import atomic_numbers, covalent_radii
import argparse
import ase
import numpy as np
import torch
import torch.utils.benchmark as benchmark
import torch.utils.benchmark.utils.common

from orb_models.forcefield.atomic_system import (
    SystemConfig,
    ase_atoms_to_atom_graphs,
)
from orb_models.forcefield import pretrained
from orb_models.forcefield.featurization_utilities import EdgeCreationMethod


def _is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    
    
def parse_extra_args(extra_kwargs: Optional[str]) -> Dict:
    """Parse a single string of space-separated key=value pairs into a dictionary.

    It's convenient to use a string: --extra-kwargs "key1=True key2=42 key3=3.14"
    and then map them to a dictionary.
    """
    parsed_args = {}
    if extra_kwargs:
        for kwarg in extra_kwargs.split(" "):
            try:
                key, value = kwarg.split("=", 1)  # Split only on the first '='
                if value.isdigit():
                    value = int(value)  # type: ignore
                elif _is_float(value):
                    value = float(value)  # type: ignore
                elif value.lower() in ["true", "false"]:
                    value = value.lower() == "true"  # type: ignore
                elif value.lower() == "none":
                    value = None  # type: ignore
                else:
                    pass  # Keep as a string
                parsed_args[key] = value
            except ValueError:
                raise ValueError(
                    f"Invalid format for extra argument: '{kwarg}'. Expected format: key=value"
                )
    return parsed_args


class Timer(benchmark.Timer):
    """
    A subclass of torch.utils.benchmark.Timer that allows for repeat calls.

    The PyTorch Timer class removes the repeat method of timeit.Timer.
    We want to use it to return all call times of the functions, not just the median.
    So we reimplement the repeat method here.
    """

    def repeat(self, repeat: int = -1, number: int = -1):
        """Repeat the measurement the given number of times."""
        times: List[float] = []
        with torch.utils.benchmark.utils.common.set_torch_threads(
            self._task_spec.num_threads
        ):
            for _ in range(repeat):
                time_spent = self._timeit(number)
                times.append(time_spent)
        return benchmark.Measurement(
            number_per_run=number, raw_times=times, task_spec=self._task_spec
        )


@dataclass
class Measurement:
    """A single measurement of a model."""

    name: str
    natoms: int
    all_natoms: List[int]
    time_ms: float
    raw_times_ms: List[float]
    memory_gb: Optional[float] = None
    extras: dict = field(default_factory=dict)


def measure_max_memory_usage(forward_fn, device):
    """Measure the peak memory usage (GB) of the forward pass."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    forward_fn(idx=0)
    torch.cuda.synchronize()
    max_forward_memory = torch.cuda.max_memory_allocated(device) / (1024**3)
    return max_forward_memory


def take_measurement(
    name: str,
    num_atoms: int,
    all_natoms: List[int],
    fn: Callable,
    fn_warmup: Callable,
    num_threads: int,
    device: str,
    warmup_repeats: int = 5,
    num_evals: int = 50,
) -> Measurement:
    """Take measurements of a cuda function."""
    try:
        # Warm-up steps
        Timer(
            stmt="fn_warmup()",
            globals={"fn_warmup": fn_warmup},
            num_threads=num_threads,
        ).timeit(warmup_repeats)

        # measure memory usage
        if device != "cpu":
            mem_usage = measure_max_memory_usage(fn, device)
        else:
            mem_usage = -1

        # Time the model forward
        times = Timer(
            stmt="fn(globals()['idx']); globals()['idx'] += 1",
            globals={"fn": fn, "idx": 0},
            num_threads=num_threads,
        ).repeat(num_evals, 1)
        return Measurement(
            name,
            num_atoms,
            all_natoms,
            1e3 * times.median,
            [1e3 * t for t in times.raw_times],
            mem_usage,
        )

    except (torch.cuda.OutOfMemoryError, RuntimeError):
        # TorchScript throws RuntimeError for OOM, while PyTorch throws cuda.OutOfMemoryError
        return Measurement(name, num_atoms, all_natoms, -1, [-1], -1)


def calculate_lattice_constant(num_atoms, elements=["Au", "Ag", "Cu"]):
    """Calculate the lattice constant for a given number of atoms and elements."""
    radii = [covalent_radii[atomic_numbers[el]] for el in elements]
    avg_radius = np.mean(radii)
    base_lattice_constant = 4 * avg_radius / np.sqrt(2)
    cells_per_side = np.cbrt(num_atoms / 4)
    cells_per_side = int(np.ceil(cells_per_side))
    total_lattice_constant = base_lattice_constant * cells_per_side
    return total_lattice_constant


def create_random_crystal(
    num_atoms, elements=["Au", "Ag", "Cu"], pbc=True
):
    """Create a random crystal structure with the given number of atoms and elements."""
    lattice_constant = calculate_lattice_constant(num_atoms, elements)
    cell_size = int(np.ceil(np.cbrt(num_atoms / 4)))
    base_crystal = bulk("Cu", "fcc", a=lattice_constant / cell_size)
    supercell = base_crystal * (cell_size, cell_size, cell_size)

    while len(supercell) < num_atoms:
        cell_size += 1
        supercell = base_crystal * (cell_size, cell_size, cell_size)

    indices = np.random.choice(len(supercell), num_atoms, replace=False)
    random_crystal = supercell[indices]
    random_elements = np.random.choice(elements, num_atoms)
    random_crystal.set_chemical_symbols(random_elements)
    random_crystal.center()
    random_crystal.set_pbc(pbc)

    return random_crystal


def create_random_crystals_list(
    num_atoms, num_crystals, elements=["Au", "Ag", "Cu"], pbc=True
):
    """Create a list of random crystals with the given number of atoms and crystals."""
    return [
        create_random_crystal(num_atoms, elements=elements, pbc=pbc)
        for _ in range(num_crystals)
    ]


def load_orb(name: str, device: str, precision: str = "float32-high", compile: bool = True):
    """Load the ORB model."""
    params = locals()
    params.pop("device")

    orb = getattr(pretrained, name)(device=device, precision=precision)

    if compile:
        orb.compile(mode="default", dynamic=True)

    orb = orb.to(device)
    orb.eval()
    for param in orb.parameters():
        param.requires_grad = False

    params["n_params"] = int(sum(p.numel() for p in orb.parameters()) / 1e6)
    params["device"] = device

    return orb, params


def benchmark_orb_forward(
    atoms_list: List[ase.Atoms],
    warmup_atoms: List[ase.Atoms],
    num_threads: int,
    device: str,
    extra_kwargs: Dict,
    warmup_repeats: int = 5,
    num_evals: int = 1,
) -> List[Measurement]:
    """Benchmark the orbital model for a given atomic system."""
    name = extra_kwargs.pop("name", "orb_v3_direct_20_omat")
    precision = extra_kwargs.pop("precision", "float32-high")
    compile = extra_kwargs.pop("compile", True)

    orb, params = load_orb(name=name, device=device, precision=precision, compile=compile)

    # Featurize atoms
    warmup_batches = [
        ase_atoms_to_atom_graphs(atoms, system_config=orb._system_config, device=device).to(device)  # type: ignore
        for atoms in warmup_atoms
    ]
    batches = [
        ase_atoms_to_atom_graphs(atoms, system_config=orb._system_config, device=device).to(device)  # type: ignore
        for atoms in atoms_list
    ]
    num_edges = [len(batch.edge_features["vectors"]) for batch in batches]
    num_edges_std = np.std(num_edges, ddof=1)
    min_num_edges = np.min(num_edges)
    max_num_edges = np.max(num_edges)

    def model_forward(idx: int):
        orb.predict(batches[idx])

    def model_forward_warmup():
        for batch in warmup_batches:
            orb.predict(batch)

    measurement = take_measurement(
        "orb",
        len(atoms_list[0].positions),
        [len(atoms.positions) for atoms in atoms_list],
        model_forward,
        model_forward_warmup,
        num_threads=num_threads,
        device=device,
        warmup_repeats=warmup_repeats,
        num_evals=num_evals,
    )
    measurement.extras.update(params)
    measurement.extras.update(
        {
            "num_edges_std": round(num_edges_std, 2),
            "min_num_edges": min_num_edges,
            "max_num_edges": max_num_edges,
        }
    )

    return [measurement]


def benchmark_orb_featurize(
    atoms_list: List[ase.Atoms],
    warmup_atoms: List[ase.Atoms],
    num_threads: int,
    device: str,
    extra_kwargs: Dict,
    warmup_repeats: int = 5,
    num_evals: int = 1,
) -> List[Measurement]:
    """Benchmark orb featurize."""
    edge_method: EdgeCreationMethod = extra_kwargs.pop("edge_method", None)
    half_supercell = extra_kwargs.pop("half_supercell", None)
    radius = extra_kwargs.pop("radius", 6.0)
    max_num_neighbors = extra_kwargs.pop("max_num_neighbors", 20)
    system_config = SystemConfig(radius=radius, max_num_neighbors=max_num_neighbors)

    def featurize(idx: int):
        ase_atoms_to_atom_graphs(
            atoms_list[idx],
            system_config=system_config,
            edge_method=edge_method,
            half_supercell=half_supercell,
            device=device,  # type: ignore
        ).to(device)

    def featurize_warmup():
        for atoms in warmup_atoms:
            ase_atoms_to_atom_graphs(
                atoms,
                system_config=system_config,
                edge_method=edge_method,
                half_supercell=half_supercell,
                device=device,  # type: ignore
            ).to(device)

    measurement = take_measurement(
        "orb-featurize",
        len(atoms_list[0].positions),
        [len(atoms.positions) for atoms in atoms_list],
        featurize,
        featurize_warmup,
        num_threads=num_threads,
        device=device,
        warmup_repeats=warmup_repeats,
        num_evals=num_evals,
    )
    measurement.extras.update(
        {
            "radius": system_config.radius,
            "max_num_neighbors": system_config.max_num_neighbors,
            "half_supercell": half_supercell,
            "edge_method": edge_method,
            "device": device,
        }
    )

    return [measurement]


def benchmark_inference_with_respect_to_natoms(
    method: str = "orb-forward",
    extra_kwargs: Optional[str] = None,
    num_evals: int = 50,
    num_atoms: List[int] = [100, 1_000, 5_000, 10_000, 50_000, 100_000],
    device: Optional[str] = None,
    num_threads: int = 4,
):
    """Time inference on an artifical system of varying num atoms (batch_size=1).

    Args:
        method: The model to benchmark. Choose from
        device: The device to run the benchmark on. Defaults to 'cuda' if available.
        num_evals: The number of evaluations to run for each number of atoms.
        num_atoms: A list of numbers of atoms to use for the benchmark.
        num_atoms_variability_frac: The fraction of natoms that we can go over or under the num_atoms.
            So with num_atoms=20 and num_atoms_variability_frac=0.1, we will evaluate between 18 and 22 atoms.
        num_threads: The number of threads to use for the benchmark.
        extra_kwargs: Additional keyword arguments to pass to the model.
    """
    extra_kwargs = parse_extra_args(extra_kwargs)  # type: ignore
    pbc = extra_kwargs.pop("pbc", True)  # type: ignore
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    benchmark_fn = {
        "orb-forward": benchmark_orb_forward,
        "orb-featurize": benchmark_orb_featurize,
    }[method]

    results: List[Measurement] = []
    for i in num_atoms:
        print(f"Benchmarking {method} with {i} atoms")

        # Create separate warmup crystals that are not used for benchmarking
        warmup_atoms = create_random_crystals_list(
            i, num_crystals=1, pbc=pbc
        )
        # Create random crystals to avoid evaluating all runs on the same tensor
        atoms_list = create_random_crystals_list(
            i, num_crystals=num_evals, pbc=pbc
        )
        # Run the benchmark
        results.extend(
            benchmark_fn(
                atoms_list,
                warmup_atoms,
                num_threads,
                device,
                deepcopy(extra_kwargs),  # type: ignore
                warmup_repeats=5,
                num_evals=num_evals,
            )
        )

    print("\n Output")
    # Create a readable output format without pandas
    headers = ["name", "natoms", "time_ms", "memory_gb"]
    # Add extra keys from the first result's extras
    if results:
        extra_keys = list(results[0].extras.keys())
        headers.extend(extra_keys)
    
    # Print headers
    print(" | ".join(headers))
    print("-" * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)))
    
    # Print each result
    for r in results:
        row = [
            r.name,
            str(r.natoms),
            f"{r.time_ms:.2f}" if r.time_ms > 0 else "OOM",
            f"{r.memory_gb:.2f}" if r.memory_gb and r.memory_gb > 0 else "N/A"
        ]
        # Add extras
        for key in extra_keys:
            value = r.extras.get(key, "")
            row.append(str(value))
        
        print(" | ".join(row))


if __name__ == "__main__":
    """Run speed and memory benchmark for the ORB model."""
    parser = argparse.ArgumentParser(
        description="Benchmark speed and memory of an ORB model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method", default="orb-forward", type=str, help="Method to benchmark."
    )
    parser.add_argument(
        "--extra_kwargs", default="", type=str, help="Extra kwargs in key=value format to unpack into the method."
    )
    parser.add_argument(
        "--num_evals", default=50, type=int, help="Number of evaluations to run."
    )
    parser.add_argument(
        "--num_atoms", 
        default=[100, 1000, 5000, 10000, 50000, 100000], 
        type=int, 
        nargs="+", 
        help="Number of atoms to benchmark."
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Device to run the benchmark on."
    )
    parser.add_argument(
        "--num_threads", default=4, type=int, help="Number of threads to use for the benchmark."
    )
    args = parser.parse_args()
    benchmark_inference_with_respect_to_natoms(
        args.method,
        args.extra_kwargs,
        args.num_evals,
        args.num_atoms,
        args.device,
        args.num_threads,
    )
