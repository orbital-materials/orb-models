import random
import numpy
import torch
import wandb
import warnings
from ase.calculators.mixing import SumCalculator

# from core.external_models.mace.model import load_mace_calculator
from orb_models.forcefield.calculator import ORBCalculator
import importlib
from orb_models.forcefield.pretrained import load_model_for_inference

warnings.filterwarnings(
    "ignore", category=UserWarning, module="plotly.matplotlylib.renderer"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="plotly.matplotlylib.mplexporter.exporter"
)
warnings.filterwarnings(
    "ignore",
    message=".*set_ticklabels.*should only be used with a fixed number of ticks.*",
)

BENCHMARKS = [
    "diatomics",
    "isolated_molecules",
    "mini_adsorption",
    "simple_md",
    "vibration",
]


def run_orbbench_mini(
    weights_path: str = "orb_v2",
    benchmarks: str = "all",
    analytic_d3: bool = False,
    seed: int = 42,
    device: str = "cuda:1",
):
    """Run the ORB Bench Mini suite of benchmarks.

    Args:
        model_name_or_uri (str): The name or URI of the model to benchmark.
        benchmarks (str, optional): comma-separated string of benchmarks.
            Valid options are in the BENCHMARKS global variable.
        analytic_d3 (bool): Whether to use analytic D3 (with mace's 'fast' settings).
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    model_name = model_name_or_uri.split("/")[-1]
    wandb.init(project="orbbench-mini", name=model_name)
    if benchmarks == "all":
        benchmark_list = BENCHMARKS
    else:
        benchmark_list = benchmarks.split(",")
        is_valid = all(benchmark in BENCHMARKS for benchmark in benchmark_list)
        if not is_valid:
            raise ValueError(
                f"Invalid benchmark(s) specified. Valid options are: {BENCHMARKS}"
            )

    # if "mace" in model_name_or_uri:
    #     size = model_name_or_uri.split("-")[-1]
    #     calculator = load_mace_calculator(size, None)
    # elif "sevennet" in model_name_or_uri:
    #     from sevenn.sevennet_calculator import SevenNetCalculator

    #     calculator = SevenNetCalculator()
    # else:
    # It's an orb model
    model = load_model_for_inference(model, weights_path, device)
    calculator = ORBCalculator(orbff, device=device)  # type: ignore

    # if analytic_d3:
    #     d3_calc = get_d3_calculator(settings="mace")
    #     calculator = SumCalculator([calculator, d3_calc])  # type: ignore

    for benchmark in benchmark_list:
        print(f"Running {benchmark} benchmark...")
        benchmark_module = importlib.import_module(
            f"core.workflows.orbbench_mini.{benchmark}"
        )
        benchmark_module.main(calculator)


if __name__ == "__main__":
    run_orbbench_mini()
