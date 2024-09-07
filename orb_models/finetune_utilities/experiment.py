"""Experiment utilities."""

import dataclasses
import random
import re
from typing import Dict, List, Optional, TypeVar, Union

import hydra
import numpy
import omegaconf
import torch
import wandb
from omegaconf import DictConfig
from orb_models.utils import env
from wandb import wandb_run

T = TypeVar("T")


@dataclasses.dataclass
class WandbArtifactTypes:
    """Artifact types for wandb."""

    MODEL = "model"
    CONFIG = "config"
    DATASET = "dataset"
    SCREENING = "screening"


def prefix_keys(
    dict_to_prefix: Dict[str, T], prefix: str, sep: str = "/"
) -> Dict[str, T]:
    """Add a prefix to dictionary keys with a seperator."""
    return {f"{prefix}{sep}{k}": v for k, v in dict_to_prefix.items()}


def seed_everything(seed: int, rank: int) -> None:
    """Set the seed for all pseudo random number generators."""
    random.seed(seed + rank)
    numpy.random.seed(seed + rank)
    torch.manual_seed(seed + rank)


def get_device(
    requested_device: Optional[Union[torch.device, str, int]] = None
) -> torch.device:
    """Get a torch device, defaulting to gpu if available."""
    if requested_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def to_numpy(x):
    """If x is a tensor, convert it to a float (if 1 element) or np array (if > 1 element)."""
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().numpy()
    return x


def init_wandb_from_config(args, job_type: str) -> wandb_run.Run:
    """Initialise wandb from config."""
    run_name = args.get("name")
    project = args.get("project")
    if not run_name:
        run_name = f"{job_type}-{env.get_timestamp()}"
    if not project:
        project = "orb-experiment"
    wandb.init(  # type: ignore
        job_type=job_type,
        dir=env.wandb_root(),
        name=run_name,
        project=project,
        entity=args.entity,
        mode=args.mode,
        group=args.get("group"),
        sync_tensorboard=True,
    )
    assert wandb.run is not None
    return wandb.run


@dataclasses.dataclass
class OptimConfig(DictConfig):
    """The optimizer and learning rate scheduler config.

    Args:
        optimizer: Configuration and kwargs for a torch.optim.Optimizer.
        lr_scheduler: Configuration and kwargs for a torch.optim.LRScheduler.
        ema: Configuration and kwargs for Exponential Moving Averages of weights during training.
        init: Regex : initializer mappings for initializing a model.
        parameter_groups: Regex: optimizer kwargs for creating multiple seperate
            parameter groups during optimization.
        verbose: Whether to list parameter groups and model init regexes. For large
            models this can be verbose.
    """

    optimizer: DictConfig
    lr_scheduler: DictConfig
    ema_decay: Optional[float]
    init: Optional[DictConfig]
    parameter_groups: Optional[DictConfig]
    verbose: bool = False


@dataclasses.dataclass
class InitConfig(omegaconf.DictConfig):
    r"""A config for initializing a model.

    an example configuration file:
    ```

    init:
      regexes:
        '*.bias':
          _partial_: True
          _target_: torch.nn.init.normal_
          mean: 0.1
        '*.weight':
          _partial_: True
          _target_: torch.nn.init.zeros_
    prevent_regex:
      - 'model\.mlp.*'

    ```
    where the key in each key, value pair under the regexes parameters is the regex that matches to
    parameters, and the value specifies an initalizer, as well as possible partial arguments. To determine
    valid partial args, refer to the torch.nn.init documentation.

    Notes:
        - **Regexes are not valid yaml keys; you must wrap these in specifically single quotes,
          as demonstrated above.**
        - The _partial_ keyword from hydra allows the creation of `functools.partial` functions. This is
          necessary if you wish to pass other keyword arguments to the torch initialization functions.
        - You must use the versions of the torch initializers which modify parameters inplace.

    Args:
        regexes : `List[Tuple[str, Initializer]]`, optional (default = `[]`)
            A list mapping parameter regexes to initializers.  We will check each parameter against
            each regex in turn, and apply the initializer paired with the first matching regex, if
            any.
        prevent_regexes: `List[str]`, optional (default=`None`)
            Any parameter name matching one of these regexes will not be initialized, regardless of
            whether it matches one of the regexes passed in the `regexes` parameter.
    """

    regexes: Dict[str, omegaconf.DictConfig]
    prevent_regexes: Optional[List[str]]


def initialize_model(optim_config: OptimConfig, module: torch.nn.Module) -> None:
    """Initialize a model from a config.

    Applies initializers to parameters of a Module based on regexes. Any parameter not
    matching a regex will not be initialized, instead using whatever the default
    initialization was in code.

    Args:
        init_config: The config specifying the init.
        module: The model/module to initialize.

    Returns:
        None, modifies model inplace!
    """
    init_config: InitConfig = optim_config.get("init", None)
    if init_config is None:
        return

    verbose = optim_config.verbose

    initializers = hydra.utils.instantiate(init_config.regexes)

    prevent_regexes = init_config.get("prevent_regexes", None)
    prevent_regex = None
    if prevent_regexes:
        prevent_regex = "(" + ")|(".join(prevent_regexes) + ")"

    hydra.utils.log.info("Initializing parameters")
    unused_regexes = {k for k, v in initializers.items()}
    uninitialized_parameters = set()
    # Store which initializers were applied to which parameters.
    for name, parameter in module.named_parameters():
        for initializer_regex, initializer in initializers.items():
            allow = prevent_regex is None or not bool(re.search(prevent_regex, name))
            if allow and re.search(initializer_regex, name):
                if verbose:
                    hydra.utils.log.info(
                        "Initializing %s using %s initializer", name, initializer_regex
                    )
                initializer(parameter)
                unused_regexes.discard(initializer_regex)
                break
        else:  # no break
            uninitialized_parameters.add(name)
    for regex in unused_regexes:
        hydra.utils.log.warning(
            "Did not use initialization regex that was passed: %s", regex
        )

    if verbose:
        hydra.utils.log.info(
            "Done initializing parameters; the following parameters are using their "
            "default initialization from their code"
        )
        uninitialized_parameter_list = list(uninitialized_parameters)
        uninitialized_parameter_list.sort()
        for name in uninitialized_parameter_list:
            hydra.utils.log.info("   %s", name)
