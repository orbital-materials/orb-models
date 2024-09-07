import re
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import hydra
import omegaconf
import torch
from orb_models.finetune_utilities.ema import ExponentialMovingAverage as EMA

Metric = Union[torch.Tensor, int, float]
MetricCollection = Union[Metric, Mapping[str, Metric]]
TensorDict = Mapping[str, Optional[torch.Tensor]]


OptimizerKwargs = Dict[str, Any]


class ParameterGroup(omegaconf.DictConfig):
    """Param type for specifying Param groups."""

    optimizer_kwargs: OptimizerKwargs
    filter_string: str


ParameterGroups = List[ParameterGroup]


def make_parameter_groups(
    module: torch.nn.Module, groups: ParameterGroups, verbose: bool = False
) -> List[Dict[str, Any]]:
    """Construct parameter groups for model optimization.

    Args:
        module: A torch Module with parameters to optimize.
        groups: A dictionary of regexes mapping to optimizer kwargs.
            See below for more info on how the groups are constructed.

    Takes a module and a parameter grouping (as specified below), and prepares them to be passed
    to the `__init__` function of a `torch.Optimizer`. This means separating the parameters into
    groups with the given regexes, and prepping whatever keyword arguments are given for those
    regexes in `groups`.

    Returns:
        The parameter groups ready to be passed to an optimizer.

    `groups` contains:
    ```
    {
        "regex1": {"lr": 1e-3},
        "regex2": {"lr": 1e-4}
    }
    ```
    All of key-value pairs specified in each of these dictionaries will be passed to the optimizer.
    If there are multiple groups specified, this is a list of dictionaries, where each
    dict contains a "parameter group" and groups specific options, e.g., {'params': [list of
    parameters], 'lr': 1e-3, ...}.  Any config option not specified in the additional options (e.g.
    for the default group) is inherited from the top level arguments given in the constructor. See:
    https://pytorch.org/docs/stable/optim.html#per-parameter-options
    """
    parameter_groups: List[Dict[str, Any]] = [
        {"params": [], **g["optimizer_kwargs"]} for g in groups
    ]
    # In addition to any parameters that match group specific regex,
    # we also need a group for the remaining "default" group.
    # Those will be included in the last entry of parameter_groups.
    parameter_groups.append({"params": []})

    regex_use_counts: Dict[str, int] = {}
    parameter_group_names: List[set] = [set() for _ in range(len(groups) + 1)]

    for name, param in module.named_parameters():
        # Determine the group for this parameter.
        group_index = None
        regex_names = [g["filter_string"] for g in groups]
        for k, regex in enumerate(regex_names):
            if regex not in regex_use_counts:
                regex_use_counts[regex] = 0
            if re.search(regex, name):
                if group_index is not None and group_index != k:
                    raise ValueError(
                        "{} was specified in two separate parameter groups".format(name)
                    )
                group_index = k
                regex_use_counts[regex] += 1

        if group_index is not None:
            # we have a group
            parameter_groups[group_index]["params"].append(param)
            parameter_group_names[group_index].add(name)
        else:
            # the default group
            parameter_groups[-1]["params"].append(param)
            parameter_group_names[-1].add(name)

    # log the remaining parameter groups
    hydra.utils.log.info("Constructed parameter groups:")
    for k in range(len(parameter_groups)):
        group_options = {
            key: val for key, val in parameter_groups[k].items() if key != "params"
        }
        hydra.utils.log.info("Group %s, options: %s", k, group_options)
        if verbose:
            hydra.utils.log.info("Parameters: ")
            for p in list(parameter_group_names[k]):
                hydra.utils.log.info(p)

    # check for unused regex
    for regex, count in regex_use_counts.items():
        if count == 0:
            hydra.utils.log.warning(
                "Parameter group regex %s does not match any parameter name.",
                regex,
            )
    return parameter_groups


def get_optim(
    learning_rate: float, model: torch.nn.Module
) -> Tuple[
    torch.optim.Optimizer,
    Optional[torch.optim.lr_scheduler._LRScheduler],
    Optional[EMA],
]:
    """Configure optimizers, LR schedulers and EMA from a Hydra config."""
    parameter_groups = [
        {
            "filter_string": "(.*bias|.*layer_norm.*|.*batch_norm.*)",
            "optimizer_kwargs": {"weight_decay": 0.0},
        }
    ]
    params = make_parameter_groups(model, parameter_groups)
    opt = torch.optim.Adam(params, lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    ema_decay = 0.999
    ema = EMA(model.parameters(), ema_decay)

    return opt, scheduler, ema
