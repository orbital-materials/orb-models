
<p align="center">
  <img src="./logo_color_text.png" alt="Orbital Materials" width="600"/>
</p>
<br/>

# Pretrained models for atomic simulations

![example workflow](https://github.com/orbital-materials/orb-models/actions/workflows/test.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/orb-models.svg)](https://badge.fury.io/py/orb-models)

### Install

```bash
pip install orb-models
pip install "pynanoflann@git+https://github.com/dwastberg/pynanoflann#egg=af434039ae14bedcbb838a7808924d6689274168",
```
Pynanoflann is not available on PyPI, so you must install it from the git repository.

Orb models are expected to work on MacOS and Linux. Windows support is not guaranteed.

### Updates

**Oct 2024**: We have released a new version of the models, `orb-v2`. This version has 2 major changes:
- v2 models use a smoothed cosine distance cutoff for the attention mechanism. This is a more physically motivated cutoff that is better suited for MPNNs.
- The force predictions now have net zero forces, meaning they are much more stable for MD simulations.
- The models are generally more accurate (Increase in 2-3% on the matbench discovery dataset).

These models are substantially better for all use cases, so we have removed the v1 models from the new orb-models package. To load the v1 models, please install the v0.3.2 version of orb-models.

**Sept 2024**: v1 models released - state of the art performance on the matbench discovery dataset.


### Pretrained models

We provide several pretrained models that can be used to calculate energies, forces & stresses of atomic systems. All models are provided in the `orb_models.forcefield.pretrained` module.

- `orb-v2` - trained on [MPTraj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842?file=41619375) + [Alexandria](https://alexandria.icams.rub.de/).
- `orb-mptraj-only-v2` - trained on the MPTraj dataset only to reproduce our second Matbench Discovery result. We do not recommend using this model for general use.
- `orb-d3-v2` - trained on MPTraj + Alexandria with integrated D3 corrections. In general, we recommend using this model, particularly for systems where dispersion interactions are important. This model was trained to predict D3-corrected targets and hence is the same speed as `orb-v2`. Incorporating D3 into the model like this is substantially faster than using analytical D3 corrections.
- `orb-d3-{sm,xs}-v2` - Smaller versions of `orb-d3-v2`. The `sm` model has 10 layers, whilst the `xs` model has 5 layers.

For more information on the models, please see the [MODELS.md](MODELS.md) file.


### Usage

Note: These examples are designed to run on the `main` branch of orb-models. If you are using a pip installed version of `orb-models`, you may want to look at the corresponding [README.md from that tag](https://github.com/orbital-materials/orb-models/tags).

#### Direct usage
```python

import ase
from ase.build import bulk

from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs

device = "cpu"  # or device="cuda"
orbff = pretrained.orb_v2(device=device)
atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)
graph = atomic_system.ase_atoms_to_atom_graphs(atoms, device=device)

# Optionally, batch graphs for faster inference
# graph = batch_graphs([graph, graph, ...])

result = orbff.predict(graph)

# Convert to ASE atoms (unbatches the results and transfers to cpu if necessary)
atoms = atomic_system.atom_graphs_to_ase_atoms(
    graph,
    energy=result["graph_pred"],
    forces=result["node_pred"],
    stress=result["stress_pred"]
)
```

#### Usage with ASE calculator

```python
import ase
from ase.build import bulk

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

device="cpu" # or device="cuda"
orbff = pretrained.orb_v2(device=device) # or choose another model using ORB_PRETRAINED_MODELS[model_name]()
calc = ORBCalculator(orbff, device=device)
atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)

atoms.set_calculator(calc)
atoms.get_potential_energy()
```

You can use this calculator with any ASE calculator-compatible code. For example, you can use it to perform a geometry optimization:

```python
from ase.optimize import BFGS

# Rattle the atoms to get them out of the minimum energy configuration
atoms.rattle(0.5)
print("Rattled Energy:", atoms.get_potential_energy())

calc = ORBCalculator(orbff, device="cpu") # or device="cuda"
dyn = BFGS(atoms)
dyn.run(fmax=0.01)
print("Optimized Energy:", atoms.get_potential_energy())
```


### Finetuning
You can finetune the model using your custom dataset.
The dataset should be an [ASE sqlite database](https://wiki.fysik.dtu.dk/ase/ase/db/db.html#module-ase.db.core).
```python
python finetune.py --dataset=<dataset_name> --data_path=<your_data_path>
```
After the model is finetuned, checkpoints will, by default, be saved to the ckpts folder in the directory you ran the finetuning script from. 

You can use the new model and load the checkpoint by:
```python
from orb_models.forcefield import pretrained

model = pretrained.orb_v2(weights_path=<path_to_ckpt>)
```

> ⚠ **Caveats**
>
> Our finetuning script is designed for simplicity and advanced users may wish to develop it further. Please be aware that:
> - The script assumes that your ASE database rows contain **energy, forces, and stress** data. To train on molecular data without stress, you will need to edit the code.
> - **Early stopping** is not implemented. However, you can use the command line argument `save_every_x_epochs` (default is 5), so "retrospective" early stopping can be applied by selecting a suitable checkpoint.
> - The **learning rate schedule is hardcoded** to be `torch.optim.lr_scheduler.OneCycleLR` with `pct_start=0.05`. The `max_lr`/`min_lr` will be 10x greater/smaller than the `lr` specified via the command line. To get the best performance, you may wish to try other schedulers.
> - The defaults of `--num_steps=100` and `--max_epochs=50` are small. This may be suitable for very small finetuning datasets (e.g. 100s of systems), but you will likely want to increase the number of steps for larger datasets (e.g. 1000s of datapoints).
> - The script only tracks a limited set of metrics (energy/force/stress MAEs) which may be insufficient for some downstream use-cases. For instance, if you wish to finetune a model for Molecular Dynamics simulations, we have found (anecdotally) that models that are just on the cusp of overfitting to force MAEs can be substantially worse for simulations. Ideally, more robust "rollout" metrics would be included in the finetuning training loop. In lieu of this, we recommend more aggressive early-stopping i.e. using models several epochs prior to any sign of overfitting.

### Citing

We are currently preparing a preprint for publication.


### License

ORB models are licensed under the Apache License, Version 2.0. Please see the [LICENSE](LICENSE) file for details.

**If you have an interesting use case or benchmark for an Orb model, please let us know!** We are happy to work with the community to make these models useful for as many applications as possible.

### Community

Please join the discussion on Discord by following [this](https://discord.gg/SyD6vWSSTB) link.