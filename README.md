
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

### Pretrained models

We provide several pretrained models that can be used to calculate energies, forces & stresses of atomic systems. All models are provided in the `orb_models.forcefield.pretrained` module. 

- `orb-v1` - trained on [MPTraj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842?file=41619375) + [Alexandria](https://alexandria.icams.rub.de/).
- `orb-mptraj-only-v1` - trained on the MPTraj dataset only to reproduce our second Matbench Discovery result. We do not recommend using this model for general use.
- `orb-d3-v1` - trained on MPTraj + Alexandria with integrated D3 corrections. In general, we recommend using this model, particularly for systems where dispersion interactions are important. This model was trained to predict D3-corrected targets and hence is the same speed as `orb-v1`. Incorporating D3 into the model like this is substantially faster than using analytical D3 corrections.
- `orb-d3-{sm,xs}-v1` - Smaller versions of `orb-d3-v1`. The `sm` model has 10 layers, whilst the `xs` model has 5 layers.

For more information on the models, please see the [MODELS.md](MODELS.md) file.


### Usage

#### Direct usage
```python

import ase
from ase.build import bulk

from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs

device = "cpu"  # or device="cuda"
orbff = pretrained.orb_v1(device=device)
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
orbff = pretrained.orb_v1(device=device) # or choose another model using ORB_PRETRAINED_MODELS[model_name]()
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
```python
python finetune.py --dataset=<dataset_name> --data_path=<your_data_path>
```
After the model is finetuned, checkpoints will be saved. You can us the new model and load the checkpoint by modifying the `weights_path` in `pretrained.py`. 

### Citing

We are currently preparing a preprint for publication.


### License

ORB models are licensed under the ORB Community License Agreement, Version 1. Please see the [LICENSE](LICENSE) file for details.

**If you have an interesting use case or benchmark for an Orb model, please let us know!** We are happy to work with the community to make these models useful for as many applications as possible.
Please fill in the [commercial license form](https://www.orbitalmaterials.com/commercial-license-form), or open an issue on GitHub.

### Community

Please join the discussion on Discord by following [this](https://discord.gg/4EYXRA7x) link.