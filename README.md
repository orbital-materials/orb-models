
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

We provide several pretrained models that can be used to calculate energies and forces of atomic systems. All models are provided in the `orb_models.forcefield.pretrained` module. 

- `orb-v1` - The main model, trained on the full dataset.
- `orb-mptraj-only-v1` - The main model, trained on the MPTraj dataset only to reproduce our second Matbench Discovery result. We do not recommend using this model for general use.

- `orb-d3-v1` - The main model, trained on the full dataset with integrated D3 corrections. This model should be used for systems where dispersion interactions are important. In general, we recommend using this model. This model is the same speed as the main model, as we include the D3 corrections in the model itself. This is substantially faster than using analytical D3 corrections.
- `orb-d3-{sm,xs}-v1` - These models are pretrained on bulk data, and finetuned on adsorption + bulk systems. The `sm` models are 10 layer models, and the `xs` models are 5 layer models.

For more information on the models, please see the [MODELS.md](MODELS.md) file.


### Usage

#### Direct usage
```python

import ase
from ase.build import bulk
from orb_models.forcefield import pretrained
from orb_models.forcefield import atomic_system
from orb_models.forcefield.base import batch_graphs

orbff = pretrained.orb_v1()
atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)
graph = atomic_system.ase_atoms_to_atom_graphs(atoms)

# Optionally, batch graphs for faster inference
# graph = batch_graphs([graph, graph, ...])

result = orbff.predict(graph)

# Convert to ASE atoms (this will also unbatch the results)
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


orbff = pretrained.orb_v1() # Or choose another model using ORB_PRETRAINED_MODELS[model_name]()
calc = ORBCalculator(orbff, device="cpu")  # or device="cuda"
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

calc = ORBCalculator(orbff, device="cpu")
dyn = BFGS(atoms)
dyn.run(fmax=0.01)
print("Optimized Energy:", atoms.get_potential_energy())
```


### Citing

We are currently preparing a preprint for publication.


### License

ORB models are licensed under the ORB Community License Agreement, Version 1. Please see the [LICENSE](LICENSE) file for details.

**If you have an interesting use case or benchmark for an Orb model, please let us know!** We are happy to work with the community to make these models useful for as many applications as possible.
Please fill in the [commercial license form](https://www.orbitalmaterials.com/commercial-license-form), or open an issue on GitHub.

### Community

Please join the discussion on Discord by following [this](https://discord.gg/4EYXRA7x) link.