
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
```

Orb models are expected to work on MacOS and Linux. Windows support is not guaranteed.

For large system (≳5k atoms PBC, or ≳30k atoms non-PBC) simulations we recommend installing [cuML](https://docs.rapids.ai/install/) (requires CUDA), which can significantly reduce graph creation time (2-10x) and improve GPU memory efficiency (2-100x):
```bash
pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu11==25.2.*"  # For cuda versions >=11.4, <11.8
pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu12==25.2.*"  # For cuda versions >=12.0, <13.0
```

Alternatively, you can use Docker to run orb-models; [see instructions below](#docker).

### Updates

**August 2025**: Release of the OrbMol potentials (blog post forthcoming). 

* Trained on the [Open Molecules 2025 (OMol25)](https://arxiv.org/pdf/2505.08762) dataset—over 100M high-accuracy DFT calculations (ωB97M-V/def2-TZVPD) on diverse molecular systems including metal complexes, biomolecules, and electrolytes.
* Architecturally similar to the highly-performant Orb-v3 models, but now explicit total charges and spins can be passed as input.  
* To get started with these models, see: [How to specify total charge and spin for OrbMol](#how-to-specify-total-charge-and-spin-for-orbmol).

**April 2025**: Release of the [Orb-v3 set of potentials](https://arxiv.org/abs/2504.06231).

**Oct 2024**: Release of the [Orb-v2 set of potentials](https://arxiv.org/abs/2410.22570). 

**Sept 2024**: Release of v1 models - state of the art performance on the matbench discovery dataset.


### Available models
See [MODELS.md](MODELS.md) for a full list of available models along with guidance.


### Usage

Note: These examples are designed to run on the `main` branch of orb-models. If you are using a pip installed version of `orb-models`, you may want to look at the corresponding [README.md from that tag](https://github.com/orbital-materials/orb-models/tags).

#### Direct usage
```python

import ase
from ase.build import bulk

from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs

device = "cpu"  # or device="cuda"
orbff = pretrained.orb_v3_conservative_inf_omat(
  device=device,
  precision="float32-high",   # or "float32-highest" / "float64
)
atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)
graph = atomic_system.ase_atoms_to_atom_graphs(atoms, orbff.system_config, device=device)

# If you have several graphs, batch them like so:
# graph = batch_graphs([graph1, graph2, ...])

result = orbff.predict(graph, split=False)

# Convert to ASE atoms (unbatches the results and transfers to cpu if necessary)
atoms = atomic_system.atom_graphs_to_ase_atoms(
    graph,
    energy=result["energy"],
    forces=result["grad_forces"],
    stress=result["grad_stress"]
)
```

#### Usage with ASE calculator

```python
import ase
from ase.build import bulk

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

device="cpu" # or device="cuda"
# or choose another model using ORB_PRETRAINED_MODELS[model_name]()
orbff = pretrained.orb_v3_conservative_inf_omat(
  device=device,
  precision="float32-high",   # or "float32-highest" / "float64
)
calc = ORBCalculator(orbff, device=device)
atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)

atoms.calc = calc
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

Or you can use it to run MD simulations. The script, an example input xyz file and a Colab notebook demonstration are available in the [examples directory.](./examples) This should work with any input, simply modify the input_file and cell_size parameters. We recommend using constant volume simulations.

#### How to specify total charge and spin for OrbMol

The OrbMol models *require* total charge and spin to be specified. This can be done by setting them in `atoms.info` dictionary.

```python
import ase
from ase.build import molecule
from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs

device = "cpu"  # or device="cuda"
orbff = pretrained.orb_v3_conservative_omol(
  device=device,
  precision="float32-high",   # or "float32-highest" / "float64
)
atoms = molecule("C6H6")
atoms.info["charge"] = 1.0  # total charge
atoms.info["spin"] = 0.0  # total spin
graph = atomic_system.ase_atoms_to_atom_graphs(atoms, orbff.system_config, device=device)

result = orbff.predict(graph, split=False)
```

#### Confidence head (Orb-v3 Models Only)

Orb-v3 models have a confidence head which produces a per-atom discrete confidence measure based on a classifier head which learns to predict the binned MAE between predicted and true forces during training. This classifier head has 50 bins, linearly spaced between 0 and 0.4A.


```python
import ase
from ase.build import molecule
from seaborn import heatmap # optional, for visualization only
import matplotlib.pyplot as plt # optional, for visualization only
import numpy

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

device="cpu" # or device="cuda"
# or choose another model using ORB_PRETRAINED_MODELS[model_name]()
orbff = pretrained.orb_v3_conservative_inf_omat(
  device=device,
)
calc = ORBCalculator(orbff, device=device)
# Use a molecule (OOD for Orb, so confidence plot is
# more interesting than a bulk crystal)
atoms = molecule("CH3CH2Cl")
atoms.calc = calc

forces = atoms.get_forces()
confidences = calc.results["confidence"]
predicted_bin_per_atom = numpy.argmax(confidences, axis=-1)

print(forces.shape, confidences.shape) # (num_atoms, 3), (num_atoms, 50)
print(predicted_bin_per_atom) # List of length num_atoms
heatmap(confidences)
plt.xlabel('Confidence Bin')
plt.ylabel('Atom Index')
plt.title('Confidence Heatmap')
plt.show()

```


### Floating Point Precision

As shown in usage snippets above, we support 3 floating point precision types: `"float32-high"`, `"float32-highest"` and `"float64"`.

The default value of `"float32-high"` is recommended for maximal acceleration when using A100 / H100 Nvidia GPUs. However, we have observed some performance loss for high-precision calculations involving second and third order properties of the PES. In these cases, we recommend `"float32-highest"`. 

In stark constrast to other universal forcefields, we have not found any benefit to using `"float64"`.

### Finetuning
You can finetune the model using your custom dataset.
The dataset should be an [ASE sqlite database](https://wiki.fysik.dtu.dk/ase/ase/db/db.html#module-ase.db.core).
```python
python finetune.py --dataset=<dataset_name> --data_path=<your_data_path> --base_model=<base_model>
```
Where base_model is an element of `orb_models.forcefield.pretrained.ORB_PRETRAINED_MODELS.keys()`.

After the model is finetuned, checkpoints will, by default, be saved to the ckpts folder in the directory you ran the finetuning script from. You can use the new model and load the checkpoint by:
```python
from orb_models.forcefield import pretrained

model = getattr(pretrained, <base_model>)(
  weights_path=<path_to_ckpt>, 
  device="cpu",               # or device="cuda"
  precision="float32-high",   # or precision="float32-highest"
)
```

> ⚠ **Caveats**
>
> Our finetuning script is designed for simplicity. We strongly advise users to customise it further for their use-case to get the best performance. Please be aware that:
> - The script assumes that your ASE database rows contain **energy, forces, and stress** data. To train on molecular data without stress, you will need to edit the code.
> - **Early stopping** is not implemented. However, you can use the command line argument `save_every_x_epochs` (default is 5), so "retrospective" early stopping can be applied by selecting a suitable checkpoint.
> - The **learning rate schedule is hardcoded** to be `torch.optim.lr_scheduler.OneCycleLR` with `pct_start=0.05`. The `max_lr`/`min_lr` will be 10x greater/smaller than the `lr` specified via the command line. To get the best performance, you may wish to try other schedulers.
> - The defaults of `--num_steps=100` and `--max_epochs=50` are small. This may be suitable for very small finetuning datasets (e.g. 100s of systems), but you will likely want to increase the number of steps for larger datasets (e.g. 1000s of datapoints).
> - The script only tracks a limited set of metrics (energy/force/stress MAEs) which may be insufficient for some downstream use-cases. For instance, if you wish to finetune a model for Molecular Dynamics simulations, we have found (anecdotally) that models that are just on the cusp of overfitting to force MAEs can be substantially worse for simulations. Ideally, more robust "rollout" metrics would be included in the finetuning training loop. In lieu of this, we recommend more aggressive early-stopping i.e. using models several epochs prior to any sign of overfitting.


## Docker

You can run orb-models using Docker, which provides a consistent environment with all dependencies pre-installed:

1. Build the Docker image locally:

   ```bash
   docker build -t orb_models .
   ```
2. Run the Docker container:

   ```bash
   docker run --gpus all --rm --name orb_models -it orb_models /bin/bash
   ```


### Citing

Preprints describing the models in more detail can be found at:
* **Orb-v3:** https://arxiv.org/abs/2504.06231
* **Orb-v2:** https://arxiv.org/abs/2410.22570

```bibtex
@misc{rhodes2025orbv3atomisticsimulationscale,
      title={Orb-v3: atomistic simulation at scale}, 
      author={Benjamin Rhodes and Sander Vandenhaute and Vaidotas Šimkus and James Gin and Jonathan Godwin and Tim Duignan and Mark Neumann},
      year={2025},
      eprint={2504.06231},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2504.06231}, 
}

@misc{neumann2024orbfastscalableneural,
      title={Orb: A Fast, Scalable Neural Network Potential}, 
      author={Mark Neumann and James Gin and Benjamin Rhodes and Steven Bennett and Zhiyi Li and Hitarth Choubisa and Arthur Hussey and Jonathan Godwin},
      year={2024},
      eprint={2410.22570},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2410.22570}, 
}
```

### License

ORB models are licensed under the Apache License, Version 2.0. Please see the [LICENSE](LICENSE) file for details.

**If you have an interesting use case or benchmark for an Orb model, please let us know!** We are happy to work with the community to make these models useful for as many applications as possible.

### Community

Please join the discussion on Discord by following [this](https://discord.gg/SyD6vWSSTB) link.
