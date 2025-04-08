
<p align="center">
  <img src="./logo_color_text.png" alt="Orbital Materials" width="600"/>
</p>
<br/>

## Announcement

**Important Notice**: All models pre version 4.1.0 will reach End of Life (EOL) on 26th January due to removal of the old model weights. Please ensure to update to the latest version to continue receiving support and updates.


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

### Updates

**April 2025**: We have released the Orb-v3 set of potentials. These models improve substantially over Orb-v2, in particular:

- Model compilation using PyTorch 2.6.0+, enabling faster inference while maintaining support for dynamic graph sizes
- Wider architecture (1024 vs 512) with fewer layers (5 vs 15) compared to v2, resulting in 2-3x faster performance with similar parameter count
- Two variants available: direct models and conservative models (forces/stress computed via backpropagation)
- Trained on the larger, more diverse OMat24 dataset
- Improved edge embeddings using Bessel-Spherical Harmonic outer products (8 Bessel bases, Lmax=3)
- Enhanced stability through Huber loss and a ZBL pair repulsion term added to forces
- Models available with both unlimited neighbors and 20-neighbor maximum configurations
- New confidence head providing intrinsic uncertainty estimates for predictions


**Oct 2024**: We have released a new version of the models, `orb-v2`. This version has 2 major changes:
- v2 models use a smoothed cosine distance cutoff for the attention mechanism. This is a more physically motivated cutoff that is better suited for MPNNs.
- The force predictions now have net zero forces, meaning they are much more stable for MD simulations.
- The models are generally more accurate (Increase in 2-3% on the matbench discovery dataset).

These models are substantially better for all use cases, so we have removed the v1 models from the new orb-models package. To load the v1 models, please install the v0.3.2 version of orb-models.

**Sept 2024**: v1 models released - state of the art performance on the matbench discovery dataset.


### Pretrained models

We provide several pretrained models that can be used to calculate energies, forces & stresses of atomic systems. All models are provided in the `orb_models.forcefield.pretrained` module.

#### V3 Models
V3 models use the following naming convention:

```orb-v3-X-Y-Z```

where:
- `X`: Model type (`direct` or `conservative`) - determines how forces/stress are computed
- `Y`: Maximum neighbors per atom (`20` or `inf`)
- `Z`: Training dataset (`omat` or `mpa`)

For example, `orb-v3-conservative-inf-omat` is a model that:
- Computes forces/stress as gradients of energy
- Has effectively infinite neighbors (120 in practice)
- Was trained on the OMat24 dataset


Orb-v3 models are **compiled** by default and use Pytorch's dynamic batching, which means that they do not need to recompile as graph sizes change. However, the first call to the model will be slower, as the graph is compiled by torch.


**We suggest using models trained on OMAT24**, as these models are more performant and the data they are trained on uses newer pseudopotentials in VASP (PBE54 vs PBE52)*. `-mpa` models should be used if compatability with benchmarks (for example, Matbench Discovery) is required.

#### V2 Models

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
orbff = pretrained.orb_v3_conservative_inf_omat(
  device=device,
  precision="float32-high",   # or "float32-highest" / "float64
)
atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)
graph = atomic_system.ase_atoms_to_atom_graphs(atoms, orbff.system_config, device=device)
atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)
graph = atomic_system.ase_atoms_to_atom_graphs(atoms, orbff.system_config, device=device)

# Optionally, batch graphs for faster inference
# graph = batch_graphs([graph, graph, ...])

result = orbff.predict(graph, split=False)

# Convert to ASE atoms (unbatches the results and transfers to cpu if necessary)
atoms = atomic_system.atom_graphs_to_ase_atoms(
    graph,
    energy=result["energy"],
    forces=result["forces"],
    stress=result["stress"]
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
Where base_model is one of:
- "orb_v3_conservative_inf_omat"
- "orb_v3_conservative_20_omat"
- "orb_v3_direct_inf_omat"
- "orb_v3_direct_20_omat"
- "orb_v2"

After the model is finetuned, checkpoints will, by default, be saved to the ckpts folder in the directory you ran the finetuning script from.

You can use the new model and load the checkpoint by:
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




### Citing

A preprint describing the model in more detail can be found here: https://arxiv.org/abs/2410.22570

```bibtex
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