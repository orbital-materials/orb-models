## Pretrained models

We provide several pretrained models that can be used to calculate energies, forces & stresses of atomic systems. All models are provided in the `orb_models.forcefield.pretrained` module.

### OrbMol Models

These models are a continuation of the `orb-v3` series, but are trained on the [Open Molecules 2025 (OMol25)](https://arxiv.org/pdf/2505.08762) dataset—over 100M high-accuracy DFT calculations (ωB97M-V/def2-TZVPD) on diverse molecular systems including metal complexes, biomolecules, and electrolytes.

There are two options:
* `orb-v3-conservative-omol`
* `orb-v3-direct-omol`

See below for more explanation of this naming convention. Both models have `inf` neighbors, ensuring a continuous PES.

### [V3 Models](https://arxiv.org/abs/2504.06231)
V3 models use the following naming convention: ```orb-v3-X-Y-Z``` where:
- `X`: Model type - `direct` or `conservative`. Conservative models compute forces and stress via backpropagation, which is a physically motivated choice that appears necessary for certain types of simulation such as NVE Molecular dynamics. Conservative models are signficantly slower and use more memory than their direct counterparts.

- `Y`: Maximum neighbors per atom: `20` or `inf`. A finite cutoff of `20` induces discontinuties in the PES, which can lead to significant inaccuracies for certain types of highly sensitive calculations (e.g. calculations involving Hessians). However, finite cutoffs reduce the amount of edge processing in the network, reducing latency and memory use.

- `Z`: Training dataset - `omat` or `mpa`. Both of these dataset consist of small bulk crystal structures. We find that models trained on such data can generalise reasonably well to non-periodic systems (organic molecules) or partially periodic systems (slabs), but caution is advised in these scenarios.

#### Features:
- Model compilation using PyTorch 2.6.0+, enabling faster inference while maintaining support for dynamic graph sizes
- Wider architecture (1024 vs 512) with fewer layers (5 vs 15) compared to v2, resulting in 2-3x faster performance with similar parameter count
- Two variants available: direct models and conservative models (forces/stress computed via backpropagation)
- Trained on the larger, more diverse OMat24 dataset
- Improved edge embeddings using Bessel-Spherical Harmonic outer products (8 Bessel bases, Lmax=3)
- Enhanced stability through Huber loss and a ZBL pair repulsion term added to forces
- Models available with both unlimited neighbors and 20-neighbor maximum configurations
- New confidence head providing intrinsic uncertainty estimates for predictions

#### Advice / Caveats
- Consider using `orb-v3-conservative-120-omat` for initial testing, specifying `precision='float32-highest'` when loading the model. This is the most computational expensive but accurate configuration. If this level of accuracy meets your needs, then other models and precisions can be investigated to improve speed and system-size scalability.
- We do not advise using the `-mpa` models unless they are required for compatability with benchmarks (for example, Matbench Discovery). They are generally less performant.
- Orb-v3 models are **compiled** by default and use Pytorch's dynamic batching, which means that they do not need to recompile as graph sizes change. However, the first call to the model will be slower, as the graph is compiled by torch.

### [V2 Models](https://arxiv.org/abs/2410.22570)

- `orb-v2` - trained on [MPTraj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842?file=41619375) + [Alexandria](https://alexandria.icams.rub.de/).
- `orb-mptraj-only-v2` - trained on the MPTraj dataset only to reproduce our second Matbench Discovery result. We do not recommend using this model for general use.
- `orb-d3-v2` - trained on MPTraj + Alexandria with integrated D3 corrections. In general, we recommend using this model, particularly for systems where dispersion interactions are important. This model was trained to predict D3-corrected targets and hence is the same speed as `orb-v2`. Incorporating D3 into the model like this is substantially faster than using analytical D3 corrections.
- `orb-d3-{sm,xs}-v2` - Smaller versions of `orb-d3-v2`. The `sm` model has 10 layers, whilst the `xs` model has 5 layers.

#### Features
- v2 models use a smoothed cosine distance cutoff for the attention mechanism, ensuring a continuous PES.
- The force predictions now have net zero forces, meaning they are much more stable for MD simulations.
- The models are generally more accurate (Increase in 2-3% on the matbench discovery dataset).

### [V1 Models](https://arxiv.org/abs/2410.22570)

Our initial release. These models were state of the art performance on the matbench discovery dataset at time of release, but have since been superceeded and removed.
