# Finetuning Guide: Custom Loss Weights & Reference Energies

This guide explains how to use `finetune.py` to finetune Orb models with custom loss weights and reference energies.

## Features

### 1. Custom Loss Weights
Control the relative importance of energy, forces, and stress in the loss function.

### 2. Custom Reference Energies
- **Load from file**: Provide your own reference energies
- **Fixed or trainable**: Keep them fixed during training or optimize them
- **Multiple formats**: Supports JSON and text file formats

## Data Format

The finetuning script expects data in **ASE SQLite database format**. This is a standard format used by the Atomic Simulation Environment (ASE) library.

### Required Properties

Each structure in your database should have:
- **Positions**: Atomic positions (automatically stored with the Atoms object)
- **Atomic numbers**: Element types (automatically stored with the Atoms object)
- **Cell**: Unit cell vectors (for periodic systems)
- **Energy**: Total energy of the structure (in eV)
- **Forces**: Forces on each atom (in eV/Å), shape `(n_atoms, 3)`
- **Stress** (optional): Stress tensor in Voigt notation (6 components), in eV/Å³

### Creating an ASE Database

Here's how to convert your data into the required format:

```python
import ase
import ase.db
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

# Create a database file
db = ase.db.connect('my_training_data.db')

# For each structure in your dataset:
for structure in your_structures:
    # Create an ASE Atoms object
    atoms = Atoms(
        symbols=structure['symbols'],        # e.g., ['H', 'H', 'O']
        positions=structure['positions'],    # shape: (n_atoms, 3) in Angstroms
        cell=structure['cell'],              # shape: (3, 3) in Angstroms
        pbc=True                              # Set to True for periodic systems
    )
    
    # Attach energy, forces, and stress using SinglePointCalculator
    calc = SinglePointCalculator(
        atoms=atoms,
        energy=structure['energy'],          # Total energy in eV
        forces=structure['forces'],          # shape: (n_atoms, 3) in eV/Å
        stress=structure['stress']           # shape: (6,) in eV/Å³ (Voigt notation)
    )
    atoms.calc = calc
    
    # Write to database
    db.write(atoms)

print(f"Created database with {len(db)} structures")
```

### Example: Converting from XYZ with External Properties

If you have structures in XYZ format with energies/forces in separate files:

```python
import ase.io
import ase.db
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

# Read structures
atoms_list = ase.io.read('structures.xyz', index=':')

# Load your energies and forces (example)
energies = np.loadtxt('energies.txt')  # One energy per structure
forces_list = [...]  # List of (n_atoms, 3) arrays

# Create database
db = ase.db.connect('training_data.db')

for atoms, energy, forces in zip(atoms_list, energies, forces_list):
    calc = SinglePointCalculator(
        atoms=atoms,
        energy=energy,
        forces=forces
    )
    atoms.calc = calc
    db.write(atoms)
```

### Stress Tensor Format

If your data includes stress, it should be in **Voigt notation** (6 components):
```python
stress = [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]  # in eV/Å³
```

If you have a full 3×3 stress tensor, convert it to Voigt notation:
```python
stress_voigt = [
    stress_3x3[0, 0],  # σ_xx
    stress_3x3[1, 1],  # σ_yy
    stress_3x3[2, 2],  # σ_zz
    stress_3x3[1, 2],  # σ_yz
    stress_3x3[0, 2],  # σ_xz
    stress_3x3[0, 1],  # σ_xy
]
```

### Verifying Your Database

Check that your database is formatted correctly:

```python
import ase.db

db = ase.db.connect('my_training_data.db')
print(f"Total structures: {len(db)}")

# Check first structure
row = db.get(1)
atoms = row.toatoms()

print(f"Formula: {row.formula}")
print(f"Energy: {row.energy} eV")
print(f"Forces shape: {row.forces.shape}")
print(f"Has stress: {row.stress is not None}")
print(f"Number of atoms: {row.natoms}")
```

Expected output:
```
Total structures: 1000
Formula: H2O
Energy: -14.2234 eV
Forces shape: (3, 3)
Has stress: True
Number of atoms: 3
```

## Usage

### Basic Finetuning with Custom Loss Weights

```bash
python finetune.py \
  --data_path /path/to/your/dataset.db \
  --base_model orb_v3_conservative_omol \
  --energy_loss_weight 0.1 \
  --forces_loss_weight 1.0 \
  --stress_loss_weight 0.0 \
  --batch_size 100 \
  --max_epochs 50
```

### Finetuning with Custom Fixed Reference Energies

```bash
python finetune.py \
  --data_path /path/to/your/dataset.db \
  --base_model orb_v3_conservative_omol \
  --custom_reference_energies /path/to/reference_energies.json \
  --energy_loss_weight 0.1 \
  --forces_loss_weight 1.0
```

### Finetuning with Trainable Reference Energies

```bash
python finetune.py \
  --data_path /path/to/your/dataset.db \
  --base_model orb_v3_conservative_omol \
  --custom_reference_energies /path/to/reference_energies.json \
  --trainable_reference_energies \
  --energy_loss_weight 0.1 \
  --forces_loss_weight 1.0
```

### Make Pretrained Reference Energies Trainable (No Custom File)

```bash
python finetune.py \
  --data_path /path/to/your/dataset.db \
  --base_model orb_v3_conservative_omol \
  --trainable_reference_energies \
  --energy_loss_weight 0.1 \
  --forces_loss_weight 1.0
```

## Reference Energy File Formats

### JSON Format

You can use either element symbols or atomic numbers as keys:

**With element symbols:**
```json
{
  "H": -13.6,
  "C": -1030.5,
  "N": -1400.0,
  "O": -2000.0
}
```

**With atomic numbers:**
```json
{
  "1": -13.6,
  "6": -1030.5,
  "7": -1400.0,
  "8": -2000.0
}
```

### Text Format

**With element symbols:**
```text
H -13.6
C -1030.5
N -1400.0
O -2000.0
```

**With atomic numbers:**
```text
1 -13.6
6 -1030.5
7 -1400.0
8 -2000.0
```

Lines starting with `#` are treated as comments and ignored.

## Arguments

### Loss Weights

- `--energy_loss_weight`: Weight for energy loss (default: uses model default, usually 1.0)
- `--forces_loss_weight`: Weight for forces loss (automatically uses correct key for model type)
- `--stress_loss_weight`: Weight for stress loss (set to 0 to disable)

### Reference Energies

- `--custom_reference_energies`: Path to reference energies file (JSON or text format)
- `--trainable_reference_energies`: Make reference energies trainable during finetuning

### Model Types

The script automatically handles the differences between conservative and direct models:

- **Conservative models** (e.g., `orb_v3_conservative_omol`):
  - Use `grad_forces` and `grad_stress` as **loss-weight keys**
  - Compute forces via automatic differentiation

- **Direct models** (e.g., `orb_v3_direct_omol`):
  - Use `forces` and `stress` as **loss-weight keys**
  - Predict forces directly

When you specify loss weights via command line (e.g., `--forces_loss_weight 10.0`), the script automatically maps this to the correct key (`grad_forces` for conservative models, `forces` for direct models).

## Using the API Directly (Without the Finetuning Script)

If you prefer to write your own finetuning script, you can use the clean API directly:

```python
from orb_models.forcefield import pretrained

# Load model with custom configuration
model = pretrained.orb_v3_conservative_omol(
    device='cuda',
    precision='float32-high',
    train=True,
    train_reference_energies=True,  # Make reference energies trainable
    loss_weights={
        'energy': 1.0,
        'grad_forces': 10.0,      # Use 'grad_forces' for conservative models
        'grad_stress': 100.0       # Use 'grad_stress' for conservative models
    }
)

# For direct models, use 'forces' and 'stress' keys:
model = pretrained.orb_v3_direct_omol(
    device='cuda',
    train=True,
    loss_weights={
        'energy': 1.0,
        'forces': 10.0,    # Use 'forces' for direct models
        'stress': 100.0    # Use 'stress' for direct models  
    }
)

# The model is now ready for training with your custom configuration!
```

This approach is more "pythonic" and clearly documents what configuration options are available. It also encapsulates the implementation details, making your code less fragile to internal changes.

## How It Works

### Reference Energies

1. **Without custom reference energies**: The model uses the pretrained reference energies from the checkpoint
2. **With `--custom_reference_energies`**: Your custom values replace the pretrained ones
3. **With `--trainable_reference_energies`** (or `train_reference_energies=True` in the API): Reference energies become learnable parameters that will be optimized during training

### Loading Finetuned Models

When you save a checkpoint after finetuning, the reference energies (whether custom or trained) are saved in the state dict. When you load the checkpoint later:

```python
import torch
from orb_models.forcefield import pretrained

# Load model architecture (set train=False for inference)
model = pretrained.orb_v3_conservative_omol(train=False)

# Load your finetuned checkpoint
model.load_state_dict(torch.load('path/to/finetuned_checkpoint.pt'))

# The custom/trained reference energies and any modified parameters are now loaded!
```

You can also specify loss weights when loading for further finetuning:

```python
# Load for continued finetuning with different loss weights
model = pretrained.orb_v3_conservative_omol(
    train=True,
    loss_weights={'energy': 0.5, 'grad_forces': 20.0}
)
model.load_state_dict(torch.load('path/to/finetuned_checkpoint.pt'))
```

## Example Workflows

### Scenario 1: Using the finetuning script

Finetuning on ORCA wB97M-V data with different reference scheme:

1. Create your reference energies file (`my_refs.json`):
```json
{
  "H": -13.6,
  "C": -1030.5,
  "N": -1400.0,
  "O": -2000.0
}
```

2. Run finetuning with fixed references:
```bash
python finetune.py \
  --data_path my_dataset.db \
  --base_model orb_v3_conservative_omol \
  --custom_reference_energies my_refs.json \
  --energy_loss_weight 1.0 \
  --forces_loss_weight 10.0 \
  --max_epochs 50
```

3. Use the finetuned model:
```python
from orb_models.forcefield import pretrained
import torch

model = pretrained.orb_v3_conservative_omol(train=False)
model.load_state_dict(torch.load('checkpoints/my_finetuned_model.pt'))
# Reference energies from my_refs.json are now loaded!
```

### Scenario 2: Writing your own finetuning script with the API

```python
import torch
from torch.utils.data import DataLoader
from orb_models.forcefield import pretrained
from orb_models.dataset.ase_sqlite_dataset import AseSqliteDataset

# Load model with configuration
model = pretrained.orb_v3_conservative_omol(
    device='cuda',
    train=True,
    train_reference_energies=False,  # Fixed reference energies
    loss_weights={
        'energy': 1.0,
        'grad_forces': 10.0,
    }
)

# Load your data
dataset = AseSqliteDataset(
    name='my_dataset',
    path='my_dataset.db',
    system_config=model.system_config,
    target_config={'graph': ['energy'], 'node': ['forces']}
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Set up optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(50):
    for batch in dataloader:
        batch = batch.to('cuda')
        output = model.loss(batch)
        loss = output.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save checkpoint
torch.save(model.state_dict(), 'my_finetuned_model.pt')
```

## Tips

1. **Energy vs Forces weighting**: If forces are more important for your application, use higher `--forces_loss_weight` (e.g., 10.0) and lower `--energy_loss_weight` (e.g., 0.1)

2. **Fixed vs Trainable references**: 
   - Use **fixed** if you know the correct reference energies for your method
   - Use **trainable** if you want the model to learn the best reference energies from your data

3. **Starting from pretrained**: If you don't provide custom reference energies, the model starts with the pretrained values (e.g., ORCA-fitted for OMol models, VASP for OMAT models)

4. **Stress training**: Set `--stress_loss_weight 0.0` if your dataset doesn't include stress information

