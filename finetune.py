"""Finetuning loop with custom loss weights and reference energy control."""

import argparse
import logging
import os
from typing import Dict, Optional, Union

import torch
import tqdm
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

try:
    import wandb
except ImportError:
    raise ImportError(
        "wandb is not installed. Please install it with `pip install wandb`."
    )
from wandb import wandb_run

from orb_models import utils
from orb_models.dataset import augmentations
from orb_models.dataset.ase_sqlite_dataset import AseSqliteDataset
from orb_models.forcefield import atomic_system, base, pretrained, property_definitions
from orb_models.forcefield.conservative_regressor import ConservativeForcefieldRegressor
from orb_models.forcefield.direct_regressor import DirectForcefieldRegressor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def init_wandb_from_config(dataset: str, job_type: str, entity: str) -> wandb_run.Run:
    """Initialise wandb."""
    wandb.init(  # type: ignore
        job_type=job_type,
        dir=os.path.join(os.getcwd(), "wandb"),
        name=f"{dataset}-{job_type}",
        project="orb-experiment",
        entity=entity,
        mode="online",
        sync_tensorboard=False,
    )
    assert wandb.run is not None
    return wandb.run


def finetune(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    lr_scheduler: Optional[_LRScheduler] = None,
    num_steps: Optional[int] = None,
    clip_grad: Optional[float] = None,
    log_freq: float = 10,
    device: torch.device = torch.device("cpu"),
    epoch: int = 0,
):
    """Train for a fixed number of steps.

    Args:
        model: The model to optimize.
        optimizer: The optimizer for the model.
        dataloader: A Pytorch Dataloader, which may be infinite if num_steps is passed.
        lr_scheduler: Optional, a Learning rate scheduler for modifying the learning rate.
        num_steps: The number of training steps to take. This is required for distributed training,
            because controlling parallism is easier if all processes take exactly the same number of steps (
            this particularly applies when using dynamic batching).
        clip_grad: Optional, the gradient clipping threshold.
        log_freq: The logging frequency for step metrics.
        device: The device to use for training.
        epoch: The number of epochs the model has been fintuned.

    Returns
        A dictionary of metrics.
    """
    run: Optional[wandb_run.Run] = wandb.run

    if clip_grad is not None:
        hook_handles = utils.gradient_clipping(model, clip_grad)

    metrics = utils.ScalarMetricTracker()

    # Set the model to "train" mode.
    model.train()

    # Get tqdm for the training batches
    batch_generator = iter(dataloader)
    num_training_batches: Union[int, float]
    if num_steps is not None:
        num_training_batches = num_steps
    else:
        try:
            num_training_batches = len(dataloader)
        except TypeError:
            raise ValueError("Dataloader has no length, you must specify num_steps.")

    batch_generator_tqdm = tqdm.tqdm(batch_generator, total=num_training_batches)

    i = 0
    batch_iterator = iter(batch_generator_tqdm)
    while True:
        if num_steps and i == num_steps:
            break

        optimizer.zero_grad(set_to_none=True)

        step_metrics = {
            "batch_size": 0.0,
            "batch_num_edges": 0.0,
            "batch_num_nodes": 0.0,
        }

        # Reset metrics so that it reports raw values for each step but still do averages on
        # the gradient accumulation.
        if i % log_freq == 0:
            metrics.reset()

        batch = next(batch_iterator)
        batch = batch.to(device)
        step_metrics["batch_size"] += len(batch.n_node)
        step_metrics["batch_num_edges"] += batch.n_edge.sum()
        step_metrics["batch_num_nodes"] += batch.n_node.sum()

        with torch.autocast("cuda", enabled=False):
            batch_outputs = model.loss(batch)
            loss = batch_outputs.loss
            metrics.update(batch_outputs.log)
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")
        loss.backward()

        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metrics.update(step_metrics)

        if i != 0 and i % log_freq == 0:
            metrics_dict = metrics.get_metrics()
            if run is not None:
                step = (epoch * num_training_batches) + i
                if run.sweep_id is not None:
                    run.log(
                        {"loss": metrics_dict["loss"]},
                        commit=False,
                    )
                run.log(
                    {"step": step},
                    commit=False,
                )
                run.log(utils.prefix_keys(metrics_dict, "finetune_step"), commit=True)

        # Finished a single full step!
        i += 1

    if clip_grad is not None:
        for h in hook_handles:
            h.remove()

    return metrics.get_metrics()


def build_train_loader(
    dataset_name: str,
    dataset_path: str,
    num_workers: int,
    batch_size: int,
    system_config: atomic_system.SystemConfig,
    augmentation: Optional[bool] = True,
    target_config: Optional[Dict] = None,
    **kwargs,
) -> DataLoader:
    """Builds the train dataloader from a config file.

    Args:
        dataset_name: The name of the dataset.
        dataset_path: Dataset path.
        num_workers: The number of workers for each dataset.
        batch_size: The batch_size config for each dataset.
        augmentation: If rotation augmentation is used.
        target_config: The target config.
        system_config: The system config.

    Returns:
        The train Dataloader.
    """
    log_train = "Loading train datasets:\n"
    aug = []
    if augmentation:
        aug = [augmentations.rotate_randomly]

    target_config = property_definitions.instantiate_property_config(target_config)
    dataset = AseSqliteDataset(
        dataset_name,
        dataset_path,
        system_config=system_config,
        target_config=target_config,
        augmentations=aug,
        **kwargs,
    )

    log_train += f"Total train dataset size: {len(dataset)} samples"
    logging.info(log_train)

    sampler = RandomSampler(dataset)

    batch_sampler = BatchSampler(
        sampler,
        batch_size=batch_size,
        drop_last=False,
    )

    train_loader: DataLoader = DataLoader(
        dataset,
        num_workers=num_workers,
        worker_init_fn=utils.worker_init_fn,
        collate_fn=base.batch_graphs,
        batch_sampler=batch_sampler,
        timeout=10 * 60 if num_workers > 0 else 0,
    )
    return train_loader


def load_custom_reference_energies(filepath: str) -> torch.Tensor:
    """
    Load custom reference energies from a file.
    
    Supports two formats:
    1. JSON: {"1": -13.6, "6": -1030.5, ...} or {"H": -13.6, "C": -1030.5, ...}
    2. Text: One line per element: "element_number energy" or "element_symbol energy"
    
    Args:
        filepath: Path to the reference energies file
        
    Returns:
        Tensor of shape [118] with reference energies
    """
    import json
    
    # Element symbol to atomic number mapping
    ELEMENT_SYMBOLS = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
        'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
        'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
        'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
        'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
        'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
        'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
        'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
        'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
        'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
        'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
        'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118,
    }
    
    ref_energies = torch.zeros(118)
    
    # Try to load as JSON first
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for key, value in data.items():
            # Try as atomic number first
            try:
                z = int(key)
                if 1 <= z <= 118:
                    ref_energies[z] = float(value)
            except ValueError:
                # Try as element symbol
                if key in ELEMENT_SYMBOLS:
                    z = ELEMENT_SYMBOLS[key]
                    ref_energies[z] = float(value)
                else:
                    logging.warning(f"Unknown element symbol or invalid atomic number: {key}")
        
        logging.info(f"Loaded reference energies from JSON file: {filepath}")
        
    except json.JSONDecodeError:
        # Try as text file format
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) != 2:
                    logging.warning(f"Skipping invalid line: {line}")
                    continue
                
                element, energy = parts
                try:
                    # Try as atomic number
                    z = int(element)
                    if 1 <= z <= 118:
                        ref_energies[z] = float(energy)
                except ValueError:
                    # Try as element symbol
                    if element in ELEMENT_SYMBOLS:
                        z = ELEMENT_SYMBOLS[element]
                        ref_energies[z] = float(energy)
                    else:
                        logging.warning(f"Unknown element: {element}")
        
        logging.info(f"Loaded reference energies from text file: {filepath}")
    
    return ref_energies


def configure_model(model, args, device):
    """Configure model with custom loss weights and reference energies."""
    is_conservative = isinstance(model, ConservativeForcefieldRegressor)
    is_direct = isinstance(model, DirectForcefieldRegressor)
    
    logging.info(f"Model type: {type(model).__name__}")
    logging.info(f"  Conservative: {is_conservative}")
    logging.info(f"  Direct: {is_direct}")
    
    # Configure loss weights
    if args.energy_loss_weight is not None or args.forces_loss_weight is not None or args.stress_loss_weight is not None:
        logging.info("=" * 60)
        logging.info("Configuring custom loss weights...")
        
        # Set energy weight
        if args.energy_loss_weight is not None:
            model.loss_weights["energy"] = args.energy_loss_weight
            logging.info(f"  energy: {args.energy_loss_weight}")
        
        # Set forces weight (key depends on model type)
        if args.forces_loss_weight is not None:
            if is_conservative:
                model.loss_weights["grad_forces"] = args.forces_loss_weight
                logging.info(f"  grad_forces: {args.forces_loss_weight}")
            elif is_direct:
                model.loss_weights["forces"] = args.forces_loss_weight
                logging.info(f"  forces: {args.forces_loss_weight}")
        
        # Set stress weight (key depends on model type)
        if args.stress_loss_weight is not None:
            if is_conservative:
                model.loss_weights["grad_stress"] = args.stress_loss_weight
                logging.info(f"  grad_stress: {args.stress_loss_weight}")
            elif is_direct and "stress" in model.heads:
                model.loss_weights["stress"] = args.stress_loss_weight
                logging.info(f"  stress: {args.stress_loss_weight}")
        
        logging.info(f"Final loss_weights: {model.loss_weights}")
        logging.info("=" * 60)
    
    # Configure reference energies
    if args.custom_reference_energies:
        logging.info("=" * 60)
        logging.info(f"Loading custom reference energies from: {args.custom_reference_energies}")
        custom_refs = load_custom_reference_energies(args.custom_reference_energies)
        custom_refs = custom_refs.to(device)
        
        # Set the custom reference energies
        model.heads["energy"].reference.linear.weight.data = custom_refs.unsqueeze(0)
        
        # Log some values for verification
        logging.info("Custom reference energies set:")
        for z in [1, 6, 7, 8]:  # H, C, N, O
            val = custom_refs[z].item()
            if val != 0:
                logging.info(f"  Element {z}: {val:.4f} eV")
        
        # Make trainable if requested
        if args.trainable_reference_energies:
            logging.info("Making custom reference energies trainable...")
            model.heads["energy"].reference.linear.weight.requires_grad = True
        else:
            logging.info("Custom reference energies are FIXED (not trainable)")
            model.heads["energy"].reference.linear.weight.requires_grad = False
        logging.info("=" * 60)
    
    elif args.trainable_reference_energies:
        logging.info("=" * 60)
        logging.info("Making reference energies trainable (starting from pretrained values)...")
        model.heads['energy'].reference.linear.weight.requires_grad = True
        ref_params = model.heads['energy'].reference.linear.weight.numel()
        logging.info(f"  Added {ref_params} trainable reference energy parameters")
        
        # Show some example values
        ref_weights = model.heads['energy'].reference.linear.weight.data.squeeze()
        logging.info(f"  Current H (element 1): {ref_weights[1].item():.2f}")
        logging.info(f"  Current C (element 6): {ref_weights[6].item():.2f}")
        logging.info(f"  Current N (element 7): {ref_weights[7].item():.2f}")
        logging.info(f"  Current O (element 8): {ref_weights[8].item():.2f}")
        logging.info("=" * 60)
    
    return model


def run(args):
    """Training Loop.

    Args:
        config (DictConfig): Config for training loop.
    """
    device = utils.init_device(device_id=args.device_id)
    utils.seed_everything(args.random_seed)

    # Setting this is 2x faster on A100 and H100
    # GPUs and does not appear to hurt training
    precision = "float32-high"

    # Instantiate model
    base_model = args.base_model
    model = getattr(pretrained, base_model)(
        device=device, precision=precision, train=True
    )
    
    # Configure model with custom settings
    model = configure_model(model, args, device)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {model_params:,} trainable parameters.")

    # Move model to correct device.
    model.to(device=device)
    total_steps = args.max_epochs * args.num_steps
    optimizer, lr_scheduler = utils.get_optim(args.lr, total_steps, model)

    wandb_run = None
    # Logger instantiation/configuration
    if args.wandb:
        logging.info("Instantiating WandbLogger.")
        wandb_run = init_wandb_from_config(
            dataset=args.dataset, job_type="finetuning", entity=args.wandb_entity
        )

        wandb.define_metric("step")
        wandb.define_metric("finetune_step/*", step_metric="step")

    graph_targets = ["energy", "stress"] if model.has_stress else ["energy"]
    loader_args = dict(
        dataset_name=args.dataset,
        dataset_path=args.data_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        target_config={"graph": graph_targets, "node": ["forces"]},
    )
    train_loader = build_train_loader(
        **loader_args,
        system_config=model.system_config,
        augmentation=True,
    )
    logging.info("Starting training!")

    num_steps = args.num_steps

    start_epoch = 0

    for epoch in range(start_epoch, args.max_epochs):
        print(f"Start epoch: {epoch} training...")
        finetune(
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            lr_scheduler=lr_scheduler,
            clip_grad=args.gradient_clip_val,
            device=device,
            num_steps=num_steps,
            epoch=epoch,
        )

        # Save every 5 epochs and final epoch
        if (epoch % args.save_every_x_epochs == 0) or (epoch == args.max_epochs - 1):
            # create ckpts folder if it does not exist
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            torch.save(
                model.state_dict(),
                os.path.join(args.checkpoint_path, f"checkpoint_epoch{epoch}.ckpt"),
            )
            logging.info(f"Checkpoint saved to {args.checkpoint_path}")

    if wandb_run is not None:
        wandb_run.finish()


def main():
    """Main."""
    parser = argparse.ArgumentParser(
        description="Finetune orb model with custom loss weights and reference energy control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--random_seed", default=1234, type=int, help="Random seed for finetuning."
    )
    parser.add_argument(
        "--device_id", default=0, type=int, help="GPU index to use if GPU is available."
    )
    parser.add_argument(
        "--wandb",
        default=True,
        action="store_true",
        help="If the run is logged to Weights and Biases (requires installation).",
    )
    parser.add_argument(
        "--wandb_entity",
        default="orbitalmaterials",
        type=str,
        help="Entity to log the run to in Weights and Biases.",
    )
    parser.add_argument(
        "--dataset",
        default="mp-traj",
        type=str,
        help="Dataset name for wandb run logging.",
    )
    parser.add_argument(
        "--data_path",
        default=os.path.join(os.getcwd(), "datasets/mptraj/finetune.db"),
        type=str,
        help="Dataset path to an ASE sqlite database (you must convert your data into this format).",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of cpu workers for the pytorch data loader.",
    )
    parser.add_argument(
        "--batch_size", default=100, type=int, help="Batch size for finetuning."
    )
    parser.add_argument(
        "--gradient_clip_val", default=0.5, type=float, help="Gradient clip value."
    )
    parser.add_argument(
        "--max_epochs",
        default=50,
        type=int,
        help="Maximum number of epochs to finetune.",
    )
    parser.add_argument(
        "--save_every_x_epochs",
        default=5,
        type=int,
        help="Save model every x epochs.",
    )
    parser.add_argument(
        "--num_steps",
        default=100,
        type=int,
        help="Num steps of in each epoch.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=os.path.join(os.getcwd(), "ckpts"),
        type=str,
        help="Path to save the model checkpoint.",
    )
    parser.add_argument(
        "--lr",
        default=3e-4,
        type=float,
        help="Learning rate. 3e-4 is purely a sensible default; you may want to tune this for your problem.",
    )
    parser.add_argument(
        "--base_model",
        default="orb_v3_conservative_inf_omat",
        type=str,
        help="Base model to finetune.",
        choices=[
            "orb_v3_conservative_inf_omat",
            "orb_v3_conservative_20_omat",
            "orb_v3_direct_inf_omat",
            "orb_v3_direct_20_omat",
            "orb_v3_conservative_omol",
            "orb_v3_direct_omol",
            "orb_v2",
        ],
    )
    
    # Loss weight arguments
    parser.add_argument(
        "--energy_loss_weight",
        default=None,
        type=float,
        help="Weight for energy loss. If not specified, uses model default (usually 1.0).",
    )
    parser.add_argument(
        "--forces_loss_weight",
        default=None,
        type=float,
        help="Weight for forces loss. Automatically uses 'forces' or 'grad_forces' depending on model type. If not specified, uses model default (usually 1.0).",
    )
    parser.add_argument(
        "--stress_loss_weight",
        default=None,
        type=float,
        help="Weight for stress loss. Automatically uses 'stress' or 'grad_stress' depending on model type. Set to 0 to disable stress training. If not specified, uses model default.",
    )
    
    # Reference energy arguments
    parser.add_argument(
        "--trainable_reference_energies",
        action="store_true",
        help="Make reference energies trainable. They will be optimized during finetuning to match your dataset's reference energy scheme.",
    )
    parser.add_argument(
        "--custom_reference_energies",
        default=None,
        type=str,
        help="Path to file with custom reference energies. Supports JSON format {'H': -13.6, 'C': -1030.5, ...} or text format 'H -13.6\\nC -1030.5\\n...'. Use with --trainable_reference_energies to make them trainable, or leave that flag off to keep them fixed.",
    )
    
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

