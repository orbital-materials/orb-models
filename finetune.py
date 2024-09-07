"""Core training loop."""

import os
import logging
import argparse
import time

import torch
from orb_models.forcefield import pretrained
from orb_models.finetune_utilities import experiment, optim, checkpointer
from orb_models.dataset import data_loaders
from orb_models.finetune_utilities import steps
from orb_models import utils


def run(args):
    """Training Loop.

    Args:
        config (DictConfig): Config for training loop.
    """
    device = utils.init_device()
    experiment.seed_everything(args.random_seed, utils.get_local_rank())

    # Make sure to use this flag for matmuls on A100 and H100 GPUs.
    torch.set_float32_matmul_precision("high")

    # Instantiate model
    model = pretrained.orb_v1(device=device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {model_params} trainable parameters.")

    # Initialize and construct harness for training.
    optim_config: experiment.OptimConfig = args.optim  # type: ignore
    experiment.initialize_model(optim_config, model)

    # Move model to correct device.
    model.to(device=device)
    optimizer, lr_scheduler, ema = optim.get_optim(lr=args.learning_rate)

    wandb_run = None
    # Logger instantiation/configuration
    if args.wandb:
        import wandb

        logging.info("Instantiating WandbLogger.")
        wandb_run = experiment.init_wandb_from_config(args, job_type="finetuning")

        wandb.define_metric("global_step")
        wandb.define_metric("epochs")
        wandb.define_metric("train_step/*", step_metric="global_step")
        wandb.define_metric("learning_rates/*", step_metric="global_step")
        wandb.define_metric("finetune/*", step_metric="epochs")
        wandb.define_metric("key-metrics/*", step_metric="epochs")

    loader_args = dict(
        dataset=args.dataset,
        num_workers=args.num_workers,
        batch_size_dict=args.batch_size,
    )
    train_loader = data_loaders.build_train_loader(
        **loader_args,
        augmentation=getattr(args, "augmentation", True),
    )

    lr_must_reduce_per_epoch = isinstance(
        lr_scheduler,
        (
            torch.optim.lr_scheduler.ReduceLROnPlateau,
            torch.optim.lr_scheduler.StepLR,
            torch.optim.lr_scheduler.ExponentialLR,
            torch.optim.lr_scheduler.CosineAnnealingLR,
        ),
    )
    logging.info("Starting training!")

    num_steps = len(train_loader)

    start_epoch = 0

    for epoch in range(start_epoch, args.max_epochs):
        t1 = time.time()
        avg_train_metrics = steps.fintune(
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            ema=ema,
            lr_scheduler=None if lr_must_reduce_per_epoch else lr_scheduler,
            clip_grad=args.gradient_clip_val,
            device=device,
            log_freq=args.get("log_freq", 100),
            num_steps=num_steps,
            epoch=epoch,
        )
        t2 = time.time()
        train_times = {}
        train_times["avg_time_per_step"] = (t2 - t1) / num_steps
        train_times["total_time"] = t2 - t1

        if wandb.run is not None:
            wandb.run.log(
                experiment.prefix_keys(avg_train_metrics, "train"), commit=False
            )
            wandb.run.log(
                experiment.prefix_keys(train_times, "train", sep="-"),
                commit=False,
            )
            wandb.run.log({"epoch": epoch}, commit=True)

        if epoch == args.max_epoch - 1:
            model_checkpointer = checkpointer.from_config(
                args.checkpoint_dir, args.checkpoint_path
            )
            model_checkpointer.checkpoint(
                model,
                num_steps * epoch,
                optimizer,
                lr_scheduler,
                ema,
            )

    if wandb_run is not None:
        wandb_run.finish()


def main():
    """Main."""
    parser = argparse.ArgumentParser(
        description="Finetune orb model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--random_seed", default=1234, type=int, help="Random seed for finetuning."
    )
    parser.add_argument(
        "--wandb",
        default=False,
        action="store_true",
        help="If the run is logged to wandb.",
    )
    parser.add_argument("--dataset", default="QM9", type=str, help="Dataset name.")
    parser.add_argument(
        "--num_workers", default=8, type=int, help="Number of workers for data loader."
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
        "--checkpoint_dir",
        default="checkpoints",
        type=str,
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=os.getcwd(),
        type=str,
        help="Path to save the model checkpoint.",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
