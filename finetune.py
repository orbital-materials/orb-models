"""Core training loop."""

import os
import logging
import argparse
import time

import torch
from orb_models.forcefield import pretrained
from orb_models.finetune_utilities import experiment, optim
from orb_models.dataset import data_loaders
from orb_models.finetune_utilities import steps
from orb_models import utils


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
    for param in model.parameters():
        param.requires_grad = True
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {model_params} trainable parameters.")

    # Move model to correct device.
    model.to(device=device)
    optimizer, lr_scheduler, ema = optim.get_optim(args.lr, model)

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
        path=args.data_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        target_config={"graph": ["energy", "stress"], "node": ["forces"]},
    )
    train_loader = data_loaders.build_train_loader(
        **loader_args,
        augmentation=getattr(args, "augmentation", True),
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
            lr_scheduler=lr_scheduler,
            clip_grad=args.gradient_clip_val,
            device=device,
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

        if epoch == args.max_epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                'ema_state_dict': ema.state_dict() if ema else None,
            }
            torch.save(checkpoint, args.checkpoint_path)
            logging.info(f"Checkpoint saved to {args.checkpoint_path}")

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
    parser.add_argument("--dataset", default="mp-traj", type=str, help="Dataset name.")
    parser.add_argument(
        "--data_path",
        default=os.path.join(os.getcwd(), "datasets/mptraj/finetune.db"),
        type=str,
        help="Dataset path.",
    )
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
        "--checkpoint_path",
        default=os.path.join(os.getcwd(), "checkpoints"),
        type=str,
        help="Path to save the model checkpoint.",
    )
    parser.add_argument(
        "--lr",
        default=3e-04,
        type=float,
        help="Learning rate",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
