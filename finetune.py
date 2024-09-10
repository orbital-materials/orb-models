"""Finetuning loop."""

import argparse
import logging
import os
from typing import Optional, Union, cast

import torch
import tqdm
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

import wandb
from orb_models.dataset import data_loaders
from orb_models import utils
from orb_models.forcefield import pretrained
from wandb import wandb_run

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
    run: Optional[wandb_run.Run] = cast(Optional[wandb_run.Run], wandb.run)

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

        with torch.cuda.amp.autocast(enabled=False):
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
                global_step = (epoch * num_training_batches) + i
                if run.sweep_id is not None:
                    run.log(
                        {"loss": metrics_dict["loss"]},
                        commit=False,
                    )
                run.log(
                    {"global_step": global_step},
                    commit=False,
                )
                run.log(utils.prefix_keys(metrics_dict, "train_step"), commit=False)
                # Log learning rates.
                run.log(
                    {
                        f"pg_{idx}": group["lr"]
                        for idx, group in enumerate(optimizer.param_groups)
                    },
                )

        # Finished a single full step!
        i += 1

    if clip_grad is not None:
        for h in hook_handles:
            h.remove()

    return metrics.get_metrics()


def run(args):
    """Training Loop.

    Args:
        config (DictConfig): Config for training loop.
    """
    device = utils.init_device()
    utils.seed_everything(args.random_seed)

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
    total_steps = args.max_epochs * args.num_steps
    optimizer, lr_scheduler = utils.get_optim(args.lr, total_steps, model)

    wandb_run = None
    # Logger instantiation/configuration
    if args.wandb:
        import wandb

        logging.info("Instantiating WandbLogger.")
        wandb_run = utils.init_wandb_from_config(job_type="finetuning")

        wandb.define_metric("global_step")
        wandb.define_metric("epochs")
        wandb.define_metric("train_step/*", step_metric="global_step")
        wandb.define_metric("learning_rates/*", step_metric="global_step")
        wandb.define_metric("finetune/*", step_metric="epochs")

    loader_args = dict(
        dataset=args.dataset,
        path=args.data_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        target_config={"graph": ["energy", "stress"], "node": ["forces"]},
    )
    train_loader = data_loaders.build_train_loader(
        **loader_args,
        augmentation=True,
    )
    logging.info("Starting training!")

    num_steps = args.num_steps

    start_epoch = 0

    for epoch in range(start_epoch, args.max_epochs):
        print(f"Start epoch: {epoch} training...")
        avg_train_metrics = finetune(
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            lr_scheduler=lr_scheduler,
            clip_grad=args.gradient_clip_val,
            device=device,
            num_steps=num_steps,
            epoch=epoch,
        )

        if args.wandb:
            wandb.run.log(
                utils.prefix_keys(avg_train_metrics, "finetune"), commit=False
            )
            wandb.run.log({"epoch": epoch}, commit=True)

        # Save checkpoint from last epoch
        if epoch == args.max_epochs - 1:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": (
                    lr_scheduler.state_dict() if lr_scheduler else None
                ),
            }
            torch.save(
                checkpoint,
                os.path.join(args.checkpoint_path, f"checkpoint_epoch{epoch}.ckpt"),
            )
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
        default=True,
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
        "--batch_size", default=10, type=int, help="Batch size for finetuning."
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
        "--num_steps",
        default=100,
        type=int,
        help="Num steps of in each epoch.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=os.getcwd(),
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
