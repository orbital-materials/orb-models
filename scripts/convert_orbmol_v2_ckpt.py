"""Convert a wandb-format orbmol-v2 checkpoint to a flat state_dict for orb-models.

The wandb checkpoint contains keys [step, state_dict, optimizer, lr_scheduler, ema].
This script applies the EMA shadow_params to state_dict and saves only the flat
state_dict, which is what orb_models.forcefield.pretrained.orbmol_v2() expects.

Usage:
    python scripts/convert_orbmol_v2_ckpt.py /path/to/wandb_model.ckpt /path/to/output.ckpt
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Wandb-format checkpoint (.ckpt with state_dict + ema keys)")
    parser.add_argument("output", help="Output flat state_dict file")
    args = parser.parse_args()

    ck = torch.load(args.input, map_location="cpu", weights_only=False)
    sd = dict(ck["state_dict"])

    if "ema" in ck and "shadow_params" in ck["ema"]:
        for param_name, shadow_param in ck["ema"]["shadow_params"].items():
            sd[param_name] = shadow_param.data.clone()
        print(f"applied EMA: {len(ck['ema']['shadow_params'])} shadow params")
    else:
        print("no EMA in checkpoint; using raw state_dict")

    torch.save(sd, args.output)
    print(f"wrote {len(sd)} keys to {args.output}")


if __name__ == "__main__":
    main()
