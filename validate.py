"""Evaluate a saved FusionNet checkpoint on an independent dataset."""

import argparse
import json

import torch

from engine import validate
from loss import FusionLoss
from model import FusionNet
from train import add_data_arguments, check_data_arguments, make_dataset, make_loader, select_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-dir", default="data/val")
    add_data_arguments(parser)
    # Reuse checkpoint data/loss settings unless explicitly overridden.
    settings = ("ms_channels", "ms_dir", "pan_dir", "upms_dir", "prior_dir",
                "normalization", "max_value", "window_size")
    parser.set_defaults(**{key: None for key in settings})
    args = parser.parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    defaults = {"ms_channels": 4, "ms_dir": "MS", "pan_dir": "PAN", "upms_dir": "UPMS",
                "prior_dir": "PRIOR", "normalization": "minmax", "max_value": 1.0, "window_size": 11}
    for key in settings:
        if getattr(args, key) is None:
            setattr(args, key, config.get(key, defaults[key]))
    check_data_arguments(args)
    device = select_device(args.device)
    dataset = make_dataset(args.val_dir, args)
    loader = make_loader(dataset, args, device)
    model = FusionNet(pan_channels=1, ms_channels=args.ms_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = validate(model, loader, FusionLoss(args.window_size).to(device), device)
    print(json.dumps({"samples": len(dataset), **metrics}, indent=2))


if __name__ == "__main__":
    main()
