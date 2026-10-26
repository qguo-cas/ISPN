"""Train FusionNet with an independent validation split."""

import argparse
import csv
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataload import FusionDataset
from engine import run_epoch, validate
from loss import FusionLoss
from model import FusionNet


def add_data_arguments(parser):
    parser.add_argument("--ms-channels", type=int, default=4, help="Use every input spectral channel.")
    parser.add_argument("--ms-dir", default="MS")
    parser.add_argument("--pan-dir", default="PAN")
    parser.add_argument("--upms-dir", default="UPMS")
    parser.add_argument("--prior-dir", default="PRIOR")
    parser.add_argument("--normalization", choices=("minmax", "scale", "none"), default="minmax")
    parser.add_argument("--max-value", type=float, default=1.0, help="Divisor for scale normalization.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--window-size", type=int, default=11)


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu.")
    return device


def make_dataset(root, args):
    return FusionDataset(root, ms_channels=args.ms_channels, ms_dir=args.ms_dir,
                         pan_dir=args.pan_dir, upms_dir=args.upms_dir, prior_dir=args.prior_dir,
                         normalization=args.normalization, max_value=args.max_value)


def make_loader(dataset, args, device, shuffle=False):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=device.type == "cuda")


def check_data_arguments(args):
    if args.ms_channels < 1 or args.batch_size < 1 or args.num_workers < 0 or args.window_size < 1:
        raise ValueError("Channels, batch size and window size must be positive; workers must be nonnegative.")
    if not math.isfinite(args.max_value) or args.max_value <= 0:
        raise ValueError("--max-value must be finite and positive.")


def check_independent_splits(train_set, val_set):
    if train_set.root == val_set.root:
        raise ValueError("Training and validation must use separate directories.")
    train_paths = {path.resolve() for sample in train_set.samples for path in sample}
    val_paths = {path.resolve() for sample in val_set.samples for path in sample}
    if train_paths & val_paths:
        raise ValueError("Training and validation reference the same image files.")


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, args):
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss, "config": vars(args)}, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default="data/train")
    parser.add_argument("--val-dir", default="data/val")
    parser.add_argument("--output-dir", default="runs/fusion")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    add_data_arguments(parser)
    args = parser.parse_args(argv)
    check_data_arguments(args)
    if args.epochs < 1 or args.save_every < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, save-every and lr must be positive and finite")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    train_set, val_set = make_dataset(args.train_dir, args), make_dataset(args.val_dir, args)
    check_independent_splits(train_set, val_set)
    train_loader = make_loader(train_set, args, device, shuffle=True)
    val_loader = make_loader(val_set, args, device)
    model = FusionNet(pan_channels=1, ms_channels=args.ms_channels).to(device)
    criterion = FusionLoss(args.window_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Avoid silently replacing a previous experiment's weights or logs.
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Choose a new --output-dir.")
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    best_val_loss = float("inf")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
            val_metrics = validate(model, val_loader, criterion, device)
            row = {"epoch": epoch,
                   **{f"train_{key}": value for key, value in train_metrics.items()},
                   **{f"val_{key}": value for key, value in val_metrics.items()}}
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(output_dir / "best.pth", model, optimizer, epoch, best_val_loss, args)
            save_checkpoint(output_dir / "last.pth", model, optimizer, epoch, best_val_loss, args)
            if epoch % args.save_every == 0:
                save_checkpoint(output_dir / f"epoch_{epoch:04d}.pth", model, optimizer,
                                epoch, best_val_loss, args)
            print(f"Epoch {epoch}/{args.epochs} | train={train_metrics['loss']:.6f} | "
                  f"val={val_metrics['loss']:.6f} | best={best_val_loss:.6f}", flush=True)
    print(f"Saved checkpoints and metrics to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
