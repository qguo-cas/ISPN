"""Epoch execution shared by training and standalone validation."""

import torch


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {}
    sample_count = 0
    with torch.set_grad_enabled(training):
        for _, pan, upms, prior in loader:
            pan, upms, prior = (tensor.to(device, non_blocking=True)
                                for tensor in (pan, upms, prior))
            if training:
                optimizer.zero_grad(set_to_none=True)
            values = criterion(model(pan, upms, prior), pan, upms)
            if not all(torch.isfinite(value).all().item() for value in values.values()):
                raise FloatingPointError("Non-finite loss; check input data and normalization.")
            if training:
                values["loss"].backward()
                optimizer.step()
            batch_size = pan.shape[0]
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value.detach().item() * batch_size
            sample_count += batch_size
    if sample_count == 0:
        raise ValueError("Cannot run an epoch on an empty dataset.")
    return {name: value / sample_count for name, value in totals.items()}


def validate(model, loader, criterion, device):
    """Evaluate without gradients or BatchNorm updates; restore caller mode."""
    was_training = model.training
    try:
        return run_epoch(model, loader, criterion, device)
    finally:
        model.train(was_training)
