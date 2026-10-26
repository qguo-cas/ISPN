"""Spectral angle loss over all input channels."""

import math

from torch import nn


class SAMLoss(nn.Module):
    def forward(self, prediction, reference):
        inner = (prediction * reference).sum(dim=1)
        norm_prediction = (prediction.square().sum(dim=1) + 1e-8).sqrt()
        norm_reference = (reference.square().sum(dim=1) + 1e-8).sqrt()
        # Preserve the original nonnegative-cosine convention and 90-degree scaling.
        cosine = (inner / (norm_prediction * norm_reference + 1e-8)).clamp(0, 0.99999)
        return cosine.acos().mean() * (2 / math.pi)
