"""Shared training and validation objective; no spectral band selection."""

from torch import nn
from torch.nn import functional as F

from .spatial_loss import UIQILoss
from .spectral_loss import SAMLoss


class FusionLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.spatial = SSIMLoss()
        self.spectral = SAMLoss()
        self.window_size = window_size

    def forward(self, outputs, pan, upms):
        fused, reconstructed_pan, reconstructed_ms, gray_ms = outputs
        spatial = self.spatial(pan, reconstructed_pan, self.window_size)
        spectral = self.spectral(reconstructed_ms, upms)
        residual_ms = upms - gray_ms
        residual_fused = fused - pan.expand_as(fused)
        # Preserve width-axis distributions and KL direction, averaging per
        # image so validation does not depend on batch partitioning.
        kl_map = F.kl_div(F.log_softmax(residual_ms, dim=-1),
                          F.softmax(residual_fused, dim=-1), reduction="none")
        prior = kl_map.flatten(1).sum(dim=1).mean()
        # The old objective applied 0.1 to KL twice, giving an effective 0.01.
        total = 0.1 * spatial + 0.8 * spectral + 0.1 * prior
        return {"loss": total, "spatial": spatial, "spectral": spectral, "prior": prior}
