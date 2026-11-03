"""Differentiable, device-independent local universal image quality loss."""

import torch
from torch import nn
from torch.nn import functional as F


class SSIMLoss(nn.Module):
    def forward(self, reference, prediction, ws=11):
        if reference.shape != prediction.shape or reference.ndim != 4:
            raise ValueError("UIQI inputs must have identical BCHW shapes.")
        if ws < 1 or min(reference.shape[-2:]) < ws:
            raise ValueError(f"UIQI window {ws} exceeds image dimensions {reference.shape[-2:]}.")
        mean_x = F.avg_pool2d(reference, ws, stride=1)
        mean_y = F.avg_pool2d(prediction, ws, stride=1)
        var_x = (F.avg_pool2d(reference.square(), ws, stride=1) - mean_x.square()).clamp_min(0)
        var_y = (F.avg_pool2d(prediction.square(), ws, stride=1) - mean_y.square()).clamp_min(0)
        covariance = F.avg_pool2d(reference * prediction, ws, stride=1) - mean_x * mean_y
        eps = torch.finfo(reference.dtype).eps
        mean_power = mean_x.square() + mean_y.square()
        variance = var_x + var_y
        luminance = torch.where(mean_power > eps, 2 * mean_x * mean_y / mean_power.clamp_min(eps),
                                torch.ones_like(mean_power))
        structure = torch.where(variance > eps, 2 * covariance / variance.clamp_min(eps),
                                torch.ones_like(variance))
        return 1 - (luminance * structure).mean()
