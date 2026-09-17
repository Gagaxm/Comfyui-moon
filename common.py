"""
Shared internal helpers used by more than one node-category module.

Not a node file: nothing here is registered in NODE_CLASS_MAPPINGS.
"""

import math
import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma, device, dtype):
    radius = max(1, int(math.ceil(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel, radius


def gaussian_blur(x, sigma, wrap_mode="replicate"):
    """Separable gaussian blur, multi-channel (groups=C), respects wrap_mode
    so it stays consistent with each node's circular tiling support."""
    if sigma <= 0:
        return x
    device, dtype = x.device, x.dtype
    kernel1d, radius = gaussian_kernel1d(sigma, device, dtype)
    C = x.shape[1]
    kx = kernel1d.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    ky = kernel1d.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    pad_mode = "circular" if wrap_mode == "circular" else "replicate"

    x = F.pad(x, (radius, radius, 0, 0), mode=pad_mode)
    x = F.conv2d(x, kx, groups=C)
    x = F.pad(x, (0, 0, radius, radius), mode=pad_mode)
    x = F.conv2d(x, ky, groups=C)
    return x
