"""
Shared internal helpers used by more than one node-category module.

Not a node file: nothing here is registered in NODE_CLASS_MAPPINGS.
"""

import logging
import math
import torch
import torch.nn.functional as F

# Console policy for this pack: no print(). Information for the user goes to
# node outputs and/or node_text(); the console (logging) is for real problems
# (warning) and developer diagnostics (debug/info).
log = logging.getLogger("comfyui-moon")

# Add to a node's INPUT_TYPES as "hidden": HIDDEN_UNIQUE_ID, accept a
# `unique_id=None` keyword in the node function, and pass it to node_text().
HIDDEN_UNIQUE_ID = {"unique_id": "UNIQUE_ID"}


def node_text(unique_id, text):
    """Shows `text` on the node in the UI, through the same channel core nodes
    use for this (PromptServer.send_progress_text, e.g. Get Image Size).

    Best effort by design: never raises, so a change in ComfyUI's server API
    only removes the text, never breaks the node. The text is not saved with
    the workflow and is not re-sent when the node's output comes from cache;
    anything the graph needs must also be a real output."""
    if unique_id is None:
        return
    try:
        from server import PromptServer
        PromptServer.instance.send_progress_text(str(text), str(unique_id))
    except Exception as e:  # noqa: BLE001 -- display is optional
        log.debug("node text unavailable: %s", e)

MAX_QUANTILE_ELEMENTS = 16_000_000

def quantile_safe(flat, q, max_elements=MAX_QUANTILE_ELEMENTS):
    """flat: (N,) or (B, N) float tensor; q: 1D tensor of quantiles in [0, 1].
    Subsamples the last dim if it exceeds torch.quantile's element limit.
    Returns shape (len(q),) or (len(q), B)."""
    n = flat.shape[-1]
    if n > max_elements:
        g = torch.Generator(device=flat.device)
        g.manual_seed(0)
        idx = torch.randint(0, n, (max_elements,), device=flat.device, generator=g)
        flat = flat[..., idx]
    return torch.quantile(flat, q, dim=-1)


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
