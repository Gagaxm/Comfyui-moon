"""
Test: does sharing GroupNorm statistics across VAE decode tiles remove the
seam artifact (Problem A) seen with vae.enable_tiling()?

Standalone script, independent of ComfyUI internals -- only needs the
PBRFusion4 .safetensors file (only the vae/ prefixed weights are actually
loaded; unet/text_encoder are skipped to save time and VRAM, since Problem A
is purely a VAE decode artifact, confirmed independent of the UNet output).

Two input modes:

  1. --latent path/to/latent.pt
     A saved PBRFusion4 UNet output latent (e.g. dumped from
     MoonPBRFusion4Depth.infer() right before the VAE decode step, via
     `torch.save(pred_latents.cpu(), "latent.pt")`). Closest to the real
     artifact since it's an actual model prediction.

  2. --image path/to/image.png
     Encodes the image with the non-tiled VAE encoder to get a latent, then
     runs the same comparison. Faster sanity check that doesn't need a UNet
     pass, but the latent statistics won't exactly match a real PBRFusion4
     prediction.

For the chosen latent, three decodes are produced and compared:

  - baseline          : single-shot decode (vae.decode with tiling off)
  - tiled              : tiled decode via vae.tiled_decode(), independent
                          per-tile GroupNorm (today's behavior -- expected
                          to show the seam)
  - tiled_shared_gn    : tiled decode with GroupNorm statistics estimated
                          once (on the full latent, or a downsampled version
                          with distribution-shift correction if it doesn't
                          fit) and shared across every tile

Outputs: PNG dumps of all three decodes plus a diff heatmap, and printed
max/mean abs diff of tiled vs baseline and tiled_shared_gn vs baseline, so
the seam's presence/absence can be judged both visually and numerically.

Usage:
    python test_shared_groupnorm_tiled_vae.py \
        --safetensors D:/path/to/PBRFusion4.safetensors \
        --image D:/path/to/galets_composite.png \
        --tile-size 512 --tile-overlap 0.5 \
        --out ./gn_test_out

    python test_shared_groupnorm_tiled_vae.py \
        --safetensors D:/path/to/PBRFusion4.safetensors \
        --latent D:/path/to/pred_latents.pt \
        --estimate-max-side 512 \
        --out ./gn_test_out
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Minimal VAE-only loader (subset of MoonPBRFusion4Loader._load_pbrfusion4_components,
# stripped of ComfyUI dependencies and of unet/text_encoder loading).
# ---------------------------------------------------------------------------

def load_vae_only(safetensors_path, dtype, device):
    from safetensors.torch import load_file, safe_open
    from diffusers import AutoencoderKL

    with safe_open(safetensors_path, framework="pt") as f:
        metadata = f.metadata()

    all_weights = load_file(safetensors_path, device="cpu")
    vae_sd = {k[len("vae."):]: v for k, v in all_weights.items() if k.startswith("vae.")}
    del all_weights

    vae_config = json.loads(metadata["vae/config.json"])
    vae = AutoencoderKL(**{k: v for k, v in vae_config.items() if not k.startswith("_")})
    vae.load_state_dict(vae_sd)
    del vae_sd

    vae.to(dtype=dtype, device=device)
    vae.eval()
    return vae


# ---------------------------------------------------------------------------
# Shared GroupNorm statistics: the actual technique under test.
# ---------------------------------------------------------------------------

def _group_norm_stats(x, num_groups, eps):
    """
    Per-group (mean, var) of x, same reshape trick nn.GroupNorm uses internally.
    Works for any number of trailing spatial dims -- conv GroupNorm layers get
    (B, C, H, W), but attention blocks' group_norm gets a flattened (B, C, N).
    """
    b, c = x.shape[0], x.shape[1]
    channels_per_group = c // num_groups
    x_reshaped = x.contiguous().view(1, b * num_groups, channels_per_group, *x.shape[2:])
    reduce_dims = [d for d in range(x_reshaped.dim()) if d != 1]
    var, mean = torch.var_mean(x_reshaped, dim=reduce_dims, unbiased=False)
    return mean, var


def _apply_fixed_group_norm(x, num_groups, mean, var, weight, bias, eps):
    """GroupNorm with externally supplied (mean, var) instead of the input's own."""
    b, c = x.shape[0], x.shape[1]
    channels_per_group = c // num_groups
    x_reshaped = x.contiguous().view(1, b * num_groups, channels_per_group, *x.shape[2:])
    out = F.batch_norm(
        x_reshaped, mean.to(x), var.to(x), weight=None, bias=None,
        training=False, momentum=0.0, eps=eps,
    )
    out = out.view(b, c, *x.shape[2:])
    broadcast_shape = (1, -1) + (1,) * (x.dim() - 2)
    if weight is not None:
        out = out * weight.view(*broadcast_shape).to(x)
    if bias is not None:
        out = out + bias.view(*broadcast_shape).to(x)
    return out


class SharedGroupNormStats:
    """
    Forces every nn.GroupNorm submodule of `module` to use pre-recorded
    (mean, var) statistics instead of computing them from each call's local
    input. Works via forward hooks, so it's agnostic to the module's internal
    naming/structure -- no need to know diffusers' up_blocks/mid_block layout.

    Usage:
        stats = SharedGroupNormStats(vae.decoder)
        stats.record(lambda: vae.decode(some_full_latent, return_dict=False))
        with stats:
            tiled_output = vae.decode(z, return_dict=False)[0]  # tiling enabled
    """

    def __init__(self, module):
        self.module = module
        self.norm_layers = [m for m in module.modules() if isinstance(m, torch.nn.GroupNorm)]
        if not self.norm_layers:
            raise RuntimeError("No nn.GroupNorm layers found in the given module.")
        self._stats = {}
        self._override_handles = []

    def record(self, forward_fn):
        """Run forward_fn() once, capturing (mean, var) per GroupNorm layer."""
        self._stats.clear()

        def make_hook(layer):
            def hook(mod, inputs, output):
                mean, var = _group_norm_stats(inputs[0], mod.num_groups, mod.eps)
                self._stats[layer] = (mean.detach(), var.detach())
            return hook

        handles = [layer.register_forward_hook(make_hook(layer)) for layer in self.norm_layers]
        try:
            forward_fn()
        finally:
            for h in handles:
                h.remove()

        missing = [l for l in self.norm_layers if l not in self._stats]
        if missing:
            raise RuntimeError(
                f"{len(missing)} GroupNorm layers were never hit during record() -- "
                f"forward_fn likely doesn't exercise the full decoder path."
            )

    def __enter__(self):
        def make_hook(layer):
            def hook(mod, inputs, output):
                mean, var = self._stats[layer]
                return _apply_fixed_group_norm(inputs[0], mod.num_groups, mean, var, mod.weight, mod.bias, mod.eps)
            return hook

        self._override_handles = [layer.register_forward_hook(make_hook(layer)) for layer in self.norm_layers]
        return self

    def __exit__(self, *exc):
        for h in self._override_handles:
            h.remove()
        self._override_handles = []


def estimate_stats(vae, z, stats, max_side_for_estimate=None):
    """
    Record GroupNorm stats via a single non-tiled vae.decode(). If the full
    latent doesn't fit for a non-tiled pass, downsample it first (with the
    distribution-shift correction from Kahsolt's Tiled VAE technique) so the
    estimate isn't biased by the downsampling itself.
    """
    was_tiling = vae.use_tiling
    vae.disable_tiling()
    try:
        z_est = z
        if max_side_for_estimate is not None:
            scale = max_side_for_estimate / max(z.shape[-2], z.shape[-1])
            if scale < 1.0:
                down = F.interpolate(z, scale_factor=scale, mode="nearest-exact")
                std_old, mean_old = torch.std_mean(z, dim=[0, 2, 3], keepdim=True)
                std_new, mean_new = torch.std_mean(down, dim=[0, 2, 3], keepdim=True)
                z_est = (down - mean_new) / std_new * std_old + mean_old
                z_est = torch.clamp(z_est, min=z.min(), max=z.max())
        with torch.inference_mode():
            stats.record(lambda: vae.decode(z_est, return_dict=False))
    finally:
        if was_tiling:
            vae.enable_tiling()


# ---------------------------------------------------------------------------
# Comparison run
# ---------------------------------------------------------------------------

def decoded_to_pil(img):
    """[-1, 1] -> [0, 1] -> uint8 PIL image, matching PBRFusion4's own remap convention."""
    img = (img / 2.0 + 0.5).clamp(0, 1)
    img = (img[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(img)


def run_comparison(vae, z, out_dir, max_side_for_estimate=None):
    os.makedirs(out_dir, exist_ok=True)

    vae.disable_tiling()
    with torch.inference_mode():
        baseline = vae.decode(z, return_dict=False)[0]
    vae.enable_tiling()  # tiling stays on for the two tiled runs below

    with torch.inference_mode():
        tiled = vae.decode(z, return_dict=False)[0]

    stats = SharedGroupNormStats(vae.decoder)
    estimate_stats(vae, z, stats, max_side_for_estimate=max_side_for_estimate)
    with stats, torch.inference_mode():
        tiled_shared = vae.decode(z, return_dict=False)[0]

    for name, img in [("baseline", baseline), ("tiled", tiled), ("tiled_shared_gn", tiled_shared)]:
        decoded_to_pil(img).save(os.path.join(out_dir, f"{name}.png"))

    def report(name, img):
        diff = (img - baseline).abs()
        print(f"{name:18s}: max abs diff = {diff.max().item():.5f}   mean abs diff = {diff.mean().item():.6f}")
        return diff

    diff_tiled = report("tiled", tiled)
    report("tiled_shared_gn", tiled_shared)

    # Locate the row with the largest baseline-vs-tiled discrepancy -- should
    # line up with the visually observed seam row.
    row_scores = diff_tiled.mean(dim=1)[0].mean(dim=-1)
    worst_row = int(row_scores.argmax().item())
    print(f"Row with largest baseline-vs-tiled discrepancy: {worst_row} "
          f"(score={row_scores[worst_row].item():.5f})")

    diff_map = (tiled - baseline).abs().mean(dim=1, keepdim=True)
    diff_map = (diff_map / diff_map.max().clamp(min=1e-8)).clamp(0, 1)
    diff_img = (diff_map[0, 0].float().cpu().numpy() * 255).astype("uint8")
    Image.fromarray(diff_img).save(os.path.join(out_dir, "diff_tiled_vs_baseline.png"))

    print(f"\nOutputs written to: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_image_as_rgb_tensor(path, device, dtype):
    import numpy as np
    img = Image.open(path).convert("RGB")
    arr = torch.from_numpy(np.array(img))  # (H, W, 3) uint8
    rgb = arr.to(device=device, dtype=dtype).permute(2, 0, 1).unsqueeze(0) / 255.0
    rgb = rgb * 2.0 - 1.0
    return rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--safetensors", required=True, help="Path to PBRFusion4.safetensors")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--image", default=None, help="Input image; VAE-encodes it (non-tiled) to get a latent")
    p.add_argument("--latent", default=None, help="Saved latent tensor (.pt) to decode directly")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--tile-overlap", type=float, default=0.25)
    p.add_argument("--estimate-max-side", type=int, default=None,
                    help="If the full-res non-tiled decode doesn't fit in VRAM, cap the GroupNorm "
                         "estimation pass to this many pixels on the longer side")
    p.add_argument("--out", default="./gn_test_out")
    args = p.parse_args()

    if not args.image and not args.latent:
        raise SystemExit("Provide either --image or --latent")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    print("Loading VAE...")
    vae = load_vae_only(args.safetensors, dtype, device)
    vae.enable_slicing()
    vae.enable_tiling()
    vae.tile_overlap_factor = args.tile_overlap
    if hasattr(vae, "tile_sample_min_size"):
        vae.tile_sample_min_size = args.tile_size

    if args.latent:
        z = torch.load(args.latent, map_location=device).to(dtype=dtype, device=device)
    else:
        rgb = load_image_as_rgb_tensor(args.image, device, dtype)
        was_tiling = vae.use_tiling
        vae.disable_tiling()
        with torch.inference_mode():
            z = vae.encode(rgb).latent_dist.sample() * vae.config.scaling_factor
        if was_tiling:
            vae.enable_tiling()

    print(f"Latent shape: {tuple(z.shape)}")
    run_comparison(vae, z, args.out, max_side_for_estimate=args.estimate_max_side)


if __name__ == "__main__":
    main()