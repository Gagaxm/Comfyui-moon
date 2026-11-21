"""
PBRFusion4 nodes for Comfyui-moon.

Loads and runs PBRFusion4 (Lotus-D discriminative depth model) directly via
diffusers, bypassing ComfyUI's own sampler/pipeline wrapping so the exact
operations confirmed against EnVision-Research/Lotus's LotusDPipeline can be
reproduced one-to-one (see MoonPBRFusion4Depth below).

VRAM lifecycle: the UNet and the VAE are NOT kept resident on the GPU by this
module. Each one is wrapped in a `_Holder` + ComfyUI `ModelPatcher`, so that
`comfy.model_management` owns their placement: they are moved to the GPU by
`load_models_gpu()` right before inference, and ComfyUI can offload them back
to the CPU when another model needs the VRAM (or when the user clicks
"Unload Models"). The computation itself (vae.encode / unet / vae.decode) is
unchanged and still calls the diffusers modules directly.
"""

import json
import math
import os
import tempfile

import torch
import torch.nn.functional as F
import comfy.model_management as mm
import folder_paths
from comfy.model_patcher import ModelPatcher
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextConfig, CLIPTokenizer
from safetensors.torch import load_file, safe_open


# ---------------------------------------------------------------------------
# VRAM budget per working pixel (padded working size, batch of 1).
#
# Measured on an RTX 4070 Ti (fp16, lean VAE path below, see
# tests/pbrfusion4_bench): the activation peak is linear in the pixel count
# from 1024 to 3072 px -- ~580-700 B/px for VAE encode (the UNet stays below
# that, ~260 B/px) and ~1030-1090 B/px for VAE decode. The values below add a
# small margin. They are used twice:
#   - as `memory_required` for comfy.model_management.load_models_gpu(), so
#     ComfyUI unloads other models (and the UNet before the decode) only when
#     that is actually needed;
#   - as a preemptive check: if the free VRAM is still below the budget, the
#     attempt is treated like an OOM and retried at a lower resolution.
#     On Windows the NVIDIA driver's default "sysmem fallback" policy turns a
#     real OOM into a silent spill to system RAM (10x slower) instead of an
#     exception, so the reactive OOM catch alone is not enough there.
# ---------------------------------------------------------------------------
_ENCODE_BYTES_PER_PIXEL = 720
_DECODE_BYTES_PER_PIXEL = 1150
# Prints the activation peak of each inference (to re-calibrate the budgets
# above on another GPU / dtype).
_LOG_PEAK_VRAM = False


# ---------------------------------------------------------------------------
# No VAE tiling.
#
# vae.enable_tiling() is intentionally never used here: diffusers' tiled_decode
# produces visible seams at tile boundaries (independent per-tile GroupNorm
# statistics, compounded by the mid_block's self-attention operating per-tile
# instead of globally -- confirmed empirically).
# The lean encode/decode below is exact instead (global GroupNorm statistics,
# global attention), only the convolutions run in strips.
#
# The primary defense against excessive VRAM/compute is the explicit
# `max_resolution` parameter on MoonPBRFusion4Depth. The preemptive budget
# check and the OOM catch-and-shrink are secondary safety nets for cases where
# the selected working resolution does not fit because other models or nodes
# are using VRAM.
# ---------------------------------------------------------------------------


class _NotEnoughVRAM(RuntimeError):
    """Raised before running a phase whose VRAM budget does not fit."""


def _is_cuda_oom(exc):
    """
    torch.cuda.OutOfMemoryError covers most cases, but some CUDA/cuDNN
    allocation failures still surface as a plain RuntimeError with the
    message intact -- catch both so a real OOM is never left unhandled.
    """
    if isinstance(exc, (torch.cuda.OutOfMemoryError, _NotEnoughVRAM)):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# ---------------------------------------------------------------------------
# Lean (exact) VAE encode / decode.
#
# The VAE's full-resolution stages dominate the VRAM peak: diffusers keeps the
# GroupNorm output, the SiLU output, the 2x nearest-upsampled tensor and the
# conv output alive at once, each one H x W x 128..256 channels. This is NOT
# tiling: GroupNorm statistics are computed over the WHOLE tensor (strip-wise
# Welford, fp32 accumulation, merged exactly), the attention blocks run untouched at latent resolution, and only
# the pointwise norm/act/upsample + 3x3 conv chain is evaluated in horizontal
# strips with a 1-row halo. Every output pixel therefore sees exactly the same
# inputs as in the monolithic forward -- no seams, same result up to fp16
# rounding -- while the per-op temporaries shrink to strip size.
# ---------------------------------------------------------------------------

# Target size of one strip's temporaries (64 MB measured fastest; the VRAM
# peak is set by the full-size tensors, not by the strips).
_STRIP_BYTES = 64 * 1024 * 1024


def _strip_rows(x, upsample=False):
    b, c, _, w = x.shape
    per_row = b * c * w * 4 * (4 if upsample else 1)
    return max(8, _STRIP_BYTES // per_row)


def _channel_var_mean(x):
    """Per-(sample, channel) population variance and mean, fp32 results.

    torch.var_mean on the whole tensor would materialize a full-size copy, so
    it runs strip by strip (fp32 accumulation inside the kernel) and the strip
    statistics are merged exactly (Chan et al.: total variance = mean of the
    strip variances + variance of the strip means, weighted by row count).
    If the fp16 statistics overflow anywhere, everything is recomputed from
    fp32 strips (one host sync per call, not per strip)."""
    h = x.shape[2]
    rows = _strip_rows(x)
    starts = range(0, h, rows)
    counts = [min(rows, h - r) for r in starts]

    def stats(upcast):
        out = [torch.var_mean(x[:, :, r:r + rows].float() if upcast else x[:, :, r:r + rows],
                              dim=(2, 3), correction=0) for r in starts]
        return (torch.stack([m for _, m in out]).float(),
                torch.stack([v for v, _ in out]).float())

    means, varis = stats(upcast=False)
    if x.dtype != torch.float32 and not (torch.isfinite(means).all() and torch.isfinite(varis).all()):
        means, varis = stats(upcast=True)
    w = torch.tensor(counts, device=x.device, dtype=torch.float32)[:, None, None] / h
    mean = (w * means).sum(0)
    var = (w * (varis + (means - mean).square())).sum(0)
    return var, mean


def _groupnorm_affine(norm, x):
    """Global GroupNorm stats folded into per-(sample, channel) scale/shift.

    Group variance = mean of the channel variances + variance of the channel
    means (law of total variance: both terms >= 0, no cancellation)."""
    b, c, _, _ = x.shape
    g = norm.num_groups
    var_c, mean_c = _channel_var_mean(x)
    var_c, mean_c = var_c.view(b, g, c // g), mean_c.view(b, g, c // g)
    mean_g = mean_c.mean(-1, keepdim=True)
    var_g = var_c.mean(-1, keepdim=True) + (mean_c - mean_g).square().mean(-1, keepdim=True)
    scale = torch.rsqrt(var_g + norm.eps).expand(b, g, c // g).reshape(b, c)
    if norm.affine:
        scale = scale * norm.weight.float()[None]
    shift = -mean_g.expand(b, g, c // g).reshape(b, c) * scale
    if norm.affine:
        shift = shift + norm.bias.float()[None]
    return scale[:, :, None, None], shift[:, :, None, None]


def _strip_conv(x, conv, norm=None, upsample=False, residual=None, out_scale=1.0):
    """conv(silu(norm(x))) / conv(upsample2x(x)) / conv(x), strip by strip.

    With `residual` (same shape as the output), the result is accumulated IN
    PLACE into `residual` -- (residual + conv(...)) / out_scale -- and
    `residual` is returned: no extra full-size output buffer (the ResnetBlock2D
    skip path). Safe because the strips read `x`, never `residual`."""
    k = conv.kernel_size[0]
    halo = k // 2
    assert conv.stride == (1, 1) and conv.padding == (halo, halo) and conv.padding_mode == "zeros"
    b, _, h, w = x.shape
    ho, wo = (2 * h, 2 * w) if upsample else (h, w)
    if norm is not None:
        scale, shift = _groupnorm_affine(norm, x)
        scale, shift = scale.to(x.dtype), shift.to(x.dtype)
    if residual is not None:
        assert residual.shape == (b, conv.out_channels, ho, wo)
        out = residual
    else:
        out = torch.empty(b, conv.out_channels, ho, wo, device=x.device, dtype=x.dtype,
                          memory_format=torch.channels_last)
    rows = _strip_rows(x, upsample) * (2 if upsample else 1)
    for r0 in range(0, ho, rows):
        r1 = min(r0 + rows, ho)
        a, e = max(r0 - halo, 0), min(r1 + halo, ho)
        if upsample:
            ia, ie = a // 2, (e + 1) // 2
            xs = F.interpolate(x[:, :, ia:ie], scale_factor=2.0, mode="nearest")
            xs = xs[:, :, a - 2 * ia:a - 2 * ia + (e - a)]
        else:
            xs = x[:, :, a:e]
        if norm is not None:
            xs = F.silu(torch.addcmul(shift, xs, scale), inplace=True)
        # Columns: zero padding inside the conv. Rows: real halo rows from the
        # neighbouring strips, zero rows only at the image's top/bottom edges.
        pad_top, pad_bot = halo - (r0 - a), halo - (e - r1)
        if pad_top or pad_bot:
            xs = F.pad(xs, (0, 0, pad_top, pad_bot))
        ys = F.conv2d(xs, conv.weight, conv.bias, padding=(0, halo))
        del xs
        if residual is not None:
            dst = out[:, :, r0:r1]
            dst += ys
            if out_scale != 1.0:
                dst /= out_scale
        else:
            out[:, :, r0:r1] = ys
        del ys
    return out


def _lean_resnet(blk, holder):
    """diffusers ResnetBlock2D (VAE variant: no temb, no up/down) -- exact.

    Takes its input as a one-element list and empties it, so the caller holds
    no reference: the input buffer is reused in place as the output (identity
    skip) or freed as soon as it is no longer needed."""
    assert blk.time_emb_proj is None and not blk.up and not blk.down
    x = holder.pop()
    skip = _strip_conv(x, blk.conv_shortcut) if blk.conv_shortcut is not None else None
    h = _strip_conv(x, blk.conv1, norm=blk.norm1)
    if skip is None:
        skip = x
    del x
    return _strip_conv(h, blk.conv2, norm=blk.norm2, residual=skip,
                       out_scale=blk.output_scale_factor)


def _lean_vae_encode_mean(vae, rgb):
    """Encoder + quant_conv, returns the posterior MEAN (deterministic)."""
    enc = vae.encoder
    x = enc.conv_in(rgb.contiguous(memory_format=torch.channels_last))
    for down in enc.down_blocks:
        for res in down.resnets:
            holder = [x]
            del x
            x = _lean_resnet(res, holder)
        if down.downsamplers is not None:
            for ds in down.downsamplers:
                x = ds(x)
    x = enc.mid_block(x)
    x = _strip_conv(x, enc.conv_out, norm=enc.conv_norm_out)
    if vae.quant_conv is not None:
        x = vae.quant_conv(x)
    return x[:, : x.shape[1] // 2]


def _lean_vae_decode(vae, z):
    dec = vae.decoder
    if vae.post_quant_conv is not None:
        z = vae.post_quant_conv(z)
    x = dec.conv_in(z.contiguous(memory_format=torch.channels_last))
    x = dec.mid_block(x, None)
    for up in dec.up_blocks:
        for res in up.resnets:
            holder = [x]
            del x
            x = _lean_resnet(res, holder)
        if up.upsamplers is not None:
            for us in up.upsamplers:
                assert us.interpolate and not us.use_conv_transpose and us.norm is None
                x = _strip_conv(x, us.conv, upsample=True)
    return _strip_conv(x, dec.conv_out, norm=dec.conv_norm_out)


def _gaussian_denoise(x, sigma):
    """Separable gaussian blur, reflect borders. x: (B, C, H, W) float.

    Targets the VAE decoder grain, which lives below ~1 working pixel (plus a
    weak 8 px latent-grid component): sigma 0.8 removes ~65% of the sub-pixel
    noise while the 1-4 px relief detail drops by ~3% only."""
    r = min(math.ceil(3 * sigma), x.shape[2] - 1, x.shape[3] - 1)
    if sigma <= 0 or r < 1:
        return x
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def _task_embedding(device, dtype):
    """Depth task embedding, per the original Lotus task conditioning scheme."""
    task = torch.tensor([1, 0], device=device, dtype=dtype).unsqueeze(0)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


def _working_size(h, w, max_resolution):
    """Native size if the longest side fits, else proportional downscale
    (exact aspect ratio; padding to the VAE multiple happens separately)."""
    longest_side = max(h, w)
    if longest_side <= max_resolution:
        return h, w
    scale = max_resolution / longest_side
    return max(1, round(h * scale)), max(1, round(w * scale))


def _check_vram(device, budget, phase):
    if torch.device(device).type != "cuda":
        return
    free = mm.get_free_memory(device)
    if free < budget:
        raise _NotEnoughVRAM(
            f"{phase} needs ~{budget / 1e9:.2f} GB of free VRAM, {free / 1e9:.2f} GB available"
        )


def _infer_at_resolution(model, image, max_resolution_px, denoise=0.0):
    """One image (batch of 1) -> raw decoded depth (1, h, w, 3) on the CPU."""
    unet_patcher = model["unet_patcher"]
    vae_patcher = model["vae_patcher"]
    empty_embed = model["empty_embed"]
    dtype = model["dtype"]
    device = model["device"]
    scaling_factor = model["vae_scaling_factor"]
    multiple = 2 ** (len(vae_patcher.model.net.config.block_out_channels) - 1)

    _, in_h, in_w, _ = image.shape
    h, w = _working_size(in_h, in_w, max_resolution_px)
    if (h, w) != (in_h, in_w):
        image = F.interpolate(
            image.permute(0, 3, 1, 2), size=(h, w), mode="bicubic", antialias=True
        ).clamp(0, 1).permute(0, 2, 3, 1)

    # Pad (instead of resizing) to the VAE multiple and crop back after the
    # decode: output pixels stay aligned 1:1 with the input pixels.
    pad_h, pad_w = (-h) % multiple, (-w) % multiple
    pixels = (h + pad_h) * (w + pad_w)

    encode_budget = pixels * _ENCODE_BYTES_PER_PIXEL
    mm.load_models_gpu([unet_patcher, vae_patcher], memory_required=encode_budget,
                       force_full_load=True)
    _check_vram(device, encode_budget, "VAE encode")
    unet = unet_patcher.model.net
    vae = vae_patcher.model.net

    log_peak = _LOG_PEAK_VRAM and torch.device(device).type == "cuda"
    if log_peak:
        torch.cuda.synchronize(device)
        base = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        rgb = image.to(device=device, dtype=dtype).permute(0, 3, 1, 2) * 2.0 - 1.0
        if pad_h or pad_w:
            mode = "reflect" if min(h, w) > multiple else "replicate"
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h), mode=mode)

        # Posterior MEAN, not a sample: deterministic. The sampling noise
        # (std ~1e-3 in latent units) only moved the depth by ~1e-4 on average.
        rgb_latents = _lean_vae_encode_mean(vae, rgb) * scaling_factor
        del rgb

        timestep = torch.tensor([999], device=device).long()
        task_emb = _task_embedding(device, dtype)
        pred_latents = unet(
            rgb_latents, timestep,
            encoder_hidden_states=empty_embed,
            class_labels=task_emb,
            return_dict=False,
        )[0]
        del rgb_latents
        pred_latents = pred_latents / scaling_factor

        # The UNet is no longer needed: ComfyUI offloads it only if the decode
        # budget does not fit next to it.
        decode_budget = pixels * _DECODE_BYTES_PER_PIXEL
        mm.load_models_gpu([vae_patcher], memory_required=decode_budget, force_full_load=True)
        _check_vram(device, decode_budget, "VAE decode")
        decoded = _lean_vae_decode(vae, pred_latents).float()
        del pred_latents
        # Before the crop: the bottom/right borders see the decoded padding
        # (reflected content) instead of an artificial reflection of the crop.
        decoded = _gaussian_denoise(decoded, denoise)
        decoded = decoded[:, :, :h, :w]
        output = (decoded / 2.0 + 0.5).permute(0, 2, 3, 1).cpu()
        del decoded

    if log_peak:
        peak = torch.cuda.max_memory_allocated(device) - base
        print(f"[MoonPBRFusion4Depth] activation peak {peak / 1e9:.2f} GB at "
              f"{w + pad_w}x{h + pad_h} px ({peak / pixels:.0f} B/px; budgets: encode "
              f"{_ENCODE_BYTES_PER_PIXEL}, decode {_DECODE_BYTES_PER_PIXEL} B/px)")
    return output


def _infer_with_fallback(model, image, res, denoise=0.0):
    """Runs one image, shrinking the working resolution on OOM / short VRAM.
    Returns (output, resolution actually used)."""
    longest = max(image.shape[1], image.shape[2])
    while True:
        working = min(res, longest)
        reason = None
        try:
            return _infer_at_resolution(model, image, working, denoise), working
        except RuntimeError as e:
            if not _is_cuda_oom(e):
                raise
            reason = str(e).splitlines()[0]
        # Retry OUTSIDE the except block: the traceback would otherwise
        # keep the failed attempt's GPU tensors alive.
        mm.soft_empty_cache()
        if working <= 256:
            raise RuntimeError(
                "MoonPBRFusion4Depth: out of VRAM even at minimal resolution."
            )
        res = max(256, int(working * 0.75))
        print(f"[MoonPBRFusion4Depth] {reason} -- retrying at {res} px")


class MoonPBRFusion4Depth:
    """
    PBRFusion4 Depth

    Discriminative (Lotus-D) single-step inference. Confirmed against the
    official EnVision-Research/Lotus pipeline.py (class LotusDPipeline): no
    concatenation with a second latent, no scheduler.step() -- the UNet is
    called once on the encoded RGB latent, and its output IS the predicted
    target latent, not a noise residual.

    Output is intentionally raw: only the fixed VAE decode convention
    ([-1, 1] -> [0, 1]) is applied, no clamp, no min/max stretch, no channel
    reduction. Normalization belongs in a downstream Depth -> Height node.
    The only optional filter is `denoise` (off by default): a gaussian blur
    whose sigma is in WORKING pixels, the scale of the VAE decoder grain --
    which is why it runs here, before any resize, and not downstream.

    `max_resolution` controls the maximum working resolution used by PBRFusion4.
    Inputs below this limit are processed at their native resolution. Inputs
    above it are downscaled proportionally before inference.
    `match_input_resolution` is an optional final bicubic resize that restores
    the original input dimensions. It does not add generative detail.

    The UNet and VAE are loaded onto the GPU through ComfyUI's model
    management (load_models_gpu, full load) for the duration of the call and
    can be offloaded by ComfyUI afterwards.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pbrfusion4_model": ("PBRFUSION4_MODEL",),
                "image": ("IMAGE",),
                "max_resolution": ([
                    "512 px",
                    "768 px",
                    "1024 px",
                    "1280 px",
                    "1536 px",
                    "2048 px",
                    "2560 px",
                    "3072 px",
                    "4096 px",
                ], {
                    "default": "2048 px",
                    "tooltip": "Maximum working resolution for PBRFusion4. "
                               "Inputs below this limit are processed at their native "
                               "resolution; larger inputs are downscaled proportionally. "
                               "VRAM grows with the pixel count (~1.1 KB per working pixel on "
                               "top of ~1.9 GB of weights): 2048 px fits comfortably in 12 GB, "
                               "3072 px fits once ComfyUI offloads the UNet. If the free VRAM "
                               "is too low, the resolution is reduced automatically."
                }),
                "match_input_resolution": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ON: resize the decoded depth back to the input image's "
                               "exact dimensions using a plain bicubic resize. "
                               "OFF: keep the decoded depth at the PBRFusion4 working "
                               "resolution. Bicubic resize adds no generative detail."
                }),
                "denoise": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Optional gaussian denoise of the decoded depth, sigma in "
                               "working pixels (0 = off, raw output). Removes the fine grain "
                               "of the VAE decoder before any resize: 0.8 removes most of it "
                               "while keeping the 1-4 px relief detail; above ~1.5 the relief "
                               "starts to soften."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("decoded_depth",)
    FUNCTION = "infer"
    CATEGORY = "moon/depth"
    DESCRIPTION = (
        "PBRFusion4-D depth inference. The model runs at the input resolution "
        "up to the selected maximum. Larger inputs are "
        "downscaled proportionally before inference. Optionally resize the decoded "
        "result back to the input image's exact dimensions using a plain bicubic "
        "resize. No normalization or channel reduction is applied; the only "
        "optional filter is a light gaussian denoise of the VAE decoder grain."
    )

    def infer(self, pbrfusion4_model, image, max_resolution, match_input_resolution, denoise=0.0):
        image = image[..., :3]
        input_h, input_w = image.shape[1], image.shape[2]
        res = int(max_resolution.split()[0])

        # One image at a time: the VRAM peak does not grow with the batch size.
        # `res` only ever decreases, so a fallback applies to the rest of the batch.
        outputs = []
        for i in range(image.shape[0]):
            out, res = _infer_with_fallback(pbrfusion4_model, image[i:i + 1], res, denoise)
            outputs.append(out)

        if match_input_resolution:
            target = (input_h, input_w)
        else:
            # All equal unless a fallback happened mid-batch: the last output
            # then has the smallest size.
            target = tuple(outputs[-1].shape[1:3])
        outputs = [
            o if tuple(o.shape[1:3]) == target else F.interpolate(
                o.permute(0, 3, 1, 2), size=target, mode="bicubic", antialias=True
            ).permute(0, 2, 3, 1)
            for o in outputs
        ]

        mm.soft_empty_cache()
        return (torch.cat(outputs, dim=0),)

# ---------------------------------------------------------------------------
# PBRFusion4 Loader
#
# Loads the PBRFusion4 discriminative depth model (Lotus-D architecture) from
# a single .safetensors package (unet + vae + text_encoder, prefixed and
# embedded as safetensors metadata).
#
# This is a minimal, from-scratch re-implementation of the loading logic found
# in Night1099/COMFYUI-PBRFusion4 (pbrfusion4_simple_nodes.py ->
# _load_from_single_safetensors). We do NOT vendor the Lotus/ directory or its
# pipeline.py: MoonPBRFusion4Depth calls unet/vae directly, reproducing only
# the exact operations confirmed in EnVision-Research/Lotus's LotusDPipeline
# (no scheduler, no concat -- see MoonPBRFusion4Depth for details).
#
# Key difference vs. the upstream node: the CLIP text encoder and tokenizer
# are only used once, to compute the empty-prompt embedding ("" conditioning,
# the only prompt PBRFusion4 is ever run with). That embedding is cached and
# the text encoder/tokenizer are freed immediately after, instead of being
# kept resident in VRAM for every inference call.
#
# Ownership: the UNet and VAE are wrapped in ModelPatchers (see _Holder) and
# the cache below stores those patchers. Weights are created on ComfyUI's
# offload device (normally the CPU) and only moved to the GPU by
# load_models_gpu() at inference time.
# ---------------------------------------------------------------------------

PBRFUSION4_MODEL_DIR = os.path.join(folder_paths.models_dir, "pbrfusion4")
os.makedirs(PBRFUSION4_MODEL_DIR, exist_ok=True)

_MODEL_CACHE = {}


class _Holder(torch.nn.Module):
    """Thin nn.Module container so a diffusers UNet/VAE can be wrapped in a
    ComfyUI ModelPatcher.

    ModelPatcher assigns `model.device = ...` on load/unload. On a diffusers
    ModelMixin `device` is a read-only property, so the assignment would
    raise. This container accepts the attribute; the real network is reached
    through `.net`.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net


def _build_empty_prompt_embedding(text_encoder, tokenizer, device, dtype):
    """Encode the empty string once -- the only prompt PBRFusion4 ever uses.
    Matches DirectDiffusionPipeline.encode_prompt's default padding_type."""
    tokens = tokenizer(
        [""],
        padding="do_not_pad",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embed = text_encoder(tokens.input_ids.to(device))[0]
    return embed.to(dtype=dtype, device=device)


def _load_pbrfusion4_components(safetensors_path, dtype, device):
    """
    Split the single-file PBRFusion4 checkpoint into its components and
    reconstruct unet / vae / text_encoder / tokenizer from the configs
    embedded as safetensors metadata. Mirrors the key-prefix convention
    used by the original PBRFusion4 ComfyUI node (unet./vae./text_encoder.).

    `device` is where the UNet and VAE weights are placed: the loader passes
    ComfyUI's offload device so nothing is resident on the GPU until
    load_models_gpu() is called at inference time.
    """

    with safe_open(safetensors_path, framework="pt") as f:
        metadata = f.metadata()

    all_weights = load_file(safetensors_path, device="cpu")

    def split(prefix):
        return {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}

    unet_sd = split("unet.")
    vae_sd = split("vae.")
    te_sd = split("text_encoder.")
    del all_weights

    # Modules are built on the meta device and the checkpoint tensors are
    # assigned directly (strict): skips the random weight initialization that
    # would be overwritten anyway (~5-6 s for UNet + CLIP on the CPU).
    unet_config = json.loads(metadata["unet/config.json"])
    with torch.device("meta"):
        unet = UNet2DConditionModel(**{k: v for k, v in unet_config.items() if not k.startswith("_")})
    unet.load_state_dict(unet_sd, assign=True)
    del unet_sd

    vae_config = json.loads(metadata["vae/config.json"])
    with torch.device("meta"):
        vae = AutoencoderKL(**{k: v for k, v in vae_config.items() if not k.startswith("_")})
    vae.load_state_dict(vae_sd, assign=True)
    del vae_sd

    te_config = CLIPTextConfig(**json.loads(metadata["text_encoder/config.json"]))
    with torch.device("meta"):
        text_encoder = CLIPTextModel(te_config)
    # Some checkpoints store CLIP weights with a "text_model." prefix already
    # stripped, some don't -- normalize before loading.
    if any(k.startswith("text_model.") for k in te_sd) and not any(
        k.startswith("text_model.") for k in text_encoder.state_dict()
    ):
        te_sd = {(k[len("text_model."):] if k.startswith("text_model.") else k): v for k, v in te_sd.items()}
    text_encoder.load_state_dict(te_sd, assign=True)
    del te_sd
    # Non-persistent buffer (not in the checkpoint): rebuild it. Located by
    # name because its parent path differs between transformers versions.
    for name, buf in list(text_encoder.named_buffers()):
        if buf.is_meta and name.endswith("position_ids"):
            parent = text_encoder.get_submodule(name.rpartition(".")[0])
            parent.register_buffer(
                "position_ids", torch.arange(buf.shape[-1]).expand((1, -1)), persistent=False
            )
    for module in (unet, vae, text_encoder):
        leftover = [n for n, t in list(module.named_parameters()) + list(module.named_buffers()) if t.is_meta]
        if leftover:
            raise RuntimeError(f"PBRFusion4 loader: tensors not loaded from the checkpoint: {leftover[:5]}")

    tokenizer_files = {k: v for k, v in metadata.items() if k.startswith("tokenizer/")}
    with tempfile.TemporaryDirectory(prefix="pbrfusion4_tokenizer_") as tmpdir:
        for key, content in tokenizer_files.items():
            with open(os.path.join(tmpdir, key.split("/", 1)[1]), "w", encoding="utf-8") as fh:
                fh.write(content)
        tokenizer = CLIPTokenizer.from_pretrained(tmpdir)

    unet.to(dtype=dtype, device=device)
    # channels_last: the lean VAE path keeps its activations in NHWC, which
    # avoids cuDNN's NCHW<->NHWC transposes around every convolution.
    vae.to(dtype=dtype, device=device, memory_format=torch.channels_last)
    # AttnProcessor2_0 (PyTorch SDPA) for the UNet attention.
    unet.set_attn_processor(AttnProcessor2_0())
    return unet, vae, text_encoder, tokenizer


class MoonPBRFusion4Loader:
    """
    Loads PBRFusion4 (Lotus-D discriminative depth model) and caches the
    empty-prompt CLIP embedding. Text encoder and tokenizer are discarded
    right after, since PBRFusion4 is only ever conditioned on an empty prompt.

    The UNet and VAE are handed to ComfyUI's model management (ModelPatcher):
    they stay on the offload device (CPU) until inference, and ComfyUI can
    offload them again when VRAM is needed elsewhere. The patchers are cached
    per (model_name, dtype, device) for the lifetime of the process, so a
    second run reuses them without rebuilding the diffusers modules. The CPU
    copy (roughly the model file size) therefore stays in system RAM.
    """

    @classmethod
    def INPUT_TYPES(cls):
        safetensors_files = [
            f for f in os.listdir(PBRFUSION4_MODEL_DIR)
            if f.endswith(".safetensors")
        ] or ["PBRFusion4.safetensors"]
        return {
            "required": {
                "model_name": (safetensors_files,),
                "dtype": (["fp16", "fp32"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("PBRFUSION4_MODEL",)
    RETURN_NAMES = ("pbrfusion4_model",)
    FUNCTION = "load"
    CATEGORY = "moon/depth"
    DESCRIPTION = "Loads the PBRFusion4 discriminative depth model (unet + vae + cached empty-prompt embedding). Weights are managed by ComfyUI's model management: on the GPU only during inference. No text encoder is kept resident after loading."

    def load(self, model_name, dtype):
        load_device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        torch_dtype = torch.float16 if dtype == "fp16" else torch.float32

        cache_key = (model_name, dtype, str(load_device))
        if cache_key in _MODEL_CACHE:
            return (_MODEL_CACHE[cache_key],)

        safetensors_path = os.path.join(PBRFUSION4_MODEL_DIR, model_name)
        unet, vae, text_encoder, tokenizer = _load_pbrfusion4_components(
            safetensors_path, torch_dtype, offload_device
        )

        empty_embed = _build_empty_prompt_embedding(text_encoder, tokenizer, device="cpu", dtype=torch_dtype)
        # Tiny tensor, independent of the model management: kept on the load device.
        empty_embed = empty_embed.to(device=load_device, dtype=torch_dtype)
        # One-time diagnostic printout -- confirms real scaling_factor /
        # channel counts / dtypes against what MoonPBRFusion4Depth assumes.
        # Expect unet.config.in_channels == vae.config.latent_channels (no
        # doubling): the D-variant does not concatenate a second latent.
        print("[PBRFusion4 Loader]")
        print(f"  Model:              {model_name}")
        print(f"  VAE scaling_factor: {vae.config.scaling_factor}")
        print(f"  UNet in_channels:   {unet.config.in_channels}")
        print(f"  UNet dtype:         {next(unet.parameters()).dtype}")
        print(f"  VAE dtype:          {next(vae.parameters()).dtype}")
        print(f"  Empty embed shape:  {tuple(empty_embed.shape)}")
        print(f"  Load / offload:     {load_device} / {offload_device}")

        # Text encoder / tokenizer are no longer needed: PBRFusion4 is only
        # ever conditioned on the empty string.
        del text_encoder
        del tokenizer
        mm.soft_empty_cache()

        scaling_factor = vae.config.scaling_factor
        unet_patcher = ModelPatcher(_Holder(unet), load_device=load_device, offload_device=offload_device)
        vae_patcher = ModelPatcher(_Holder(vae), load_device=load_device, offload_device=offload_device)

        model = {
            "unet_patcher": unet_patcher,
            "vae_patcher": vae_patcher,
            "empty_embed": empty_embed,
            "dtype": torch_dtype,
            "device": load_device,
            "vae_scaling_factor": scaling_factor,
        }
        _MODEL_CACHE[cache_key] = model
        return (model,)


NODE_CLASS_MAPPINGS = {
    "MoonPBRFusion4Loader": MoonPBRFusion4Loader,
    "MoonPBRFusion4Depth": MoonPBRFusion4Depth,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonPBRFusion4Loader": "PBRFusion4 Loader",
    "MoonPBRFusion4Depth": "PBRFusion4 Depth",
}