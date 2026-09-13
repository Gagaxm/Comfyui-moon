"""
PBRFusion4 nodes for Comfyui-moon.

Loads and runs PBRFusion4 (Lotus-D discriminative depth model) directly via
diffusers, bypassing ComfyUI's own model/VAE wrapping so the exact operations
confirmed against EnVision-Research/Lotus's LotusDPipeline can be reproduced
one-to-one (see MoonPBRFusion4Depth below).
"""

import json
import os
import tempfile

import torch
import torch.nn.functional as F
import comfy.model_management as mm
import folder_paths
from diffusers.models.attention_processor import AttnProcessor2_0


# ---------------------------------------------------------------------------
# VAE decode: no tiling, adapts to available VRAM reactively.
#
# vae.enable_tiling() is intentionally never used here: diffusers' tiled_decode
# produces visible seams at tile boundaries (independent per-tile GroupNorm
# statistics, compounded by the mid_block's self-attention operating per-tile
# instead of globally -- confirmed empirically, see
# tests/test_shared_groupnorm_tiled_vae.py).
#
# The primary defense against excessive VRAM/compute is now the explicit
# `resolution` parameter on MoonPBRFusion4Depth (see below) -- the user
# chooses a known-safe working resolution instead of the node silently
# adapting. The OOM catch-and-shrink below is a secondary safety net only,
# for the rare case where even the chosen resolution doesn't fit (e.g. VRAM
# taken by other models in the same workflow).
# ---------------------------------------------------------------------------

# id(vae) -> largest decoded output size (max(H, W) in pixels) confirmed to
# work without tiling this session. Reset when ComfyUI restarts: VRAM headroom
# at startup can differ between sessions, so each session re-learns its own
# limit rather than assuming a fixed cap.
_MAX_DECODE_SIZE_CACHE = {}


def _is_cuda_oom(exc):
    """
    torch.cuda.OutOfMemoryError covers most cases, but some CUDA/cuDNN
    allocation failures still surface as a plain RuntimeError with the
    message intact -- catch both so a real OOM is never left unhandled.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _decode_no_tiling_with_fallback(vae, pred_latents, scaling_factor):
    vae.disable_tiling()
    spatial_scale = 2 ** (len(vae.config.block_out_channels) - 1)
    latents = pred_latents
    cache_key = id(vae)
    known_good = _MAX_DECODE_SIZE_CACHE.get(cache_key)

    if known_good is not None and max(latents.shape[-2:]) * spatial_scale > known_good:
        scale = known_good / (max(latents.shape[-2:]) * spatial_scale)
        latents = F.interpolate(latents, scale_factor=scale, mode="nearest-exact")
        print(f"[MoonPBRFusion4Depth] Downscaling to {latents.shape[-2] * spatial_scale}x"
              f"{latents.shape[-1] * spatial_scale}px (largest confirmed working size this session).")

    while True:
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(latents.device)

            decoded = vae.decode(latents / scaling_factor, return_dict=False)[0]

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated(latents.device)
                print(f"[MoonPBRFusion4Depth] Decode peak VRAM: {peak / 1e9:.2f} GB for "
                      f"{latents.shape[-2] * spatial_scale}x{latents.shape[-1] * spatial_scale}px.")

            _MAX_DECODE_SIZE_CACHE[cache_key] = max(latents.shape[-2:]) * spatial_scale
            return decoded

        except RuntimeError as e:
            if not _is_cuda_oom(e):
                raise  # a real bug, not a VRAM issue -- don't swallow it

            mm.soft_empty_cache()
            h, w = latents.shape[-2:]
            if min(h, w) <= 8:
                raise RuntimeError(
                    "MoonPBRFusion4Depth: out of VRAM even at minimal resolution -- "
                    "this GPU cannot run a non-tiled decode for this model."
                )
            _MAX_DECODE_SIZE_CACHE[cache_key] = max(h, w) * spatial_scale - spatial_scale
            latents = F.interpolate(latents, scale_factor=0.5, mode="nearest-exact")
            print(f"[MoonPBRFusion4Depth] OOM -- reducing to {latents.shape[-2] * spatial_scale}x"
                  f"{latents.shape[-1] * spatial_scale}px and retrying.")


def _task_embedding(device, dtype):
    """Depth task embedding, per the original Lotus task conditioning scheme."""
    task = torch.tensor([1, 0], device=device, dtype=dtype).unsqueeze(0)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


def _resize_to_resolution(image, resolution, spatial_scale):
    """Resize (B, H, W, C) so the longer side equals `resolution`, rounded to a multiple
    of spatial_scale (required for VAE encode/decode divisibility). Aspect ratio preserved."""
    b, h, w, c = image.shape
    scale = resolution / max(h, w)
    new_h = max(spatial_scale, round(h * scale / spatial_scale) * spatial_scale)
    new_w = max(spatial_scale, round(w * scale / spatial_scale) * spatial_scale)
    image_bchw = image.permute(0, 3, 1, 2)
    resized = F.interpolate(image_bchw, size=(new_h, new_w), mode="bicubic", antialias=True)
    return resized.permute(0, 2, 3, 1).clamp(0, 1)


def _resize_to_shape(image, h, w):
    """Resize (B, H, W, C) to an exact (h, w) -- used to match decoded_depth back to the
    original input resolution. Plain bicubic resize, not a learned upscaler: restores pixel
    dimensions for a 1:1 drop-in output, recovers no generative detail beyond what the
    'resolution' working size already produced."""
    image_bchw = image.permute(0, 3, 1, 2)
    resized = F.interpolate(image_bchw, size=(h, w), mode="bicubic", antialias=True)
    return resized.permute(0, 2, 3, 1).clamp(0, 1)


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

    `resolution` and `match_input_resolution` are a deliberate, documented
    exception to "no parameters beyond loading config": they control
    inference conditions (the resolution the model actually runs at), not
    post-processing or result interpretation. This matches the standard
    convention across ComfyUI's preprocessor-style nodes (e.g.
    comfyui_controlnet_aux) for this exact class of problem -- a heavy
    per-pixel model whose VRAM/time cost depends on working resolution.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pbrfusion4_model": ("PBRFUSION4_MODEL",),
                "image": ("IMAGE",),
                "resolution": ("INT", {
                    "default": 1024, "min": 512, "max": 2048, "step": 64,
                    "tooltip": "Longer-side working resolution. 1024 confirmed fast (~13s) and "
                               "safe on 12GB VRAM; up to ~1536 also works well. Beyond ~1536, "
                               "the VAE mid_block attention cost grows with the 4th power of "
                               "resolution and can stall or exhaust VRAM."
                }),
                "match_input_resolution": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ON (default): decoded_depth is resized back to the input "
                               "image's exact resolution with a plain bicubic resize -- "
                               "simple, fast, 1:1 drop-in output, no generative detail beyond "
                               "'resolution'. OFF: output stays at the working resolution -- "
                               "upscale it yourself downstream with a dedicated model for "
                               "finer detail recovery."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("decoded_depth",)
    FUNCTION = "infer"
    CATEGORY = "moon/pbr"
    DESCRIPTION = "Decoded output of the PBRFusion4-D prediction latent. No normalization, filtering, channel reduction or depth-specific post-processing is applied beyond the resolution controls below."

    def infer(self, pbrfusion4_model, image, resolution, match_input_resolution):
        unet = pbrfusion4_model["unet"]
        vae = pbrfusion4_model["vae"]
        empty_embed = pbrfusion4_model["empty_embed"]
        dtype = pbrfusion4_model["dtype"]
        device = pbrfusion4_model["device"]
        scaling_factor = pbrfusion4_model["vae_scaling_factor"]

        spatial_scale = 2 ** (len(vae.config.block_out_channels) - 1)
        input_h, input_w = image.shape[1], image.shape[2]
        image = _resize_to_resolution(image, resolution, spatial_scale)
        print(f"[MoonPBRFusion4Depth] Input {input_w}x{input_h} -> working "
              f"{image.shape[2]}x{image.shape[1]}" +
              (f" -> output {input_w}x{input_h}" if match_input_resolution
               else " -> output stays at working resolution (match_input_resolution off)"))

        # ComfyUI IMAGE: (B, H, W, C) in [0, 1] -> Lotus convention: (B, C, H, W) in [-1, 1]
        rgb = image.to(device=device, dtype=dtype).permute(0, 3, 1, 2)
        rgb = rgb * 2.0 - 1.0

        with torch.inference_mode():
            rgb_latents = vae.encode(rgb).latent_dist.sample()
            rgb_latents = rgb_latents * scaling_factor

            timestep = torch.tensor([999], device=device).long()
            # batch size is always 1 for PBRFusion4, but we expand the prompt and task embeddings to match the input batch size for generality.
            batch_size = rgb_latents.shape[0]
            prompt_embeds = empty_embed.expand(batch_size, -1, -1)
            task_emb = _task_embedding(device, dtype).expand(batch_size, -1)

            pred_latents = unet(
                rgb_latents,
                timestep,
                encoder_hidden_states=prompt_embeds,
                class_labels=task_emb,
                return_dict=False,
            )[0]

            # rgb / rgb_latents are unused from here on -- freed before the memory-heavy
            # decode step, not after, since that's when VRAM is most contended.
            del rgb, rgb_latents
            mm.soft_empty_cache()

            decoded = _decode_no_tiling_with_fallback(vae, pred_latents, scaling_factor)

        # Fixed VAE convention remap only -- no clamp, no data-dependent stretch.
        output = (decoded / 2.0 + 0.5).permute(0, 2, 3, 1).float().cpu()
        if match_input_resolution:
            output = _resize_to_shape(output, input_h, input_w)

        del pred_latents, decoded
        mm.soft_empty_cache()
        return (output,)


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
# ---------------------------------------------------------------------------

PBRFUSION4_MODEL_DIR = os.path.join(folder_paths.models_dir, "pbrfusion4")
os.makedirs(PBRFUSION4_MODEL_DIR, exist_ok=True)

_MODEL_CACHE = {}


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
    """
    from safetensors.torch import load_file, safe_open
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextConfig, CLIPTokenizer

    with safe_open(safetensors_path, framework="pt") as f:
        metadata = f.metadata()

    all_weights = load_file(safetensors_path, device="cpu")

    def split(prefix):
        return {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}

    unet_sd = split("unet.")
    vae_sd = split("vae.")
    te_sd = split("text_encoder.")
    del all_weights

    unet_config = json.loads(metadata["unet/config.json"])
    unet = UNet2DConditionModel(**{k: v for k, v in unet_config.items() if not k.startswith("_")})
    unet.load_state_dict(unet_sd)
    del unet_sd

    vae_config = json.loads(metadata["vae/config.json"])
    vae = AutoencoderKL(**{k: v for k, v in vae_config.items() if not k.startswith("_")})
    vae.load_state_dict(vae_sd)
    del vae_sd

    te_config = CLIPTextConfig(**json.loads(metadata["text_encoder/config.json"]))
    text_encoder = CLIPTextModel(te_config)
    # Some checkpoints store CLIP weights with a "text_model." prefix already
    # stripped, some don't -- normalize before loading.
    if any(k.startswith("text_model.") for k in te_sd) and not any(
        k.startswith("text_model.") for k in text_encoder.state_dict()
    ):
        te_sd = {(k[len("text_model."):] if k.startswith("text_model.") else k): v for k, v in te_sd.items()}
    text_encoder.load_state_dict(te_sd)
    del te_sd

    tokenizer_files = {k: v for k, v in metadata.items() if k.startswith("tokenizer/")}
    with tempfile.TemporaryDirectory(prefix="pbrfusion4_tokenizer_") as tmpdir:
        for key, content in tokenizer_files.items():
            with open(os.path.join(tmpdir, key.split("/", 1)[1]), "w", encoding="utf-8") as fh:
                fh.write(content)
        tokenizer = CLIPTokenizer.from_pretrained(tmpdir)

    unet.to(dtype=dtype, device=device)
    vae.to(dtype=dtype, device=device)
    # AttnProcessor2_0 reduces VRAM usage during UNet inference.
    unet.set_attn_processor(AttnProcessor2_0())
    # vae.enable_slicing() splits along the batch dimension -- PBRFusion4 always runs at
    # batch_size=1, so this has no real memory effect in practice. Left enabled since it's
    # harmless and free, but it should not be relied on as VRAM protection (see
    # _decode_no_tiling_with_fallback and MoonPBRFusion4Depth's `resolution` parameter for
    # the mechanisms that actually do that job).
    vae.enable_slicing()
    return unet, vae, text_encoder, tokenizer


class MoonPBRFusion4Loader:
    """
    Loads PBRFusion4 (Lotus-D discriminative depth model) and caches the
    empty-prompt CLIP embedding. Text encoder and tokenizer are discarded
    right after, since PBRFusion4 is only ever conditioned on an empty prompt.
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
    CATEGORY = "moon/pbr"
    DESCRIPTION = "Loads the PBRFusion4 discriminative depth model (unet + vae + cached empty-prompt embedding). No text encoder is kept resident after loading."

    def load(self, model_name, dtype):
        device = mm.get_torch_device()
        torch_dtype = torch.float16 if dtype == "fp16" else torch.float32

        cache_key = (model_name, dtype, str(device))
        if cache_key in _MODEL_CACHE:
            return (_MODEL_CACHE[cache_key],)

        safetensors_path = os.path.join(PBRFUSION4_MODEL_DIR, model_name)
        unet, vae, text_encoder, tokenizer = _load_pbrfusion4_components(
            safetensors_path, torch_dtype, device
        )

        empty_embed = _build_empty_prompt_embedding(text_encoder, tokenizer, device="cpu", dtype=torch_dtype)
        empty_embed = empty_embed.to(device=device, dtype=torch_dtype)
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

        # Text encoder / tokenizer are no longer needed: PBRFusion4 is only
        # ever conditioned on the empty string.
        del text_encoder
        del tokenizer
        mm.soft_empty_cache()

        model = {
            "unet": unet,
            "vae": vae,
            "empty_embed": empty_embed,
            "dtype": torch_dtype,
            "device": device,
            "vae_scaling_factor": vae.config.scaling_factor,
        }
        _MODEL_CACHE[cache_key] = model
        return (model,)


class MoonMeanChannels:
    """
    Test/diagnostic node: collapses an IMAGE's channels to their mean,
    then broadcasts back to 3 identical channels (keeps IMAGE type
    compatible with downstream nodes that expect RGB shape).

    Used to isolate whether inter-channel noise (e.g. in PBRFusion4's
    decoded_depth, which is nominally grayscale but has small real
    differences between R/G/B) is the source of artifacts that appear
    after channel-sensitive processing like frequency band extraction.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mean_image",)
    FUNCTION = "run"
    CATEGORY = "moon/debug"
    DESCRIPTION = "Collapses channels to their mean (true average, not luminance-weighted), broadcast back to 3 channels."

    def run(self, image):
        mean = image.mean(dim=-1, keepdim=True)
        return (mean.expand(-1, -1, -1, 3).contiguous(),)


NODE_CLASS_MAPPINGS = {
    "MoonPBRFusion4Loader": MoonPBRFusion4Loader,
    "MoonPBRFusion4Depth": MoonPBRFusion4Depth,
    "MoonMeanChannels": MoonMeanChannels,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonPBRFusion4Loader": "PBRFusion4 Loader",
    "MoonPBRFusion4Depth": "PBRFusion4 Depth",
    "MoonMeanChannels": "Mean Channel (debug)",
}