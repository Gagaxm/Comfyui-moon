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
"""


import json
import os
import tempfile
import torch
import comfy.model_management as mm
import folder_paths
from diffusers.models.attention_processor import AttnProcessor2_0



def _task_embedding(device, dtype):
    """Depth task embedding, per the original Lotus task conditioning scheme."""
    task = torch.tensor([1, 0], device=device, dtype=dtype).unsqueeze(0)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


class MoonPBRFusion4Depth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pbrfusion4_model": ("PBRFUSION4_MODEL",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("raw_depth",)
    FUNCTION = "infer"
    CATEGORY = "moon/pbr/pbrfusion4"
    DESCRIPTION = "Single-step PBRFusion4 discriminative depth inference. No resize, no normalization -- raw VAE-decoded output."

    def infer(self, pbrfusion4_model, image):
        unet = pbrfusion4_model["unet"]
        vae = pbrfusion4_model["vae"]
        empty_embed = pbrfusion4_model["empty_embed"]
        dtype = pbrfusion4_model["dtype"]
        device = pbrfusion4_model["device"]
        scaling_factor = pbrfusion4_model["vae_scaling_factor"]

        # ComfyUI IMAGE: (B, H, W, C) in [0, 1] -> Lotus convention: (B, C, H, W) in [-1, 1]
        rgb = image.to(device=device, dtype=dtype).permute(0, 3, 1, 2)
        rgb = rgb * 2.0 - 1.0

        with torch.inference_mode():
            rgb_latents = vae.encode(rgb).latent_dist.sample()
            rgb_latents = rgb_latents * scaling_factor

            timestep = torch.tensor([999], device=device).long()

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

            decoded = vae.decode(pred_latents / scaling_factor, return_dict=False)[0]

        # Fixed VAE convention remap only -- no clamp, no data-dependent stretch.
        output = (decoded / 2.0 + 0.5).permute(0, 2, 3, 1).float().cpu()
        # Cleanup: free VRAM from the intermediate latents and input image.
        del rgb, rgb_latents, pred_latents
        mm.soft_empty_cache()
        return (output,)




"""
PBRFusion4 Loader

Loads the PBRFusion4 discriminative depth model (Lotus-D architecture) from
a single .safetensors package (unet + vae + text_encoder, prefixed and
embedded as safetensors metadata).

This is a minimal, from-scratch re-implementation of the loading logic found
in Night1099/COMFYUI-PBRFusion4 (pbrfusion4_simple_nodes.py ->
_load_from_single_safetensors). We do NOT vendor the Lotus/ directory or its
pipeline.py: MoonPBRFusion4Depth calls unet/vae directly, reproducing only
the exact operations confirmed in EnVision-Research/Lotus's LotusDPipeline
(no scheduler, no concat -- see MoonPBRFusion4Depth for details).

Key difference vs. the upstream node: the CLIP text encoder and tokenizer
are only used once, to compute the empty-prompt embedding ("" conditioning,
the only prompt PBRFusion4 is ever run with). That embedding is cached and
the text encoder/tokenizer are freed immediately after, instead of being
kept resident in VRAM for every inference call.
"""

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
    tmpdir = tempfile.mkdtemp(prefix="pbrfusion4_tokenizer_")
    for key, content in tokenizer_files.items():
        with open(os.path.join(tmpdir, key.split("/", 1)[1]), "w", encoding="utf-8") as fh:
            fh.write(content)
    tokenizer = CLIPTokenizer.from_pretrained(tmpdir)

    unet.to(dtype=dtype, device=device)
    vae.to(dtype=dtype, device=device)
    # The AttnProcessor2_0 is used to reduce VRAM usage during inference, especially for large UNet models.
    unet.set_attn_processor(AttnProcessor2_0())
    vae.enable_slicing()
    vae.enable_tiling()

    text_encoder.to(dtype=dtype, device=device)

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
    CATEGORY = "moon/pbr/pbrfusion4"
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

        empty_embed = _build_empty_prompt_embedding(text_encoder, tokenizer, device, torch_dtype)

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


NODE_CLASS_MAPPINGS = {
    "MoonPBRFusion4Loader": MoonPBRFusion4Loader,
    "MoonPBRFusion4Depth": MoonPBRFusion4Depth,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonPBRFusion4Loader": "PBRFusion4 Loader",
    "MoonPBRFusion4Depth": "PBRFusion4 Depth",
}
