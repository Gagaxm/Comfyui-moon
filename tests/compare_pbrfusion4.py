"""
compare_pbrfusion4.py

Pixel-diff validation: runs the exact same image through
(A) the official PBRFusion4 pipeline (reference, via pipe(...) directly,
    bypassing infer_depth_pipe's mean()/normalize post-processing)
(B) MoonPBRFusion4Depth's manual re-implementation (unet + vae only)

Both paths share the SAME loaded weights (same pipe.unet / pipe.vae
instances) so any difference comes from the inference logic itself,
not from a loading discrepancy.

Usage:
    python compare_pbrfusion4.py path/to/test_image.png
"""

import sys
import numpy as np
import torch
from PIL import Image
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# --- adjust these imports to match your local paths ---
sys.path.insert(0, r"D:\ComfyUI")   # ComfyUI root -> make "comfy" importable
sys.path.insert(0, r"D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-PBRFusion4") # nom exact de ton dossier custom_nodes
sys.path.insert(0, r"D:\regis\Comfyui repos\ComfyUI-moon")

from pbrfusion4_simple_nodes import _load_from_single_safetensors, get_device
from nodes_pbrfusion4 import _build_empty_prompt_embedding
# --------------------------------------------------------

SAFETENSORS_PATH = r"D:\ComfyUI\ComfyUI\models\pbrfusion4\PBRFusion4.safetensors"
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "test_image.png"
DTYPE = torch.float16


def load_image_tensor(path, device, dtype):
    """Mirrors infer_depth_pipe's preprocessing exactly, optimize=False (no resize)."""
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32)
    img = img.astype(np.float16)  # matches official node's hardcoded cast
    t = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    t = t / 127.5 - 1.0
    return t.to(device=device, dtype=dtype), img.shape[:2]


def reference_pass(pipe, rgb_in, device):
    """Same pre/inference call as infer_depth_pipe(), stopped BEFORE
    the depth-specific mean()/normalize post-processing."""
    task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(device)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

    with torch.autocast(device.type):
        pred = pipe(
            rgb_in=rgb_in,
            prompt='',
            num_inference_steps=1,
            output_type='np',
            timesteps=[999],
            task_emb=task_emb,
            processing_res=0, # 0 = no resize, same as MoonPBRFusion4Depth
        ).images[0]  # (H, W, 3), already [0,1]-clamped by image_processor.postprocess

    return pred


def moon_pass(unet, vae, empty_embed, rgb_in, device, dtype):
    """Same logic as MoonPBRFusion4Depth.infer(), clamped here for fair comparison."""
    scaling_factor = vae.config.scaling_factor
    task = torch.tensor([1, 0], device=device, dtype=dtype).unsqueeze(0)
    task_emb = torch.cat([torch.sin(task), torch.cos(task)], dim=-1)

    with torch.no_grad():
        rgb_latents = vae.encode(rgb_in).latent_dist.sample()
        rgb_latents = rgb_latents * scaling_factor

        batch_size = rgb_latents.shape[0]
        prompt_embeds = empty_embed.expand(batch_size, -1, -1)
        task_emb = task_emb.expand(batch_size, -1)

        timestep = torch.tensor([999], device=device).long()
        pred_latents = unet(
            rgb_latents, timestep,
            encoder_hidden_states=prompt_embeds,
            class_labels=task_emb,
            return_dict=False,
        )[0]

        decoded = vae.decode(pred_latents / scaling_factor, return_dict=False)[0]

    output = (decoded / 2.0 + 0.5).clamp(0, 1)  # clamp added ONLY for this comparison
    return output.permute(0, 2, 3, 1).float().cpu().numpy()[0]


def main():
    device = get_device()
    pipe = _load_from_single_safetensors(SAFETENSORS_PATH, DTYPE, device)

    # --- fix VRAM temporaire pour le pass référence, pas présent dans le node officiel ---
    from diffusers.models.attention_processor import AttnProcessor2_0
    pipe.unet.set_attn_processor(AttnProcessor2_0())
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    # 

    rgb_in, orig_shape = load_image_tensor(IMAGE_PATH, device, DTYPE)

    ref = reference_pass(pipe, rgb_in, device)

    empty_embed = _build_empty_prompt_embedding(pipe.text_encoder, pipe.tokenizer, device, DTYPE)
    moon = moon_pass(pipe.unet, pipe.vae, empty_embed, rgb_in, device, DTYPE)

    diff = np.abs(ref.astype(np.float32) - moon.astype(np.float32))

    print(f"Image size compared: {ref.shape}")
    print(f"Max abs diff:  {diff.max():.6f}")
    print(f"Mean abs diff: {diff.mean():.6f}")
    print(f"MSE:           {np.mean(diff ** 2):.8f}")
    for c, name in enumerate(("R", "G", "B")):
        print(f"  channel {name} - max: {diff[..., c].max():.6f}  mean: {diff[..., c].mean():.6f}")

    Image.fromarray((ref * 255).astype(np.uint8)).save(SCRIPT_DIR / "compare_reference.png")
    Image.fromarray((moon * 255).astype(np.uint8)).save(SCRIPT_DIR / "compare_moon.png")
    diff_vis = (diff / (diff.max() + 1e-8) * 255).astype(np.uint8)
    Image.fromarray(diff_vis).save(SCRIPT_DIR / "compare_diff_heatmap.png")
    print("\nSaved: compare_reference.png, compare_moon.png, compare_diff_heatmap.png")


if __name__ == "__main__":
    main()