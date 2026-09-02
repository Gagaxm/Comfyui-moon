"""
Included nodes:
/image/ Image Blur
/tiling/ Periodic+Smooth Decomposition (Moisan)
/image/ Split RGB and Alpha
/tiling/ Circular Pad (wrap, for tiling)
/tiling/ Circular Unpad (crop back)
/image/ Publish Image
/image/ Previous Render Buffer
"""

import math
import torch
import os
import numpy as np
from PIL import Image
import torch.nn.functional as F

def _nhwc_to_nchw(img):
    return img.permute(0, 3, 1, 2).contiguous()
 
 
def _nchw_to_nhwc(img):
    return img.permute(0, 2, 3, 1).contiguous()
 
 
def _build_1d_kernel(radius, blur_type, device, dtype):
    samples = int(math.ceil(max(radius, 0.0)))
    if samples <= 0:
        return None
    sigma = radius / 2.0
    idx = torch.arange(-samples, samples + 1, device=device, dtype=dtype)
    if blur_type == "Gaussian":
        weights = torch.exp(-(idx * idx) / (2.0 * sigma * sigma))
    else:  # Box
        weights = torch.ones_like(idx)
    weights = weights / weights.sum()
    return weights
 
 
def _separable_blur(img_nchw, kernel_1d, wrap_mode):
    C = img_nchw.shape[1]
    pad = (kernel_1d.shape[0] - 1) // 2
    pad_mode = "circular" if wrap_mode == "circular" else "replicate"
 
    # horizontal pass
    kh = kernel_1d.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    x = F.pad(img_nchw, (pad, pad, 0, 0), mode=pad_mode)
    x = F.conv2d(x, kh, groups=C)
 
    # vertical pass
    kv = kernel_1d.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    x = F.pad(x, (0, 0, pad, pad), mode=pad_mode)
    x = F.conv2d(x, kv, groups=C)
    return x
 
 
def _radial_blur(img_nchw, radius):
    B, C, H, W = img_nchw.shape
    device, dtype = img_nchw.device, img_nchw.dtype
 
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, H, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, W, device=device, dtype=dtype),
        indexing="ij",
    )
    dx = xx - 0.5
    dy = yy - 0.5
    dist = torch.sqrt(dx * dx + dy * dy)
    dist_safe = torch.clamp(dist, min=1e-4)
    dirx = dx / dist_safe
    diry = dy / dist_safe
 
    RADIAL_SAMPLES = 12
    RADIAL_STRENGTH = 0.0003
    angle_step = radius * RADIAL_STRENGTH
    neg_angle = -RADIAL_SAMPLES * angle_step
    cos_na, sin_na = math.cos(neg_angle), math.sin(neg_angle)
    rotx = dirx * cos_na - diry * sin_na
    roty = dirx * sin_na + diry * cos_na
    cos_step, sin_step = math.cos(angle_step), math.sin(angle_step)
 
    acc = torch.zeros_like(img_nchw)
    total_w = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
 
    for i in range(-RADIAL_SAMPLES, RADIAL_SAMPLES + 1):
        u = 0.5 + rotx * dist
        v = 0.5 + roty * dist
        grid = torch.stack([u * 2.0 - 1.0, v * 2.0 - 1.0], dim=-1)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        sampled = F.grid_sample(
            img_nchw, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        w = 1.0 - abs(i) / RADIAL_SAMPLES
        acc += sampled * w
        total_w += w
        rotx, roty = rotx * cos_step - roty * sin_step, rotx * sin_step + roty * cos_step
 
    out = acc / total_w.clamp(min=0.001)
    # pass again (dist < 1e-4): central pixel unchanged"
    center_mask = (dist < 1e-4).view(1, 1, H, W)
    out = torch.where(center_mask, img_nchw, out)
    return out
 
 
class MoonImageBlur:
    """MoonImageBlur — Python/torch implementation of the 'Image Blur' algorithm.

    Accurately replicates three modes:
    - Gaussian: Standard Gaussian blur
    - Box: Separable 2-pass box blur
    - Radial: Rotational sampling around the center (12 samples per side)

    Implementation details:
    - Sample count: ceil(radius), sigma = radius / 2
    - Radial mode: 12 samples per side, angular step = radius * 0.0003
    - Edge handling: Default behavior matches 'replicate' (clamping to edge values)
    - Optional 'circular' mode avoids needing external CircularPad/Unpad operations
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_type": (["Gaussian", "Box", "Radial"], {"default": "Gaussian"}),
                "radius": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 512.0, "step": 0.5}),
                "wrap_mode": (["replicate", "circular"], {"default": "replicate"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blur"
    CATEGORY = "moon/image"
    DESCRIPTION = "Blurs an image using Gaussian, Box, or Radial."

    def blur(self, image, blur_type, radius, wrap_mode):
        img_nchw = _nhwc_to_nchw(image)
 
        if blur_type == "Radial":
            out = _radial_blur(img_nchw, radius)
        else:
            kernel = _build_1d_kernel(radius, blur_type, img_nchw.device, img_nchw.dtype)
            if kernel is None:
                out = img_nchw
            else:
                out = _separable_blur(img_nchw, kernel, wrap_mode)
 
        return (_nchw_to_nhwc(out),)


class PeriodicSmoothDecomposition:
    """
    Decomposes an image into a seamlessly-tileable periodic component
    and a smooth low-frequency component (Moisan, 2011).

    Periodic + Smooth image decomposition for ComfyUI.

Implements: L. Moisan, "Periodic Plus Smooth Image Decomposition",
Journal of Mathematical Imaging and Vision 39(2), 161-179, 2011.

Splits an image u into:
  - a periodic component p, which tiles seamlessly (same content/detail
    as the original, but the border discontinuity is removed)
  - a smooth component s, a very low-frequency image whose only job is
    to absorb the value/gradient jump between opposite borders (u = p + s)

This is a single closed-form pass in Fourier space (no iteration, no
model, deterministic), so it's cheap even at 4K compared to any
render-at-2x-then-crop approach.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "clamp_periodic": ("BOOLEAN", {"default": True}),
                "renormalize_periodic": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("periodic", "smooth")
    FUNCTION = "decompose"
    CATEGORY = "moon/tiling"
    DESCRIPTION = (
        "Periodic+smooth decomposition (Moisan 2011). 'periodic' tiles "
        "seamlessly on both X and Y; 'smooth' is the low-frequency "
        "correction that was removed (useful for debugging/visualizing "
        "the border gradient that was absorbed)."
    )

    def decompose(self, image: torch.Tensor, clamp_periodic: bool, renormalize_periodic: bool):
        # ComfyUI IMAGE tensors are (B, H, W, C) float32 in [0, 1]
        device = image.device
        compute_dtype = torch.float64  # precision matters right at the borders

        u = image.to(compute_dtype)
        B, H, W, C = u.shape

        # Work in (B, C, H, W) so fft2 operates on the last two dims per image/channel
        u = u.permute(0, 3, 1, 2).contiguous()

        v = self._boundary_jump(u)
        v_fft = torch.fft.fft2(v)
        s_fft = self._solve_smooth(v_fft, H, W, device, compute_dtype)
        s = torch.fft.ifft2(s_fft).real
        p = u - s

        p = p.permute(0, 2, 3, 1)
        s = s.permute(0, 2, 3, 1)

        if renormalize_periodic:
            # Rescale p back to the original image's min/max range per-image.
            # Off by default: only useful if extreme border gradients pushed
            # p noticeably outside [0, 1] and you'd rather rescale than clip.
            p_min = p.amin(dim=(1, 2, 3), keepdim=True)
            p_max = p.amax(dim=(1, 2, 3), keepdim=True)
            u_min = u.permute(0, 2, 3, 1).amin(dim=(1, 2, 3), keepdim=True)
            u_max = u.permute(0, 2, 3, 1).amax(dim=(1, 2, 3), keepdim=True)
            scale = (u_max - u_min) / (p_max - p_min).clamp_min(1e-8)
            p = (p - p_min) * scale + u_min

        if clamp_periodic:
            p = p.clamp(0.0, 1.0)

        return (p.to(torch.float32), s.to(torch.float32))

    @staticmethod
    def _boundary_jump(u: torch.Tensor) -> torch.Tensor:
        """Builds v, the image encoding only the border discontinuities of u.
        u: (B, C, H, W)
        """
        v = torch.zeros_like(u)
        v[:, :, 0, :] = u[:, :, -1, :] - u[:, :, 0, :]
        v[:, :, -1, :] = u[:, :, 0, :] - u[:, :, -1, :]
        v[:, :, :, 0] += u[:, :, :, -1] - u[:, :, :, 0]
        v[:, :, :, -1] += u[:, :, :, 0] - u[:, :, :, -1]
        return v

    @staticmethod
    def _solve_smooth(v_fft: torch.Tensor, H: int, W: int, device, dtype) -> torch.Tensor:
        """Closed-form solution for the smooth component in Fourier space."""
        q = torch.arange(H, device=device, dtype=dtype).reshape(H, 1)
        r = torch.arange(W, device=device, dtype=dtype).reshape(1, W)
        denom = (
            2 * torch.cos(2 * math.pi * q / H)
            + 2 * torch.cos(2 * math.pi * r / W)
            - 4
        ).reshape(1, 1, H, W)

        denom_safe = denom.clone()
        denom_safe[0, 0, 0, 0] = 1.0  # placeholder, DC term is zeroed out below

        s_fft = v_fft / denom_safe
        s_fft[:, :, 0, 0] = 0
        return s_fft


class CircularPad:
    """
    Pads an image by wrapping content from the opposite edge (circular
    padding), so that any downstream filter (blur, sharpen, any
    convolution-based node) sees correct neighboring pixels near the
    border instead of replicated/reflected ones. Use together with
    CircularUnpad, wrapped around the filter, to keep a seamless
    texture seamless through the filter.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pad_x": ("INT", {"default": 32, "min": 0, "max": 4096}),
                "pad_y": ("INT", {"default": 32, "min": 0, "max": 4096}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("padded_image", "pad_x", "pad_y")
    FUNCTION = "pad"
    CATEGORY = "moon/tiling"
    DESCRIPTION = (
        "Wrap-pads the image (uses the opposite edge as context) so a "
        "downstream filter doesn't break tiling. pad_x/pad_y should be "
        ">= your filter's radius. Outputs pad_x/pad_y again so you can "
        "wire them straight into CircularUnpad."
    )

    def pad(self, image: torch.Tensor, pad_x: int, pad_y: int):
        t = image.permute(0, 3, 1, 2)
        if pad_x > 0 or pad_y > 0:
            t = torch.nn.functional.pad(
                t, (pad_x, pad_x, pad_y, pad_y), mode="circular"
            )
        return (t.permute(0, 2, 3, 1), pad_x, pad_y)


class CircularUnpad:
    """
    Crops back to the original size after CircularPad + a filter.
    Use the same pad_x/pad_y as the matching CircularPad (wire its
    outputs directly in).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pad_x": ("INT", {"default": 32, "min": 0, "max": 4096}),
                "pad_y": ("INT", {"default": 32, "min": 0, "max": 4096}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "unpad"
    CATEGORY = "moon/tiling"
    DESCRIPTION =(
        "Wrap-pads the image (uses the opposite edge as context) so a "
        "downstream filter doesn't break tiling. pad_x/pad_y should be "
        ">= your filter's radius. Use CircularPad first."
        )

    def unpad(self, image: torch.Tensor, pad_x: int, pad_y: int):
        B, H, W, C = image.shape
        y0, y1 = pad_y, H - pad_y
        x0, x1 = pad_x, W - pad_x
        return (image[:, y0:y1, x0:x1, :],)


class ImageSplitRGBAndAlpha:
    """
    Extracts the RGB channels as a ComfyUI IMAGE and the Alpha channel 
    as a proper ComfyUI 3D MASK tensor [B, H, W].
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("RGB", "Alpha")
    FUNCTION = "split_channels"
    CATEGORY = "moon/image"
    DESCRIPTION = "Splits an RGBA image into a separate RGB image and a proper ComfyUI alpha mask."

    def split_channels(self, image):
        # ComfyUI image format is [B, H, W, C]
        # If the image lacks an alpha channel, generate a solid white mask
        if image.shape[-1] < 4:
            mask = torch.ones((image.shape[0], image.shape[1], image.shape[2]), dtype=torch.float32, device=image.device)
            return (image, mask)
        
        # 1. Extract RGB channels and discard the original alpha channel
        rgb_image = image[:, :, :, :3]
        
        # 2. Extract Alpha channel as a native ComfyUI 3D Mask [B, H, W]
        alpha_mask = image[:, :, :, 3]
        
        return (rgb_image, alpha_mask)

class PublishImage:
    """
    Publie une image (PNG 8-bit, RGB ou RGBA) vers un chemin fixe,
    en parallèle d'un SaveImageAdvanced. Écrase le fichier existant
    à chaque run (utile pour un fichier surveillé par Maya).
    """
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "dest_folder": ("STRING", {"default": ""}),
                "dest_filename": ("STRING", {"default": "publish"}),
                "active": ("BOOLEAN", {"default": True}),
            }
        }
 
    RETURN_TYPES = ()
    FUNCTION = "publish"
    OUTPUT_NODE = True
    CATEGORY = "moon/image"
    DESCRIPTION = "Save batch in PNG 8-bit to a fixed name (overwrite), independently of SaveImageAdvanced."
 
    def publish(self, images, dest_folder, dest_filename, active):
        if not active:
            return {}
 
        if not dest_folder:
            print("[PublishImage] dest_folder vide, publication ignorée.")
            return {}
 
        os.makedirs(dest_folder, exist_ok=True)
 
        base_name = os.path.splitext(dest_filename)[0]
        batch_size = images.shape[0]
 
        for i in range(batch_size):
            arr = images[i].cpu().numpy()
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
 
            channels = arr.shape[-1]
            if channels == 4:
                mode = "RGBA"
            elif channels == 3:
                mode = "RGB"
            elif channels == 1:
                mode = "L"
                arr = arr[..., 0]
            else:
                print(f"[PublishImage] Nombre de canaux inattendu ({channels}), image ignorée.")
                continue
 
            img = Image.fromarray(arr, mode=mode)
 
            suffix = f"_{i:02d}" if batch_size > 1 else ""
            fname = f"{base_name}{suffix}.png"
            fpath = os.path.join(dest_folder, fname)
 
            img.save(fpath, format="PNG")
            print(f"[PublishImage] Écrit : {fpath}")
 
        return {}


_BUFFER_STORE: dict[str, torch.Tensor] = {}
_BUFFER_OWNERS: dict[str, str] = {}  # key -> owning node's unique_id


class MoonPreviousRenderBuffer:
    """
    Returns the image (or batch) stored from the PREVIOUS execution, then
    overwrites the buffer with the current one for the NEXT execution.
    No disk I/O -- pure in-memory RAM buffer.

    Module-level, in-memory buffer store (RAM, not VRAM). Persists across
    queued executions for the lifetime of the ComfyUI server process
    (resets on restart). Same technique used by feedback-loop workflows
    (e.g. AnimateDiff) to carry state between runs.

    If the incoming batch shape doesn't match the stored buffer (different
    batch size, resolution, or channel count -- i.e. a genuinely different
    source), the comparison is auto-disabled for this run: `previous_image`
    falls back to `new_image` (a no-op diff) and `comparison_valid` is
    False, so downstream nodes can react. The last valid buffer is kept
    untouched rather than overwritten by the mismatched batch, so a
    transient change (e.g. temporarily testing with 1 image instead of
    a batch of 4) doesn't permanently wipe your comparison history.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "new_image": ("IMAGE",),
            },
            "optional": {
                # Leave blank to auto-scope by node id (recommended).
                "key": ("STRING", {"default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("current_image", "previous_image", "comparison_valid")
    FUNCTION = "run"
    CATEGORY = "moon/io"

    def run(self, new_image, unique_id, key=""):
               
        effective_key = key.strip() or f"node_{unique_id}"

        # Soft fallback on key collision: never break the whole run over
        # a naming conflict, just fall back to a per-node buffer.
        owner = _BUFFER_OWNERS.get(effective_key)
        if owner is not None and owner != unique_id:
            print(
                f"[MoonPreviousRenderBuffer] Key '{effective_key}' already "
                f"used by node {owner}; falling back to per-node buffer "
                f"for node {unique_id}."
            )
            effective_key = f"node_{unique_id}"
        _BUFFER_OWNERS[effective_key] = unique_id

        stored = _BUFFER_STORE.get(effective_key)
        new_cpu = new_image.detach().cpu()

        if stored is not None and stored.shape == new_cpu.shape:
            # Same source shape: genuine comparison.
            previous = stored.clone()
            comparison_valid = True
        else:
            # No buffer yet, or shape mismatch (different batch size,
            # resolution, ...): disable the comparison for this run
            # instead of faking one. Keep the existing buffer (if any)
            # untouched -- don't let a one-off mismatch erase history.
            previous = new_cpu.clone()
            comparison_valid = False

        # Only commit to the buffer on a clean run, OR if there was
        # nothing stored yet (first run ever for this key).
        if comparison_valid or stored is None:
            _BUFFER_STORE[effective_key] = new_cpu.clone()

        # Bypass output stays on the input's original device -- no
        # reason to force it to CPU, unlike the buffered copy.
        return (new_image, previous, comparison_valid)


NODE_CLASS_MAPPINGS = {
    "MoonImageBlur": MoonImageBlur,
    "PeriodicSmoothDecomposition": PeriodicSmoothDecomposition,
    "ImageSplitRGBAndAlpha": ImageSplitRGBAndAlpha,
    "CircularPad": CircularPad,
    "CircularUnpad": CircularUnpad,
    "PublishImage": PublishImage,
    "MoonPreviousRenderBuffer": MoonPreviousRenderBuffer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonImageBlur": "Image Blur",
    "PeriodicSmoothDecomposition": "Periodic+Smooth Decomposition (Moisan)",
    "ImageSplitRGBAndAlpha": "Split RGB and Alpha",
    "CircularPad": "Circular Pad (wrap, for tiling)",
    "CircularUnpad": "Circular Unpad (crop back)",
    "PublishImage": "Publish Image",
    "MoonPreviousRenderBuffer": "Previous Render Buffer",
}