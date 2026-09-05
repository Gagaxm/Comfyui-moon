"""
Included nodes:
/image/ Image Blur
/tiling/ Periodic+Smooth Decomposition (Moisan)
/image/ Split RGB and Alpha
/tiling/ Circular Pad (wrap, for tiling)
/tiling/ Circular Unpad (crop back)
/io/ Publish Image
/io/ Previous Render Buffer
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
                "blur_type": (["Gaussian", "Box", "Radial"], {
                    "default": "Gaussian",
                    "tooltip": "'Gaussian'/'Box': separable 2-pass blur (samples = ceil(radius), sigma = radius/2). 'Radial': rotational sampling around the image center (12 samples per side) — creates a motion-blur-like sweep instead of a uniform blur."
                }),
                "radius": ("FLOAT", {
                    "default": 20.0, "min": 0.0, "max": 512.0, "step": 0.5,
                    "tooltip": "Blur strength in pixels. For 'Gaussian'/'Box', this sets the kernel size/sigma. For 'Radial', it scales the angular sweep step, not a pixel distance."
                }),
                "wrap_mode": (["replicate", "circular"], {
                    "default": "replicate",
                    "tooltip": "Edge handling for 'Gaussian'/'Box' only (ignored by 'Radial', which always samples with a border clamp). 'replicate' matches the original shader's edge behavior. 'circular' makes the blur seamless on its own, without an external CircularPad/CircularUnpad sandwich."
                }),
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
                "clamp_periodic": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clamp the 'periodic' output to [0, 1] after decomposition. Turn off only if you specifically want to inspect/use out-of-range values (e.g. for further float processing)."
                }),
                "renormalize_periodic": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Rescale 'periodic' back to the original image's min/max range per-image, instead of clamping. Only useful if extreme border gradients pushed 'periodic' noticeably outside [0, 1] and you'd rather rescale than clip. Off by default."
                }),
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
                "pad_x": ("INT", {
                    "default": 32, "min": 0, "max": 4096,
                    "tooltip": "Horizontal wrap-padding in pixels. Should be >= the radius of the filter you're sandwiching (e.g. >= blur radius), or the filter will still see replicated/reflected pixels near the border."
                }),
                "pad_y": ("INT", {
                    "default": 32, "min": 0, "max": 4096,
                    "tooltip": "Vertical wrap-padding in pixels. Should be >= the radius of the filter you're sandwiching (e.g. >= blur radius), or the filter will still see replicated/reflected pixels near the border."
                }),
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
                "pad_x": ("INT", {
                    "default": 32, "min": 0, "max": 4096,
                    "tooltip": "Horizontal crop amount, in pixels. Must match the pad_x used in the matching CircularPad — wire CircularPad's pad_x output directly here instead of typing it twice."
                }),
                "pad_y": ("INT", {
                    "default": 32, "min": 0, "max": 4096,
                    "tooltip": "Vertical crop amount, in pixels. Must match the pad_y used in the matching CircularPad — wire CircularPad's pad_y output directly here instead of typing it twice."
                }),
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
                "dest_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute folder path to write the PNG(s) to. Created automatically if it doesn't exist. Publication is skipped (with a console message) if left empty."
                }),
                "dest_filename": ("STRING", {
                    "default": "publish",
                    "tooltip": "Base filename (extension ignored/replaced with .png). Overwritten on every run. For a batch >1, each image gets a numeric suffix (_00, _01, ...)."
                }),
                "active": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Toggle off to disable publishing without disconnecting the node from the graph."
                }),
            }
        }
 
    RETURN_TYPES = ()
    FUNCTION = "publish"
    OUTPUT_NODE = True
    CATEGORY = "moon/io"
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
_BUFFER_KEEP_STATE: dict[str, bool] = {}  # key -> keep flag used on the previous run
_BUFFER_FROZEN: dict[str, torch.Tensor] = {}  # key -> frozen image (when keep was first activated)
 
 
class MoonPreviousRenderBuffer:
    """
    Returns the image (or batch) stored from the PREVIOUS execution, then
    overwrites the buffer with the current one for the NEXT execution.
    In-memory only (RAM, not VRAM, no disk I/O), similar in spirit to how
    feedback-loop workflows carry state between runs.
 
    Shape mismatch (different batch size, resolution, or channel count)
    disables the comparison for this run only: `previous_image` falls
    back to `new_image` and `comparison_valid` is False. This is a shape
    check only, not a guarantee that the image is from a genuinely
    different source or that the upstream workflow ran correctly. The
    stored buffer is left untouched on mismatch, so a one-off shape
    change doesn't wipe your comparison history.
 
    Explicit keys are a deliberate sharing mechanism: nodes using the
    same key intentionally read/write the same buffer, with no ownership
    arbitration. Leave `key` blank for automatic per-node scoping.
 
    `keep` freezes the buffer: on the run it flips OFF->ON, the buffer
    commits current_image one last time, then stays frozen (replaying
    that same previous_image) until flipped back OFF, at which point
    normal per-run replacement resumes. `current_image` is always a
    straight bypass, independent of `keep`.
 
    Buffers are process-lifetime and never expire on their own -- use
    MoonClearRenderBuffer to free them manually if needed.
    """
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "new_image": ("IMAGE",),
            },
            "optional": {
                "key": ("STRING", {
                    "default": "",
                    "tooltip": "Buffer key to compare against. Leave blank to auto-scope by this node's own id (recommended, avoids collisions). Set explicitly if you need several nodes to share the same buffer -- sharing is intentional, there's no conflict protection."
                }),
                "keep": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Freeze the buffer. When ON, the buffer keeps the image from the first run where keep was activated, and stays frozen until turned back OFF."
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("current_image", "previous_image", "comparison_valid")
    FUNCTION = "run"
    CATEGORY = "moon/io"
 
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-run: this node's entire purpose is to reflect the
        # previous execution, so ComfyUI's upstream-cache-hit shortcut
        # (which would skip run() and leave the buffer stale) must be
        # bypassed unconditionally.
        return float("nan")
 
    @staticmethod
    def _to_owned_cpu(image):
        """
        Detach and move to CPU, guaranteeing an independent tensor
        (no shared storage with `image`), regardless of its input device.
 
        .cpu() only copies when the source isn't already on CPU -- on an
        already-CPU input it returns the same object, storage included.
        So a GPU input gets independence for free via detach().cpu();
        a CPU input needs one extra clone() to avoid aliasing the
        bypassed `current_image` output.
        """
        was_cpu = image.device.type == "cpu"
        owned = image.detach().cpu()
        if was_cpu:
            owned = owned.clone()
        return owned
 
    def run(self, new_image, unique_id, key="", keep=False):
        effective_key = key.strip() or f"node_{unique_id}"

        new_cpu = self._to_owned_cpu(new_image)
        was_keeping = _BUFFER_KEEP_STATE.get(effective_key, False)
        is_frozen = _BUFFER_FROZEN.get(effective_key) is not None

        # --- Gestion du gel ---
        if keep and not was_keeping:
            # Premier passage à keep=True : on gèle l'image courante
            _BUFFER_FROZEN[effective_key] = new_cpu.clone()
        elif not keep and was_keeping:
            # keep repasse à False : on dé-gèle et on nettoie
            _BUFFER_FROZEN.pop(effective_key, None)

        # --- Récupération du previous_image ---
        if is_frozen:
            # Buffer gelé : on utilise toujours l'image gelée
            previous = _BUFFER_FROZEN[effective_key].clone()
            comparison_valid = (_BUFFER_FROZEN[effective_key].shape == new_cpu.shape)
        else:
            # Fonctionnement normal
            stored = _BUFFER_STORE.get(effective_key)
            if stored is not None and stored.shape == new_cpu.shape:
                previous = stored.clone()
                comparison_valid = True
            else:
                previous = new_cpu.clone()
                comparison_valid = False

            # Mise à jour du buffer normal
            if comparison_valid or stored is None:
                _BUFFER_STORE[effective_key] = new_cpu

        # Mise à jour de l'état keep
        _BUFFER_KEEP_STATE[effective_key] = keep

        return (new_image, previous, comparison_valid)
 
 
class MoonClearRenderBuffer:
    """
    Utility node to free buffers held by MoonPreviousRenderBuffer.
 
    Buffers are process-lifetime and never expire on their own, so if you
    accumulate many keys (e.g. after renaming nodes or iterating on a
    graph) at 4K they can add up in RAM. Wire this in and run it once to
    clear a specific key, or leave `key` blank to clear everything.
 
    This node is a manual, explicit action -- it does not run
    automatically and does not track which keys are "stale"; that
    judgment is left to you.
    """
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("IMAGE",),
            },
            "optional": {
                "key": ("STRING", {
                    "default": "",
                    "tooltip": "Buffer key to clear. Leave blank to clear ALL stored buffers (every key, every node)."
                }),
            },
        }
 
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("trigger",)
    FUNCTION = "run"
    CATEGORY = "moon/io"
 
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
 
    def run(self, trigger, key=""):
        effective_key = key.strip()
 
        if effective_key:
            removed = _BUFFER_STORE.pop(effective_key, None) is not None
            _BUFFER_KEEP_STATE.pop(effective_key, None)
            print(
                f"[MoonClearRenderBuffer] Cleared key '{effective_key}' "
                f"({'was set' if removed else 'was already empty'})."
            )
        else:
            count = len(_BUFFER_STORE)
            _BUFFER_STORE.clear()
            _BUFFER_KEEP_STATE.clear()
            print(f"[MoonClearRenderBuffer] Cleared all {count} buffer(s).")
 
        return (trigger,)


NODE_CLASS_MAPPINGS = {
    "MoonImageBlur": MoonImageBlur,
    "PeriodicSmoothDecomposition": PeriodicSmoothDecomposition,
    "ImageSplitRGBAndAlpha": ImageSplitRGBAndAlpha,
    "CircularPad": CircularPad,
    "CircularUnpad": CircularUnpad,
    "PublishImage": PublishImage,
    "MoonPreviousRenderBuffer": MoonPreviousRenderBuffer,
    "MoonClearRenderBuffer": MoonClearRenderBuffer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonImageBlur": "Image Blur",
    "PeriodicSmoothDecomposition": "Periodic+Smooth Decomposition (Moisan)",
    "ImageSplitRGBAndAlpha": "Split RGB and Alpha",
    "CircularPad": "Circular Pad (wrap, for tiling)",
    "CircularUnpad": "Circular Unpad (crop back)",
    "PublishImage": "Publish Image",
    "MoonPreviousRenderBuffer": "Previous Render Buffer",
    "MoonClearRenderBuffer": "Clear Render Buffer",
}