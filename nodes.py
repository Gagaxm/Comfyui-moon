"""
PeriodicSmoothDecomposition
ImageSplitRGBAndAlpha
CircularPad
CircularUnpad
PublishImage
"""

import math
import torch
import os
import numpy as np
from PIL import Image



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

class ChannelMeanStats:
    """
    Computes the per-channel (R, G, B) mean pixel value of an image.
Built to check whether a normal map's R/G channels are centered
around 127.5 (the neutral "flat surface" value), or whether it
carries a directional bias (common with AI-generated normal maps
like DeepBump).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }
 
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("r_mean_255", "g_mean_255", "b_mean_255", "report")
    FUNCTION = "compute"
    CATEGORY = "moon/pbr"
 
    def compute(self, image, mask=None):
        # image: (B, H, W, C) float tensor, values in [0, 1]
        img = image
 
        if mask is not None:
            m = mask
            if m.dim() == 3:
                m = m.unsqueeze(-1)  # (B, H, W, 1)
            m = m.expand(-1, -1, -1, img.shape[-1])
            weighted_sum = (img * m).sum(dim=(0, 1, 2))
            weight_total = m.sum(dim=(0, 1, 2)).clamp(min=1e-6)
            means = weighted_sum / weight_total
        else:
            means = img.mean(dim=(0, 1, 2))
 
        means_255 = (means * 255.0).tolist()
        # pad in case of a grayscale (1-channel) input
        while len(means_255) < 3:
            means_255.append(0.0)
 
        r, g, b = means_255[0], means_255[1], means_255[2]
 
        deviation_r = r - 127.5
        deviation_g = g - 127.5
 
        report = (
            f"R mean: {r:.2f}  (offset from 127.5: {deviation_r:+.2f})\n"
            f"G mean: {g:.2f}  (offset from 127.5: {deviation_g:+.2f})\n"
            f"B mean: {b:.2f}"
        )
 
        return (r, g, b, report)


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
    DESCRIPTION = "Sauvegarde le batch en PNG 8-bit vers un nom fixe (overwrite), indépendamment de SaveImageAdvanced."
 
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


NODE_CLASS_MAPPINGS = {
    "PeriodicSmoothDecomposition": PeriodicSmoothDecomposition,
    "ChannelMeanStats": ChannelMeanStats,
    "ImageSplitRGBAndAlpha": ImageSplitRGBAndAlpha,
    "CircularPad": CircularPad,
    "CircularUnpad": CircularUnpad,
    "PublishImage": PublishImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PeriodicSmoothDecomposition": "Periodic+Smooth Decomposition (Moisan)",
    "ChannelMeanStats": "Channel Mean Stats (RGB)",
    "ImageSplitRGBAndAlpha": "Split RGB and Alpha",
    "CircularPad": "Circular Pad (wrap, for tiling)",
    "CircularUnpad": "Circular Unpad (crop back)",
    "PublishImage": "Publish Image",
}