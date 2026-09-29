"""
Included nodes:
/image/ Image Blur
/image/ Split RGB and Alpha
/image/ Exposure / Offset / Gamma
/image/ Mean Channel
/image/ Preview Crop (1:1 Pixel)
"""

import math
import torch
import torch.nn.functional as F
import comfy.model_management as model_management

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



class MoonExposureOffsetGamma:
    """
    Exposure / Offset / Gamma color correction, ported from the ComfyUI-moon
    native GLSLShader "Exposition" blueprint.

    Two modes are available via `color_space`:

    - "linear" (default): operates directly on the tensor values, exactly
      like the original node. This is the safe default for non-color data
      (heightmaps, masks, roughness/normal channels, etc.) where the values
      are NOT meant to be treated as gamma-encoded sRGB and where forcing a
      2.2 gamma round-trip would silently corrupt the data.

        color = src.rgb * pow(2.0, exposure)
        color = color + offset
        color = pow(max(color, 0.0), 1.0 / gamma)
        color = clamp(color, 0, 1)

    - "srgb": reproduces the behavior of professional exposure tools
      (Photoshop's Exposure dialog, GIMP/GEGL's gegl:exposure operation)
      on 8/16-bit display-referred images. Both operate in a linearized
      working space rather than the image's own gamma-encoded space.

      The Exposure + Offset step below is ported from GEGL's own
      "gegl:exposure" operation (GIMP's engine), not from a naive
      multiply-then-add. GEGL's key insight: Offset is not a plain additive
      shift, it is a *black-point remap that also renormalizes gain* so the
      (shifted) white point stays pinned at 1.0. This is what a raw
      "+ offset" followed by a clamp cannot do: it lifts shadows without
      ever clipping highlights, because the gain compensates automatically.

        linear = pow(src.rgb, 2.2)              // decode sRGB -> linear
        white  = pow(2.0, -exposure)
        gain   = 1.0 / max(white + offset, 1e-6)
        linear = (linear + offset) * gain       // black-point remap + gain
        linear = clamp(linear, 0, 1)
        linear = pow(max(linear, 1e-4), 1.0 / gamma)   // gamma, no extra round-trip
        color  = pow(linear, 1.0 / 2.2)         // re-encode linear -> sRGB
        color  = clamp(color, 0, 1)

      Gamma is kept as a fully separate, uncoupled power-law step, mirroring
      how GEGL itself keeps gamma correction as its own independent
      operation rather than fusing it into the exposure math.

      Use this mode ONLY when `image` genuinely represents a display-referred
      sRGB color (e.g. an albedo/base color pass), never on heightmaps,
      masks, or other linear/data channels.

    Note: This is a pointwise operation (no spatial neighborhood is sampled), so
    it is wrap-agnostic and has no wrap_mode input.
    """

    CATEGORY = "moon/image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"

    # sRGB gamma approximation used for the decode/encode round-trip.
    # (A simplified 2.2 power curve, not the piecewise sRGB transfer
    # function, which matches how Photoshop's own reverse-engineered
    # formula for this adjustment has been documented to behave.)
    _SRGB_GAMMA = 2.2

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Photoshop's Exposure dialog documents these exact ranges:
                # Exposure [-20, 20], Offset [-0.5, 0.5], Gamma [0.01, 9.99].
                "exposure": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "offset": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.001}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 9.99, "step": 0.01}),
                "color_space": (["linear", "srgb"], {"default": "linear"}),
            },
        }

    def apply(self, image, exposure, offset, gamma, color_space="linear"):
        device = model_management.get_torch_device()
        img = image.to(device)

        has_alpha = img.shape[-1] == 4
        rgb = img[..., :3]
        alpha = img[..., 3:4] if has_alpha else None

        safe_gamma = max(gamma, 1e-4)
        g = self._SRGB_GAMMA

        if color_space == "srgb":
            # Decode sRGB (encoded) -> linear before exposure/offset.
            linear = torch.clamp(rgb, min=0.0).pow(g)

            # Exposure + Offset, ported from GEGL's gegl:exposure operation:
            # a black-point remap with self-compensating gain, rather than
            # a plain multiply-then-add. This is what lets Offset lift
            # shadows without ever needing to clip the highlights.
            white = 2.0 ** (-exposure)
            diff = max(white + offset, 1e-6)
            gain = 1.0 / diff
            linear = (linear + offset) * gain
            linear = torch.clamp(linear, 0.0, 1.0)

            # Gamma is applied directly on this intermediate, without an
            # extra round-trip (matches both Photoshop's documented Gamma
            # slider behavior and GEGL's separate, uncoupled gamma op:
            # pure black/white are left untouched). Only clamp to 0 to keep
            # pow() in a valid domain — do NOT floor to a small epsilon
            # here, that would silently override whatever the exposure/
            # offset stage just computed (this was the earlier bug).
            linear = torch.clamp(linear, min=0.0).pow(1.0 / safe_gamma)

            # Re-encode linear -> sRGB.
            color = linear.pow(1.0 / g)

        else:
            # Original behavior, unchanged: operate directly on the raw
            # tensor values. Safe for heightmaps / masks / non-color data.
            color = rgb * (2.0 ** exposure)
            color = color + offset
            color = torch.clamp(color, min=0.0).pow(1.0 / safe_gamma)

        color = torch.clamp(color, 0.0, 1.0)

        out = torch.cat([color, alpha], dim=-1) if has_alpha else color
        return (out.cpu(),)

    
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
    CATEGORY = "moon/image"
    DESCRIPTION = "Collapses channels to their mean (true average, not luminance-weighted), broadcast back to 3 channels."

    def run(self, image):
        mean = image.mean(dim=-1, keepdim=True)
        return (mean.expand(-1, -1, -1, 3).contiguous(),)


class MoonPreviewCrop:
    """Crop image (and optional mask) to a fixed square size for 1:1 pixel preview.

    Bypass conditions:
    - If both input dimensions are already <= crop_size, image and mask pass through unchanged.
    - If the mask's resolution doesn't match the image's resolution (invalid/placeholder mask),
      the mask passes through unchanged even when the image is cropped.
    - bypass disables cropping entirely regardless of resolution.
    - If no mask is connected, a 64x64 black mask is returned (ComfyUI's "no mask" convention).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_size": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "anchor_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Position of the crop's center point on X: 0 = crop pinned to "
                               "the left edge, 0.5 = crop centered, 1 = crop pinned to the right edge."}),
                "anchor_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Position of the crop's center point on Y: 0 = crop pinned to "
                               "the top edge, 0.5 = crop centered, 1 = crop pinned to the bottom edge."}),
                "bypass": ("BOOLEAN", {"default": False,
                    "tooltip": "Force passthrough: image and mask returned unchanged regardless of size."}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "crop"
    CATEGORY = "moon/image"
    DESCRIPTION = "Crops image+mask to a fixed square size for 1:1 pixel preview; passes through unchanged if already smaller or equal."

    def crop(self, image, crop_size, anchor_x, anchor_y, bypass, mask=None):
        if mask is None:
            mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")

        _, img_h, img_w, _ = image.shape

        if bypass or (img_h <= crop_size and img_w <= crop_size):
            return (image, mask)

        out_h = crop_size if img_h > crop_size else img_h
        out_w = crop_size if img_w > crop_size else img_w
        y0 = round((img_h - out_h) * anchor_y)
        x0 = round((img_w - out_w) * anchor_x)

        cropped_image = image[:, y0:y0 + out_h, x0:x0 + out_w, :]

        mask_h, mask_w = mask.shape[-2], mask.shape[-1]
        if mask_h == img_h and mask_w == img_w:
            cropped_mask = mask[:, y0:y0 + out_h, x0:x0 + out_w]
        else:
            cropped_mask = mask  # mismatched (or placeholder 64x64) mask, left untouched

        return (cropped_image, cropped_mask)

NODE_CLASS_MAPPINGS = {
    "MoonImageBlur": MoonImageBlur,
    "ImageSplitRGBAndAlpha": ImageSplitRGBAndAlpha,
    "MoonExposureOffsetGamma": MoonExposureOffsetGamma,
    "MoonMeanChannels": MoonMeanChannels,
    "MoonPreviewCrop": MoonPreviewCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonImageBlur": "Image Blur",
    "ImageSplitRGBAndAlpha": "Split RGB and Alpha",
    "MoonExposureOffsetGamma": "Exposure / Offset / Gamma",
    "MoonMeanChannels": "Mean Channel",
    "MoonPreviewCrop": "Preview Crop (1:1 Pixel)",
}
