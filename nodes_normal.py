"""
Included nodes:
/normal/ Normal From Height (Scharr)
/normal/ Blend Normal
/normal/ Normal Map Recenter
"""

import torch
import torch.nn.functional as F
import comfy.model_management as model_management

from .common import gaussian_blur

LUMA = (0.2126, 0.7152, 0.0722)  # Rec.709/BT.709 luma weights (R,G,B) — used consistently across the whole pack (see also nodes_height.py, nodes_ao.py)
 
_SCHARR_X = [
    [3, 0, -3],
    [10, 0, -10],
    [3, 0, -3],
]
_SCHARR_Y = [
    [3, 10, 3],
    [0, 0, 0],
    [-3, -10, -3],
]
 
 

class MoonNormalFromHeight:
    """Heightmap -> Normal map (Scharr)
 
    Pipeline:
        lum = luminance(image)
        grad_base   = Scharr(lum) * scalar                        (macro relief)
        grad_detail = Scharr(lum - gaussian_blur(lum, detail_radius)) * detail
                                                                    (fine relief, high-frequency band
                                                                     isolated separately from scalar)
        grad = grad_base + grad_detail
        if flip: grad = grad.yx
        if invert_height: grad = -grad          (flips convexity: bumps <-> dents)
        normal = normalize(-grad.x, -grad.y, 1.0)
        if normal_format == "opengl": normal.y = -normal.y   (ONLY the green channel changes,
                                                                independent of invert_height)
        if intensity != 1.0: normal.xy *= intensity; normal.z rebuilt; renormalize
                                                                (post-process, predictable even
                                                                 when scalar/detail are already extreme)
        encode 0..1
        recenter: always-on, subtracts global X/Y mean bias, rebuilds Z (final step)
    """
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detail": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Fine (high-frequency) relief strength, computed on a band separated "
                               "from 'scalar' via a gaussian high-pass (see detail_radius). "
                               "Independent from scalar."
                }),
                "detail_radius": ("FLOAT", {
                    "default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1,
                    "tooltip": "This is the SIGMA of the gaussian blur used to isolate the "
                               "high-frequency band controlled by 'detail' — not a hard pixel "
                               "radius (the actual kernel radius used internally is ~3x this value). "
                               "Small = very fine detail, large = mid-frequency relief."
                }),
                "scalar": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 50.0, "step": 0.01,
                    "tooltip": "Macro relief strength (overall slope of the full heightmap)."
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Post-process applied AFTER normalization: rescales the normal "
                               "vector's X/Y, rebuilds Z, and renormalizes. More predictable than "
                               "scalar/detail when the relief is already strong. High values can "
                               "saturate the normal toward grazing angles (Z clamped to 0) rather "
                               "than scaling linearly forever."
                }),
                "flip": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Swaps the gradient's X/Y channels (grad = grad.yx). Use if "
                               "ridges/valleys look rotated 90° from the expected lighting direction."
                }),
                "invert_height": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Flips convexity (bumps <-> dents). Has NOTHING to do with the "
                               "OpenGL/DirectX convention — use 'normal_format' for that."
                }),
                "normal_format": (["opengl", "directx"], {
                    "default": "opengl",
                    "tooltip": "Normal map convention. Only the GREEN channel differs between the "
                               "two (R and B are identical)."
                }),
                "wrap_mode": (["replicate", "circular"], {
                    "default": "replicate",
                    "tooltip": "Edge handling for the Scharr gradient AND for the detail_radius blur. "
                               "'circular' for seamless tiling."
                }),
            }
        }
 
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "moon/normal"
    DESCRIPTION = "Converts a heightmap to a normal map using the Scharr operator, with detail/macro separation, OpenGL/DirectX format, intensity post-process and integrated recenter."
 
    def convert(self, image, detail, detail_radius, scalar, intensity, flip,
                invert_height, normal_format, wrap_mode):
        device = model_management.get_torch_device()
        dtype = image.dtype

        img_nchw = image.permute(0, 3, 1, 2).contiguous().to(device)
        r, g, b = img_nchw[:, 0:1], img_nchw[:, 1:2], img_nchw[:, 2:3]
        lum = r * LUMA[0] + g * LUMA[1] + b * LUMA[2]

        kx = torch.tensor(_SCHARR_X, device=device, dtype=dtype).view(1, 1, 3, 3)
        ky = torch.tensor(_SCHARR_Y, device=device, dtype=dtype).view(1, 1, 3, 3)
 
        pad_mode = "circular" if wrap_mode == "circular" else "replicate"
 
        def scharr(field):
            field_p = F.pad(field, (1, 1, 1, 1), mode=pad_mode)
            gx = F.conv2d(field_p, kx)
            gy = F.conv2d(field_p, ky)
            return gx, gy
 
        # --- macro relief ---
        gx_base, gy_base = scharr(lum)
        gx = gx_base * scalar
        gy = gy_base * scalar
 
        # --- fine relief (high-frequency band, separated from scalar) ---
        if detail != 0.0:
            highpass = lum - gaussian_blur(lum, detail_radius, wrap_mode)
            gx_detail, gy_detail = scharr(highpass)
            gx = gx + gx_detail * detail
            gy = gy + gy_detail * detail
 
        if flip:
            gx, gy = gy, gx  # grad = grad.yx
 
        if invert_height:
            gx, gy = -gx, -gy  # flips convexity only
 
        # The Scharr kernels above produce the *negative* image-space derivatives
        # because conv2d performs cross-correlation without kernel flipping — so gx/gy
        # already carry the sign required for N=(-dh/dx,-dh/dy,1). Do NOT negate again
        # here (that was a sign-flip bug in an earlier revision of this file): if you
        # ever redefine the kernels or the conv method, re-derive this sign, don't
        # assume it still holds.
        nx = gx
        ny = gy
        nz = torch.ones_like(nx)
 
        if normal_format == "opengl":
            ny = -ny  # ONLY the green channel differs between OpenGL and DirectX
 
        normal = torch.cat([nx, ny, nz], dim=1)
        normal = F.normalize(normal, dim=1)
 
        # --- intensity: post-process AFTER normalization ---
        if intensity != 1.0:
            xy = normal[:, 0:2] * intensity
            z_sq = (1.0 - (xy ** 2).sum(dim=1, keepdim=True)).clamp(min=0.0)
            z = torch.sqrt(z_sq)
            normal = torch.cat([xy, z], dim=1)
            normal = F.normalize(normal, dim=1)
 
        normal = normal * 0.5 + 0.5
        out = normal.permute(0, 2, 3, 1).contiguous()  # back to BHWC, encoded 0..1
 
        # --- recenter (final step, always-on global-offset, no parameters) ---
        out = self._recenter(out)
 
        return (out.cpu(),)
 
    @staticmethod
    def _recenter(image):
        """Subtracts the global X/Y mean bias (measured on the pre-encoded, already
        normalized vector) and rebuilds Z. NOTE: because renormalization is non-linear
        (per-pixel division by a varying length), zeroing the pre-normalize mean does
        NOT strictly guarantee a zero mean on the final encoded channels — this is a
        cheap, usually-small corrective step, not an exact guarantee. With
        wrap_mode="circular" the underlying gradient (gx/gy, before normalize) already
        has a mathematically exact zero mean for a truly periodic heightmap, since the
        Scharr kernel coefficients themselves sum to zero — so on well-tiling, moderate
        relief content this step should have little effect. Its effect grows with
        stronger relief (scalar/detail/intensity), where the normalize non-linearity
        matters more. Always on, global_offset only — cheap and safe for tileable
        textures with no intentional overall slope (see convert() caveat)."""
        img = image.clone()
 
        x = img[..., 0] * 2.0 - 1.0
        y = img[..., 1] * 2.0 - 1.0
 
        bias_x = x.mean(dim=(1, 2), keepdim=True)
        bias_y = y.mean(dim=(1, 2), keepdim=True)
 
        x_corrected = x - bias_x
        y_corrected = y - bias_y
 
        z_sq = (1.0 - x_corrected ** 2 - y_corrected ** 2).clamp(min=0.0)
        z_corrected = torch.sqrt(z_sq)
        length = torch.sqrt(
            x_corrected ** 2 + y_corrected ** 2 + z_corrected ** 2
        ).clamp(min=1e-6)
        x_corrected = x_corrected / length
        y_corrected = y_corrected / length
        z_corrected = z_corrected / length
 
        out = img.clone()
        out[..., 0] = (x_corrected + 1.0) * 0.5
        out[..., 1] = (y_corrected + 1.0) * 0.5
        if img.shape[-1] >= 3:
            out[..., 2] = (z_corrected + 1.0) * 0.5
 
        return out



def _unpack(c):
    return c * 2.0 - 1.0


def _pack(n):
    n = F.normalize(n, dim=-1)
    return n * 0.5 + 0.5


class MoonBlendNormal:
    """Blend of two normal maps
    Purely pointwise node (no neighborhood read) : direct conversion, no
    tiling implication, no CircularPad/Unpad sandwich required.

    3 modes, faithful to the original shader :

    linear : mix() between the two unpack/renormalized normal maps (ratio 0-1)
    whiteout : UDN, attenuation of detail towards neutral (0.5,0.5,1.0) before combination
    reoriented : RNM (Stephen Hill), same attenuation as whiteout followed by reprojection
    Default : mode = "reoriented" (RNM), intensity = 1.0
    """

    MODE_RANGES = {
        "linear": (0.0, 1.0),
        "whiteout": (0.0, 2.0),
        "reoriented": (0.0, 2.0),
    }

    @classmethod
    def _mode_labels(cls):
        return {
            m: f"{m} ({lo:g}\u2013{hi:g})"
            for m, (lo, hi) in cls.MODE_RANGES.items()
        }

    @classmethod
    def INPUT_TYPES(cls):
        labels = cls._mode_labels()
        return {
            "required": {
                "base_normal": ("IMAGE",),
                "detail_normal": ("IMAGE",),
                "mode": (list(labels.values()), {
                    "default": labels["linear"],
                    "tooltip": "Blend algorithm. 'linear': straight mix of the two unpacked/renormalized normals (intensity 0-1 = mix ratio). 'whiteout' (UDN): adds X/Y, multiplies Z — simple, but can flatten strong detail. 'reoriented' (RNM, Stephen Hill, default): reprojects the detail normal into the base normal's frame — best detail preservation, especially at higher intensity."
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "How much the detail normal is blended in. 'linear' mode: 0-1 is the meaningful range (0 = pure base, 1 = pure detail). 'whiteout'/'reoriented': 0-2 is meaningful (1 = full detail strength, above 1 exaggerates it)."
    }),
}
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"
    CATEGORY = "moon/normal"
    DESCRIPTION = "Blend two normal maps together, with three modes"

    def blend(self, base_normal, detail_normal, mode, intensity):
        # displayed label -> stable internal mode, derived automatically
        label_to_mode = {v: k for k, v in self._mode_labels().items()}
        mode = label_to_mode[mode]

        lo, hi = self.MODE_RANGES[mode]
        intensity = max(lo, min(intensity, hi))

        base_rgb = base_normal[..., :3]
        detail_rgb = detail_normal[..., :3]

        if mode == "linear":
            base_n = F.normalize(_unpack(base_rgb), dim=-1)
            detail_n = F.normalize(_unpack(detail_rgb), dim=-1)
            combined = base_n * (1.0 - intensity) + detail_n * intensity
        else:
            neutral = torch.tensor(
                [0.5, 0.5, 1.0], device=base_normal.device, dtype=base_normal.dtype
            )
            detail_att = detail_rgb * intensity + neutral * (1.0 - intensity)
            base_n = _unpack(base_rgb)
            detail_n = _unpack(detail_att)

            if mode == "whiteout":
                xy = base_n[..., 0:2] + detail_n[..., 0:2]
                z = base_n[..., 2:3] * detail_n[..., 2:3]
                combined = torch.cat([xy, z], dim=-1)
            else:  # reoriented (RNM)
                t = base_n[..., 0:2] * detail_n[..., 2:3] + detail_n[..., 0:2]
                combined = torch.cat([t, base_n[..., 2:3]], dim=-1)

        out_rgb = _pack(combined)

        if base_normal.shape[-1] == 4:
            out = torch.cat([out_rgb, base_normal[..., 3:4]], dim=-1)
        else:
            out = out_rgb

        return (out,)



class NormalMapRecenter:
    """
    NormalMapRecenter - custom ComfyUI node

    Corrects a directional bias on the R/G channels of a tangent-space
    normal map so the surface reads as neutral/flat on average, instead
    of tilted in one direction. This is a common artifact of AI-based
    normal map generation (e.g. DeepBump), which can imprint a
    low-frequency bias from the source photo's lighting into the output.

    Two modes:
    - "global_offset": subtracts the single average X/Y bias across the
    whole image. Fast, correct if the bias is uniform everywhere.
    - "highpass_blur": subtracts a heavily blurred (low-frequency) version
    of the X/Y bias instead of a single global average. Use this when
    the bias varies across the image (e.g. per-tile drift from
    DeepBump's tiled inference) rather than being a flat, uniform tilt.

    After correcting X/Y, the node renormalizes the (X, Y, Z) vector back
    to unit length by default, recomputing Z, so the result stays a valid
    normal map rather than just a color-shifted image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["global_offset", "highpass_blur"], {
                    "default": "global_offset",
                    "tooltip": "'global_offset': subtracts a single average X/Y bias over the whole image — fast, correct for a uniform tilt. 'highpass_blur': subtracts a heavily blurred (low-frequency) version of the bias instead — use when the bias varies across the image (e.g. per-tile drift from DeepBump's tiled inference)."
                }),
                "blur_sigma": ("FLOAT", {
                    "default": 32.0, "min": 1.0, "max": 512.0, "step": 1.0,
                    "tooltip": "Gaussian blur radius (pixels) used to estimate the low-frequency bias in 'highpass_blur' mode. Ignored in 'global_offset' mode. Raise for a smoother/wider bias estimate, lower to track more local variation."
                }),
                "renormalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Recompute Z and renormalize (X,Y,Z) to unit length after correcting X/Y, so the output stays a valid normal map. Turn off only if you want a raw color shift without enforcing a valid normal."
                }),
},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal_corrected",)
    FUNCTION = "correct"
    CATEGORY = "moon/normal"

    def correct(self, image, mode, blur_sigma, renormalize):
        device = model_management.get_torch_device()
        img = image.to(device=device).clone()

        # Decode R, G to signed vector components [-1, 1]
        x = img[..., 0] * 2.0 - 1.0
        y = img[..., 1] * 2.0 - 1.0

        if mode == "global_offset":
            bias_x = x.mean(dim=(1, 2), keepdim=True)
            bias_y = y.mean(dim=(1, 2), keepdim=True)
        else:  # highpass_blur
            bias_x = gaussian_blur(x.unsqueeze(1), blur_sigma).squeeze(1)
            bias_y = gaussian_blur(y.unsqueeze(1), blur_sigma).squeeze(1)

        x_corrected = x - bias_x
        y_corrected = y - bias_y

        if renormalize:
            z_sq = (1.0 - x_corrected ** 2 - y_corrected ** 2).clamp(min=0.0)
            z_corrected = torch.sqrt(z_sq)
            length = torch.sqrt(
                x_corrected ** 2 + y_corrected ** 2 + z_corrected ** 2
            ).clamp(min=1e-6)
            x_corrected = x_corrected / length
            y_corrected = y_corrected / length
            z_corrected = z_corrected / length
        else:
            x_corrected = x_corrected.clamp(-1.0, 1.0)
            y_corrected = y_corrected.clamp(-1.0, 1.0)
            z_corrected = img[..., 2] * 2.0 - 1.0

        out = img.clone()
        out[..., 0] = (x_corrected + 1.0) * 0.5
        out[..., 1] = (y_corrected + 1.0) * 0.5
        if img.shape[-1] >= 3:
            out[..., 2] = (z_corrected + 1.0) * 0.5

        return (out.cpu(),)



NODE_CLASS_MAPPINGS = {
    "MoonNormalFromHeight": MoonNormalFromHeight,
    "MoonBlendNormal": MoonBlendNormal,
    "NormalMapRecenter": NormalMapRecenter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonNormalFromHeight": "Normal From Height (Scharr)",
    "MoonBlendNormal": "Blend Normal",
    "NormalMapRecenter": "Normal Map Recenter",
}
