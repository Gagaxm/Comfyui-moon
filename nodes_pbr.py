"""
Included nodes:
Normal From Height (Scharr)
Blend Normal
Normal Map Recenter
Channel Mean Stats (RGB)
Horizon Ambient Occlusion
"""
import math
import torch
import torch.nn.functional as F

LUMA = (0.299, 0.587, 0.114)
 
_SCHARR_X = [[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]]
_SCHARR_Y = [[3.0, 10.0, 3.0], [0.0, 0.0, 0.0], [-3.0, -10.0, -3.0]]
 
 
class MoonNormalFromHeight:
    """Heightmap -> Normal map (Scharr)
    (height gradient using Scharr 3x3 kernels, converted into a tangent-space normal map).

    scharr_x[x+1][y+1] and scharr_y[x+1][y+1] once converted into row-major matrices [row=y][col=x] for conv2d (which performs cross-correlation,
    so no kernel flipping is needed),
    result in:
        Scharr_X = [[ 3,  0, -3],
                    [10,  0,-10],
                    [ 3,  0, -3]]

        Scharr_Y = [[ 3, 10,  3],
                    [ 0,  0,  0],
                    [-3,-10, -3]]

    Logic:
        grad = heightGradient(luminance, detail) * scalar
        if flip: grad = grad.yx
        if not invert_height: grad = -grad
        normal = normalize(-grad.x, -grad.y, 1.0) * 0.5 + 0.5

    By default: flip=False and invert_height=False (same default values as the PrimitiveBoolean nodes in the original subgraph)
    """
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scalar": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 50.0, "step": 0.01}),
                "detail": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01}),
                "flip": ("BOOLEAN", {"default": False}),
                "invert_height": ("BOOLEAN", {"default": False}),
                "wrap_mode": (["replicate", "circular"], {"default": "replicate"}),
            }
        }
 
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "moon/pbr"
    DESCRIPTION = "Converts a heightmap to a normal map using the Scharr operator."

    def convert(self, image, scalar, detail, flip, invert_height, wrap_mode):
        device, dtype = image.device, image.dtype
 
        img_nchw = image.permute(0, 3, 1, 2).contiguous()
        r, g, b = img_nchw[:, 0:1], img_nchw[:, 1:2], img_nchw[:, 2:3]
        lum = r * LUMA[0] + g * LUMA[1] + b * LUMA[2]
 
        kx = torch.tensor(_SCHARR_X, device=device, dtype=dtype).view(1, 1, 3, 3)
        ky = torch.tensor(_SCHARR_Y, device=device, dtype=dtype).view(1, 1, 3, 3)
 
        pad_mode = "circular" if wrap_mode == "circular" else "replicate"
        lum_p = F.pad(lum, (1, 1, 1, 1), mode=pad_mode)
 
        gx = F.conv2d(lum_p, kx) * detail
        gy = F.conv2d(lum_p, ky) * detail
 
        gx = gx * scalar
        gy = gy * scalar
 
        if flip:
            gx, gy = gy, gx  # grad = grad.yx
 
        if not invert_height:
            gx, gy = -gx, -gy
 
        nx = -gx
        ny = -gy
        nz = torch.ones_like(nx)
 
        normal = torch.cat([nx, ny, nz], dim=1)
        normal = F.normalize(normal, dim=1)
        normal = normal * 0.5 + 0.5
 
        out = normal.permute(0, 2, 3, 1).contiguous()
        return (out,)


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

    # Source unique de vérité : mode interne -> plage valide d'intensity
    MODE_RANGES = {
        "linear": (0.0, 1.0),
        "whiteout": (0.0, 2.0),
        "reoriented": (0.0, 2.0),
    }

    @classmethod
    def _mode_labels(cls):
        # mode interne -> libellé affiché (généré, jamais recopié à la main)
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
                "mode": (list(labels.values()), {"default": labels["linear"]}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"
    CATEGORY = "moon/pbr"
    DESCRIPTION = "Blend two normal maps together, with three modes"

    def blend(self, base_normal, detail_normal, mode, intensity):
        # libellé affiché -> mode interne stable, dérivé automatiquement
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

def _gaussian_kernel1d(sigma, radius, device, dtype):
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur(img, sigma):
    # img: (B, C, H, W), blurs each channel independently
    if sigma <= 0:
        return img
    radius = max(1, int(sigma * 3))
    kernel_1d = _gaussian_kernel1d(sigma, radius, img.device, img.dtype)

    channels = img.shape[1]
    kernel_x = kernel_1d.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, -1, 1).expand(channels, 1, -1, 1)

    pad = radius
    img = F.pad(img, (pad, pad, 0, 0), mode="reflect")
    img = F.conv2d(img, kernel_x, groups=channels)
    img = F.pad(img, (0, 0, pad, pad), mode="reflect")
    img = F.conv2d(img, kernel_y, groups=channels)
    return img


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
                "mode": (["global_offset", "highpass_blur"], {"default": "global_offset"}),
                "blur_sigma": ("FLOAT", {"default": 32.0, "min": 1.0, "max": 512.0, "step": 1.0}),
                "renormalize": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal_corrected",)
    FUNCTION = "correct"
    CATEGORY = "moon/pbr"

    def correct(self, image, mode, blur_sigma, renormalize):
        # image: (B, H, W, C) float tensor in [0, 1]
        img = image.clone()

        # Decode R, G to signed vector components [-1, 1]
        x = img[..., 0] * 2.0 - 1.0
        y = img[..., 1] * 2.0 - 1.0

        if mode == "global_offset":
            bias_x = x.mean(dim=(1, 2), keepdim=True)
            bias_y = y.mean(dim=(1, 2), keepdim=True)
        else:  # highpass_blur
            bias_x = _gaussian_blur(x.unsqueeze(1), blur_sigma).squeeze(1)
            bias_y = _gaussian_blur(y.unsqueeze(1), blur_sigma).squeeze(1)

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

        return (out,)

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



class HorizonAO:
    """
    Physically-motivated ambient occlusion computed directly from a height
    map via horizon mapping (multi-step horizon search per direction).

    Implements the classic horizon-mapping AO technique (Zhukov, Iones & Kronin,
    "An Ambient Light Illumination Model", 1998; the same principle behind
    real-time HBAO): for each texel, walk outward in several directions,
    find the steepest elevation angle ("horizon angle") reached by neighboring
    height values, and use sin(horizon_angle) as the occlusion contributed by
    that direction. Averaging over all directions gives a physically motivated
    occlusion factor - not a heuristic edge/curvature approximation.

    Unlike a single-sample-per-direction shortcut, this walks multiple
    distances per direction so a close, steep obstacle isn't missed just
    because it doesn't land exactly at the max radius.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("IMAGE",),
                "radius": ("INT", {
                    "default": 16, "min": 1, "max": 256, "step": 1,
                    "tooltip": "Max search distance in pixels for the horizon walk."
                }),
                "directions": ("INT", {
                    "default": 16, "min": 4, "max": 32, "step": 1,
                    "tooltip": "Number of angular directions sampled around each texel."
                }),
                "steps": ("INT", {
                    "default": 8, "min": 2, "max": 32, "step": 1,
                    "tooltip": "Distance steps per direction (the horizon walk). More = catches closer/steeper obstacles more reliably."
                }),
                "height_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Converts height value units to the same spatial units as pixel distance. Raise if the AO looks too weak, lower if it looks too strong/noisy."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01,
                    "tooltip": "Overall AO intensity multiplier."
                }),
                "detail_bias": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Concentrates distance samples closer to the texel to catch finer micro-relief. 0 = linear spacing (original behavior), 1 = samples heavily biased toward short distances. Doesn't cost extra compute (same step count)."
                }),
                "min_radius": ("INT", {
                    "default": 0, "min": 0, "max": 255, "step": 1,
                    "tooltip": "Ignores height variation closer than this distance. Removes fine micro-detail from the AO calculation without blurring the height map - large shape edges stay sharp. 0 = samples starting from 1px (original behavior)."
                }),
                "wrap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ON = circular/seamless sampling (use for a periodic/tileable height map - no Circular Pad/Unpad needed). OFF = edge-replicate padding."
                }),
            },
            "optional": {
                "normal": ("IMAGE",),
                "normal_bias": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Optional stylistic extra darkening on faces not pointing up, using the normal map's Z channel. 0 = pure horizon-mapping result, no bias."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("ao",)
    FUNCTION = "generate"
    CATEGORY = "moon/pbr"
    DESCRIPTION = (
        "Horizon-mapping AO: for each texel, walks outward in multiple "
        "directions and distances along the height map, finds the "
        "steepest horizon angle per direction, and uses sin(angle) as "
        "the physically motivated occlusion contribution."
    )

    def generate(self, height, radius, directions, steps, height_scale, strength,
                 detail_bias, min_radius, wrap, normal=None, normal_bias=0.0):
        device = height.device
        dtype = torch.float32

        h = height.to(dtype)
        if h.shape[-1] >= 3:
            weights = torch.tensor([0.299, 0.587, 0.114], device=device, dtype=dtype)
            h = torch.sum(h[..., :3] * weights, dim=-1, keepdim=True)
        elif h.shape[-1] != 1:
            h = h[..., 0:1]

        B, H, W, _ = h.shape
        h_chw = h.permute(0, 3, 1, 2)  # (B,1,H,W)

        pad = radius if not wrap else 0
        if not wrap:
            h_padded = torch.nn.functional.pad(h_chw, (pad, pad, pad, pad), mode="replicate")
        else:
            h_padded = h_chw

        occlusion_sum = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

        # detail_bias=0 -> exponent=1 (linear spacing, original behavior)
        # detail_bias=1 -> exponent=3 (samples heavily concentrated near the texel)
        spacing_exponent = 1.0 + detail_bias * 2.0
        min_radius = min(min_radius, max(radius - 1, 0))  # keep a usable sampling range

        for d in range(directions):
            angle = 2.0 * math.pi * d / directions
            ux, uy = math.cos(angle), math.sin(angle)

            dir_max_slope = torch.full((B, 1, H, W), float("-inf"), device=device, dtype=dtype)

            for s in range(1, steps + 1):
                t = s / steps
                dist = min_radius + (radius - min_radius) * (t ** spacing_exponent)
                ox = int(round(ux * dist))
                oy = int(round(uy * dist))
                if ox == 0 and oy == 0:
                    continue

                if wrap:
                    shifted = torch.roll(h_padded, shifts=(-oy, -ox), dims=(2, 3))
                    center = h_chw
                else:
                    y0, x0 = pad + oy, pad + ox
                    shifted = h_padded[:, :, y0:y0 + H, x0:x0 + W]
                    center = h_chw

                height_diff = (shifted - center) * height_scale
                slope = height_diff / dist
                dir_max_slope = torch.maximum(dir_max_slope, slope)

            dir_max_slope = torch.clamp(dir_max_slope, min=0.0)
            horizon_angle = torch.atan(dir_max_slope)
            occlusion_sum = occlusion_sum + torch.sin(horizon_angle)

        occlusion_avg = occlusion_sum / directions
        occlusion_avg = torch.clamp(occlusion_avg * strength, 0.0, 1.0)
        ao = 1.0 - occlusion_avg

        if normal is not None and normal_bias > 0.0:
            n = normal.to(dtype)
            normal_z = n[..., 2:3].permute(0, 3, 1, 2)
            if normal_z.shape[2:] != (H, W):
                normal_z = torch.nn.functional.interpolate(
                    normal_z, size=(H, W), mode="bilinear", align_corners=False
                )
            bias = torch.clamp((1.0 - normal_z) * normal_bias, 0.0, 1.0)
            ao = ao * (1.0 - bias)

        ao = ao.permute(0, 2, 3, 1)
        ao_rgb = ao.repeat(1, 1, 1, 3)
        return (ao_rgb,)


NODE_CLASS_MAPPINGS = {
    "MoonNormalFromHeight": MoonNormalFromHeight,
    "MoonBlendNormal": MoonBlendNormal,
    "NormalMapRecenter": NormalMapRecenter,
    "ChannelMeanStats": ChannelMeanStats,
    "MoonAO": HorizonAO,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonNormalFromHeight": "Normal From Height (Scharr)",
    "MoonBlendNormal": "Blend Normal",
    "NormalMapRecenter": "Normal Map Recenter",
    "ChannelMeanStats": "Channel Mean Stats (RGB)",
    "MoonAO": "Horizon Ambient Occlusion",
}