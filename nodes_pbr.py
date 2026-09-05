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

LUMA = (0.2126, 0.7152, 0.0722)  # Rec.709/BT.709 luma weights (R,G,B) — replace if your original used different ones (e.g. Rec.601: 0.299/0.587/0.114)
 
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
 
 
def _gaussian_kernel1d(sigma, device, dtype):
    radius = max(1, int(math.ceil(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel, radius
 
 
def _gaussian_blur(x, sigma, wrap_mode="replicate"):
    """Separable gaussian blur, multi-channel (groups=C), respects wrap_mode
    so it stays consistent with the node's circular tiling support."""
    if sigma <= 0:
        return x
    device, dtype = x.device, x.dtype
    kernel1d, radius = _gaussian_kernel1d(sigma, device, dtype)
    C = x.shape[1]
    kx = kernel1d.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    ky = kernel1d.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    pad_mode = "circular" if wrap_mode == "circular" else "replicate"
 
    x = F.pad(x, (radius, radius, 0, 0), mode=pad_mode)
    x = F.conv2d(x, kx, groups=C)
    x = F.pad(x, (0, 0, radius, radius), mode=pad_mode)
    x = F.conv2d(x, ky, groups=C)
    return x
 
 
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
    CATEGORY = "moon/pbr"
    DESCRIPTION = "Converts a heightmap to a normal map using the Scharr operator, with detail/macro separation, OpenGL/DirectX format, intensity post-process and integrated recenter."
 
    def convert(self, image, detail, detail_radius, scalar, intensity, flip,
                invert_height, normal_format, wrap_mode):
        device, dtype = image.device, image.dtype
 
        img_nchw = image.permute(0, 3, 1, 2).contiguous()
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
            highpass = lum - _gaussian_blur(lum, detail_radius, wrap_mode)
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
 
        return (out,)
 
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
                "mask": ("MASK", {
                    "tooltip": "Optional mask restricting the mean calculation to a specific region. If omitted, the mean is computed over the whole image."
                }),
            }
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
 
    V2 changes vs the original implementation:
      - Samples are taken with sub-pixel bilinear interpolation (grid_sample)
        instead of being rounded to the nearest integer pixel. This removes
        the directional aliasing bias toward cardinal/diagonal angles that
        integer offsets caused at small radii.
      - The elevation angle is now computed against the *actual* sampled
        distance, not the pre-rounding target distance (a real source of
        systematic error in v1: rounding ox/oy to the nearest pixel changed
        the true sample distance, but v1 kept dividing by the un-rounded
        target distance).
      - A local tangent-plane correction is applied: the horizon angle is
        compared against the surface's own local slope (from the supplied
        normal map, or estimated from the height map itself) instead of a
        fixed global horizontal plane. This follows the tangent-angle term
        t(theta) from the original HBAO formulation, so a smooth-but-tilted
        surface no longer reads as artificially occluded on one side and
        lit on the other.
      - An optional distance falloff attenuates occluders that sit near the
        edge of the search radius, instead of weighting every occluder
        inside the radius equally.
      - The sampling domain starts at the configured `min_radius` (with a
        1px floor) instead of at 0, so the nearest sample is never further
        out than later samples and never sits at a sub-pixel distance that
        would amplify interpolation noise into a spurious elevation angle.
      - The gradient feeding the tangent-plane correction is smoothed over
        `tangent_scale * radius` pixels instead of measured at a raw 1px
        scale. Without this, the tangent estimate lives at the exact same
        scale as the sharpest occluders (e.g. the contact seam between two
        touching shapes) and cancels them out — visible as AO going
        completely white everywhere except a thin residual line right on
        contact seams, which is what an un-smoothed tangent estimate
        produces.
 
    Reference: Zhukov, Iones & Kronin, "An Ambient Light Illumination
    Model" (1998); Bavoil, Sainz & Dimitrov, "Image-Space Horizon-Based
    Ambient Occlusion" (2008).
 
    INTERFACE CHANGE vs v1: the `normal_bias` widget is gone. In v1 it was
    a post-hoc multiplicative darkening based on normal.z, applied after
    (and on top of) the horizon computation. That was redundant with the
    height-derived slope already baked into the AO, and could double-count
    the same geometry. In v2, `normal` (if connected) feeds directly into
    the tangent-plane term described above, so there is nothing left for a
    separate bias multiplier to do. A new `distance_falloff` toggle was
    added. Existing saved workflow JSON that wires a value into
    `normal_bias` will need that link removed/rewired.
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
                    "tooltip": "Distance steps per direction, spread between min_radius and "
                               "radius. The nearest step is floored at ~1px so it can never "
                               "sit at a sub-pixel distance."
                }),
                "height_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Converts height value units to the same spatial units as pixel "
                               "distance. Raise if the AO looks too weak, lower if it looks too "
                               "strong/noisy."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01,
                    "tooltip": "Overall AO intensity multiplier."
                }),
                "detail_bias": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Concentrates distance samples closer to the texel to catch "
                               "finer micro-relief. 0 = linear spacing, 1 = samples heavily "
                               "biased toward short distances."
                }),
                "min_radius": ("INT", {
                    "default": 0, "min": 0, "max": 255, "step": 1,
                    "tooltip": "Ignores height variation closer than this distance. Large shape "
                               "edges stay sharp; fine micro-detail is excluded from the AO "
                               "calculation."
                }),
                "wrap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ON = circular/seamless sampling (use for a periodic/tileable "
                               "height map). OFF = edge-replicate padding."
                }),
                "distance_falloff": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Attenuate an occluder's contribution as it approaches the edge "
                               "of the search radius, instead of weighting every occluder "
                               "inside the radius equally."
                }),
                "tangent_scale": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Smooths the local-slope estimate used for the tangent-plane "
                               "correction over roughly tangent_scale * radius pixels. Without "
                               "this, the tangent estimate is computed at the same 1px scale as "
                               "the sharpest occluders (e.g. the contact seam between two "
                               "touching shapes), so it cancels out exactly the occlusion the "
                               "node is supposed to detect there. 0 = raw per-pixel slope (can "
                               "wash out AO in creases/contacts); higher = smoother, more "
                               "macro-only tangent estimate."
                }),
            },
            "optional": {
                "normal": ("IMAGE",),
            },
        }
 
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("ao",)
    FUNCTION = "generate"
    CATEGORY = "moon/pbr"
    DESCRIPTION = (
        "Horizon-mapping AO: for each texel, walks outward in multiple "
        "directions along the height map, finds the horizon angle relative "
        "to the local tangent plane, and integrates sin(horizon) - sin(tangent) "
        "as the physically motivated occlusion contribution."
    )
 
    @staticmethod
    def _blur(x, kernel_radius, pad_mode, output_margin=0):
        """
        Separable gaussian blur over a *bounded* quantity (height, or unit
        normal components) — deliberately applied BEFORE any differentiation
        or division, never after. Blurring an already-differentiated or
        already-divided field is unsafe here: a single steep edge (e.g. a
        near-vertical seam between two touching pebbles) can produce an
        arbitrarily large raw gradient value (or, for a normal map, nz can
        sit near zero and make -nx/nz spike), and averaging an unbounded
        spike smears it into a large blob covering the whole kernel
        footprint. Height and unit-normal components are bounded, so
        blurring them first can only ever average toward the local mean,
        never blow up.
 
        Returns a tensor covering the original H,W plus `output_margin`
        extra pixels on every side (still filled with real, correctly
        padded data — not zeros), so a subsequent finite-difference can be
        taken safely all the way to the true image edges.
        """
        if kernel_radius <= 0:
            return x if output_margin == 0 else F.pad(x, (output_margin,) * 4, mode=pad_mode)
 
        device, dtype = x.device, x.dtype
        C = x.shape[1]
        ksize = 2 * kernel_radius + 1
        sigma = max(kernel_radius / 3.0, 1e-3)
        coords = torch.arange(ksize, device=device, dtype=dtype) - kernel_radius
        kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
 
        pad_amount = kernel_radius + output_margin
        padded = F.pad(x, (pad_amount, pad_amount, pad_amount, pad_amount), mode=pad_mode)
        kernel_h = kernel_1d.view(1, 1, 1, ksize).expand(C, 1, 1, ksize).contiguous()
        kernel_v = kernel_1d.view(1, 1, ksize, 1).expand(C, 1, ksize, 1).contiguous()
        blurred = F.conv2d(padded, kernel_h, groups=C)
        blurred = F.conv2d(blurred, kernel_v, groups=C)
        return blurred
 
    def generate(self, height, radius, directions, steps, height_scale, strength,
                 detail_bias, min_radius, wrap, distance_falloff, tangent_scale, normal=None):
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
 
        # Padding margin: must cover the largest possible offset (radius)
        # plus one extra texel so bilinear sampling never reads outside the
        # padded buffer.
        pad = int(math.ceil(radius)) + 1
        pad_mode = "circular" if wrap else "replicate"
        h_padded = F.pad(h_chw, (pad, pad, pad, pad), mode=pad_mode)
        Hp, Wp = H + 2 * pad, W + 2 * pad
 
        # --- Local tangent plane (surface slope) --------------------------
        # v1 always compared the horizon angle against a flat, global
        # reference plane (z = 0). That over- or under-estimates occlusion
        # on any surface that is itself locally tilted, even if perfectly
        # smooth. HBAO's tangent angle t(theta) fixes this by measuring
        # occlusion relative to the surface's own local slope instead.
        #
        # The slope estimate is smoothed over ~tangent_scale * radius
        # pixels so it represents a macro tilt, not the same 1px-scale
        # curvature the horizon search itself is trying to detect (see
        # `tangent_scale` tooltip). Critically, the blur is applied to the
        # bounded source signal (height, or the normal map's unit-vector
        # components) BEFORE differentiating/dividing — never after — to
        # avoid smearing a single steep edge into a large halo (see
        # `_blur` docstring).
        tangent_kernel_radius = int(round(max(radius, 1) * tangent_scale))
 
        if normal is not None:
            n = normal.to(dtype)
            if n.shape[1] != H or n.shape[2] != W:
                n = F.interpolate(
                    n.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
            # A ComfyUI IMAGE normal map is stored as [0,1] per channel
            # (R=nx*0.5+0.5, G=ny*0.5+0.5, B=nz*0.5+0.5). It must be decoded
            # back to a unit vector in [-1,1] before use — a flat normal
            # (0.5, 0.5, 1.0) read raw as (nx=0.5, ny=0.5, nz=1.0) would
            # otherwise fabricate a 45-degree slope on a perfectly flat
            # surface, which was a real bug in an earlier version.
            n = n * 2.0 - 1.0
            n = F.normalize(n, dim=-1, eps=1e-6)
            n_chw = n.permute(0, 3, 1, 2)  # (B,3,H,W)
 
            # Blur the unit-vector components (bounded in [-1,1]) rather
            # than the derived slope (-nx/nz, unbounded whenever nz is
            # near zero on a near-vertical patch of surface).
            n_smooth = self._blur(n_chw, tangent_kernel_radius, pad_mode, output_margin=0)
            n_norm = torch.linalg.vector_norm(n_smooth, dim=1, keepdim=True).clamp(min=1e-6)
            n_smooth = n_smooth / n_norm
 
            nx, ny, nz = n_smooth[:, 0:1], n_smooth[:, 1:2], n_smooth[:, 2:3]
            nz = torch.clamp(nz, min=1e-3)
            grad_x = -nx / nz
            grad_y = -ny / nz
        else:
            # Blur the height map itself (bounded) before differentiating,
            # requesting a 1px margin so the central difference can still
            # be taken right up to the image edges.
            h_smooth = self._blur(h_chw, tangent_kernel_radius, pad_mode, output_margin=1)
            grad_x = (h_smooth[:, :, 1:1 + H, 2:2 + W]
                      - h_smooth[:, :, 1:1 + H, 0:W]) * 0.5 * height_scale
            grad_y = (h_smooth[:, :, 2:2 + H, 1:1 + W]
                      - h_smooth[:, :, 0:H, 1:1 + W]) * 0.5 * height_scale
 
        # --- Distance schedule ---------------------------------------------
        # detail_bias=0 -> exponent=1 (linear spacing)
        # detail_bias=1 -> exponent=3 (samples concentrated near the texel)
        spacing_exponent = 1.0 + detail_bias * 2.0
        min_radius = min(min_radius, max(radius - 1, 0))
 
        # The sampling domain starts at the configured minimum effective
        # distance (`min_radius`), with a floor of 1px so it never starts
        # at, or below, zero. Fixed v2.0 bug: that floor used to be spliced
        # into an already-built [0, radius] schedule by overwriting
        # dists[0], which could make the "closest" sample land further out
        # than the second and third samples (non-monotonic schedule), and
        # separately allowed sub-pixel distances (e.g. 0.05px) elsewhere in
        # the schedule when detail_bias was high — dividing by a near-zero
        # distance amplifies any bilinear-interpolation noise into a huge,
        # spurious elevation angle. The domain is now built directly as
        # [near_dist, radius], so it is monotonic by construction and never
        # goes below 1px.
        near_dist = max(float(min_radius), 1.0)
        if steps <= 1 or near_dist >= radius:
            dists = torch.tensor([float(radius)], device=device, dtype=dtype)
        else:
            t = torch.linspace(0.0, 1.0, steps, device=device, dtype=dtype)
            dists = near_dist + (radius - near_dist) * t.pow(spacing_exponent)
        S = dists.shape[0]
 
        # Base pixel-center grid in padded-image coordinates, reused for
        # every direction/step by adding a continuous (float) offset —
        # no rounding to integer pixels anywhere in this path.
        ys = torch.arange(H, device=device, dtype=dtype) + pad
        xs = torch.arange(W, device=device, dtype=dtype) + pad
        base_y, base_x = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)
 
        occlusion_sum = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
 
        for d in range(directions):
            angle = 2.0 * math.pi * d / directions
            ux, uy = math.cos(angle), math.sin(angle)
 
            # All `steps` samples for this direction are gathered with a
            # single batched grid_sample call (steps folded into the batch
            # axis) instead of one small tensor op per step.
            off_x = dists * ux  # (S,)
            off_y = dists * uy  # (S,)
            samp_x = base_x.unsqueeze(0) + off_x.view(S, 1, 1)  # (S,H,W)
            samp_y = base_y.unsqueeze(0) + off_y.view(S, 1, 1)
 
            norm_x = (samp_x / (Wp - 1)) * 2.0 - 1.0
            norm_y = (samp_y / (Hp - 1)) * 2.0 - 1.0
            grid = torch.stack((norm_x, norm_y), dim=-1)  # (S,H,W,2)
            grid = grid.unsqueeze(1).expand(S, B, H, W, 2).reshape(S * B, H, W, 2)
 
            src = h_padded.unsqueeze(0).expand(S, B, 1, Hp, Wp).reshape(S * B, 1, Hp, Wp)
            sampled = F.grid_sample(
                src, grid, mode="bilinear", padding_mode="border", align_corners=True
            )
            sampled = sampled.view(S, B, 1, H, W)
 
            height_diff = (sampled - h_chw.unsqueeze(0)) * height_scale
            dist_view = dists.view(S, 1, 1, 1, 1)
            # Elevation angle relative to the ACTUAL sampled distance, not
            # the pre-rounding target distance (the v1 bug).
            elevation = torch.atan(height_diff / dist_view)
 
            # Horizon = steepest occluder found along this direction.
            max_elev, max_idx = elevation.max(dim=0)  # (B,1,H,W)
            dist_at_max = dists[max_idx.clamp(max=S - 1)]
 
            tangent_angle = torch.atan(grad_x * ux + grad_y * uy)  # (B,1,H,W)
 
            # HBAO occlusion integrand for one direction: sin(h) - sin(t),
            # clamped to zero when the surface itself is already steeper
            # than any occluder found (nothing actually blocks the sky
            # there, so no extra darkening should be added).
            contribution = torch.clamp(
                torch.sin(max_elev) - torch.sin(tangent_angle), min=0.0, max=1.0
            )
 
            if distance_falloff:
                falloff = torch.clamp(1.0 - dist_at_max / radius, min=0.0, max=1.0)
                contribution = contribution * falloff
 
            occlusion_sum = occlusion_sum + contribution
 
        occlusion_avg = occlusion_sum / directions
        occlusion_avg = torch.clamp(occlusion_avg * strength, 0.0, 1.0)
        ao = 1.0 - occlusion_avg
 
        ao = ao.permute(0, 2, 3, 1)
        ao_rgb = ao.repeat(1, 1, 1, 3)
        return (ao_rgb,)



class MoonCavityMap:
    """
    Concavity/curvature detector — complementary to HorizonAO, not a
    replacement for it.

    HorizonAO answers "is there something taller than me nearby, in some
    direction?" (a horizon-mapping / directional-elevation question). That
    question can legitimately come back "no" for a basin that is flat, or
    only gently sloped, at the bottom of a groove between two convex
    shapes — even though that basin is clearly the lowest point of its
    neighborhood and should read as occluded. Detecting "am I lower than
    my neighborhood on average?" is a different question (curvature /
    concavity), not a directional-elevation one, and horizon mapping
    cannot answer it by construction, however the sampling is tuned.

    This node outputs two independent estimates of that curvature signal,
    for side-by-side comparison rather than a single opinionated result:

      - cavity_from_height: derived from the height map. A pixel that
        sits below the local (blurred) average of its neighborhood reads
        as concave. Implemented as blur(height) - height, so the blur is
        applied to the bounded height signal itself, not to a derivative
        (see HorizonAO's `_blur` for why that ordering matters).

      - cavity_from_normal: derived from the normal map, using the
        divergence of its projected (nx, ny) field: div = d(nx)/dx +
        d(ny)/dy. Converging normals (pointing toward each other) signal
        a concave basin; diverging normals signal a convex bump. This can
        pick up curvature that a separately-generated height map missed
        or smoothed away, if height and normal came from independent
        estimators.

    Both outputs use the same visualization convention as Substance
    Designer's curvature maps: flat = mid-gray (0.5), concave = darker,
    convex = brighter.

    This node does not combine these signals with HorizonAO's output —
    that is a separate step once it's clear which of these two (or both)
    actually captures the missed cavities on real content.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("IMAGE",),
                "radius": ("INT", {
                    "default": 8, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Neighborhood size in pixels used to detect curvature: for "
                               "cavity_from_height, the blur radius averaged against; for "
                               "cavity_from_normal, the distance between the two samples used "
                               "to estimate the normal field's divergence."
                }),
                "contrast": ("FLOAT", {
                    "default": 5.0, "min": 0.1, "max": 50.0, "step": 0.1,
                    "tooltip": "Amplifies the raw curvature signal so it's visible as an image. "
                               "Purely a visualization aid at this stage — raise it if the "
                               "output looks like flat gray, lower it if it's clipping to pure "
                               "black/white."
                }),
                "wrap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ON = circular/seamless sampling for a periodic/tileable height "
                               "map. OFF = edge-replicate padding."
                }),
            },
            "optional": {
                "normal": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("cavity_from_height", "cavity_from_normal")
    FUNCTION = "generate"
    CATEGORY = "moon/pbr"
    DESCRIPTION = (
        "Detects concave basins (lowest points of a neighborhood) via curvature, "
        "independently from the height map and from the normal map, as two "
        "separate outputs for comparison. Complementary to HorizonAO, which "
        "cannot detect a flat or gently-sloped basin by construction."
    )

    @staticmethod
    def _blur(x, kernel_radius, pad_mode):
        """Separable gaussian blur of a bounded quantity. See HorizonAO's
        `_blur` for the reasoning on why this must be applied to bounded
        signals (height, unit-normal components), never to an already
        unbounded derivative."""
        if kernel_radius <= 0:
            return x
        device, dtype = x.device, x.dtype
        C = x.shape[1]
        ksize = 2 * kernel_radius + 1
        sigma = max(kernel_radius / 3.0, 1e-3)
        coords = torch.arange(ksize, device=device, dtype=dtype) - kernel_radius
        kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        padded = F.pad(x, (kernel_radius, kernel_radius, kernel_radius, kernel_radius), mode=pad_mode)
        kernel_h = kernel_1d.view(1, 1, 1, ksize).expand(C, 1, 1, ksize).contiguous()
        kernel_v = kernel_1d.view(1, 1, ksize, 1).expand(C, 1, ksize, 1).contiguous()
        blurred = F.conv2d(padded, kernel_h, groups=C)
        blurred = F.conv2d(blurred, kernel_v, groups=C)
        return blurred

    @staticmethod
    def _to_gray_vis(x, contrast):
        vis = torch.clamp(0.5 + x * contrast, 0.0, 1.0)
        return vis

    def generate(self, height, radius, contrast, wrap, normal=None):
        device = height.device
        dtype = torch.float32
        pad_mode = "circular" if wrap else "replicate"

        h = height.to(dtype)
        if h.shape[-1] >= 3:
            weights = torch.tensor([0.299, 0.587, 0.114], device=device, dtype=dtype)
            h = torch.sum(h[..., :3] * weights, dim=-1, keepdim=True)
        elif h.shape[-1] != 1:
            h = h[..., 0:1]

        B, H, W, _ = h.shape
        h_chw = h.permute(0, 3, 1, 2)  # (B,1,H,W)

        # --- cavity_from_height: blur(height) - height ---------------------
        # Positive where a pixel sits below the local blurred average of its
        # own neighborhood, i.e. a basin. The blur is applied to height
        # itself (bounded), so it can't blow up on a steep nearby edge the
        # way blurring a derivative field would.
        h_blurred = self._blur(h_chw, radius, pad_mode)
        cavity_h = h_blurred - h_chw
        cavity_from_height = self._to_gray_vis(cavity_h, contrast)
        cavity_from_height = cavity_from_height.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)

        # --- cavity_from_normal: divergence of the projected normal field --
        if normal is not None:
            n = normal.to(dtype)
            if n.shape[1] != H or n.shape[2] != W:
                n = F.interpolate(
                    n.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
            # ComfyUI IMAGE normal maps are stored as [0,1] per channel and
            # must be decoded back to a unit vector in [-1,1] before use.
            n = n * 2.0 - 1.0
            n = F.normalize(n, dim=-1, eps=1e-6)
            n_chw = n.permute(0, 3, 1, 2)  # (B,3,H,W)

            r = max(int(radius), 1)
            n_padded = F.pad(n_chw, (r, r, r, r), mode=pad_mode)
            nx = n_padded[:, 0:1]
            ny = n_padded[:, 1:2]
            # Wide-stencil finite difference at the requested radius, taken
            # directly on the bounded (nx, ny) components — not on an
            # already-computed derivative — so there's no unbounded spike
            # to smear, and no need for a separate blur pass here.
            dnx_dx = (nx[:, :, r:r + H, 2 * r:2 * r + W]
                      - nx[:, :, r:r + H, 0:W]) / (2.0 * r)
            dny_dy = (ny[:, :, 2 * r:2 * r + H, r:r + W]
                      - ny[:, :, 0:H, r:r + W]) / (2.0 * r)
            divergence = dnx_dx + dny_dy
            # Converging normals (negative divergence) = concave basin.
            # Flip sign so positive means "concave", matching cavity_from_height.
            cavity_n = -divergence
            cavity_from_normal = self._to_gray_vis(cavity_n, contrast)
            cavity_from_normal = cavity_from_normal.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
        else:
            print("[MoonCavityMap] No normal map connected — cavity_from_normal "
                  "output is flat mid-gray (0.5) and carries no information.")
            cavity_from_normal = torch.full((B, H, W, 3), 0.5, device=device, dtype=dtype)

        return (cavity_from_height, cavity_from_normal)




NODE_CLASS_MAPPINGS = {
    "MoonNormalFromHeight": MoonNormalFromHeight,
    "MoonBlendNormal": MoonBlendNormal,
    "NormalMapRecenter": NormalMapRecenter,
    "ChannelMeanStats": ChannelMeanStats,
    "MoonAO": HorizonAO,
    "MoonCavityMap": MoonCavityMap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonNormalFromHeight": "Normal From Height (Scharr)",
    "MoonBlendNormal": "Blend Normal",
    "NormalMapRecenter": "Normal Map Recenter",
    "ChannelMeanStats": "Channel Mean Stats (RGB)",
    "MoonAO": "Horizon Ambient Occlusion",
    "MoonCavityMap": "Cavity Map (Curvature Detector)",
}