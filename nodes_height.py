"""
Included nodes:
/height/ Height Diagnostics
/height/ Frequency Bands (Macro/Mid/High)
/height/ Remap Range
"""

import math
import torch
import torch.nn.functional as F
import comfy.model_management as model_management

from .common import gaussian_blur

class MoonHeightDiagnostics:
    """
    Diagnostic node for inspecting a height map before Height -> Normal.

    Outputs:
        height_gray: grayscale height used for analysis
        residual: height - GaussianBlur(height), centered at 0.5
        gradient: gradient magnitude, black at zero slope
        laplacian: signed second derivative, centered at 0.5

    The three diagnostic fields are visualization outputs only.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("IMAGE",),
                "blur_sigma": ("FLOAT", {
                    "default": 8.0, "min": 0.5, "max": 200.0, "step": 0.5,
                    "tooltip": "Gaussian scale used for height - blurred height."
                }),
                "residual_gain": ("FLOAT", {
                    "default": 8.0, "min": 0.1, "max": 100.0, "step": 0.1,
                    "tooltip": "Display gain for signed residual. 0.5 = zero."
                }),
                "gradient_gain": ("FLOAT", {
                    "default": 4.0, "min": 0.1, "max": 100.0, "step": 0.1,
                    "tooltip": "Display gain for gradient magnitude."
                }),
                "laplacian_gain": ("FLOAT", {
                    "default": 8.0, "min": 0.1, "max": 100.0, "step": 0.1,
                    "tooltip": "Display gain for signed Laplacian. 0.5 = zero."
                }),
                "wrap_mode": (["replicate", "circular"], {
                    "default": "circular",
                    "tooltip": "Use circular for a tileable height map."
                }),
                "auto_contrast": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Use per-image 99th percentile normalization for the "
                        "diagnostic displays. Useful for inspection, but not "
                        "for comparing absolute signal strength."
                    )
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("height_gray", "residual", "gradient", "laplacian")
    FUNCTION = "analyze"
    CATEGORY = "moon/height"
    DESCRIPTION = (
        "Diagnoses local height halos, steep transitions and ringing "
        "before Height -> Normal conversion."
    )

    @staticmethod
    def _gaussian_blur(x, sigma, pad_mode):
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum()

        kx = kernel.view(1, 1, 1, -1)
        ky = kernel.view(1, 1, -1, 1)

        x = F.pad(x, (radius, radius, 0, 0), mode=pad_mode)
        x = F.conv2d(x, kx)
        x = F.pad(x, (0, 0, radius, radius), mode=pad_mode)
        return F.conv2d(x, ky)

    @staticmethod
    def _gradient(h, pad_mode):
        p = F.pad(h, (1, 1, 1, 1), mode=pad_mode)
        dx = (p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2]) * 0.5
        dy = (p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1]) * 0.5
        return dx, dy

    @staticmethod
    def _laplacian(h, pad_mode):
        p = F.pad(h, (1, 1, 1, 1), mode=pad_mode)
        c = p[:, :, 1:-1, 1:-1]
        return (
            p[:, :, 1:-1, :-2] +
            p[:, :, 1:-1, 2:] +
            p[:, :, :-2, 1:-1] +
            p[:, :, 2:, 1:-1] -
            4.0 * c
        )

    @staticmethod
    def _scale(x):
        flat = x.detach().abs().reshape(x.shape[0], -1)
        scale = torch.quantile(
            flat.float(), 0.99, dim=1, keepdim=True
        )
        return scale.reshape(-1, 1, 1, 1).to(x.dtype).clamp_min(1e-8)

    @classmethod
    def _signed_display(cls, x, gain, auto):
        if auto:
            x = x / cls._scale(x)
            return (0.5 + 0.5 * x).clamp(0.0, 1.0)
        return (0.5 + gain * x).clamp(0.0, 1.0)

    @classmethod
    def _positive_display(cls, x, gain, auto):
        if auto:
            return (x / cls._scale(x)).clamp(0.0, 1.0)
        return (gain * x).clamp(0.0, 1.0)

    @staticmethod
    def _stats(name, x):
        v = x.detach().float().reshape(x.shape[0], -1)
        q = torch.quantile(
            v,
            torch.tensor(
                [0.001, 0.01, 0.50, 0.99, 0.999],
                device=v.device, dtype=v.dtype
            ),
            dim=1,
        )
        for b in range(v.shape[0]):
            print(
                f"[MoonHeightDiagnostics] {name} batch {b}: "
                f"min={v[b].min().item():.6g} "
                f"max={v[b].max().item():.6g} "
                f"mean={v[b].mean().item():.6g} "
                f"std={v[b].std(unbiased=False).item():.6g} "
                f"p0.1={q[0,b].item():.6g} "
                f"p1={q[1,b].item():.6g} "
                f"p50={q[2,b].item():.6g} "
                f"p99={q[3,b].item():.6g} "
                f"p99.9={q[4,b].item():.6g}"
            )

    @staticmethod
    def _rgb(x):
        return (
            x.permute(0, 2, 3, 1)
             .contiguous()
             .repeat(1, 1, 1, 3)
             .cpu()
        )

    def analyze(
        self, height, blur_sigma, residual_gain, gradient_gain,
        laplacian_gain, wrap_mode, auto_contrast
    ):
        device = model_management.get_torch_device()
        x = height.to(device=device, dtype=torch.float32)

        # Use Rec.709 luminance. Alpha is intentionally ignored.
        if x.shape[-1] >= 3:
            h = (
                x[..., 0:1] * 0.2126 +
                x[..., 1:2] * 0.7152 +
                x[..., 2:3] * 0.0722
            )
        else:
            h = x[..., 0:1]

        h = h.permute(0, 3, 1, 2).contiguous()
        pad_mode = "circular" if wrap_mode == "circular" else "replicate"

        blurred = self._gaussian_blur(h, blur_sigma, pad_mode)
        residual = h - blurred

        dx, dy = self._gradient(h, pad_mode)
        gradient = torch.sqrt(dx * dx + dy * dy)

        laplacian = self._laplacian(h, pad_mode)

        self._stats("height", h)
        self._stats("residual", residual)
        self._stats("gradient", gradient)
        self._stats("laplacian", laplacian)

        return (
            self._rgb(h.clamp(0.0, 1.0)),
            self._rgb(self._signed_display(
                residual, residual_gain, auto_contrast
            )),
            self._rgb(self._positive_display(
                gradient, gradient_gain, auto_contrast
            )),
            self._rgb(self._signed_display(
                laplacian, laplacian_gain, auto_contrast
            )),
        )



class MoonRemapRange:
    """Generic linear remap: maps [in_min, in_max] -> [out_min, out_max].
    Values outside [in_min, in_max] extrapolate linearly (not clamped
    pre-remap) unless `clamp_output` is on.

    Common uses:
    - Preview a signed field (e.g. band_mid/band_high from
      MoonFrequencyBands) as viewable [0,1]: in_min=-1, in_max=1,
      out_min=0, out_max=1 (equivalent to x*0.5+0.5).
    - Recalibrate a ML model's output range if it doesn't land exactly
      on the [-1,1] or [0,1] convention this pack otherwise assumes.
    - Boost visibility of a low-contrast signal: narrow in_min/in_max
      around the actual data range instead of the full theoretical range.

    Preview/debug oriented: safe as a default remap for inspection, but
    treat any downstream node expecting a specific convention (e.g. a
    normal map decoder expecting [-1,1]) as needing the ORIGINAL signal,
    not a remapped one -- this node changes the data's meaning, not just
    its display.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "in_min": ("FLOAT", {"default": -1.0, "min": -1000.0, "max": 1000.0, "step": 0.001}),
                "in_max": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.001}),
                "out_min": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.001}),
                "out_max": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.001}),
                "clamp_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clamp the result to [out_min, out_max]. Turn off to let "
                               "values outside [in_min, in_max] extrapolate past the output "
                               "range instead of being clipped."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("remapped",)
    FUNCTION = "remap"
    CATEGORY = "moon/height"
    DESCRIPTION = "Generic linear remap: [in_min, in_max] -> [out_min, out_max]."

    def remap(self, image, in_min, in_max, out_min, out_max, clamp_output):
        span_in = in_max - in_min
        if abs(span_in) < 1e-8:
            raise ValueError("MoonRemapRange: in_min and in_max must differ.")

        t = (image - in_min) / span_in
        out = out_min + t * (out_max - out_min)

        if clamp_output:
            lo, hi = (out_min, out_max) if out_min <= out_max else (out_max, out_min)
            out = out.clamp(lo, hi)

        return (out,)


def _box_blur(x, radius, wrap_mode="replicate"):
    """Box blur using separable 1D cumulative sums.

    The 2D box kernel is mathematically equivalent to two consecutive
    1D box filters:
        horizontal -> vertical

    This avoids the large 2D summed-area table used by the previous
    implementation. Each cumulative sum grows along only one image
    dimension, greatly reducing floating-point cancellation while
    keeping the same box radius and kernel.

    Border handling matches the previous implementation:
        circular  -> periodic/tileable padding
        replicate -> edge replication

    Complexity is O(H*W) per pass and independent of the radius.
    """
    if radius <= 0:
        return x

    H, W = x.shape[2], x.shape[3]
    ksize = 2 * radius + 1
    pad_mode = "circular" if wrap_mode == "circular" else "replicate"

    # Pad once so both 1D passes use the same border semantics.
    padded = F.pad(
        x,
        (radius, radius, radius, radius),
        mode=pad_mode,
    )

    # Horizontal box sum.
    # The cumulative sum is only along the width, avoiding the
    # large 2D values produced by a full summed-area table.
    csum = padded.cumsum(dim=3)
    csum = F.pad(csum, (1, 0, 0, 0))

    horizontal = (
        csum[..., ksize:ksize + W]
        - csum[..., :W]
    )

    # Vertical box sum.
    # horizontal still contains the padded vertical border, so the
    # vertical filter sees exactly the same padded image as the
    # original 2D box filter.
    csum = horizontal.cumsum(dim=2)
    csum = F.pad(csum, (0, 0, 1, 0))

    box_sum = (
        csum[..., ksize:ksize + H, :]
        - csum[..., :H, :]
    )

    return box_sum / (ksize * ksize)


def _guided_filter(guide, src, radius, eps, wrap_mode="replicate", mean_p_cache=None):
    """He et al. guided filter. `eps` is scale-dependent on the guide's
    value range -- assumes ComfyUI's standard IMAGE convention (float32
    in [0,1]), consistent with the rest of this codebase. If you ever
    feed it a differently-scaled field, eps will need rescaling too.

    mean_p_cache: optional precomputed box_blur(src, radius) -- src is
    constant across RGF iterations (only `guide` changes), so this is
    redundant work to skip when the caller can supply it."""
    mean_I = _box_blur(guide, radius, wrap_mode)
    mean_p = mean_p_cache if mean_p_cache is not None else _box_blur(src, radius, wrap_mode)
    corr_I = _box_blur(guide * guide, radius, wrap_mode)
    corr_Ip = _box_blur(guide * src, radius, wrap_mode)

    var_I = torch.clamp(corr_I - mean_I * mean_I, min=0.0)
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_blur(a, radius, wrap_mode)
    mean_b = _box_blur(b, radius, wrap_mode)
    return mean_a * guide + mean_b


def _rolling_guidance_filter(x, sigma, iterations, eps, wrap_mode="replicate"):
    """radius intentionally re-derived from sigma per call: RGF's spatial
    scale must stay coherent between the Gaussian seed and every
    subsequent guided-filter pass, or two RGF instances at different
    sigma converge toward the same attractor regardless of their seed.
    eps remains the one genuinely independent axis."""
    radius = max(1, int(round(sigma)))
    mean_p_const = _box_blur(x, radius, wrap_mode)  # src never changes across iterations
    p = gaussian_blur(x, sigma, wrap_mode)
    for _ in range(iterations):
        p = _guided_filter(guide=p, src=x, radius=radius, eps=eps,
                            wrap_mode=wrap_mode, mean_p_cache=mean_p_const)
    return p


class MoonFrequencyBands:
    """Height -> {band_macro, band_mid, band_high, height_final}

    Multi-scale decomposition of a height/grayscale field into three
    additive bands (macro / mid / high), each with an independent gain,
    matching the InstaMAT-style Low/Medium/High relief control rather
    than a single DoG pass. Primary target: neutralizing the "pillow
    shape" artifact on convex elements (pebbles, bricks) via the mid
    band's flatten gain, upstream of MoonNormalFromHeight / MoonCavityMap.

    Pipeline:
        macro_base = base(H, sigma_macro)
        mid_base   = base(H, sigma_mid)          (sigma_mid < sigma_macro)
        band_macro = macro_base
        band_mid   = mid_base - macro_base       (signed, not clamped)
        band_high  = H - mid_base                (signed, not clamped)
        height_final = band_macro*macro_gain + band_mid*mid_flatten + band_high*high_gain

    base() is either a plain Gaussian blur, or a Rolling Guidance Filter
    (edge-aware: removes structure by scale while preserving contours --
    see _rolling_guidance_filter). RGF is the more expensive but more
    relevant option for pebble/brick-type content.

    Operates generically per-channel (no luminance conversion): if the
    input is a single-channel height field replicated to 3 identical
    channels, each channel decomposes identically and the result stays
    coherent. This keeps the node reusable beyond height maps (e.g.
    albedo) unlike HorizonAO/MoonCavityMap which need a scalar field.

    band_macro/band_mid/band_high are returned RAW (signed, pre-gain,
    pre-visualization-remap) for numerical inspection/export -- matching
    the validate-by-pixel-export habit used elsewhere in this pack. They
    are NOT remapped to [0,1] for display; expect out-of-range values
    when viewing band_mid/band_high directly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("IMAGE",),
                "filter_type": (["gaussian", "rgf"], {
                    "default": "gaussian",
                    "tooltip": "'gaussian': plain separable blur per band -- band_mid/band_high "
                               "are then a strict, linear frequency-band decomposition. 'rgf': "
                               "Rolling Guidance Filter, edge-aware (preserves pebble/brick "
                               "contours) but non-linear -- band_mid/band_high become "
                               "'scale residuals', not strict spectral bands. More expensive."
                }),
                "sigma_macro": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 200.0, "step": 0.5,
                    "tooltip": "Scale (pixels) of the macro/low-frequency band. Must be > sigma_mid. "
                               "For RGF, this is the Gaussian seed scale only -- see rgf_guide_radius "
                               "for the edge-sensitivity control."
                }),
                "sigma_mid": ("FLOAT", {
                    "default": 8.0, "min": 0.5, "max": 100.0, "step": 0.5,
                    "tooltip": "Scale (pixels) separating mid from high frequency -- roughly "
                               "the size of one relief element (a pebble, a brick)."
                }),
                "rgf_iterations": ("INT", {
                    "default": 3, "min": 1, "max": 10,
                    "tooltip": "Guided-filter re-projection passes. Ignored when filter_type='gaussian'."
                }),
                "rgf_eps": ("FLOAT", {
                    "default": 0.01, "min": 0.0001, "max": 1.0, "step": 0.0001,
                    "tooltip": "Guided filter regularization (prevents division by ~0 in flat "
                               "areas). Assumes height in [0,1] (standard ComfyUI IMAGE range) -- "
                               "not directly comparable to a bilateral filter's range sigma, "
                               "recalibrate visually. Ignored when filter_type='gaussian'."
                }),
                "macro_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "mid_gain": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Gain on the mid band -- lower this to reduce the 'pillow-shaped' "
                               "bulge on convex elements (pebbles, bricks) without touching "
                               "macro slope or fine grain. 0 = fully removed, 1 = unchanged."
                }),
                "high_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "tileable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ON = circular padding in all internal blur/filter passes, for "
                               "seamless tiling. Assumes the input height is itself periodic -- "
                               "if it isn't, this just moves the discontinuity to the material's "
                               "edge rather than fixing it. OFF = edge-replicate padding."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("band_macro", "band_mid", "band_high", "height_final")
    FUNCTION = "process"
    CATEGORY = "moon/height"

    def process(self, height, filter_type, sigma_macro, sigma_mid,
                rgf_iterations, rgf_eps, macro_gain, mid_gain, high_gain, tileable):

        if sigma_mid >= sigma_macro:
            raise ValueError(
                f"MoonFrequencyBands: sigma_mid ({sigma_mid}) must be < sigma_macro ({sigma_macro})."
            )

        device = model_management.get_torch_device()
        # TEMP: for debugging, measure the time and peak memory of the RGF passes
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        # END TEMP
        wrap_mode = "circular" if tileable else "replicate"
        x = height.permute(0, 3, 1, 2).contiguous().to(device)

        def base(sigma):
            if filter_type == "gaussian":
                return gaussian_blur(x, sigma, wrap_mode)
            return _rolling_guidance_filter(x, sigma, rgf_iterations, rgf_eps, wrap_mode)

        macro_base = base(sigma_macro)
        mid_base = base(sigma_mid)

        band_macro = macro_base
        band_mid = mid_base - macro_base
        band_high = x - mid_base

        height_final = (band_macro * macro_gain
                         + band_mid * mid_gain
                         + band_high * high_gain)

        def to_out(t):
            return t.permute(0, 2, 3, 1).contiguous().cpu()

        # TEMP: for debugging, measure the time and peak memory of the RGF passes
        t1.record()
        torch.cuda.synchronize()
        print(f"[MoonFrequencyBands] {t0.elapsed_time(t1):.0f} ms, "
              f"peak VRAM {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
        # END TEMP

        return (to_out(band_macro), to_out(band_mid), to_out(band_high), to_out(height_final))




NODE_CLASS_MAPPINGS = {
    "MoonHeightDiagnostics": MoonHeightDiagnostics,
    "MoonRemapRange": MoonRemapRange,
    "MoonFrequencyBands": MoonFrequencyBands,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonHeightDiagnostics": "Height Diagnostics",
    "MoonRemapRange": "Remap Range",
    "MoonFrequencyBands": "Frequency Bands (Macro/Mid/High)",
}
