"""
Included nodes:
/height/ Frequency Bands (Macro/Mid/High)
/height/ Remap Range
/height/ Auto Remap Range
/height/ Normal To Height
"""

import math
import torch
import torch.nn.functional as F
import comfy.model_management as model_management

from .common import gaussian_blur, quantile_safe, log, node_text, HIDDEN_UNIQUE_ID



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

class MoonAutoRemapRange:
    """
    Auto Remap Range — computes robust in_min/in_max bounds directly from
    the image's own pixel distribution, meant to feed MoonRemapRange
    (connect this node's in_min/in_max FLOAT outputs into MoonRemapRange's
    in_min/in_max inputs -- convert those widgets to inputs first).
 
    Motivation: a plain literal min()/max() defines the whole stretch
    range from one or two outlier pixels. On a height/depth map with very
    little actual relief, the TRUE dynamic range can be tiny, so any
    linear stretch toward [0, 1] massively amplifies sensor/quantization
    noise along with whatever real signal exists -- this is the likely
    cause of "lots of noise on low-relief maps" observed with
    MoonRemapRange's fixed constants.
 
    Two safeguards against that:
      - Percentile clipping (not raw min/max), so a couple of extreme
        pixels can't single-handedly define the stretch.
      - A minimum span floor: if the percentile-based span is narrower
        than min_span, the returned bounds are widened symmetrically
        (around the same center) until they span at least min_span.
        This caps the maximum amplification factor at 1/min_span instead
        of letting it grow unbounded as the real signal flattens out.
 
    Computed over the ENTIRE batch, not per-image. If you need a
    different range per image, run this with batch size 1 rather than
    batching unrelated maps together.
 
    channel_reduction collapses a multi-channel image to a single scalar
    field before computing statistics:
      - "mean": plain average across R/G/B. Matches a MoonMeanChannels-
        collapsed image (all channels already identical) -- the expected
        use case in this pipeline.
      - "luminance": Rec.709-weighted (0.2126/0.7152/0.0722). Use on a
        color image you haven't pre-collapsed.
      - "max_channel": per-pixel max across channels. Rarely needed,
        kept for completeness.
    """
  
    _LUMA = (0.2126, 0.7152, 0.0722)
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "percentile_low": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 49.0, "step": 0.1,
                    "tooltip": "Lower percentile (%) used as in_min, instead of the literal "
                               "minimum. 0 = literal min (no robustness against outliers)."
                }),
                "percentile_high": ("FLOAT", {
                    "default": 99.0, "min": 51.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Upper percentile (%) used as in_max, instead of the literal "
                               "maximum. 100 = literal max (no robustness against outliers)."
                }),
                "min_span": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Minimum allowed (in_max - in_min). If the measured span is "
                               "narrower (e.g. a near-flat height map), the bounds are widened "
                               "symmetrically to this floor -- caps the maximum amplification "
                               "factor at 1/min_span instead of letting a near-zero span blow "
                               "up the noise. 0 = no floor (raw percentile behavior)."
                }),
                "channel_reduction": (["mean", "luminance", "max_channel"], {
                    "default": "mean",
                    "tooltip": "How a multi-channel image is collapsed to a single scalar "
                               "field before computing percentiles. 'mean' matches a "
                               "MoonMeanChannels-collapsed input (expected use case here). "
                               "'luminance' for a color image you haven't pre-collapsed."
                }),
            },
            "hidden": HIDDEN_UNIQUE_ID,
        }
 
    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("in_min", "in_max", "report")
    FUNCTION = "compute"
    CATEGORY = "moon/height"
    DESCRIPTION = (
        "Computes robust, percentile-based in_min/in_max bounds from the image's own "
        "pixel distribution, with a minimum-span floor to avoid amplifying noise on "
        "near-flat inputs. Feed the outputs into MoonRemapRange's in_min/in_max."
    )
 
    def compute(self, image, percentile_low, percentile_high, min_span, channel_reduction,
                unique_id=None):
        if percentile_low >= percentile_high:
            raise ValueError(
                f"MoonAutoRemapRange: percentile_low ({percentile_low}) must be < "
                f"percentile_high ({percentile_high})."
            )
 
        x = image.detach().float()
 
        if x.shape[-1] >= 3:
            if channel_reduction == "luminance":
                r, g, b = x[..., 0], x[..., 1], x[..., 2]
                field = r * self._LUMA[0] + g * self._LUMA[1] + b * self._LUMA[2]
            elif channel_reduction == "max_channel":
                field = x[..., :3].amax(dim=-1)
            else:  # "mean"
                field = x[..., :3].mean(dim=-1)
        else:
            field = x[..., 0]
 
        flat = field.reshape(-1)

        # Literal min/max on the FULL field (diagnostic only, no size limit).
        literal_min = flat.amin().item()
        literal_max = flat.amax().item()

        q = torch.tensor(
            [percentile_low / 100.0, percentile_high / 100.0],
            device=flat.device, dtype=flat.dtype,
        )
        quantiles = quantile_safe(flat, q)
        in_min = quantiles[0].item()
        in_max = quantiles[1].item()

        span = in_max - in_min
        widened = span < min_span
        if widened:
            center = (in_max + in_min) / 2.0
            in_min = center - min_span / 2.0
            in_max = center + min_span / 2.0
 
        report = (
            f"literal min/max: {literal_min:.6f} / {literal_max:.6f}\n"
            f"p{percentile_low:g}/p{percentile_high:g}: "
            f"{quantiles[0].item():.6f} / {quantiles[1].item():.6f}\n"
            f"final in_min/in_max: {in_min:.6f} / {in_max:.6f}"
        )
        if widened:
            report += f"  (widened to min_span={min_span:g}, measured span was {span:.6f})"
 
        node_text(unique_id, report)
 
        return (in_min, in_max, report)


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

    Two-source fusion (optional `detail_height`):
        band_high = k * (D - base(D, sigma_mid))
    i.e. macro/mid come from `height` and the high band from a second
    height source D, with the crossover at sigma_mid. Typical pairing:
    a depth model (PBRFusion4: coherent large-scale shape, almost no
    energy below ~4 px) as `height`, and an image-to-height model
    (PBRify Height: sharp, albedo-aligned detail, unreliable large-scale
    shape) as `detail_height`. Measured on bark: the two agree at the
    element scale (corr ~0.65 at 4-16 px) and disagree above ~32 px.

    The two sources are normalized independently upstream, so their
    relative amplitude is arbitrary. With `match_detail_amplitude` ON,
    k is the least-squares gain mapping D onto `height` in the overlap
    band [sigma_mid/2, 2*sigma_mid] (gaussian DoG, per image and
    channel), so the crossover joins without a relief step; OFF, k = 1.
    The factor and the band correlation are returned as FLOAT outputs
    (detail_gain, detail_correlation) and shown on the node -- a low
    correlation means the sources disagree at the crossover scale and
    sigma_mid should move. A correlation <= 0 skips the matching (k = 1,
    logged as a warning). high_gain still applies on top.
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
                "edge_sensitivity": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Controls the RGF smoothing strength on a logarithmic scale. "
                            "Lower values preserve weaker height variations; higher values "
                            "apply stronger smoothing. Internally mapped from eps=0.0001 "
                            "at 0 to eps=1.0 at 100."
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
                # Appended last on purpose: widgets_values are positional in saved workflows.
                "match_detail_amplitude": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Only used when detail_height is connected. ON = scale detail_height "
                               "so it matches height's amplitude around sigma_mid (least squares), "
                               "so the two sources join without a relief step. OFF = use "
                               "detail_height as is."
                }),
            },
            "optional": {
                "detail_height": ("IMAGE", {
                    "tooltip": "Optional second height source for the high band (below sigma_mid). "
                               "Macro and mid still come from 'height'. E.g. height = PBRFusion4 "
                               "depth (coherent shape), detail_height = PBRify Height (sharp "
                               "detail). Resized to height's resolution if needed."
                }),
            },
            "hidden": HIDDEN_UNIQUE_ID,
        }

    # Outputs appended at the end only: links in saved workflows are by slot index.
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "FLOAT", "FLOAT")
    RETURN_NAMES = ("band_macro", "band_mid", "band_high", "height_final",
                    "detail_gain", "detail_correlation")
    OUTPUT_TOOLTIPS = (
        "Raw signed macro band (not remapped to [0,1]).",
        "Raw signed mid band (not remapped to [0,1]).",
        "Raw signed high band (not remapped to [0,1]); from detail_height when connected.",
        "Recombined height: macro*macro_gain + mid*mid_gain + high*high_gain.",
        "Gain k applied to detail_height (1.0 when not connected or not matched).",
        "Correlation between height and detail_height around sigma_mid "
        "(0.0 when not connected). Below ~0.3 the sources disagree at that scale.",
    )
    FUNCTION = "process"
    CATEGORY = "moon/height"

    def process(self, height, filter_type, sigma_macro, sigma_mid,
                rgf_iterations, edge_sensitivity, macro_gain, mid_gain, high_gain, tileable,
                match_detail_amplitude=True, detail_height=None, unique_id=None):

        t = edge_sensitivity / 100.0
        rgf_eps = 10.0 ** (-4.0 + 4.0 * t)

        if sigma_mid >= sigma_macro:
            raise ValueError(
                f"MoonFrequencyBands: sigma_mid ({sigma_mid}) must be < sigma_macro ({sigma_macro})."
            )

        device = model_management.get_torch_device()

        wrap_mode = "circular" if tileable else "replicate"
        x = height.permute(0, 3, 1, 2).contiguous().to(device)

        def base(img, sigma):
            if filter_type == "gaussian":
                return gaussian_blur(img, sigma, wrap_mode)
            return _rolling_guidance_filter(img, sigma, rgf_iterations, rgf_eps, wrap_mode)

        macro_base = base(x, sigma_macro)
        mid_base = base(x, sigma_mid)

        band_macro = macro_base
        band_mid = mid_base - macro_base
        k_out, corr_out = 1.0, 0.0
        if detail_height is None:
            band_high = x - mid_base
        else:
            d, resized = self._prepare_detail(detail_height, x)
            band_high = d - base(d, sigma_mid)
            k, corr, band_px = self._match_gain(x, d, sigma_mid, wrap_mode)
            corr_out = float(corr.mean())
            lines = []
            if resized:
                lines.append(f"detail_height resized {resized[0]} -> {resized[1]}")
            if not match_detail_amplitude:
                lines.append("amplitude matching off (k = 1)")
            elif corr_out <= 0.0:
                log.warning("MoonFrequencyBands: correlation %.3f <= 0 between height and "
                            "detail_height around sigma_mid -- amplitude matching skipped (k = 1).",
                            corr_out)
                lines.append("no positive correlation: k = 1 (not matched)")
            else:
                band_high = band_high * k
                k_out = float(k.mean())
            lines.append(f"k = {k_out:.3f}  |  correlation {corr_out:+.2f} @ {band_px}")
            if corr_out < 0.3:
                lines.append("low correlation: sources disagree here, try another sigma_mid")
            node_text(unique_id, "\n".join(lines))

        height_final = (band_macro * macro_gain
                         + band_mid * mid_gain
                         + band_high * high_gain)

        def to_out(t):
            return t.permute(0, 2, 3, 1).contiguous().cpu()

        return (to_out(band_macro), to_out(band_mid), to_out(band_high), to_out(height_final),
                k_out, corr_out)

    @staticmethod
    def _prepare_detail(detail_height, x):
        """detail_height (B,H,W,C) -> (B,C,H,W) on x's device, matching x's
        batch, resolution and channel count. Returns (d, resized) where
        resized is None or ((h0, w0), (H, W)) for the node text."""
        d = detail_height.permute(0, 3, 1, 2).contiguous().to(x.device, x.dtype)
        B, C, H, W = x.shape
        resized = None
        if d.shape[1] != C:
            d = d[:, :3].mean(1, keepdim=True).expand(-1, C, -1, -1)
        if d.shape[2:] != (H, W):
            resized = (tuple(d.shape[2:]), (H, W))
            d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        if d.shape[0] != B:
            if d.shape[0] != 1:
                raise ValueError(
                    f"MoonFrequencyBands: detail_height batch ({d.shape[0]}) must be 1 or "
                    f"match height's batch ({B}).")
            d = d.expand(B, -1, -1, -1)
        return d.contiguous(), resized

    @staticmethod
    def _match_gain(x, d, sigma_mid, wrap_mode):
        """Least-squares gain k (B,C,1,1) mapping d onto x in the band
        [sigma_mid/2, 2*sigma_mid], the band correlation (B,C,1,1), and the
        band as a label."""
        side = min(x.shape[2], x.shape[3])
        s_lo = 0.5 * sigma_mid
        s_hi = min(2.0 * sigma_mid, side / 6.0)
        bx = gaussian_blur(x, s_lo, wrap_mode) - gaussian_blur(x, s_hi, wrap_mode)
        bd = gaussian_blur(d, s_lo, wrap_mode) - gaussian_blur(d, s_hi, wrap_mode)
        bx = bx - bx.mean(dim=(2, 3), keepdim=True)
        bd = bd - bd.mean(dim=(2, 3), keepdim=True)
        xd = (bx * bd).mean(dim=(2, 3), keepdim=True)
        dd = (bd * bd).mean(dim=(2, 3), keepdim=True)
        xx = (bx * bx).mean(dim=(2, 3), keepdim=True)
        k = xd / dd.clamp(min=1e-12)
        corr = xd / (xx * dd).sqrt().clamp(min=1e-12)
        return k, corr, f"{s_lo:g}-{s_hi:g} px"


class MoonNormalToHeight:
    """Normal map -> height map (least-squares integration, Frankot-Chellappa).

    Finds the height field whose slopes best match the normal map in the
    least-squares sense, solved in closed form in the Fourier domain (one
    FFT pass, deterministic, GPU). This is the exact Poisson solution for a
    periodic texture; relaxation/multigrid schemes (e.g. the Danny Chan
    SIGGRAPH 2018 method) converge toward the same solution.

    Pipeline:
        n = rgb * 2 - 1                      (renormalized, nz clamped >= MIN_NZ)
        p = dh/dx   = -nx / nz               (x = column, to the right)
        q = dh/drow = +ny / nz  (opengl)     (row index grows downward)
                    = -ny / nz  (directx)
        H(w) = -i (wx P + wy Q) / (wx^2 + wy^2 + lambda)
        lambda = (2*pi / L)^2, L = flatten_scale% of the smallest side (0 = off)

    Same convention as MoonNormalFromHeight: a height built here and
    converted back with Normal From Height (same normal_format) gives the
    input normals back (up to its Scharr discretization).

    Improvements over DeepBump (comfy-mtb) "Normals to Height":
      - correct frequency grid (DeepBump's is off by one bin after
        ifftshift, which distorts the low frequencies: ~10% error on
        perfect synthetic normals, reconstructed exactly here);
      - true slopes nx/nz instead of nx (no flattening of steep slopes);
      - flatten_scale: Tikhonov damping of very low frequencies, where
        normals estimated from photos/AI are least reliable (large
        blotchy undulations). Comparable in purpose to Knald's
        "Planar Rigidity". Response at wavelength w: 1 / (1 + (w/L)^2) on
        the height amplitude, i.e. halved at exactly L;
      - tileable OFF: Neumann boundary via even/odd mirror extension
        (equivalent to a DCT solve) instead of assuming periodicity;
      - depth output: the relief amplitude in % of the smallest side,
        same unit as Height Ambient Occlusion's `depth` -> wire it there
        to keep the AO physically consistent with the integrated relief.

    Output height is normalized to [0, 1] with min/max computed over the
    whole batch, so a single `depth` value applies to every image.
    """

    MIN_NZ = 0.05  # caps slopes at ~87 deg; guards against nz <= 0 (invalid texels)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal": ("IMAGE",),
                "normal_format": (["opengl", "directx"], {
                    "default": "opengl",
                    "tooltip": "Convention of the input normal map. Only the green channel "
                               "differs. A wrong choice gives a warped, saddle-like height."
                }),
                "flatten_scale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 200.0, "step": 0.5,
                    "tooltip": "Flattens relief undulations larger than this size, in percent "
                               "of the texture's smallest side (0 = off, exact integration). "
                               "At exactly this size the relief is halved; much larger "
                               "undulations are removed. Use it to tame large blotches "
                               "coming from low-frequency bias in AI/photo normal maps. "
                               "Typical: 10-50."
                }),
                "tileable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ON = periodic integration (seamless, exact for tileable "
                               "normal maps). OFF = mirror boundary (no wrap-around artifacts "
                               "on non-tileable images; costs 4x the memory)."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("height", "depth")
    FUNCTION = "integrate"
    CATEGORY = "moon/height"
    DESCRIPTION = (
        "Integrates a normal map into a height map (least-squares, Frankot-Chellappa, "
        "GPU). Optional low-frequency flattening. 'depth' output = relief amplitude in % "
        "of the texture side, ready for Height Ambient Occlusion."
    )

    @staticmethod
    def _slopes(normal, normal_format, min_nz):
        n = normal[..., :3] * 2.0 - 1.0
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        nz = n[..., 2].clamp(min=min_nz)
        p = -n[..., 0] / nz
        q = n[..., 1] / nz
        if normal_format == "directx":
            q = -q
        return p, q

    @staticmethod
    def _mirror(p, q):
        """Even extension of the height <=> odd extension of the slope normal to
        each mirror axis. Output is (B, 2H, 2W) and periodic."""
        p_top = torch.cat([p, -p.flip(-1)], dim=-1)
        p = torch.cat([p_top, p_top.flip(-2)], dim=-2)
        q_top = torch.cat([q, q.flip(-1)], dim=-1)
        q = torch.cat([q_top, -q_top.flip(-2)], dim=-2)
        return p, q

    @staticmethod
    def _integrate_periodic(p, q, lam):
        B, H, W = p.shape
        wx = (2 * math.pi * torch.fft.fftfreq(W, device=p.device, dtype=p.dtype)).view(1, 1, W)
        wy = (2 * math.pi * torch.fft.fftfreq(H, device=p.device, dtype=p.dtype)).view(1, H, 1)
        P = torch.fft.fft2(p)
        Q = torch.fft.fft2(q)
        den = wx ** 2 + wy ** 2 + lam
        den[..., 0, 0] = 1.0
        Z = (-1j) * (wx * P + wy * Q) / den
        Z[..., 0, 0] = 0.0
        return torch.fft.ifft2(Z).real

    def integrate(self, normal, normal_format, flatten_scale, tileable):
        device = model_management.get_torch_device()
        x = normal.to(device=device, dtype=torch.float32)
        B, H, W, _ = x.shape
        side = min(H, W)

        p, q = self._slopes(x, normal_format, self.MIN_NZ)
        if not tileable:
            p, q = self._mirror(p, q)

        lam = 0.0
        if flatten_scale > 0:
            L = flatten_scale / 100.0 * side  # wavelength in pixels
            lam = (2 * math.pi / L) ** 2

        h = self._integrate_periodic(p, q, lam)
        if not tileable:
            h = h[:, :H, :W]

        h_min, h_max = h.min(), h.max()
        rng = (h_max - h_min).clamp(min=1e-12)
        h01 = (h - h_min) / rng
        depth = float(rng) / side * 100.0

        out = h01.unsqueeze(-1).expand(B, H, W, 3).contiguous().to(normal.dtype).cpu()
        return (out, depth)


NODE_CLASS_MAPPINGS = {
    "MoonPBRRemapRange": MoonRemapRange,
    "MoonPBRAutoRemapRange": MoonAutoRemapRange,
    "MoonPBRFrequencyBands": MoonFrequencyBands,
    "MoonPBRNormalToHeight": MoonNormalToHeight,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonPBRRemapRange": "Remap Range",
    "MoonPBRAutoRemapRange": "Auto Remap Range",
    "MoonPBRFrequencyBands": "Frequency Bands (Macro/Mid/High)",
    "MoonPBRNormalToHeight": "Normal To Height",
}
