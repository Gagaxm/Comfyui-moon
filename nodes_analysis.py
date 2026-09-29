"""
Included nodes:
/analysis/ Height Diagnostics
/analysis/ Normal Map Bias Check
/analysis/ Channel Distribution
"""

import math
import torch
import torch.nn.functional as F
import comfy.model_management as model_management

from .common import quantile_safe

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
    CATEGORY = "moon/analysis"
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
        scale = quantile_safe(
            flat.float(), torch.tensor([0.99], device=flat.device)
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
        q = quantile_safe(v, torch.tensor([0.001, 0.01, 0.50, 0.99, 0.999],
                                           device=v.device,
                                           dtype=v.dtype))
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

class MoonNormalMapCheck:
    """
    Checks a normal map's R/G channel bias from the neutral 127.5 midpoint
    (the flat-surface value). Also reports min/max per channel.

    Run before NormalMapRecenter to see whether a directional bias is
    present (common with AI-generated normal maps, e.g. DeepBump) and
    how strong it is, before deciding whether correction is needed.

    FLOAT outputs always remain in the native ComfyUI [0, 1] range.
    The display_scale option only affects the human-readable report.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "display_scale": (
                    ["0-1", "0-255"],
                    {
                        "default": "0-255",
                        "tooltip": (
                            "Controls the value scale used in the report. "
                            "The FLOAT outputs always remain in the [0, 1] range."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("r_mean", "g_mean", "b_mean", "report")

    FUNCTION = "compute"
    CATEGORY = "moon/analysis"
    DESCRIPTION = (
    "Checks a normal map's R/G bias from the neutral 127.5 midpoint "
    "(flat surface). Run before NormalMapRecenter to see if correction "
    "is needed."
    )

    def compute(self, image, display_scale):
        # IMAGE tensors use the shape (B, H, W, C) with values in [0, 1].
        channels = image.shape[-1]

        if channels < 3:
            raise ValueError(
                f"MoonNormalMapCheck requires an RGB or RGBA image, "
                f"but received {channels} channel(s)."
            )

        if channels > 4:
            raise ValueError(
                f"MoonNormalMapCheck supports RGB and RGBA images, "
                f"but received {channels} channel(s)."
            )

        # Compute statistics independently for each channel.
        means = image.mean(dim=(0, 1, 2))
        minimums = image.amin(dim=(0, 1, 2))
        maximums = image.amax(dim=(0, 1, 2))

        # Keep FLOAT outputs in ComfyUI's native [0, 1] representation.
        r_mean = means[0].item()
        g_mean = means[1].item()
        b_mean = means[2].item()

        # Convert values only for the human-readable report.
        if display_scale == "0-255":
            scale = 255.0
            neutral = 127.5
        else:
            scale = 1.0
            neutral = 0.5

        def format_channel(name, index, show_offset=False):
            mean = means[index].item() * scale
            minimum = minimums[index].item() * scale
            maximum = maximums[index].item() * scale

            if show_offset:
                offset = mean - neutral
                return (
                    f"{name}  mean {mean:.4f}  "
                    f"min {minimum:.4f}  "
                    f"max {maximum:.4f}  "
                    f"offset {offset:+.4f}"
                )

            return (
                f"{name}  mean {mean:.4f}  "
                f"min {minimum:.4f}  "
                f"max {maximum:.4f}"
            )

        # R/G offsets are useful for checking normal-map directional bias.
        report_lines = [
            format_channel("R", 0, show_offset=True),
            format_channel("G", 1, show_offset=True),
            format_channel("B", 2),
        ]

        # Automatically include alpha statistics for RGBA images.
        if channels == 4:
            report_lines.append(format_channel("A", 3))

        report = "\n".join(report_lines)

        return (r_mean, g_mean, b_mean, report)


class MoonChannelDistribution:
    """Analyze the value distribution of each image channel.

The text report lists every channel. Numeric outputs expose the
statistics of the selected channel, making the node useful for
inspecting parameter effects such as scalar or edge_sensitivity.
"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": (["R", "G", "B", "A"], {
                    "default": "B",
                    "tooltip": "Which channel the numeric outputs (mean/median/"
                               "percentile_low/percentile_high/std) report on. "
                               "The text report always lists every channel present."
                }),
                "percentile_low": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 49.0, "step": 0.5,
                    "tooltip": "Lower percentile of the selected channel's distribution "
                            "(e.g. 1.0 = value below which 1% of pixels fall). Feed "
                            "into MoonRemapRange's in_min for an auto-calibrated remap."
                }),
                "percentile_high": ("FLOAT", {
                    "default": 99.0, "min": 51.0, "max": 100.0, "step": 0.5,
                    "tooltip": "Upper percentile of the selected channel's distribution "
                            "(e.g. 99.0 = value below which 99% of pixels fall). Feed "
                            "into MoonRemapRange's in_max for an auto-calibrated remap."
                }),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("mean", "median", "percentile_low", "percentile_high", "standard_deviation", "report")
    FUNCTION = "analyze"
    CATEGORY = "moon/analysis"
    DESCRIPTION = "Per-channel value distribution (mean/median/percentiles/standard_deviation) for inspecting image value ranges and parameter effects."

    def analyze(self, image, channel, percentile_low, percentile_high):
        available = image.shape[-1]
        names = ["R", "G", "B", "A"][:available]

        if channel not in names:
            raise ValueError(
                f"MoonChannelDistribution: channel '{channel}' not present "
                f"in a {available}-channel image (available: {names})."
            )

        qs = torch.tensor([percentile_low / 100.0, 0.5, percentile_high / 100.0],
                           dtype=torch.float32)

        lines = []
        selected = None
        for i, name in enumerate(names):
            ch = image[..., i].reshape(-1).float()
            p_low, p50, p_high = quantile_safe(ch, qs.to(ch.device)).tolist()
            mean = ch.mean().item()
            std = ch.std(unbiased=False).item()
            lines.append(
                f"{name}  mean {mean:.4f}  median {p50:.4f}  "
                f"p{percentile_low:g} {p_low:.4f}  p{percentile_high:g} {p_high:.4f}  "
                f"deviation {std:.4f}"
            )
            if name == channel:
                selected = (mean, p50, p_low, p_high, std)

        mean, median, p_low, p_high, std = selected
        return (mean, median, p_low, p_high, std, "\n".join(lines))





NODE_CLASS_MAPPINGS = {
    "MoonHeightDiagnostics": MoonHeightDiagnostics,
    "MoonNormalMapCheck": MoonNormalMapCheck,
    "MoonChannelDistribution": MoonChannelDistribution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonHeightDiagnostics": "Height Diagnostics",
    "MoonNormalMapCheck": "Normal Map Bias Check",
    "MoonChannelDistribution": "Channel Distribution",
}