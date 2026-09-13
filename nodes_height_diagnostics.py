import math
import torch
import torch.nn.functional as F
import comfy.model_management as model_management


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
    CATEGORY = "moon/pbr"
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


NODE_CLASS_MAPPINGS = {
    "MoonHeightDiagnostics": MoonHeightDiagnostics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonHeightDiagnostics": "Height Diagnostics",
}
