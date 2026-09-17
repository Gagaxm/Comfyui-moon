"""
Included nodes:
/tiling/ Periodic+Smooth Decomposition (Moisan)
/tiling/ Circular Pad (wrap, for tiling)
/tiling/ Circular Unpad (crop back)
"""

import math
import torch

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



NODE_CLASS_MAPPINGS = {
    "PeriodicSmoothDecomposition": PeriodicSmoothDecomposition,
    "CircularPad": CircularPad,
    "CircularUnpad": CircularUnpad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PeriodicSmoothDecomposition": "Periodic+Smooth Decomposition (Moisan)",
    "CircularPad": "Circular Pad (wrap, for tiling)",
    "CircularUnpad": "Circular Unpad (crop back)",
}
