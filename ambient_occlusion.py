"""
Horizon-based Ambient Occlusion for ComfyUI.

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

import math
import torch


class HorizonAO:
    """
    Physically-motivated ambient occlusion computed directly from a height
    map via horizon mapping (multi-step horizon search per direction).
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
    "MoonAO": HorizonAO,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Moon Ambient Occlusion": "Moon Horizon AO",
}