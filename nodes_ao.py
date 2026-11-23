"""
Included nodes:
/ao/ Height Ambient Occlusion (V3, recommended)
/ao/ Cavity Map (Curvature Detector)
"""

import math
import torch
import torch.nn.functional as F
import comfy.model_management as model_management

from .common import node_text, HIDDEN_UNIQUE_ID

class MoonHeightAO:
    """
    Ambient occlusion baked from a height map with horizon search, in the
    same spirit as a mesh AO baker (Substance, Marmoset), but working on the
    height field directly.

    For every texel, a set of slices (pairs of opposite directions) is
    marched outward. Along each direction the horizon elevation rises as
    taller samples are found; every rise occludes a further band of the
    hemisphere. That band is weighted by the distance attenuation W(r) of
    the sample that caused it (per-sample attenuation, Bavoil & Sainz 2008),
    not by the distance of the final horizon point — with a single weight at
    the horizon, any concave bowl whose horizon keeps rising up to the
    radius reads as fully unoccluded.

    Two angular distributions, matching the Substance bakers:
      - cosine  : Lambert-weighted visibility of the normal-oriented
                  hemisphere (GTAO slice integral, Jimenez et al. 2016).
                  Physically meaningful; the one to use as a PBR AO channel.
      - uniform : uniform solid-angle weighting above the tangent plane
                  (classic HBAO, sin(h) - sin(t)). Softer, more linear
                  gradients; handy for masks or for multiplying into albedo.

    All lengths are relative to the smallest texture side, so the same
    settings give the same result at 1K, 2K or 4K. The surface slope is
    measured on the height map itself, in the same units as the horizon
    search, so there is no external normal map to keep consistent.
    """

    QUALITY = {
        # name: (slices, max steps per direction)
        "low": (8, 16),
        "medium": (12, 32),
        "high": (16, 64),
        "ultra": (24, 128),
    }
    FALLOFF_RANGE = 0.5  # fraction of the radius over which W fades to 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("IMAGE",),
                "depth": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 50.0, "step": 0.05,
                    "tooltip": "Relief depth: a height of 1.0 (white) stands this many percent "
                               "of the texture's smallest side above 0.0 (black). Example: a 4 m "
                               "wide texture with 5 cm deep relief -> 1.25."
                }),
                "radius": ("FLOAT", {
                    "default": 3.0, "min": 0.1, "max": 25.0, "step": 0.05,
                    "tooltip": "Occlusion search distance, in percent of the texture's smallest "
                               "side. Occluders further away than this are ignored (soft fade "
                               "over the outer half)."
                }),
                "distribution": (["cosine", "uniform"], {
                    "default": "cosine",
                    "tooltip": "cosine = physically based AO, for use as a PBR channel "
                               "(Substance baker default). uniform = softer, more linear "
                               "gradients, for masks or multiplying into albedo."
                }),
                "quality": (list(cls.QUALITY.keys()), {
                    "default": "high",
                    "tooltip": "Number of directions and samples. Higher = less banding, slower."
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Output exponent: ao ** intensity. 1 = unmodified, >1 darker, "
                               "<1 lighter. White stays white, black stays black."
                }),
                "tileable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ON = tileable sampling (circular). OFF = edge-replicate."
                }),
                # Kept last: widgets_values are positional in saved workflows.
                "depth_multiplier": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Multiplies depth before the bake (effective depth = depth x "
                               "this). Use it to correct a measured depth that is too weak "
                               "(e.g. Normal To Height's depth output on a flat AI normal map) "
                               "while keeping the measurement itself wired unchanged. Unlike "
                               "intensity, it changes the relief geometry: deeper relief widens "
                               "the occluded areas and strengthens contacts, whereas intensity "
                               "only darkens an AO whose shape stays the same."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("ao",)
    FUNCTION = "generate"
    CATEGORY = "moon/ao"
    DESCRIPTION = (
        "Bakes ambient occlusion from a height map (horizon search with per-sample "
        "attenuation). Cosine or uniform distribution; resolution-independent settings."
    )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _shift(hp, pad, H, W, dx, dy):
        """Bilinear sample of the padded height at (x+dx, y+dy) for every texel.
        The offset is the same for all texels, so this is 4 shifted views of the
        padded buffer instead of a grid_sample."""
        ix, iy = math.floor(dx), math.floor(dy)
        fx, fy = dx - ix, dy - iy
        y0, x0 = pad + iy, pad + ix
        v00 = hp[:, :, y0:y0 + H, x0:x0 + W]
        v01 = hp[:, :, y0:y0 + H, x0 + 1:x0 + 1 + W]
        v10 = hp[:, :, y0 + 1:y0 + 1 + H, x0:x0 + W]
        v11 = hp[:, :, y0 + 1:y0 + 1 + H, x0 + 1:x0 + 1 + W]
        return (v00 * ((1 - fx) * (1 - fy)) + v01 * (fx * (1 - fy))
                + v10 * ((1 - fx) * fy) + v11 * (fx * fy))

    @staticmethod
    def _distances(radius_px, max_steps):
        """Sample distances in pixels, from 1px to radius. One sample per pixel
        when affordable; otherwise quadratic spacing (dense near the texel)."""
        n_px = max(int(math.ceil(radius_px)), 1)
        if n_px <= max_steps:
            return [radius_px * (i + 1) / n_px for i in range(n_px)]
        return [1.0 + (radius_px - 1.0) * (i / (max_steps - 1)) ** 2 for i in range(max_steps)]

    @staticmethod
    def _gtao_a(h, n):
        # Cosine-weighted visibility of one half-slice up to zenith angle h,
        # for a projected normal at angle n (Jimenez et al. 2016, eq. 10).
        return 0.25 * (-torch.cos(2.0 * h - n) + torch.cos(n) + 2.0 * h * torch.sin(n))

    # ------------------------------------------------------------------- main
    def generate(self, height, depth, radius, distribution, quality, intensity, tileable,
                 depth_multiplier=1.0):
        device = model_management.get_torch_device()
        dtype = torch.float32

        h = height.to(device=device, dtype=dtype)
        if h.shape[-1] >= 3:
            w = torch.tensor([0.2126, 0.7152, 0.0722], device=device, dtype=dtype)
            h = (h[..., :3] * w).sum(-1, keepdim=True)
        else:
            h = h[..., 0:1]
        B, H, W, _ = h.shape
        side = min(H, W)

        depth_px = depth * depth_multiplier / 100.0 * side
        radius_px = min(radius / 100.0 * side, 0.5 * side - 2.0)
        radius_px = max(radius_px, 1.0)

        hc = h.permute(0, 3, 1, 2) * depth_px  # (B,1,H,W) height in pixels
        pad = int(math.ceil(radius_px)) + 2
        pad_mode = "circular" if tileable else "replicate"
        hp = F.pad(hc, (pad, pad, pad, pad), mode=pad_mode)

        # Slope in pixel units, from the same height field the horizon search uses.
        gx = (self._shift(hp, pad, H, W, 1, 0) - self._shift(hp, pad, H, W, -1, 0)) * 0.5
        gy = (self._shift(hp, pad, H, W, 0, 1) - self._shift(hp, pad, H, W, 0, -1)) * 0.5
        inv = torch.rsqrt(gx * gx + gy * gy + 1.0)
        nx, ny, nz = -gx * inv, -gy * inv, inv  # unit normal, z up

        slices, max_steps = self.QUALITY[quality]
        dists = self._distances(radius_px, max_steps)
        weights = [min(max((radius_px - r) / (self.FALLOFF_RANGE * radius_px), 0.0), 1.0)
                   for r in dists]
        cosine = distribution == "cosine"

        occ_total = torch.zeros_like(hc)
        open_total = torch.zeros_like(hc)
        half_pi = 0.5 * math.pi

        for k in range(slices):
            phi = math.pi * (k + 0.5) / slices
            ux, uy = math.cos(phi), math.sin(phi)

            if cosine:
                n_dot_d = nx * ux + ny * uy
                n_ang = torch.atan2(n_dot_d, nz)                  # projected normal angle
                plen = torch.sqrt(n_dot_d * n_dot_d + nz * nz)    # projected normal length
                open_vis = plen * (self._gtao_a(n_ang - half_pi, n_ang)
                                   + self._gtao_a(n_ang + half_pi, n_ang))
                open_total += open_vis
            else:
                g_dot_d = gx * ux + gy * uy
                open_total += 2.0

            for sgn in (1.0, -1.0):
                # Lower bound of the horizon elevation on this side.
                if cosine:
                    e_prev = -sgn * n_ang  # normal hemisphere boundary
                else:
                    e_prev = torch.atan(sgn * g_dot_d)  # tangent plane
                occ = torch.zeros_like(hc)
                # Visibility measure of the current horizon (zenith angle signed
                # toward +d for cosine; sin(elevation) for uniform).
                if cosine:
                    f_prev = self._gtao_a(sgn * (half_pi - e_prev), n_ang)
                else:
                    f_prev = torch.sin(e_prev)

                for r, wgt in zip(dists, weights):
                    if wgt <= 0.0:
                        break  # nothing beyond this distance contributes
                    dz = self._shift(hp, pad, H, W, sgn * r * ux, sgn * r * uy) - hc
                    e_prev = torch.maximum(e_prev, torch.atan2(dz, torch.full_like(dz, r)))
                    if cosine:
                        f_new = self._gtao_a(sgn * (half_pi - e_prev), n_ang)
                        occ += wgt * (f_prev - f_new)
                    else:
                        f_new = torch.sin(e_prev)
                        occ += wgt * (f_new - f_prev)
                    f_prev = f_new

                if cosine:
                    occ = occ * plen

                occ_total += occ

        ao = (1.0 - occ_total / open_total.clamp(min=1e-6)).clamp(0.0, 1.0)
        if intensity != 1.0:
            ao = ao.pow(intensity)

        ao = ao.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
        return (ao.cpu(),)


class MoonCavityMap:
    """
    Concavity/curvature detector — complementary to horizon-based AO
    (Height Ambient Occlusion), not a replacement for it.

    Horizon AO answers "is there something taller than me nearby, in some
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
        (see `_blur` for why that ordering matters).

      - cavity_from_normal: derived from the normal map, using the
        divergence of its projected (nx, ny) field: div = d(nx)/dx +
        d(ny)/dy. Converging normals (pointing toward each other) signal
        a concave basin; diverging normals signal a convex bump. This can
        pick up curvature that a separately-generated height map missed
        or smoothed away, if height and normal came from independent
        estimators. The raw 1px derivative (bounded — no division is
        involved here, unlike a -nx/nz slope term) is smoothed over
        `radius` pixels before use, exactly mirroring cavity_from_height's
        averaging: a two-point stencil sampled `radius` pixels apart would
        only ever look at 2 pixels and ignore everything in between,
        making it extremely sensitive to per-pixel normal-map noise
        (dithering, ML-model grain) rather than the actual macro curvature.

    Both outputs use the same visualization convention as Substance
    Designer's curvature maps: flat = mid-gray (0.5), concave = darker,
    convex = brighter.

    This node does not combine these signals with an AO output —
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
                    "tooltip": "Neighborhood size in pixels used to detect curvature — the "
                               "averaging radius for both outputs (blur radius for "
                               "cavity_from_height, and the smoothing applied to the "
                               "normal-derived divergence for cavity_from_normal)."
                }),
                "contrast": ("FLOAT", {
                    "default": 5.0, "min": 0.1, "max": 50.0, "step": 0.1,
                    "tooltip": "Amplifies the raw curvature signal so it's visible as an image. "
                               "Purely a visualization aid at this stage — raise it if the "
                               "output looks like flat gray, lower it if it's clipping to pure "
                               "black/white."
                }),
                "tileable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ON = circular/seamless sampling for a periodic/tileable height "
                               "map. OFF = edge-replicate padding."
                }),
            },
            "optional": {
                "normal": ("IMAGE",),
            },
            "hidden": HIDDEN_UNIQUE_ID,
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("cavity_from_height", "cavity_from_normal")
    FUNCTION = "generate"
    CATEGORY = "moon/ao"
    DESCRIPTION = (
        "Detects concave basins (lowest points of a neighborhood) via curvature, "
        "independently from the height map and from the normal map, as two "
        "separate outputs for comparison. Complementary to horizon-based AO, "
        "which cannot detect a flat or gently-sloped basin by construction."
    )

    @staticmethod
    def _blur(x, kernel_radius, pad_mode):
        """Separable gaussian blur of a bounded quantity. It must be applied
        to bounded signals (height, unit-normal components), never to an
        already unbounded derivative: a single steep edge in the radius
        would dominate the average of a derivative field."""
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

    def generate(self, height, radius, contrast, tileable, normal=None, unique_id=None):
        device = model_management.get_torch_device()
        dtype = torch.float32
        pad_mode = "circular" if tileable else "replicate"

        h = height.to(dtype).to(device)
        if h.shape[-1] >= 3:
            weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device, dtype=dtype)  # Rec.709/BT.709 luma weights (R,G,B)
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
            n = normal.to(dtype).to(device)
            if n.shape[1] != H or n.shape[2] != W:
                n = F.interpolate(
                    n.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
            # ComfyUI IMAGE normal maps are stored as [0,1] per channel and
            # must be decoded back to a unit vector in [-1,1] before use.
            n = n * 2.0 - 1.0
            n = F.normalize(n, dim=-1, eps=1e-6)
            n_chw = n.permute(0, 3, 1, 2)  # (B,3,H,W)

            # Raw 1px-centered derivative — bounded, since nx/ny are unit-
            # vector components in [-1,1] and no division is involved (this
            # is not the same situation as a -nx/nz slope term,
            # which can spike when nz is near zero). Smoothing this raw,
            # bounded divergence over `radius` pixels is therefore safe and
            # gives a proper neighborhood average — not a sparse 2-sample
            # readout — matching how cavity_from_height is averaged.
            n_padded = F.pad(n_chw, (1, 1, 1, 1), mode=pad_mode)
            nx = n_padded[:, 0:1]
            ny = n_padded[:, 1:2]
            dnx_dx = (nx[:, :, 1:1 + H, 2:2 + W] - nx[:, :, 1:1 + H, 0:W]) * 0.5
            dny_dy = (ny[:, :, 2:2 + H, 1:1 + W] - ny[:, :, 0:H, 1:1 + W]) * 0.5
            raw_divergence = dnx_dx + dny_dy
            divergence = self._blur(raw_divergence, radius, pad_mode)
            # Converging normals (negative divergence) = concave basin.
            # Flip sign so positive means "concave", matching cavity_from_height.
            cavity_n = -divergence
            cavity_from_normal = self._to_gray_vis(cavity_n, contrast)
            cavity_from_normal = cavity_from_normal.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
        else:
            node_text(unique_id, "No normal connected: cavity_from_normal is flat "
                                 "mid-gray (0.5) and carries no information.")
            cavity_from_normal = torch.full((B, H, W, 3), 0.5, device=device, dtype=dtype)

        cavity_from_height = cavity_from_height.cpu()
        cavity_from_normal = cavity_from_normal.cpu()
        return (cavity_from_height, cavity_from_normal)



NODE_CLASS_MAPPINGS = {
    "MoonPBRHeightAO": MoonHeightAO,
    "MoonPBRCavityMap": MoonCavityMap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonPBRHeightAO": "Height Ambient Occlusion",
    "MoonPBRCavityMap": "Cavity Map (Curvature Detector)",
}
