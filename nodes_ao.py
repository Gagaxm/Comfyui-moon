"""
Included nodes:
/ao/ Horizon Ambient Occlusion
/ao/ Cavity Map (Curvature Detector)
"""

import math
import torch
import torch.nn.functional as F
import comfy.model_management as model_management

class MoonAO:
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
    CATEGORY = "moon/ao"
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
        device = model_management.get_torch_device()
        # TEMP: for debugging, measure the time and peak memory of the horizon search
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        # END TEMP
        dtype = torch.float32

        h = height.to(dtype).to(device)
        if h.shape[-1] >= 3:
            weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device, dtype=dtype)  # Rec.709/BT.709 luma weights (R,G,B)
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
            n = normal.to(dtype).to(device)
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

            # Streaming max over steps instead of stacking a (S,H,W) tensor
            # and reducing afterward: this is the change that removes the
            # `steps` factor entirely from peak memory. At 4K with the
            # default steps=8, the batched version was materializing
            # samp_x/samp_y/norm_x/norm_y/grid/src/sampled/height_diff/
            # elevation all at (S,H,W) or larger -- several GB per
            # direction, times 16 directions. This sequential form keeps
            # everything at (B,1,H,W), trading more (smaller) grid_sample
            # calls for a ~steps-fold reduction in peak VRAM.
            max_elev = torch.full((B, 1, H, W), -float("inf"), device=device, dtype=dtype)
            dist_at_max = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

            for s in range(S):
                dist = dists[s]  # 0-dim tensor, stays on device (no host sync)

                samp_x = base_x + dist * ux  # (H,W)
                samp_y = base_y + dist * uy

                norm_x = (samp_x / (Wp - 1)) * 2.0 - 1.0
                norm_y = (samp_y / (Hp - 1)) * 2.0 - 1.0
                grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0).expand(B, H, W, 2)

                sampled = F.grid_sample(
                    h_padded, grid, mode="bilinear", padding_mode="border", align_corners=True
                )  # (B,1,H,W) -- h_padded used directly, no per-step copy needed

                height_diff = (sampled - h_chw) * height_scale
                elevation = torch.atan(height_diff / dist)

                update = elevation > max_elev
                max_elev = torch.where(update, elevation, max_elev)
                dist_val = dist * torch.ones_like(dist_at_max)
                dist_at_max = torch.where(update, dist_val, dist_at_max)

            tangent_angle = torch.atan(grad_x * ux + grad_y * uy)  # (B,1,H,W)

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
        ao_rgb = ao.repeat(1, 1, 1, 3).cpu()
        # TEMP: for debugging, measure the time and peak memory of the horizon search
        t1.record()
        torch.cuda.synchronize()
        print(f"[MoonAO] {t0.elapsed_time(t1):.0f} ms, "
              f"peak VRAM {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
        # END TEMP
        return (ao_rgb,)





class MoonCavityMap:
    """
    Concavity/curvature detector — complementary to MoonAO, not a
    replacement for it.

    MoonAO answers "is there something taller than me nearby, in some
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
        (see MoonAO's `_blur` for why that ordering matters).

      - cavity_from_normal: derived from the normal map, using the
        divergence of its projected (nx, ny) field: div = d(nx)/dx +
        d(ny)/dy. Converging normals (pointing toward each other) signal
        a concave basin; diverging normals signal a convex bump. This can
        pick up curvature that a separately-generated height map missed
        or smoothed away, if height and normal came from independent
        estimators. The raw 1px derivative (bounded — no division is
        involved here, unlike MoonAO's tangent term) is smoothed over
        `radius` pixels before use, exactly mirroring cavity_from_height's
        averaging: a two-point stencil sampled `radius` pixels apart would
        only ever look at 2 pixels and ignore everything in between,
        making it extremely sensitive to per-pixel normal-map noise
        (dithering, ML-model grain) rather than the actual macro curvature.

    Both outputs use the same visualization convention as Substance
    Designer's curvature maps: flat = mid-gray (0.5), concave = darker,
    convex = brighter.

    This node does not combine these signals with MoonAO's output —
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
    CATEGORY = "moon/ao"
    DESCRIPTION = (
        "Detects concave basins (lowest points of a neighborhood) via curvature, "
        "independently from the height map and from the normal map, as two "
        "separate outputs for comparison. Complementary to MoonAO, which "
        "cannot detect a flat or gently-sloped basin by construction."
    )

    @staticmethod
    def _blur(x, kernel_radius, pad_mode):
        """Separable gaussian blur of a bounded quantity. See MoonAO's
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
        device = model_management.get_torch_device()
        dtype = torch.float32
        pad_mode = "circular" if wrap else "replicate"

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
            # is not the same situation as MoonAO's -nx/nz tangent term,
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
            print("[MoonCavityMap] No normal map connected — cavity_from_normal "
                  "output is flat mid-gray (0.5) and carries no information.")
            cavity_from_normal = torch.full((B, H, W, 3), 0.5, device=device, dtype=dtype)

        cavity_from_height = cavity_from_height.cpu()
        cavity_from_normal = cavity_from_normal.cpu()
        return (cavity_from_height, cavity_from_normal)



NODE_CLASS_MAPPINGS = {
    "MoonAO": MoonAO,
    "MoonCavityMap": MoonCavityMap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonAO": "Horizon Ambient Occlusion",
    "MoonCavityMap": "Cavity Map (Curvature Detector)",
}
