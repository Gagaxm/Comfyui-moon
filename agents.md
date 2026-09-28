# agents.md — comfyui-moon

> Working notes for AI assistants working on this repository. The user-facing README stays functional and synthetic; this file holds the implementation-level context: interface contracts, conventions, wiring pitfalls and version changes. For full algorithms, read the docstrings in the `nodes_*.py` files — this file is the map, not the territory.

## Node files &amp; categories

Nodes are grouped one file per category:

| File                  | Category        | Contents                                                                                                                                     |
| --------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `nodes_image.py`      | `moon/image`    | Generic image operations (blur, exposure, split, crop, Mean Channel)                                                                         |
| `nodes_tiling.py`     | `moon/tiling`   | Seamless-tiling utilities                                                                                                                    |
| `nodes_io.py`         | `moon/io`       | Publish / buffer / state nodes                                                                                                               |
| `nodes_normal.py`     | `moon/normal`   | Normal-map generation &amp; correction                                                                                                       |
| `nodes_height.py`     | `moon/height`   | Height-map decomposition &amp; remap                                                                                                         |
| `nodes_ao.py`         | `moon/ao`       | Occlusion &amp; curvature derived from height/normal                                                                                         |
| `nodes_analysis.py`   | `moon/analysis` | Diagnostics: Height Diagnostics, Normal Map Bias Check, Channel Distribution                                                                 |
| `nodes_pbrfusion4.py` | `moon/depth`    | PBRFusion4 (Lotus-D) depth model — auto-disabled with a console notice if the dependency is missing (wrapped in try/except in `__init__.py`) |
| `common.py`           | —               | Shared internal helpers (not a node file)                                                                                                    |

**History note:** `nodes_debug.py` / `moon/debug` no longer exists. Its only node, **Mean Channel**, now lives in `nodes_image.py` (`moon/image`). Height Diagnostics moved from `moon/height` to `moon/analysis`.

## Common conventions

- **Tensors:** ComfyUI `IMAGE` = `(B, H, W, C)` in `[0, 1]`; `MASK` = `(B, H, W)`. FLOAT outputs stay in the native `[0, 1]` range.
- **`wrap_mode`** (`replicate` | `circular`): appears in every node that samples neighbors (blur, gradients, AO, decomposition). `circular` keeps seamless textures seamless; `replicate` matches the original shader's edge behavior. Pointwise nodes (Blend Normal, Exposure/Offset/Gamma, Remap) are wrap-agnostic.
- **Categories** are all `moon/<area>`; display names drop the `Moon` prefix (`MoonAO` → "Horizon Ambient Occlusion").
- **Devices:** compute runs on the torch device from `comfy.model_management`; render buffers are kept in RAM (CPU), never VRAM.

## Interface contracts &amp; wiring pitfalls

### moon/tiling

- **Periodic+Smooth Decomposition (Moisan 2011):** closed-form single FFT pass, deterministic. Outputs: `periodic` (tiles seamlessly, same detail as input) + `smooth` (the low-frequency border correction that was removed — kept for debugging/visualization).
- **Circular Pad / Circular Unpad:** sandwich any non-tiling-aware filter (blur, sharpen, any convolution) to preserve seamlessness. Pad wrap-pads using the opposite edge as context; Unpad crops back. **Wire Pad's `pad_x`/`pad_y` outputs directly into Unpad's matching inputs** — they carry the pad amounts; do not retype them.

### moon/normal

- **Normal From Height (Scharr):** 3×3 Scharr kernels via `torch.conv2d` (no GLSL/GPU-shader dependency). Input reduced to luminance first. Key inputs: `scalar` (macro gradient strength), `detail`/`detail_radius` (high-frequency band via gaussian high-pass, independent of `scalar`), `flip` (swaps X/Y gradient channels), `invert_height` (flips gradient sign — inverts convexity), `normal_format` (`opengl` | `directx` — only the green channel differs), `intensity` (post-normalization X/Y rescale + renormalize), `wrap_mode`. Always finishes with a global-offset recenter.
- **Blend Normal:** pointwise (never breaks tiling). Modes: `linear` (mix of unpacked/renormalized normals), `whiteout` (UDN — add X/Y, multiply Z), `reoriented` (RNM, Stephen Hill — default; reprojects detail into the base normal's frame). `intensity` = how much detail normal is blended in before combination.
- **Normal Map Recenter:** corrects directional bias typical of AI-generated normal maps (e.g. DeepBump). `global_offset` subtracts one average bias for the whole image; `highpass_blur` subtracts a heavily blurred version instead (for per-tile drift). `renormalize` rebuilds Z and renormalizes.

### moon/ao — V2 interface change

**The `normal_bias` widget is gone (v1 → v2).** In v1 it was a post-hoc multiplicative darkening based on normal.z, redundant with the height-derived slope and prone to double-counting. In v2 a connected `normal` input feeds the tangent-plane term directly. **Saved workflow JSONs wiring a value into `normal_bias` need that link removed.** A `distance_falloff` toggle was added.

- **Horizon Ambient Occlusion:** horizon mapping (Zhukov/Iones/Kronin 1998; Bavoil/Sainz/Dimitrov 2008). For each texel, walks outward in multiple directions, finds the horizon angle relative to the surface's own local tangent plane (not a flat global reference) and integrates `sin(horizon) - sin(tangent)`. V2: sub-pixel bilinear sampling via `grid_sample` (removes cardinal/diagonal aliasing bias), elevation computed against the actual sampled distance, optional `distance_falloff`, sampling starts at `min_radius` (1px floor), tangent estimate smoothed over `tangent_scale * radius` (an un-smoothed 1px tangent estimate cancels sharp contact seams). Key inputs: `radius`, `directions`, `steps`, `height_scale`, `detail_bias`, `wrap`.
- **Cavity Map (Curvature Detector):** complements AO, not a replacement — detects concave basins that are flat or gently sloped at the bottom, which horizon mapping cannot see by construction. Two independent outputs for side-by-side comparison: `cavity_from_height` (`blur(height) - height`) and `cavity_from_normal` (divergence of projected `(nx, ny)`). Substance Designer convention: flat = mid-gray, concave = darker, convex = brighter. Key inputs: `strength`, `min_radius`, `distance_falloff`, `tangent_scale`.

### moon/height

- **Frequency Bands (Macro/Mid/High):** additive 3-band decomposition with independent gains — InstaMAT-style relief control. Main use: neutralizing the "pillow shape" artifact on convex elements (pebbles, bricks) via the mid band. `filter_type`: plain Gaussian (strict linear spectral bands) or Rolling Guidance Filter (edge-aware, preserves contours, non-linear — bands become "scale residuals", more expensive). **`band_macro`/`band_mid`/`band_high` are raw signed outputs, NOT remapped to `[0,1]`** — expect out-of-range values; use MoonRemapRange to preview them.
- **Remap Range:** linear remap `[in_min, in_max] → [out_min, out_max]`; values outside extrapolate unless `clamp_output`. **Do not remap a signal that a downstream node expects in its original convention** (e.g. a normal decoder expecting `[-1,1]`).
- **Auto Remap Range:** computes robust bounds from the image's own distribution — **connect its `in_min`/`in_max` FLOAT outputs into MoonRemapRange's matching inputs**.

### moon/analysis

- **Height Diagnostics:** pre-conversion inspection. Outputs: `height_gray` (Rec.709 luminance used for analysis, alpha ignored), `residual` (height − gaussian blur, centered 0.5 — local halos), `gradient` (slope magnitude), `laplacian` (signed second derivative, centered 0.5 — ringing). `auto_contrast` switches to per-image 99th-percentile display normalization (inspection only). Also prints per-band stats to the console.
- **Normal Map Bias Check:** R/G channel mean/min/max from the neutral 127.5 (0.5) midpoint, with offsets — run before NormalMapRecenter to decide whether correction is needed. FLOAT outputs stay in `[0,1]`; `display_scale` only affects the report. Replaces the old "Channel Statistics" node.
- **Channel Distribution:** per-channel mean/median/percentiles/std; numeric outputs follow the selected channel. Percentile outputs pair with Remap Range for auto-calibrated remaps.

### moon/depth (PBRFusion4)

- **PBRFusion4 Loader:** loads the discriminative depth model (Lotus-D architecture: unet + vae + cached empty-prompt embedding) from a single `.safetensors` package; text encoder/tokenizer discarded right after computing the embedding — the model is only ever conditioned on an empty prompt. Model files go in `ComfyUI/models/pbrfusion4` (default `PBRFusion4.safetensors`).
- **PBRFusion4 Depth:** single-step inference, validated against the official `Envision-Research/Lotus` pipeline: the UNet is called once on the encoded RGB latent and its output IS the predicted target latent (no scheduler step, no second-latent concatenation). **Output is raw** — only the fixed VAE decode convention (`[-1,1] → [0,1]`), no clamp, no channel reduction; normalization belongs downstream. `max_resolution` = working resolution (inputs downscaled proportionally; **1536 px recommended for 12 GB VRAM**). `match_input_resolution` resizes back to the input's exact dimensions via bicubic (adds no generative detail). **VAE tiling is intentionally never used** (per-tile GroupNorm statistics cause seam artifacts); OOM is handled by catch-and-retry with resolution halving.

### moon/io

- **Publish Image:** saves a batch as 8-bit PNG to a fixed folder/filename, **overwriting on every run** — independent of `SaveImageAdvanced`, for files watched by external apps (e.g. Maya). `active` toggles it off without disconnecting.
- **Previous Render Buffer:** returns the image stored from the PREVIOUS execution, then overwrites the buffer with the current one. In-memory only (RAM, not VRAM, no disk I/O). Shape-agnostic: batch size, resolution or channel count changes do not invalidate the stored buffer. `keep` freezes the buffer until turned off.
- **Clear Render Buffer:** frees buffers held by Previous Render Buffer. **Buffers are process-lifetime and never expire on their own** — wire this in and run it once to clear a key, or leave `key` blank to clear everything.

### moon/image

- **Image Blur:** three modes — `Gaussian` and `Box` (separable two-pass, `samples = ceil(radius)`, `sigma = radius/2`), `Radial` (rotational sampling around the center, 12 samples per side via `grid_sample`). With `wrap_mode: circular` the blur is seamless on its own; no CircularPad/Unpad sandwich needed.
- **Exposure / Offset / Gamma:** pointwise, wrap-agnostic. `linear` mode (default) operates on raw tensor values — safe for non-color data (heightmaps, masks, roughness/normal channels). `srgb` mode reproduces Photoshop's Exposure dialog / GIMP-GEGL `gegl:exposure` on display-referred sRGB; Offset is a black-point remap with self-compensating gain (lifts shadows without clipping highlights), not a plain additive shift.
- **Mean Channel:** collapses channels to their true average (not luminance-weighted), broadcast to 3 channels — isolates whether inter-channel noise is the source of artifacts after channel-sensitive processing (e.g. frequency band extraction on a nominally grayscale depth output).
- **Split RGB and Alpha:** RGBA → clean RGB `IMAGE` + proper ComfyUI `MASK`. If the input has no alpha channel, outputs a solid white mask instead of erroring.
- **Preview Crop (1:1 Pixel):** crops image (and optional mask) to a fixed square; passes through unchanged if already ≤ `crop_size` or `bypass` is on. A mismatched/placeholder mask is left untouched even when the image is cropped.
