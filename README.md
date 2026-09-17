# comfyui-moon 🌕

Custom ComfyUI nodes for PBR texture workflows — seamless tiling, ambient occlusion, normal maps, and channel/publish utilities.
**Vibe coded nodes, use at your own risks.**

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Gagaxm/Comfyui-moon
```

No extra dependencies for the core nodes — uses `torch`, `numpy`, and `PIL`, all already bundled with ComfyUI. Restart ComfyUI after installing.

The PBRFusion4 nodes (`moon/depth`) additionally require `diffusers`, `transformers`, and `safetensors`. If those aren't installed, PBRFusion4 nodes are silently disabled (with a console message) and every other node in this pack loads normally.

## Node files & categories

Nodes are grouped one file per category:

| File | Category | Contents |
|---|---|---|
| `nodes_image.py` | `moon/image` | Generic image operations |
| `nodes_tiling.py` | `moon/tiling` | Seamless-tiling utilities |
| `nodes_io.py` | `moon/io` | Publish / buffer / state nodes |
| `nodes_normal.py` | `moon/normal` | Normal-map generation & correction |
| `nodes_height.py` | `moon/height` | Height-map inspection, decomposition, remap |
| `nodes_ao.py` | `moon/ao` | Occlusion & curvature derived from height/normal |
| `nodes_pbrfusion4.py` | `moon/depth` | PBRFusion4 (Lotus-D) depth model |
| `nodes_debug.py` | `moon/debug` | Debug-only helper nodes |
| `common.py` | — | Shared internal helpers (not a node file) |

## Nodes

### moon/image

#### Image Blur

Three modes: `Gaussian` and `Box` (separable, two-pass, `samples = ceil(radius)`, `sigma = radius / 2`), and `Radial` (rotational sampling around the image center, 12 samples per side via `grid_sample`).

Key inputs: `blur_type`, `radius`, `wrap_mode` (`replicate` matches the original shader's edge behavior; `circular` makes the blur seamless on its own, without an external `CircularPad`/`CircularUnpad` sandwich).

#### Split RGB and Alpha

Splits an RGBA image into a clean RGB `IMAGE` and a proper ComfyUI `MASK` tensor. If the input has no alpha channel, outputs a solid white mask instead of erroring.

#### Exposure / Offset / Gamma

Exposure/Offset/Gamma color correction. Two modes via `color_space`:
- `linear` (default): operates directly on raw tensor values — the safe choice for non-color data (heightmaps, masks, roughness/normal channels).
- `srgb`: reproduces Photoshop's Exposure dialog / GIMP-GEGL's `gegl:exposure` behavior on display-referred sRGB images (e.g. an albedo pass). Offset is a black-point remap with self-compensating gain (lifts shadows without clipping highlights), not a plain additive shift.

Purely pointwise — `wrap_mode`-agnostic, kept for chain consistency only.

#### Channel Statistics

Computes per-channel (R, G, B[, A]) mean/min/max statistics for an image, with a human-readable report in either `0-1` or `0-255` display scale. R/G offsets from neutral (127.5 / 0.5) are reported to help spot directional bias in normal maps.

#### Preview Crop (1:1 Pixel)

Crops image (and optional mask) to a fixed square size for 1:1 pixel preview. Passes through unchanged if the image is already smaller or equal to `crop_size`, or if `bypass` is on. A mismatched/placeholder mask is left untouched even when the image is cropped.

### moon/tiling

#### Periodic+Smooth Decomposition (Moisan)

Implements Moisan (2011) periodic+smooth decomposition: splits an image into a `periodic` component (tiles seamlessly, same detail as the original) and a `smooth` component (the low-frequency correction absorbed at the borders). Single closed-form FFT pass — deterministic, no iteration.

#### Circular Pad / Circular Unpad

Sandwich a non-tiling-aware filter (blur, sharpen, any convolution-based node) to keep a seamless texture seamless. `CircularPad` wrap-pads the image using the opposite edge as context; `CircularUnpad` crops back to the original size. Wire `CircularPad`'s `pad_x`/`pad_y` outputs directly into the matching `CircularUnpad`.

### moon/io

#### Publish Image

Saves a batch as 8-bit PNG to a fixed folder/filename, overwriting on every run — independent of `SaveImageAdvanced`. Useful for a fixed-name file watched by an external app (e.g. Maya). Toggle off with `active` to disable without disconnecting.

#### Previous Render Buffer

Returns the image (or batch) stored from the PREVIOUS execution, then overwrites the buffer with the current one for the NEXT execution. In-memory only (RAM, not VRAM, no disk I/O). The previous image is returned regardless of shape — a change in batch size, resolution, or channel count doesn't invalidate the stored buffer. `keep` freezes the buffer at its current content until turned back off.

#### Clear Render Buffer

Frees buffers held by Previous Render Buffer. Buffers are process-lifetime and never expire on their own; wire this in and run it once to clear a specific key, or leave `key` blank to clear everything.

### moon/normal

#### Normal From Height (Scharr)

Converts a height/albedo-luminance map into a tangent-space normal map using 3×3 Scharr kernels (torch `conv2d`, no GLSL/GPU-shader dependency). Reduces the input to luminance, computes the gradient (with separate macro/detail bands), then builds `normalize(-grad.x, -grad.y, 1.0)` packed to `[0,1]`.

Key inputs: `scalar` (macro gradient strength), `detail`/`detail_radius` (high-frequency band isolated via a gaussian high-pass, independent from `scalar`), `flip` (swaps X/Y gradient channels), `invert_height` (flips gradient sign — inverts convexity), `normal_format` (`opengl`/`directx`, only the green channel differs), `intensity` (post-normalization X/Y rescale + renormalize), `wrap_mode` (`replicate` or `circular`). Always finishes with a global-offset recenter.

#### Blend Normal

Blends a detail normal map onto a base normal map. Purely pointwise (no neighbor sampling), so tiling is never affected regardless of `wrap_mode` elsewhere in the graph.

Three modes: `linear` (straight mix of the two unpacked/renormalized normals), `whiteout` (UDN — adds X/Y, multiplies Z), `reoriented` (RNM, Stephen Hill — reprojects the detail normal into the base normal's frame; default mode). `intensity` controls how much the detail normal is blended in before combination.

#### Normal Map Recenter

Recenters a normal map's R/G channels back around the neutral 127.5 midpoint, correcting the directional bias sometimes introduced by AI-generated normal maps (e.g. DeepBump). `global_offset` mode subtracts one average bias for the whole image; `highpass_blur` mode subtracts a heavily blurred (low-frequency) version instead, for bias that varies across the image (e.g. per-tile drift). `renormalize` rebuilds Z and normalizes back to unit length.

### moon/height

#### Height Diagnostics

Inspects a height map before Height → Normal conversion. Outputs `height_gray` (luminance used for analysis), `residual` (height minus its gaussian blur, centered at 0.5 — reveals local halos), `gradient` (slope magnitude), and `laplacian` (signed second derivative, centered at 0.5 — reveals ringing). `auto_contrast` switches display gain to a per-image 99th-percentile normalization for inspection.

#### Frequency Bands (Macro/Mid/High)

Multi-scale decomposition of a height/grayscale field into three additive bands (macro/mid/high) with independent gains — an InstaMAT-style Low/Medium/High relief control rather than a single DoG pass. Primary use: neutralizing the "pillow shape" artifact on convex elements (pebbles, bricks) via the mid band's gain.

`filter_type` chooses between plain Gaussian (strict, linear spectral bands) and Rolling Guidance Filter (edge-aware, preserves contours but non-linear — bands become "scale residuals" rather than strict frequency bands, more expensive). `band_macro`/`band_mid`/`band_high` outputs are raw (signed, pre-gain) for inspection — not remapped to `[0,1]`, expect out-of-range values when viewing them directly.

#### Remap Range

Generic linear remap: `[in_min, in_max] → [out_min, out_max]`. Common uses: previewing a signed field (e.g. `band_mid`/`band_high` from Frequency Bands) as viewable `[0,1]`, recalibrating an ML model's output range, or boosting visibility of a low-contrast signal. Values outside `[in_min, in_max]` extrapolate linearly unless `clamp_output` is on. Treat any downstream node expecting a specific convention (e.g. a normal decoder expecting `[-1,1]`) as needing the *original* signal, not a remapped one.

### moon/ao

#### Horizon Ambient Occlusion

Physically-motivated ambient occlusion from a height map via horizon mapping (Zhukov, Iones & Kronin 1998 / Bavoil, Sainz & Dimitrov 2008): for each texel, walks outward in multiple directions, finds the horizon angle relative to the surface's own local tangent plane (not a flat global reference), and integrates `sin(horizon) - sin(tangent)`.

Key inputs: `radius`, `directions`, `steps` (distance samples per direction, sub-pixel bilinear, spacing biased toward the texel via `detail_bias`), `height_scale`, `min_radius` (excludes micro-detail below a threshold), `wrap` (seamless sampling), `distance_falloff`, `tangent_scale` (smooths the local-slope estimate so it doesn't cancel out sharp contact seams). Optional `normal` input feeds the tangent-plane term directly instead of estimating slope from height.

#### Cavity Map (Curvature Detector)

Complementary to Horizon Ambient Occlusion, not a replacement: detects concave basins that are flat or gently sloped at the bottom (which horizon mapping cannot see by construction) via curvature instead of directional elevation. Two independent estimates, for side-by-side comparison:
- `cavity_from_height`: `blur(height) - height` — a pixel below its local average reads as concave.
- `cavity_from_normal`: divergence of the projected `(nx, ny)` normal field — converging normals signal a basin, diverging normals a bump.

Both follow Substance Designer's curvature convention: flat = mid-gray, concave = darker, convex = brighter.

### moon/depth

#### PBRFusion4 Loader

Loads the PBRFusion4 discriminative depth model (Lotus-D architecture, unet + vae + cached empty-prompt embedding) from a single `.safetensors` package. The text encoder/tokenizer are discarded right after computing the embedding — PBRFusion4 is only ever conditioned on an empty prompt.

#### PBRFusion4 Depth

Discriminative (Lotus-D) single-step depth inference, confirmed against the official `EnVision-Research/Lotus` pipeline: the UNet is called once on the encoded RGB latent, and its output IS the predicted target latent (no scheduler step, no second-latent concatenation). Output is raw — only the fixed VAE decode convention (`[-1,1] → [0,1]`) is applied, no clamp or channel reduction; normalization belongs in a downstream node.

`max_resolution` controls PBRFusion4's working resolution (inputs above it are downscaled proportionally, 1536px recommended for 12GB VRAM). `match_input_resolution` optionally resizes the decoded output back to the input's exact dimensions via plain bicubic resize (adds no generative detail). VAE tiling is intentionally never used (causes seam artifacts from per-tile GroupNorm statistics); an OOM catch-and-retry with resolution halving is the VRAM safety net instead.

### moon/debug

#### Mean Channel (debug)

Collapses an image's channels to their true average (not luminance-weighted), broadcast back to 3 channels. Used to isolate whether inter-channel noise (e.g. in a nominally grayscale depth output with small real R/G/B differences) is the source of artifacts appearing after channel-sensitive processing like frequency band extraction.

## License

[MIT](LICENSE)
