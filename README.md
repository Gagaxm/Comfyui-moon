# comfyui-moon 🌕

Custom ComfyUI nodes for PBR texture workflows — seamless tiling, ambient occlusion, normal maps, and channel/publish utilities.
**Vibe coded nodes, use at your own risks.**

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Gagaxm/Comfyui-moon
```

No extra dependencies — uses `torch`, `numpy`, and `PIL`, all already bundled with ComfyUI. Restart ComfyUI after installing.

## Nodes

### Normal From Height (Scharr)

_Category: `moon/pbr`_

Converts a height/albedo-luminance map into a tangent-space normal map using 3×3 Scharr kernels (torch `conv2d`, no GLSL/GPU-shader dependency). Reduces the input to luminance, computes the gradient, then builds `normalize(-grad.x, -grad.y, 1.0)` packed to `[0,1]`.

Key inputs: `scalar` (overall gradient strength), `detail` (pre-scalar gradient multiplier), `flip` (swaps the X/Y gradient channels), `invert_height` (flips gradient sign — inverts convexity), `wrap_mode` (`replicate` or `circular`; `circular` gives seamless tiling directly, no external `CircularPad`/`CircularUnpad` sandwich needed for this node).

### Blend Normal

_Category: `moon/pbr`_

Blends a detail normal map onto a base normal map. Purely pointwise (no neighbor sampling), so tiling is never affected regardless of `wrap_mode` elsewhere in the graph.

Three modes: `linear` (straight mix of the two unpacked/renormalized normals), `whiteout` (UDN — adds X/Y, multiplies Z), `reoriented` (RNM, Stephen Hill — reprojects the detail normal into the base normal's frame; default mode). `intensity` controls how much the detail normal is blended in before combination.

### Normal Map Recenter

_Category: `moon/pbr`_

Recenters a normal map's R/G channels back around the neutral 127.5 midpoint, correcting the directional bias sometimes introduced by AI-generated normal maps (e.g. DeepBump).

### Horizon Ambient Occlusion

_Category: `moon/pbr`_

Physically-based ambient occlusion from a height map, using horizon mapping (Zhukov, Iones & Kronin, 1998 — same principle as HBAO): for each texel, walks outward in multiple directions and distances, finds the steepest horizon angle per direction, and averages `sin(angle)` across directions.

Key inputs: `radius`, `directions`, `steps` (distance samples per direction), `height_scale`, `detail_bias` (biases sampling toward close distances for finer relief), `min_radius` (ignores micro-detail below a threshold), `wrap` (seamless/circular sampling — no CircularPad/Unpad needed). Optional `normal` input adds a stylistic extra darkening on down-facing normals.

### Channel Mean Stats

_Category: `moon/pbr`_

Computes the per-channel (R, G, B) mean pixel value of an image.
Built to check whether a normal map's R/G channels are centered around 127.5 (the neutral "flat surface" value), or whether it carries a directional bias (common with AI-generated normal maps like DeepBump).

### Periodic+Smooth Decomposition (Moisan)

_Category: `moon/tiling`_

Implements Moisan (2011) periodic+smooth decomposition: splits an image into a `periodic` component (tiles seamlessly, same detail as the original) and a `smooth` component (the low-frequency correction absorbed at the borders). Single closed-form FFT pass — deterministic, no iteration.

### Circular Pad / Circular Unpad

_Category: `moon/tiling`_

Sandwich a non-tiling-aware filter (blur, sharpen, any convolution-based node) to keep a seamless texture seamless. `CircularPad` wrap-pads the image using the opposite edge as context; `CircularUnpad` crops back to the original size. Wire `CircularPad`'s `pad_x`/`pad_y` outputs directly into the matching `CircularUnpad`.

### Image Blur

_Category: `moon/image`_

Three modes: `Gaussian` and `Box` (separable, two-pass, `samples = ceil(radius)`, `sigma = radius / 2`), and `Radial` (rotational sampling around the image center, 12 samples per side via `grid_sample`).

Key inputs: `blur_type`, `radius`, `wrap_mode` (`replicate` matches the original shader's edge behavior; `circular` makes the blur seamless on its own, without an external `CircularPad`/`CircularUnpad` sandwich).

### Split RGB and Alpha

_Category: `moon/image`_

Splits an RGBA image into a clean RGB `IMAGE` and a proper ComfyUI `MASK` tensor. If the input has no alpha channel, outputs a solid white mask instead of erroring.

### Publish Image

_Category: `moon/image`_

Saves a batch as 8-bit PNG to a fixed folder/filename, overwriting on every run — independent of `SaveImageAdvanced`. Useful for a fixed-name file watched by an external app (e.g. Maya). Toggle off with `active` to disable without disconnecting.

### Previous Render Buffer

_Category: `moon/image`_

Returns the image (or batch) stored from the PREVIOUS execution, then overwrites the buffer with the current one for the NEXT execution.
No disk I/O -- pure in-memory RAM buffer.

## License

[MIT](LICENSE)
