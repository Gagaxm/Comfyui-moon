# 🌕 comfyui-moon 🌕

Custom ComfyUI nodes for PBR texture workflows — seamless tiling, ambient occlusion, and channel/publish utilities.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone <this-repo-url> comfyui-moon
```

No extra dependencies — uses `torch`, `numpy`, and `PIL`, all already bundled with ComfyUI. Restart ComfyUI after installing.

## Nodes

### 🌔 Moon Horizon AO (`MoonAO`) 🌖
*Category: `moon/pbr`*

Physically-based ambient occlusion from a height map, using horizon mapping (Zhukov, Iones & Kronin, 1998 — same principle as HBAO): for each texel, walks outward in multiple directions and distances, finds the steepest horizon angle per direction, and averages `sin(angle)` across directions.

Key inputs: `radius`, `directions`, `steps` (distance samples per direction), `height_scale`, `detail_bias` (biases sampling toward close distances for finer relief), `min_radius` (ignores micro-detail below a threshold), `wrap` (seamless/circular sampling — no CircularPad/Unpad needed). Optional `normal` input adds a stylistic extra darkening on down-facing normals.

### 🌔 Periodic+Smooth Decomposition (Moisan) (`PeriodicSmoothDecomposition`) 🌖
*Category: `moon/tiling`*

Implements Moisan (2011) periodic+smooth decomposition: splits an image into a `periodic` component (tiles seamlessly, same detail as the original) and a `smooth` component (the low-frequency correction absorbed at the borders). Single closed-form FFT pass — deterministic, no iteration.

### 🌔 Circular Pad (`CircularPad`) / Circular Unpad (`CircularUnpad`) 🌖
*Category: `moon/tiling`*

Sandwich a non-tiling-aware filter (blur, sharpen, any convolution-based node) to keep a seamless texture seamless. `CircularPad` wrap-pads the image using the opposite edge as context; `CircularUnpad` crops back to the original size. Wire `CircularPad`'s `pad_x`/`pad_y` outputs directly into the matching `CircularUnpad`.

### 🌔 Split RGB and Alpha (`ImageSplitRGBAndAlpha`) 🌖
*Category: `moon/conversion`*

Splits an RGBA image into a clean RGB `IMAGE` and a proper ComfyUI `MASK` tensor. If the input has no alpha channel, outputs a solid white mask instead of erroring.

### 🌔 Publish Image (`PublishImage`) 🌖
*Category: `moon/io`*

Saves a batch as 8-bit PNG to a fixed folder/filename, overwriting on every run — independent of `SaveImageAdvanced`. Useful for a fixed-name file watched by an external app (e.g. Maya). Toggle off with `active` to disable without disconnecting.

## License

TBD.
