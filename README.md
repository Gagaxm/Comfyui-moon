# comfyui-moon 🌕

**A pack of ComfyUI nodes for PBR texture work: seamless tiling, normal maps, ambient occlusion, relief control and AI depth.**

Vibe-coded nodes — results may vary, but the pack is self-contained: pure image math, no network access, no file writes outside your output folder, and no modification of ComfyUI's own behavior.

## What it does

- Makes any texture **seamlessly tileable**
- Generates and fixes **normal maps** (including AI-generated ones)
- Produces **ambient occlusion** and cavity maps from a height map
- Splits a height map into **macro / mid / high frequency bands** for relief control
- Adds an AI **depth** generator (PBRFusion4)
- Publishes renders to a fixed file **watched by your 3d software** (e.g. Maya)

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Gagaxm/Comfyui-moon
```

Restart ComfyUI. No extra dependencies for the core nodes.

**Depth nodes (`moon/depth`) additionally require the [PBRFusion4 model](https://huggingface.co/NightRaven109/PBRFusion4)**, placed in `ComfyUI/models/pbrfusion4`.

## Nodes by task

### Seamless tiling

| Node                              | What it does                                                                                                    |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Periodic+Smooth Decomposition** | Splits an image into a seamlessly tiling component + a low-frequency correction. Single pass, deterministic.    |
| **Circular Pad / Circular Unpad** | Wraps any filter (blur, sharpen…) so a seamless texture stays seamless. Wire Pad's outputs straight into Unpad. |
| **Image Blur**                    | Gaussian, box or radial blur. `wrap_mode: circular` = seamless on its own, no padding needed.                   |

### Normal maps

| Node                            | What it does                                                                                      |
| ------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Normal From Height (Scharr)** | Height → normal map, with independent macro/detail strength, OpenGL or DirectX, convexity invert. |
| **Blend Normal**                | Blends a detail normal onto a base. UDN and RNM (reoriented) modes.                               |
| **Normal Map Recenter**         | Fixes the directional bias of AI-generated normal maps (global or low-frequency correction).      |

### Relief, AO and cavities

| Node                          | What it does                                                                                                        |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Horizon Ambient Occlusion** | Physically-motivated AO from a height map (horizon mapping).                                                        |
| **Cavity Map**                | Detects flat-bottomed basins AO can't see. Substance convention: concave = darker.                                  |
| **Frequency Bands**           | Macro/mid/high decomposition with independent gains — kills the "pillow" look on convex elements (pebbles, bricks). |

### AI depth

| Node                  | What it does                                                                                                        |
| --------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **PBRFusion4 Loader** | Loads the depth model from a single `.safetensors` file.                                                            |
| **PBRFusion4 Depth**  | Single-step depth, no prompt required. 1536 px recommended for 12 GB VRAM; auto-retries at lower resolution on OOM. |

### Analysis

| Node                      | What it does                                                                                 |
| ------------------------- | -------------------------------------------------------------------------------------------- |
| **Height Diagnostics**    | Inspects a height map before conversion: halos, ringing, slope.                              |
| **Normal Map Bias Check** | Reports a normal map's R/G bias from the neutral midpoint — tells you if Recenter is needed. |
| **Channel Distribution**  | Per-channel value distribution (mean, median, percentiles) — pairs with Auto Remap Range.    |

### Image &amp; output utilities

| Node                               | What it does                                                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Exposure / Offset / Gamma**      | Color correction. `linear` mode for non-color data (height, roughness), `srgb` mode matching Photoshop's behavior. |
| **Split RGB and Alpha**            | Splits RGBA into an image + a proper ComfyUI mask.                                                                 |
| **Remap Range / Auto Remap Range** | Recalibrates level ranges (fixed or auto-computed from the image).                                                 |
| **Mean Channel**                   | Collapses channels to their true average — isolates inter-channel noise artifacts.                                 |
| **Preview Crop (1:1)**             | Fast square preview, mask untouched.                                                                               |
| **Publish Image**                  | Saves to a fixed filename, overwritten each run — for files watched by external apps.                              |
| **Previous Render Buffer**         | Keeps the previous run's render in memory (before/after comparison, etc.).                                         |
| **Clear Render Buffer**            | Frees the above buffers.                                                                                           |

## License

[MIT](LICENSE)
