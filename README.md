# comfyui-moon 🌕

**A pack of ComfyUI nodes for PBR texture work: seamless tiling, normal maps, ambient occlusion, relief control and AI depth.**

Vibe-coded nodes — results may vary, but the pack is self-contained: no network access and no file writes outside ComfyUI-managed model/output folders, except for the explicit Publish Image destination. ComfyUI's own behavior is not modified.

## What it does

- Makes any texture **seamlessly tileable**
- Generates and fixes **normal maps**
- Produces **ambient occlusion** and cavity maps from a height map
- Splits a height map into **macro / mid / high frequency bands** for relief control
- Adds an AI **depth** generator (PBRFusion4)
- Publishes renders to a fixed file **watched by your 3d software** (e.g. Maya)

## Installation

The core nodes have no extra dependencies. The depth nodes (`moon/depth`) additionally require `diffusers` (`pip install diffusers`); `transformers` and `safetensors` normally already come with ComfyUI. If these are missing, only the depth nodes are disabled, with a console message.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Gagaxm/Comfyui-moon
```

Restart ComfyUI. No extra dependencies for the core nodes.

**Depth nodes (`moon/depth`) additionally require the [PBRFusion4 model](https://huggingface.co/NightRaven109/PBRFusion4)**, placed in `ComfyUI/models/pbrfusion4`.

## Nodes list

### Seamless tiling

|                                   |                                                                                                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Periodic+Smooth Decomposition** | Removes the border discontinuities so the image tiles seamlessly. Outputs the tileable image and the low-frequency correction that was removed.              |
| **Circular Pad / Circular Unpad** | Pad wraps the edges around the image, Unpad crops it back. Place any filter (blur, sharpen…) between them to avoid seams. Wire Pad's pad_x/pad_y into Unpad. |
| **Image Blur**                    | Gaussian, box or radial blur. wrap_mode: circular keeps a tileable image seamless.                                                                           |

<br>
### Normal maps

|                                 |                                                                                                                                               |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Normal From Height (Scharr)** | Converts a height map into a normal map. Separate strength for large relief and fine detail, OpenGL/DirectX output, circular wrap for tiling. |
| **Blend Normal**                | Combines a base normal map with a detail normal map (linear, whiteout or reoriented).                                                         |
| **Normal Map Recenter**         | Removes the overall tilt of a normal map so flat areas read as neutral. Corrects one global offset, or a varying low-frequency drift.         |

<br>
### Relief, AO and cavities

|                               |                                                                                                                                                                                          |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Horizon Ambient Occlusion** | Computes ambient occlusion from a height map. An optional normal map improves the handling of sloped surfaces.                                                                           |
| **Cavity Map**                | Darkens concave areas and brightens convex ones (flat = mid-gray). Gives one output from height and one from normal. Catches flat-bottomed grooves that AO misses.                       |
| **Frequency Bands**           | Splits a height map into macro / mid / high bands, each with its own gain, and recombines them into height_final. Lower the mid gain to flatten the "pillow" look on pebbles and bricks. |

<br>
### AI depth

|                       |                                                                                                                                                                              |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PBRFusion4 Loader** | Loads the PBRFusion4 .safetensors model from models/pbrfusion4.                                                                                                              |
| **PBRFusion4 Depth**  | Estimates a depth (height) map from an RGB image in a single pass, no prompt needed. max_resolution caps the working size; the result can be resized back to the input size. |

<br>
### Analysis

|                           |                                                                                                                      |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Height Diagnostics**    | Inspects a height map before conversion: halos, ringing, slope.                                                      |
| **Normal Map Bias Check** | Reports the average R/G/B of a normal map and its offset from neutral (127.5). Tells you whether Recenter is needed. |
| **Channel Distribution**  | Reports mean, median, percentiles and deviation per channel. The numeric outputs follow the selected channel.        |

<br>
### Image &amp; output utilities

|                                    |                                                                                                                                                                                         |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Exposure / Offset / Gamma**      | Adjusts exposure, offset and gamma. linear for data maps (height, roughness), srgb for color images (matches Photoshop).                                                                |
| **Split RGB and Alpha**            | Splits RGBA into an image + a proper ComfyUI mask.                                                                                                                                      |
| **Remap Range / Auto Remap Range** | Linearly maps an input value range to a new one (e.g. -1…1 → 0…1).                                                                                                                      |
| **Mean Channel**                   | Replaces R, G and B with their average (gray, 3 identical channels; isolates inter-channel noise artifacts).                                                                            |
| **Preview Crop (1:1)**             | Crops the image to a fixed-size square (default 1024 px) to inspect it at 1:1. The anchor_x/anchor_y inputs set the position. Passes through unchanged if the image is already smaller. |
| **Publish Image**                  | Saves images as PNG to a fixed folder and filename, overwritten on each run, so external apps (e.g. Maya) can reload the file.                                                          |
| **Previous Render Buffer**         | Outputs the image from the previous run next to the current one, for before/after comparison. keep freezes the reference.                                                               |
| **Clear Render Buffer**            | Frees the memory held by Previous Render Buffer.                                                                                                                                        |

## License

[MIT](LICENSE)
