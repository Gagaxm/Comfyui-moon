## Node files & categories

Nodes are grouped one file per category:

| File                  | Category      | Contents                                         |
| --------------------- | ------------- | ------------------------------------------------ |
| `nodes_image.py`      | `moon/image`  | Generic image operations                         |
| `nodes_tiling.py`     | `moon/tiling` | Seamless-tiling utilities                        |
| `nodes_io.py`         | `moon/io`     | Publish / buffer / state nodes                   |
| `nodes_normal.py`     | `moon/normal` | Normal-map generation & correction               |
| `nodes_height.py`     | `moon/height` | Height-map inspection, decomposition, remap      |
| `nodes_ao.py`         | `moon/ao`     | Occlusion & curvature derived from height/normal |
| `nodes_pbrfusion4.py` | `moon/depth`  | PBRFusion4 (Lotus-D) depth model                 |
| `nodes_debug.py`      | `moon/debug`  | Debug-only helper nodes                          |
| `common.py`           | —             | Shared internal helpers (not a node file)        |
