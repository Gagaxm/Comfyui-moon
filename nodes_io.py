"""
Included nodes:
/io/ Publish Image
/io/ Previous Render Buffer
/io/ Clear Render Buffer
"""

import os
import torch
import numpy as np
from PIL import Image

class PublishImage:
    """
    Publishes an image (8-bit PNG, RGB or RGBA) to a fixed path,
    alongside a SaveImageAdvanced. Overwrites the existing file on
    every run (useful for a fixed-name file watched by an external
    app, e.g. Maya).
    """
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "dest_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute folder path to write the PNG(s) to. Created automatically if it doesn't exist. Publication is skipped (with a console message) if left empty."
                }),
                "dest_filename": ("STRING", {
                    "default": "publish",
                    "tooltip": "Base filename (extension ignored/replaced with .png). Overwritten on every run. For a batch >1, each image gets a numeric suffix (_00, _01, ...)."
                }),
                "active": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Toggle off to disable publishing without disconnecting the node from the graph."
                }),
            }
        }
 
    RETURN_TYPES = ()
    FUNCTION = "publish"
    OUTPUT_NODE = True
    CATEGORY = "moon/io"
    DESCRIPTION = "Save batch in PNG 8-bit to a fixed name (overwrite), independently of SaveImageAdvanced."
 
    def publish(self, images, dest_folder, dest_filename, active):
        if not active:
            return {}
 
        if not dest_folder:
            print("[PublishImage] dest_folder is empty, publish skipped.")
            return {}
 
        os.makedirs(dest_folder, exist_ok=True)
 
        base_name = os.path.splitext(dest_filename)[0]
        batch_size = images.shape[0]
 
        for i in range(batch_size):
            arr = images[i].cpu().numpy()
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
 
            channels = arr.shape[-1]
            if channels == 4:
                mode = "RGBA"
            elif channels == 3:
                mode = "RGB"
            elif channels == 1:
                mode = "L"
                arr = arr[..., 0]
            else:
                print(f"[PublishImage] Unexpected channel count ({channels}), image skipped.")
                continue
 
            img = Image.fromarray(arr, mode=mode)
 
            suffix = f"_{i:02d}" if batch_size > 1 else ""
            fname = f"{base_name}{suffix}.png"
            fpath = os.path.join(dest_folder, fname)
 
            img.save(fpath, format="PNG")
            print(f"[PublishImage] Written: {fpath}")
 
        return {}


_BUFFER_STORE: dict[str, torch.Tensor] = {}
_BUFFER_KEEP_STATE: dict[str, bool] = {}  # key -> keep flag used on the previous run
_BUFFER_FROZEN: dict[str, torch.Tensor] = {}  # key -> frozen image (when keep was first activated)

class MoonPreviousRenderBuffer:
    """
    Returns the image (or batch) stored from the PREVIOUS execution, then
    overwrites the buffer with the current one for the NEXT execution.

    In-memory only (RAM, not VRAM, no disk I/O), similar in spirit to how
    feedback-loop workflows carry state between runs.

    The previous image is returned regardless of shape. A change in batch
    size, resolution, or channel count does not invalidate or replace the
    stored previous image prematurely. The native ComfyUI "Compare Images"
    node can receive the two images independently.

    `comparison_valid` indicates whether a previous buffer existed for this
    key. It does not indicate that the two images have matching shapes.

    Explicit keys are a deliberate sharing mechanism: nodes using the
    same key intentionally read/write the same buffer, with no ownership
    arbitration. Leave `key` blank for automatic per-node scoping.

    `keep` freezes the buffer: on the run it flips OFF->ON, the current
    image is committed as the frozen image, then stays frozen until `keep`
    is turned OFF again. `current_image` is always a straight bypass,
    independent of `keep`.

    Buffers are process-lifetime and never expire on their own -- use
    MoonClearRenderBuffer to free them manually if needed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "new_image": ("IMAGE",),
            },
            "optional": {
                "key": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Buffer key to compare against. Leave blank to "
                        "auto-scope by this node's own id (recommended, "
                        "avoids collisions). Set explicitly if you need "
                        "several nodes to share the same buffer -- sharing "
                        "is intentional, there's no conflict protection."
                    )
                }),
                "keep": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Freeze the buffer. When ON, the buffer keeps the "
                        "image from the first run where keep was activated, "
                        "and stays frozen until turned back OFF."
                    )
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("current_image", "previous_image", "comparison_valid")
    FUNCTION = "run"
    CATEGORY = "moon/io"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-run: this node's purpose is to expose the image from
        # the previous execution, so the normal ComfyUI cache mechanism
        # must not prevent run() from being called.
        return float("nan")

    @staticmethod
    def _to_owned_cpu(image):
        """
        Detach and move the image to CPU, with independent storage.

        The buffer is intentionally kept in RAM rather than VRAM.
        """
        return image.detach().cpu().clone()

    def run(self, new_image, unique_id, key="", keep=False):
        effective_key = key.strip() or f"node_{unique_id}"

        new_cpu = self._to_owned_cpu(new_image)
        was_keeping = _BUFFER_KEEP_STATE.get(effective_key, False)

        # --- Freeze handling ---
        if keep and not was_keeping:
            # First run with keep enabled:
            # freeze the current image.
            _BUFFER_FROZEN[effective_key] = new_cpu.clone()

        elif not keep and was_keeping:
            # keep has just been disabled:
            # leave frozen mode and resume the normal buffer.
            _BUFFER_FROZEN.pop(effective_key, None)

        # --- Retrieve previous_image ---
        frozen = _BUFFER_FROZEN.get(effective_key)

        if frozen is not None:
            # Frozen buffer: always return the same image.
            previous = frozen.clone()
            comparison_valid = True

        else:
            # Normal operation.
            stored = _BUFFER_STORE.get(effective_key)

            if stored is not None:
                # A previous image exists, regardless of its shape.
                previous = stored.clone()
                comparison_valid = True
            else:
                # First execution for this buffer.
                previous = new_cpu.clone()
                comparison_valid = False

            # Always replace the normal buffer with the current image.
            _BUFFER_STORE[effective_key] = new_cpu

        # Store the current keep state for the next execution.
        _BUFFER_KEEP_STATE[effective_key] = keep

        return (new_image, previous, comparison_valid)


class MoonClearRenderBuffer:
    """
    Utility node to free buffers held by MoonPreviousRenderBuffer.

    Buffers are process-lifetime and never expire on their own, so if you
    accumulate many keys (e.g. after renaming nodes or iterating on a
    graph) they can add up in RAM.

    Wire this in and run it once to clear a specific key, or leave `key`
    blank to clear everything.

    This node is a manual, explicit action -- it does not run automatically
    and does not track which keys are "stale"; that judgment is left to you.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("IMAGE",),
            },
            "optional": {
                "key": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Buffer key to clear. Leave blank to clear ALL "
                        "stored and frozen buffers."
                    )
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("trigger",)
    FUNCTION = "run"
    CATEGORY = "moon/io"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, trigger, key=""):
        effective_key = key.strip()

        if effective_key:
            removed_store = (
                _BUFFER_STORE.pop(effective_key, None) is not None
            )
            removed_frozen = (
                _BUFFER_FROZEN.pop(effective_key, None) is not None
            )
            _BUFFER_KEEP_STATE.pop(effective_key, None)

            removed = removed_store or removed_frozen

            print(
                f"[MoonClearRenderBuffer] Cleared key '{effective_key}' "
                f"({'was set' if removed else 'was already empty'})."
            )

        else:
            count = len(_BUFFER_STORE) + len(_BUFFER_FROZEN)

            _BUFFER_STORE.clear()
            _BUFFER_FROZEN.clear()
            _BUFFER_KEEP_STATE.clear()

            print(
                f"[MoonClearRenderBuffer] Cleared all {count} buffer(s)."
            )

        return (trigger,)



NODE_CLASS_MAPPINGS = {
    "PublishImage": PublishImage,
    "MoonPreviousRenderBuffer": MoonPreviousRenderBuffer,
    "MoonClearRenderBuffer": MoonClearRenderBuffer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PublishImage": "Publish Image",
    "MoonPreviousRenderBuffer": "Previous Render Buffer",
    "MoonClearRenderBuffer": "Clear Render Buffer",
}
