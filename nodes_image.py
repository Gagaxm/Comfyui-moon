import torch

class MoonPreviewCrop:
    """Crop image (and optional mask) to a fixed square size for 1:1 pixel preview.

    Bypass conditions:
    - If both input dimensions are already <= crop_size, image and mask pass through unchanged.
    - If the mask's resolution doesn't match the image's resolution (invalid/placeholder mask),
      the mask passes through unchanged even when the image is cropped.
    - bypass disables cropping entirely regardless of resolution.
    - If no mask is connected, a 64x64 black mask is returned (ComfyUI's "no mask" convention).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_size": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "anchor_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Position of the crop's center point on X: 0 = crop pinned to "
                               "the left edge, 0.5 = crop centered, 1 = crop pinned to the right edge."}),
                "anchor_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Position of the crop's center point on Y: 0 = crop pinned to "
                               "the top edge, 0.5 = crop centered, 1 = crop pinned to the bottom edge."}),
                "bypass": ("BOOLEAN", {"default": False,
                    "tooltip": "Force passthrough: image and mask returned unchanged regardless of size."}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "crop"
    CATEGORY = "moon/image"
    DESCRIPTION = "Crops image+mask to a fixed square size for 1:1 pixel preview; passes through unchanged if already smaller or equal."

    def crop(self, image, crop_size, anchor_x, anchor_y, bypass, mask=None):
        if mask is None:
            mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")

        _, img_h, img_w, _ = image.shape

        if bypass or (img_h <= crop_size and img_w <= crop_size):
            return (image, mask)

        out_h = crop_size if img_h > crop_size else img_h
        out_w = crop_size if img_w > crop_size else img_w
        y0 = round((img_h - out_h) * anchor_y)
        x0 = round((img_w - out_w) * anchor_x)

        cropped_image = image[:, y0:y0 + out_h, x0:x0 + out_w, :]

        mask_h, mask_w = mask.shape[-2], mask.shape[-1]
        if mask_h == img_h and mask_w == img_w:
            cropped_mask = mask[:, y0:y0 + out_h, x0:x0 + out_w]
        else:
            cropped_mask = mask  # mismatched (or placeholder 64x64) mask, left untouched

        return (cropped_image, cropped_mask)


NODE_CLASS_MAPPINGS = {
    "MoonPreviewCrop": MoonPreviewCrop,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonPreviewCrop": "Preview Crop (1:1 Pixel)",
}