class MoonCenterCrop:
    """Center-crop image (and optional mask) to a fixed square size for 1:1 pixel preview.

    Bypass conditions:
    - If both input dimensions are already <= crop_size, image and mask pass through unchanged.
    - If the mask's resolution doesn't match the image's resolution (invalid/placeholder mask),
      the mask passes through unchanged even when the image is cropped.
    - force_bypass disables cropping entirely regardless of resolution.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_size": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "position": (["center"], {"default": "center"}),  # TODO: corner options later
                "force_bypass": ("BOOLEAN", {"default": False}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "crop"
    CATEGORY = "moon/utils"

    def crop(self, image, crop_size, position, force_bypass, mask=None):
        _, img_h, img_w, _ = image.shape

        if force_bypass or (img_h <= crop_size and img_w <= crop_size):
            return (image, mask)

        out_h = crop_size if img_h > crop_size else img_h
        out_w = crop_size if img_w > crop_size else img_w
        y0, x0 = self._offsets(img_h, img_w, out_h, out_w, position)

        cropped_image = image[:, y0:y0 + out_h, x0:x0 + out_w, :]

        cropped_mask = mask
        if mask is not None:
            mask_h, mask_w = mask.shape[-2], mask.shape[-1]
            mask_valid = (mask_h == img_h) and (mask_w == img_w)
            if mask_valid:
                cropped_mask = mask[:, y0:y0 + out_h, x0:x0 + out_w]
            # else: mismatched mask, left untouched at its own resolution

        return (cropped_image, cropped_mask)

    @staticmethod
    def _offsets(img_h, img_w, out_h, out_w, position):
        if position == "center":
            return (img_h - out_h) // 2, (img_w - out_w) // 2
        raise NotImplementedError(f"Crop position '{position}' not implemented yet")

NODE_CLASS_MAPPINGS = {
    "MoonCenterCrop": MoonCenterCrop,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonCenterCrop": "Center Crop",
}