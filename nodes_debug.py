"""
Included nodes:
/debug/ Mean Channel (debug)
"""


class MoonMeanChannels:
    """
    Test/diagnostic node: collapses an IMAGE's channels to their mean,
    then broadcasts back to 3 identical channels (keeps IMAGE type
    compatible with downstream nodes that expect RGB shape).

    Used to isolate whether inter-channel noise (e.g. in PBRFusion4's
    decoded_depth, which is nominally grayscale but has small real
    differences between R/G/B) is the source of artifacts that appear
    after channel-sensitive processing like frequency band extraction.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mean_image",)
    FUNCTION = "run"
    CATEGORY = "moon/debug"
    DESCRIPTION = "Collapses channels to their mean (true average, not luminance-weighted), broadcast back to 3 channels."

    def run(self, image):
        mean = image.mean(dim=-1, keepdim=True)
        return (mean.expand(-1, -1, -1, 3).contiguous(),)



NODE_CLASS_MAPPINGS = {
    "MoonMeanChannels": MoonMeanChannels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoonMeanChannels": "Mean Channel (debug)",
}
