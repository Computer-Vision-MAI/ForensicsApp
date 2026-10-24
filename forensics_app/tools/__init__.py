"""Register course functionality here so it appears in the sidebar."""



from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry
from .mask import MaskTool
from .histogram import HistogramTool
from .contrast_stretch import ContrastStretchTool
from .convert_mode import ConvertModeTool
from .histogram_match import HistogramMatchTool
from .canny_edge import CannyEdgeTool
from .convolution import ConvolutionTool
from .filters import FiltersTool
from .sharpen import SharpenTool
from .template_matching import TemplateMatchingTool
from .corner_detection import CornerDetectionTool
from .keypoint_detection import KeypointDetectionTool
from .feature_matching import FeatureMatchingTool
from .image_lookup import ImageLookupTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            ChannelSplitTool(),
            ChannelSwapTool(),
            ConvertModeTool(),
            MaskTool(),
            HistogramTool(),
            ContrastStretchTool(),
            HistogramMatchTool(),
            CannyEdgeTool(),
            ConvolutionTool(),
            FiltersTool(),
            SharpenTool(),
            TemplateMatchingTool(),
            CornerDetectionTool(),
            KeypointDetectionTool(),
            FeatureMatchingTool(),
            ImageLookupTool(),
        ]
    )


__all__ = [
    "ToolRegistry",
    "build_tool_registry",
    "TemplateMatchingTool",
    "CornerDetectionTool",
    "KeypointDetectionTool",
    "FeatureMatchingTool",
    "ImageLookupTool",
]