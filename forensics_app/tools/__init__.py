"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry
from .mask import MaskTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            ChannelSplitTool(),
            MaskTool()
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
