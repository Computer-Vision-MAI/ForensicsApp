"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            ChannelSplitTool(),
            ChannelSwapTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
