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
from .hog import HogTool
from .person_detection import PersonDetectionTool
from .corner_detection import CornerDetectionTool
from .keypoint_detection import KeypointDetectionTool
from .feature_matching import FeatureMatchingTool
from .image_lookup import ImageLookupTool
from .face_detection import FaceDetectionTool
from .webcam_capture import WebcamCaptureTool
from .eigenfaces_builder import EigenfacesBuilderTool
from .face_recognition import FaceRecognitionTool
from .bow_tools import BowClassifyTool, BowCodebookTool, BowHistogramTool
from .cnn_tools import CnnCompareTool
from .yolo_tools import (
    YoloBloodCellTool,
    YoloDatasetTool,
    YoloDetectTool,
    YoloEvaluateTool,
    YoloTrainTool,
)
from .cnn_model_tools import CnnClassifyTool, CnnProjectionTool
from .clip_tools import ClipSearchTool, ClipZeroShotTool
from .vit_tools import VitClassifyTool, VitCompareTool, VitPatchesTool, VitTrainTool
from .unet_tools import UnetCompareTool, UnetDatasetTool, UnetEvaluateTool, UnetSegmentTool, UnetTrainTool


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
            HogTool(),
            PersonDetectionTool(),
            CornerDetectionTool(),
            KeypointDetectionTool(),
            FeatureMatchingTool(),
            ImageLookupTool(),
            FaceDetectionTool(),
            WebcamCaptureTool(),
            EigenfacesBuilderTool(),
            FaceRecognitionTool(),
            BowCodebookTool(),
            BowHistogramTool(),
            BowClassifyTool(),
            CnnCompareTool(),
            YoloDetectTool(),
            YoloDatasetTool(),
            YoloTrainTool(),
            YoloBloodCellTool(),
            YoloEvaluateTool(),
            CnnClassifyTool(),
            CnnProjectionTool(),
            ClipZeroShotTool(),
            ClipSearchTool(),
            VitPatchesTool(),
            VitTrainTool(),
            VitClassifyTool(),
            VitCompareTool(),
            UnetDatasetTool(),
            UnetTrainTool(),
            UnetSegmentTool(),
            UnetEvaluateTool(),
            UnetCompareTool(),
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
    "FaceDetectionTool",
    "WebcamCaptureTool",
    "EigenfacesBuilderTool",
    "FaceRecognitionTool",
    "BowCodebookTool",
    "BowHistogramTool",
    "BowClassifyTool",
    "CnnCompareTool",
    "YoloDetectTool",
    "YoloDatasetTool",
    "YoloTrainTool",
    "YoloBloodCellTool",
    "YoloEvaluateTool",
    "CnnClassifyTool",
    "CnnProjectionTool",
    "ClipZeroShotTool",
    "ClipSearchTool",
    "VitPatchesTool",
    "VitTrainTool",
    "VitClassifyTool",
    "VitCompareTool",
    "UnetDatasetTool",
    "UnetTrainTool",
    "UnetSegmentTool",
    "UnetEvaluateTool",
    "UnetCompareTool",
]