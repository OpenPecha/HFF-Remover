"""Document layout detectors for headers, footers, and footnotes."""

from hff_remover.detector._base import BaseHFFDetector
from hff_remover.detector._yolo import (
    YoloDetector,
    YoloModelConfig,
    HFFDetector,
    EricYoloDetector,
    TDLADetector,
    TMBDLADetector,
    TDLA_CONFIG,
    DOCLAYOUT_YOLO_CONFIG,
    ERIC_YOLO_CONFIG,
    DOCLAYOUT_YOLO_CLASS_NAMES,
    DOCLAYOUT_YOLO_HFF_CLASSES,
    ERIC_YOLO_CLASS_NAMES,
    ERIC_YOLO_HFF_CLASSES,
    ERIC_YOLO_MODEL_TO_COCO,
    ERIC_YOLO_DEBUG_COLOR_MAP,
    TDLA_CLASS_NAMES,
    TDLA_HFF_CLASSES,
    TMB_DLA_CLASS_NAMES,
    TMB_DLA_HFF_CLASSES,
    draw_bboxes,
)
from hff_remover.detector._pp_doclayout import (
    PPDocLayoutDetector,
    PP_DOCLAYOUT_HFF_CLASSES,
)
from hff_remover.detector._surya_layout import (
    SuryaLayoutDetector,
    SURYA_HFF_CLASS_IDS,
    SURYA_LABEL_TO_OUR_CLASS,
)
from hff_remover.detector._ensemble import EnsembleDetector

# Backward-compatibility aliases
CLASS_NAMES = DOCLAYOUT_YOLO_CLASS_NAMES
HFF_CLASSES = DOCLAYOUT_YOLO_HFF_CLASSES

__all__ = [
    "BaseHFFDetector",
    "YoloDetector",
    "YoloModelConfig",
    "HFFDetector",
    "EricYoloDetector",
    "TDLADetector",
    "TMBDLADetector",
    "PPDocLayoutDetector",
    "SuryaLayoutDetector",
    "EnsembleDetector",
    "TDLA_CONFIG",
    "DOCLAYOUT_YOLO_CONFIG",
    "ERIC_YOLO_CONFIG",
    "DOCLAYOUT_YOLO_CLASS_NAMES",
    "DOCLAYOUT_YOLO_HFF_CLASSES",
    "ERIC_YOLO_CLASS_NAMES",
    "ERIC_YOLO_HFF_CLASSES",
    "ERIC_YOLO_MODEL_TO_COCO",
    "ERIC_YOLO_DEBUG_COLOR_MAP",
    "TDLA_CLASS_NAMES",
    "TDLA_HFF_CLASSES",
    "TMB_DLA_CLASS_NAMES",
    "TMB_DLA_HFF_CLASSES",
    "PP_DOCLAYOUT_HFF_CLASSES",
    "SURYA_HFF_CLASS_IDS",
    "SURYA_LABEL_TO_OUR_CLASS",
    "draw_bboxes",
    "CLASS_NAMES",
    "HFF_CLASSES",
]
