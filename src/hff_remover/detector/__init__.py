"""Document layout detectors for headers, footers, and footnotes."""

from hff_remover.detector._base import BaseHFFDetector
from hff_remover.detector._doclayout_yolo import (
    HFFDetector,
    DOCLAYOUT_YOLO_CLASS_NAMES,
    DOCLAYOUT_YOLO_HFF_CLASSES,
)
from hff_remover.detector._pp_doclayout import (
    PPDocLayoutDetector,
    PP_DOCLAYOUT_HFF_CLASSES,
)
from hff_remover.detector._yolo11_doclayout import (
    Yolo11DocLayoutDetector,
    YOLO11_DOCLAYOUT_CLASS_NAMES,
    YOLO11_DOCLAYOUT_HFF_CLASSES,
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
    "HFFDetector",
    "PPDocLayoutDetector",
    "Yolo11DocLayoutDetector",
    "SuryaLayoutDetector",
    "EnsembleDetector",
    "SURYA_HFF_CLASS_IDS",
    "SURYA_LABEL_TO_OUR_CLASS",
    "DOCLAYOUT_YOLO_CLASS_NAMES",
    "DOCLAYOUT_YOLO_HFF_CLASSES",
    "PP_DOCLAYOUT_HFF_CLASSES",
    "YOLO11_DOCLAYOUT_CLASS_NAMES",
    "YOLO11_DOCLAYOUT_HFF_CLASSES",
    "CLASS_NAMES",
    "HFF_CLASSES",
]
