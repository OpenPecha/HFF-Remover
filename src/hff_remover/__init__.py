"""HFF Remover - Remove headers, footers, and footnotes from scanned book images."""

__version__ = "0.1.0"

from hff_remover.detector import (
    HFFDetector,
    PPDocLayoutDetector,
    EnsembleDetector,
    BaseHFFDetector,
)
from hff_remover.processor import HFFProcessor
from hff_remover.batch import BatchProcessor

__all__ = [
    "HFFDetector",
    "PPDocLayoutDetector",
    "EnsembleDetector",
    "BaseHFFDetector",
    "HFFProcessor",
    "BatchProcessor",
    "__version__",
]
