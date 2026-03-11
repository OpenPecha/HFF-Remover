"""YOLO11 Document Layout based HFF detector.

Uses the Armaggheddon/yolo11-document-layout model (ultralytics YOLO11)
trained on the DocLayNet dataset with 11 layout classes.  Unlike the
DocLayout-YOLO detector, this model has *explicit* Page-header, Page-footer,
and Footnote classes — no positional heuristics are needed.

Reference: https://huggingface.co/Armaggheddon/yolo11-document-layout
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from hff_remover.detector._base import BaseHFFDetector

# DocLayNet 11-class mapping used by the YOLO11 document-layout model.
# Order follows the model's training configuration.
YOLO11_DOCLAYOUT_CLASS_NAMES: Dict[int, str] = {
    0: "Text",
    1: "Title",
    2: "Section-header",
    3: "Table",
    4: "Picture",
    5: "Caption",
    6: "List-item",
    7: "Formula",
    8: "Page-header",
    9: "Page-footer",
    10: "Footnote",
}

# HFF-relevant subset: the classes we keep for header/footer/footnote removal.
YOLO11_DOCLAYOUT_HFF_CLASSES: Dict[int, str] = {
    0: "Text",          # body text
    8: "Page-header",   # page header
    9: "Page-footer",   # page footer
    10: "Footnote",     # footnote
}

# Map from model variant name to the weight filename on the Hub.
_MODEL_VARIANT_FILES: Dict[str, str] = {
    "nano": "yolo11n_doc_layout.pt",
    "small": "yolo11s_doc_layout.pt",
    "medium": "yolo11m_doc_layout.pt",
}

_HF_REPO_ID = "Armaggheddon/yolo11-document-layout"


def _normalize_hff_label(class_id: int) -> Optional[str]:
    """Map a YOLO11 DocLayNet class_id to a normalised HFF label.

    Returns:
        Normalised label string, or ``None`` if the class is not HFF-relevant.
    """
    if class_id == 0:
        return "text-area"
    if class_id == 8:
        return "header"
    if class_id == 9:
        return "footer"
    if class_id == 10:
        return "footnote"
    return None


class Yolo11DocLayoutDetector(BaseHFFDetector):
    """Detector for headers, footers, and footnotes using YOLO11 document-layout.

    The model is available in three variants (nano / small / medium) and is
    automatically downloaded from the HuggingFace Hub when no explicit
    ``model_path`` is provided.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_variant: str = "nano",
        device: str = "cuda",
        confidence_threshold: float = 0.5,
    ) -> None:
        """Initialise the YOLO11 DocLayout detector.

        Args:
            model_path: Path to custom YOLO11 model weights.  When ``None``
                the weights for *model_variant* are downloaded from
                HuggingFace Hub automatically.
            model_variant: One of ``"nano"``, ``"small"``, or ``"medium"``.
                Only used when *model_path* is ``None``.
            device: Device for inference (``"cuda"`` or ``"cpu"``).
            confidence_threshold: Minimum confidence score for detections.
        """
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        if model_variant not in _MODEL_VARIANT_FILES:
            raise ValueError(
                f"Unknown model_variant '{model_variant}'. "
                f"Choose from {list(_MODEL_VARIANT_FILES)}"
            )

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model_variant = model_variant

        if model_path is None:
            model_path = hf_hub_download(
                repo_id=_HF_REPO_ID,
                filename=_MODEL_VARIANT_FILES[model_variant],
                repo_type="model",
            )

        self.model = YOLO(model_path)
        self.model_path = model_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_hff_detections(
        self,
        boxes: Any,
    ) -> List[Dict[str, Any]]:
        """Extract HFF detections from an ultralytics ``Boxes`` object."""
        if boxes is None:
            return []

        detections: List[Dict[str, Any]] = []
        for j in range(len(boxes)):
            class_id = int(boxes.cls[j].item())
            label = _normalize_hff_label(class_id)
            if label is None:
                continue

            bbox = boxes.xyxy[j].cpu().numpy().tolist()
            confidence = float(boxes.conf[j].item())

            detections.append({
                "bbox": bbox,
                "class_id": class_id,
                "class_name": label,
                "confidence": confidence,
            })

        return detections

    # ------------------------------------------------------------------
    # Public API (BaseHFFDetector)
    # ------------------------------------------------------------------

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1280,
    ) -> List[Dict[str, Any]]:
        """Detect headers, footers, and footnotes in an image.

        Args:
            image: Path to an image file or a numpy array (BGR format).
            image_size: Input size for the model (default ``1280``, the
                resolution the model was trained at).

        Returns:
            List of detection dicts with keys ``bbox``, ``class_id``,
            ``class_name``, and ``confidence``.
        """
        results = self.model.predict(
            image,
            imgsz=image_size,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        for result in results:
            detections.extend(self._extract_hff_detections(result.boxes))

        return detections

    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        image_size: int = 1280,
        batch_size: int = 8,
    ) -> List[List[Dict[str, Any]]]:
        """Detect headers, footers, and footnotes in multiple images.

        Args:
            images: List of image paths or numpy arrays.
            image_size: Input size for the model.
            batch_size: Number of images to process at once.

        Returns:
            List of detection lists, one per input image.
        """
        all_detections: List[List[Dict[str, Any]]] = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            results = self.model.predict(
                batch,
                imgsz=image_size,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )

            for result in results:
                all_detections.append(
                    self._extract_hff_detections(result.boxes)
                )

        return all_detections

    # ------------------------------------------------------------------
    # Extra utility (matches HFFDetector API)
    # ------------------------------------------------------------------

    def get_all_detections(
        self,
        image: Union[str, Path, np.ndarray],
        image_size: int = 1280,
    ) -> List[Dict[str, Any]]:
        """Return *all* document-layout detections (not just HFF).

        Useful for debugging or visualisation.

        Args:
            image: Path to an image file or a numpy array.
            image_size: Input size for the model.

        Returns:
            List of detection dicts for all 11 DocLayNet classes.
        """
        results = self.model.predict(
            image,
            imgsz=image_size,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                confidence = float(boxes.conf[i].item())

                detections.append({
                    "bbox": bbox,
                    "class_id": class_id,
                    "class_name": YOLO11_DOCLAYOUT_CLASS_NAMES.get(
                        class_id, f"unknown_{class_id}"
                    ),
                    "confidence": confidence,
                })

        return detections
