# HFF Remover

<h1 align="center">
  <br>
  <a href="https://openpecha.org"><img src="https://avatars.githubusercontent.com/u/82142807?s=400&u=19e108a15566f3a1449bafb03b8dd706a72aebcd&v=4" alt="OpenPecha" width="150"></a>
  <br>
</h1>

Remove **H**eaders, **F**ooters, and **F**ootnotes from scanned book images using multiple layout-detection backends.

## Overview

HFF Remover is a Python package that automatically detects and masks headers, footers, and footnotes in scanned book page images. It ships with **six pluggable detector backends** and an **ensemble/cascade strategy** that can combine any of them. All detectors normalize their output to a shared four-class label set (`header`, `footer`, `footnote`, `text-area`), enabling uniform post-processing and evaluation across backends.

The package also includes a **COCO-style mAP evaluation pipeline** and a **benchmark dataset** based on [openpecha/Tibetan-header-footer](https://huggingface.co/datasets/openpecha/Tibetan-header-footer) (879 ground-truth boxes across 4 classes) used to compare detector performance.

This project is built under the scope of the **BDRC eText Corpus Project**.

## Features

- **Six Detector Backends**: DocLayout-YOLO, YOLO11 DocLayout, PP-DocLayout-L (PaddlePaddle), Surya Layout, Eric's tiled YOLO11-nano, and ensemble/cascade combinations
- **Batch Processing**: Process thousands of images with GPU acceleration, threaded I/O, and checkpoint-based resumption
- **Evaluation Pipeline**: mAP\@0.50, mAP\@0.50:0.95 (COCO), per-class AP/Precision/Recall with detailed reporting
- **Multiple Output Formats**: COCO-format YOLO labels or masked overlay images
- **Mixed Image Formats**: JPEG, PNG, TIFF, BMP, and WebP
- **Configurable**: Adjustable confidence thresholds, padding, tiling parameters, and output quality
- **CLI & Python API**: Both command-line and programmatic interfaces

## Package Structure

```
src/hff_remover/
├── __init__.py              # Public API re-exports, version
├── cli.py                   # CLI entry point (process, single, detect, evaluate)
├── batch.py                 # BatchProcessor: threaded load/save, checkpoints, reporting
├── processor.py             # HFFProcessor (box merging), COCODatasetWriter, MaskedInferenceImageWriter
├── evaluate.py              # mAP evaluation: IoU matching, AP computation, report generation
├── utils.py                 # Image discovery, load/save, resize helpers
└── detector/
    ├── __init__.py           # Exports all detectors and constants
    ├── _base.py              # BaseHFFDetector abstract class (detect, detect_batch, label normalization)
    ├── _doclayout_yolo.py    # HFFDetector — DocLayout-YOLO (splits "abandon" → header/footer by vertical position)
    ├── _yolo11_doclayout.py  # Yolo11DocLayoutDetector — Ultralytics YOLO11 with DocLayNet-style classes
    ├── _pp_doclayout.py      # PPDocLayoutDetector — PaddleOCR PPStructure layout
    ├── _surya_layout.py      # SuryaLayoutDetector — Surya foundation model with string labels
    ├── _eric_yolo.py         # EricYoloDetector — tiled YOLO11-nano (640x640 tiles, cross-tile box merging)
    └── _ensemble.py          # EnsembleDetector — union (NMS-like suppression) or cascade (first non-empty)
```

## Installation

```bash
git clone https://github.com/OpenPecha/HFF-Remover.git
cd HFF-Remover
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended for large batches)
- PyTorch >= 2.0.0

### Dependencies

Core: `doclayout-yolo`, `torch`, `torchvision`, `opencv-python`, `Pillow`, `tqdm`, `huggingface-hub`, `ultralytics`, `paddleocr==2.9.0`, `paddlepaddle==3.0.0`, `surya-ocr>=0.17.0`

## Quick Start

### Command Line

```bash
# Process all images in a directory
hff-remover process /path/to/images --output /path/to/output

# Process with GPU and custom settings
hff-remover process /path/to/images --output /path/to/output \
    --device cuda \
    --batch-size 32 \
    --confidence 0.5 \
    --padding 5

# Resume interrupted processing
hff-remover process /path/to/images --output /path/to/output --resume

# Process a single image
hff-remover single input.jpg output.jpg

# Detect without masking (view coordinates)
hff-remover detect input.jpg

# Evaluate predictions against ground truth
hff-remover evaluate --gt-dir data/benchmark_dataset/label \
    --pred-dir data/surya_inference_1500/labels
```

### Python API

```python
from hff_remover import HFFDetector, HFFProcessor, BatchProcessor
from hff_remover.utils import load_image, save_image

detector = HFFDetector(device="cuda", confidence_threshold=0.5)
processor = HFFProcessor(padding=5)

image = load_image("input.jpg")
detections = detector.detect(image)
detections = processor.merge_nearby_detections(detections)
result = processor.mask_regions(image, detections)
save_image(result, "output.jpg")
```

### Using Different Backends

```python
from hff_remover.detector import (
    HFFDetector,            # DocLayout-YOLO
    Yolo11DocLayoutDetector, # YOLO11 DocLayout
    PPDocLayoutDetector,     # PaddlePaddle PP-DocLayout-L
    SuryaLayoutDetector,     # Surya foundation model
    EricYoloDetector,        # Tiled YOLO11-nano
    EnsembleDetector,        # Combine multiple detectors
)

# Surya (best mAP@0.50 on benchmark)
detector = SuryaLayoutDetector(confidence_threshold=0.5)

# Eric's tiled YOLO11-nano
detector = EricYoloDetector(
    model_path="data/eric_yolo_hff_best.pt",
    device="cpu",
    confidence_threshold=0.3,
)

# Ensemble: run multiple detectors, merge with IoU suppression
yolo = HFFDetector(device="cpu", confidence_threshold=0.5)
surya = SuryaLayoutDetector(confidence_threshold=0.5)
detector = EnsembleDetector(
    detectors=[yolo, surya],
    merge_strategy="union",  # or "cascade"
)
```

### Running Evaluation

```python
from hff_remover.evaluate import evaluate

result = evaluate(
    gt_dir="data/benchmark_dataset/label",
    pred_dir="data/surya_inference_1500/labels",
)

print(f"mAP@0.50:      {result.map_50:.4f}")
print(f"mAP@0.50:0.95: {result.map_50_95:.4f}")
```

See `example.py` for full inference usage and `example_evaluate.py` for detailed report generation.

## Detection Approaches

All detectors inherit from `BaseHFFDetector` and normalize their native class labels to the canonical set:

| ID | Class | Description |
|----|-------|-------------|
| 0 | `header` | Running headers, page titles |
| 1 | `text-area` | Main body text |
| 2 | `footnote` | Footnotes and table footnotes |
| 3 | `footer` | Running footers, page numbers |

### Backend Details

| Backend | Model | Key Technique |
|---------|-------|---------------|
| **DocLayout-YOLO** | `doclayout_yolo.YOLOv10` (HuggingFace weights) | Splits "abandon" class into header/footer by vertical band position |
| **YOLO11 DocLayout** | `ultralytics` YOLO11 (nano) | Explicit Page-header/Page-footer/Footnote classes from DocLayNet |
| **PP-DocLayout-L** | PaddleOCR `PPStructure` | 23-class layout model with string-label normalization |
| **Surya Layout** | Surya `LayoutPredictor` foundation model | String labels (`PageHeader`, `PageFooter`, `Footnote`) mapped to HFF |
| **Eric YOLO** | Custom YOLO11-nano trained at 640x640 | Page scaled to 2x tile width, sliding 640x640 tiles, cross-tile box merging |
| **Ensemble** | Any combination of the above | `union` (concat + NMS suppression) or `cascade` (first non-empty wins) |

Post-processing across all backends: `HFFProcessor.merge_nearby_detections` iteratively merges same-class boxes that overlap or fall within a configurable pixel margin.

## Benchmark Results

All detectors were evaluated against the same benchmark dataset (879 ground-truth boxes, 4 classes) using COCO-style mAP metrics. Full reports are available in the `data/` folder.

### Overall Performance

| Detector | mAP\@0.50 | mAP\@0.50:0.95 | Pred Boxes |
|----------|-----------|-----------------|------------|
| **Surya Layout** | **0.8221** | 0.4902 | 864 |
| **Ensemble** (YOLO+Paddle+Surya+YOLO11) | 0.7095 | 0.4349 | 984 |
| **PaddleVLM** | 0.6890 | **0.4890** | 907 |
| **Eric YOLO** (tiled) | 0.6509 | 0.4628 | 1102 |
| **dots.ocr** | 0.6128 | 0.3707 | 857 |
| **DocLayout-YOLO** | 0.3807 | 0.3244 | 814 |
| **PP-DocLayout-L** | 0.3124 | 0.1668 | 623 |
| **Gemini 2.5 Flash** | 0.2830 | 0.1215 | 908 |
| **Gemini 3.1 Pro Preview** | 0.2691 | 0.1386 | 909 |
| **YOLO11 DocLayout** | 0.1839 | 0.1078 | 448 |

### Per-Class AP @ IoU 0.50

| Detector | Header | Text-Area | Footnote | Footer |
|----------|--------|-----------|----------|--------|
| Surya Layout | 0.7777 | **0.9584** | 0.8287 | **0.7239** |
| Ensemble | 0.4192 | 0.8387 | **0.8465** | 0.7336 |
| PaddleVLM | 0.5131 | 0.9655 | 0.8261 | 0.4513 |
| Eric YOLO | 0.5242 | 0.8790 | 0.6665 | 0.5339 |
| dots.ocr | 0.6015 | 0.8974 | 0.7492 | 0.2033 |
| DocLayout-YOLO | 0.3699 | 0.5866 | 0.0000 | 0.5664 |
| PP-DocLayout-L | 0.2427 | 0.5520 | 0.0000 | 0.4548 |
| Gemini 2.5 Flash | 0.0528 | 0.9074 | 0.1581 | 0.0137 |
| Gemini 3.1 Pro | 0.3876 | 0.2369 | 0.2427 | 0.2091 |
| YOLO11 DocLayout | 0.0019 | 0.5327 | 0.0122 | 0.1888 |

**Key findings:**
- **Surya Layout** achieves the highest mAP\@0.50 (0.8221) with strong, balanced performance across all classes
- **PaddleVLM** leads on mAP\@0.50:0.95 (0.4890), indicating tighter bounding-box localization
- **Ensemble** yields the best footnote AP (0.8465) by combining detections from multiple models
- **DocLayout-YOLO** and **PP-DocLayout-L** miss footnotes entirely (AP = 0) in their native class mappings
- **Eric YOLO** (tiled) handles high-resolution pages well but over-predicts (1102 boxes vs 879 GT)

Full per-IoU-threshold breakdowns are in the individual report files under `data/`.

## Evaluation Methodology

The evaluation pipeline (`hff_remover.evaluate`) uses standard COCO-style mAP:

1. **Label format**: YOLO-normalized bounding boxes (`class_id x_center y_center width height [confidence]`) in `.txt` files, one per image
2. **Matching**: Per-image, per-class greedy IoU matching of predictions (sorted by confidence) to ground-truth boxes
3. **AP computation**: Interpolated precision-recall curve per class
4. **mAP\@0.50**: Mean of per-class AP at IoU threshold 0.50
5. **mAP\@0.50:0.95**: Mean over 10 IoU thresholds (0.50, 0.55, ..., 0.95) of the per-threshold mean AP

## CLI Reference

### `hff-remover process`

Batch-process all images in a directory.

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | Required | Output directory |
| `--device` | `cuda` | Device for inference (`cuda` or `cpu`) |
| `--batch-size` | `8` | Images per batch |
| `--confidence` | `0.5` | Minimum detection confidence |
| `--padding` | `0` | Extra pixels around detected regions |
| `--image-size` | `1024` | Model input size |
| `--quality` | `95` | Output image quality (0-100) |
| `--resume` | `false` | Resume from checkpoint |
| `--no-recursive` | `false` | Don't search subdirectories |

### `hff-remover single`

Process a single image.

```bash
hff-remover single input.jpg output.jpg --device cuda
```

### `hff-remover detect`

Detect HFF regions without masking.

```bash
hff-remover detect input.jpg --output detections.json
```

### `hff-remover evaluate`

Run mAP evaluation on prediction labels.

```bash
hff-remover evaluate --gt-dir data/benchmark_dataset/label \
    --pred-dir data/surya_inference_1500/labels
```

## Data Directory

The `data/` folder contains the benchmark dataset and inference outputs:

| Path | Description |
|------|-------------|
| `benchmark_dataset/` | Ground-truth images and YOLO-format labels (879 boxes, 4 classes) |
| `*_inference/` | Inference outputs from each detector (images + labels) |
| `*_report.txt` | Evaluation reports comparing each detector against benchmark |

## Development

```bash
pip install -e ".[dev]"
pytest
pytest --cov=hff_remover
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) for the document layout model
- [Surya](https://github.com/VikParuchuri/surya) for the layout foundation model
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for PP-DocLayout
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO11
- [OpenPecha](https://openpecha.org) for supporting this project
