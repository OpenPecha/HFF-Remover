# HFF Remover - Header, Footer, and Footnote Removal Tool

## Description

Build a Python package that automatically detects and removes headers, footers, and footnotes from scanned book images using DocLayout-YOLO. The tool is designed to process large batches of images (200K+) efficiently with GPU acceleration support.

**Key Features:**
- Automatic detection of document layout elements using DocLayout-YOLO model
- White masking of detected HFF (Header, Footer, Footnote) regions
- Batch processing with checkpoint/resume support for large datasets
- Support for mixed image formats (JPEG, PNG, TIFF, BMP, WebP)
- Both CLI and Python API interfaces
- Configurable confidence thresholds and padding options

**Target Classes:**
| Class ID | Element | Description |
|----------|---------|-------------|
| 2 | abandon | Headers, footers, page numbers |
| 7 | table_footnote | Footnotes in tables |

---

## Implementation

### 1. Project Setup
- [x] Create project structure based on OpenPecha template
- [x] Update `pyproject.toml` with project name (`hff-remover`)
- [x] Add dependencies: `doclayout-yolo`, `torch`, `torchvision`, `opencv-python`, `Pillow`, `tqdm`, `huggingface-hub`
- [x] Configure CLI entry point (`hff-remover`)
- [x] Update README with usage documentation

### 2. Core Detection Module (`detector.py`)
- [x] Implement `HFFDetector` class
- [x] Add DocLayout-YOLO model loading from HuggingFace Hub
- [x] Implement single image detection (`detect()`)
- [x] Implement batch detection (`detect_batch()`)
- [x] Filter detections for HFF classes only (abandon, table_footnote)
- [x] Support configurable confidence threshold
- [x] Support both CPU and CUDA devices

### 3. Image Processing Module (`processor.py`)
- [x] Implement `HFFProcessor` class
- [x] Add white masking for detected regions (`mask_regions()`)
- [x] Add smooth edge masking option (`mask_regions_smooth()`)
- [x] Support configurable padding around detected regions
- [x] Support custom mask colors
- [x] Implement clean region mask generation

### 4. Utility Module (`utils.py`)
- [x] Implement image format detection
- [x] Add recursive image search in directories
- [x] Implement image loading for mixed formats (including TIFF via PIL)
- [x] Implement image saving with quality control
- [x] Add output path generation preserving directory structure
- [x] Add image resize and bbox scaling utilities

### 5. Batch Processing Module (`batch.py`)
- [x] Implement `BatchProcessor` class
- [x] Add directory processing with progress tracking
- [x] Implement checkpoint save/load for resume support
- [x] Add multi-threaded I/O for parallel loading/saving
- [x] Implement processing statistics tracking
- [x] Add JSON report generation
- [x] Support graceful interruption with checkpoint save

### 6. Command Line Interface (`cli.py`)
- [x] Implement `process` command for batch processing
- [x] Implement `single` command for single image processing
- [x] Implement `detect` command for detection-only mode
- [x] Add all configuration options (device, batch-size, confidence, padding, etc.)
- [x] Add resume flag for checkpoint recovery
- [x] Add verbose output option

### 7. Testing
- [x] Add unit tests for `HFFProcessor`
- [x] Add unit tests for `utils` module
- [x] Add unit tests for `BatchProcessor` and checkpoint
- [x] Add unit tests for `HFFDetector`

### 8. Documentation & Examples
- [x] Update README with installation instructions
- [x] Add CLI usage examples
- [x] Add Python API examples
- [x] Create standalone `example.py` script

---

## Usage

### CLI
```bash
# Batch process
hff-remover process /path/to/images --output /path/to/output --device cuda --batch-size 32

# Single image
hff-remover single input.jpg output.jpg

# Detection only
hff-remover detect input.jpg --output detections.json
```

### Python API
```python
from hff_remover import HFFDetector, HFFProcessor
from hff_remover.utils import load_image, save_image

detector = HFFDetector(device="cpu", confidence_threshold=0.5)
processor = HFFProcessor(padding=5)

image = load_image("input.jpg")
detections = detector.detect(image)
result = processor.mask_regions(image, detections)
save_image(result, "output.jpg")
```

---

## Performance Estimates

| Hardware | Speed | 200K Images |
|----------|-------|-------------|
| NVIDIA V100 | ~100 img/s | ~33 min |
| NVIDIA T4 | ~50 img/s | ~67 min |
| CPU (8 cores) | ~2-5 img/s | ~11-28 hours |
