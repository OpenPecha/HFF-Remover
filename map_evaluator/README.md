
# YOLO Label Evaluation – mAP Calculator

This repository provides a lightweight Python script to evaluate YOLO object detection results by comparing ground truth labels with inference labels and computing:

- Per-class Average Precision (AP)
- Mean Average Precision (mAP)

The implementation uses IoU-based matching and 11-point interpolation.

---

## 📂 Project Structure

```
├── compute_map.py
├── gt.txt
├── pred.txt
└── README.md
```

---

## ⚙️ Requirements

Only Python + NumPy:

```bash
pip install numpy
```

---

## 🚀 Usage

```bash
python compute_map.py gt.txt pred.txt
```

### Output Example

```
Class 0 AP: 0.83
Class 1 AP: 0.76

mAP: 0.795
```
