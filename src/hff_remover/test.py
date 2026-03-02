"""
Smoke test for SuryaLayoutDetector and detector.py YOLO output.
Uses exactly the same API from detector.py (save_detections_yolo_format,
save_yolo_data_yaml, SURYA_HFF_CLASS_NAMES) and verifies the saved files
so you can check if detector.py is working correctly.
"""
from pathlib import Path

from hff_remover.detector import (
    SuryaLayoutDetector,
    save_detections_yolo_format,
    save_yolo_data_yaml,
    SURYA_HFF_CLASS_NAMES,
)

# --- 1) Same as detector.py: resolve image path ---
image_path = Path(__file__).resolve().parent.parent.parent / "test_images" / "1.jpg"
if not image_path.exists():
    image_path = Path(r"D:\budhaghat\HFF-Remover\test_images\1.jpg")
if not image_path.exists():
    raise FileNotFoundError(f"Test image not found. Put an image at {image_path}")

# --- 2) Same as detector.py: detect HFF ---
detector = SuryaLayoutDetector(confidence_threshold=0.1)
hff_detections = detector.detect(image_path)

print("HFF detections (from SuryaLayoutDetector.detect()):")
print("-" * 60)
for i, det in enumerate(hff_detections):
    print(f"  [{i}] class_id={det['class_id']!r}, class_name={det['class_name']!r}, confidence={det['confidence']:.4f}")
    print(f"      bbox={det['bbox']}")
print("-" * 60)
print(f"Total: {len(hff_detections)} HFF detection(s)")

# Class IDs (same as in detector.py SURYA_HFF_CLASS_IDS / SURYA_HFF_CLASS_NAMES)
print("\nClass IDs (0=header, 1=footer, 2=footnote, 3=page_number):")
for cid, name in sorted(SURYA_HFF_CLASS_NAMES.items()):
    print(f"  {cid} = {name}")

# --- 3) Same as detector.py: save YOLO .txt (one line per box: class_id cx cy w h) ---
output_txt = image_path.with_suffix(".txt")
save_detections_yolo_format(hff_detections, image_path, output_txt)
print(f"\nSaved YOLO labels (detector.save_detections_yolo_format): {output_txt}")

# --- 4) Same as detector.py: save data.yaml ---
labels_dir = output_txt.parent
data_yaml_path = labels_dir / "data.yaml"
save_yolo_data_yaml(data_yaml_path, images_dir=labels_dir)
print(f"Saved data.yaml (detector.save_yolo_data_yaml): {data_yaml_path}")

# --- 5) Verify .txt: read back and check format (detector.py output check) ---
txt_content = output_txt.read_text(encoding="utf-8").strip()
txt_lines = [ln.strip() for ln in txt_content.splitlines() if ln.strip()]
print(f"\n--- Check 1: YOLO .txt ---")
print(f"  Lines in file: {len(txt_lines)} (expected {len(hff_detections)})")
all_txt_ok = True
for i, line in enumerate(txt_lines):
    parts = line.split()
    if len(parts) != 5:
        print(f"  Line {i}: INVALID (expected 5 numbers, got {len(parts)})")
        all_txt_ok = False
    else:
        cid, cx, cy, w, h = parts
        name = SURYA_HFF_CLASS_NAMES.get(int(cid), "?")
        print(f"  Line {i}: class_id={cid} ({name}) cx={cx} cy={cy} w={w} h={h}")
        if int(cid) not in SURYA_HFF_CLASS_NAMES:
            all_txt_ok = False
if len(txt_lines) != len(hff_detections):
    all_txt_ok = False
print(f"  .txt check: {'OK' if all_txt_ok else 'FAIL'}")

# --- 6) Verify data.yaml: read back and check (detector.py output check) ---
print(f"\n--- Check 2: data.yaml ---")
yaml_content = data_yaml_path.read_text(encoding="utf-8")
print(yaml_content)
all_yaml_ok = True
if "nc: 4" not in yaml_content:
    print("  data.yaml check: FAIL (nc: 4 not found)")
    all_yaml_ok = False
for cid, name in SURYA_HFF_CLASS_NAMES.items():
    if f"{cid}: {name}" not in yaml_content and f"{cid}: {name}" not in yaml_content.replace("  ", " "):
        # allow slight spacing
        if name not in yaml_content:
            all_yaml_ok = False
if all_yaml_ok and "names:" in yaml_content:
    print("  data.yaml check: OK")
else:
    if not all_yaml_ok:
        print("  data.yaml check: FAIL (names or nc mismatch)")

# --- 7) Summary: detector.py working or not ---
print("\n" + "=" * 60)
if all_txt_ok and all_yaml_ok:
    print("detector.py check: OK (YOLO .txt and data.yaml correct)")
else:
    print("detector.py check: FAIL (see above)")
print("=" * 60)
