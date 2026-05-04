# Axelera Pythonic API Cookbooks

Example applications using the **Axelera Voyager SDK Pythonic API** (SDK 1.6+).
These scripts run inference on Axelera hardware using a clean, composable pipeline API — no Ultralytics or heavy frameworks needed at runtime.

---

## Prerequisites

- Axelera Voyager SDK 1.6+ installed (provides `axelera-runtime`, `axelera-rt`)
- Python 3.8+
- OpenCV and NumPy (see [Installation](#installation))
- A compiled `.axm` model file (see [Compiling Models](#compiling-models))

---

## Installation

```bash
# Activate the SDK virtual environment (created by the Axelera installer)
source pythonic_api/bin/activate

# Install standard dependencies
pip install -r requirements.txt
```

> **Note:** `axelera-runtime` and `axelera-rt` are bundled with the Voyager SDK installer
> and are already present in the `pythonic_api` virtual environment. They are not on PyPI.

---

## Examples

### 1. YOLO11n Object Detection

**File:** `yolo11n_detection.py`

Runs YOLO11n detection on a video file or live camera stream.
Displays results in a window and prints rolling-average FPS to the console.

**Requirements:** compiled model at `yolo11n_axelera_model/yolo11n.axm`

```bash
# Run on a video file
python yolo11n_detection.py video.mp4

# Run on webcam (camera index 0)
python yolo11n_detection.py 0
```

---

### 2. YOLO11n Instance Segmentation

**File:** `yolo11n_segmentation.py`

Runs YOLO11n instance segmentation on a video file or live camera stream.
Overlays per-instance masks, bounding boxes, labels, and a live FPS counter on the frame.

**Requirements:** compiled model at `yolo11n-seg_axelera_model/yolo11n-seg.axm`

```bash
# Run on a video file
python yolo11n_segmentation.py --model yolo11n-seg_axelera_model/yolo11n-seg.axm --source video.mp4

# Run on webcam
python yolo11n_segmentation.py --model yolo11n-seg_axelera_model/yolo11n-seg.axm --source 0
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Path to compiled `.axm` segmentation model |
| `--source` | `0` | Video file path or camera index |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |

---

## Compiling Models

Use `axcompile` (included in the SDK) to compile ONNX models for Axelera hardware:

```bash
# Detection
axcompile yolo11n.onnx --output yolo11n_axelera_model/

# Segmentation
axcompile yolo11n-seg.onnx --output yolo11n-seg_axelera_model/
```

Pre-compiled `.axm` files are **not** tracked in this repository (see `.gitignore`).
Download them from the Axelera model zoo or compile them yourself.

---

## Controls

| Key | Action |
|-----|--------|
| `q` or `ESC` | Quit |

---

## FPS Measurement

Both scripts compute a **30-frame rolling average** for stable FPS readings:

- Timing starts just before the pipeline call and ends after it returns.
- FPS is printed to stdout every frame.
- `yolo11n_segmentation.py` also overlays FPS in the top-left corner of the display.
