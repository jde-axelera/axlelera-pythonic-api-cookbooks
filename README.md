# Axelera Pythonic API Cookbooks

Example applications using the **Axelera Voyager SDK Pythonic API** (SDK 1.6+).
Inference runs on Axelera hardware via a composable pipeline API — no Ultralytics or heavy
frameworks required at runtime.

---

## Prerequisites

- Axelera Voyager SDK 1.6+ (provides `axelera-runtime`, `axelera-rt`)
- Python 3.8+
- OpenCV and NumPy (see [Installation](#installation))
- At least one compiled `.axm` model (see [Compiling Models](#compiling-models))

---

## Installation

```bash
# Activate the SDK virtual environment (created by the Axelera installer)
source pythonic_api/bin/activate

# Install standard dependencies
pip install -r requirements.txt
```

> **Note:** `axelera-runtime` and `axelera-rt` ship with the Voyager SDK installer
> and are pre-installed in `pythonic_api/`. They are not available on PyPI.

---

## Unified Inference Script — `inference.py`

A single script handles **both detection and segmentation** for any COCO-compatible model.
Task and algorithm are **auto-detected** from the model's `metadata.yaml`; no flags needed
for standard models.

### Quick start

```bash
# Detection (YOLO11n)
python inference.py --model yolo11n_axelera_model --source video.mp4

# Segmentation (YOLO11n-seg)
python inference.py --model yolo11n-seg_axelera_model --source 0

# Detection (YOLO26n, end-to-end NMS baked in)
python inference.py --model yolo26n_axelera_model --source video.mp4
```

### Headless benchmark (no display, prints per-stage FPS table)

```bash
python inference.py --model yolo11n_axelera_model --source video.mp4 --benchmark 300
```

Example output:

```
Model  : yolo11n_axelera_model/yolo11n.axm
Task   : detect
Algo   : yolo11   end2end=False   classes=80
Input  : 640×640

Benchmark: 300 frames in 4.8s

====================================================
Stage          Latency (us)   Effective FPS
====================================================
Preprocess          1,102           907.4
Inference          14,823            67.5
Postprocess         1,641           609.4
====================================================
End-to-end                           62.3
====================================================
```

### Manual override (custom models without metadata.yaml)

```bash
python inference.py --model my_model.axm --task detect --algo yolov8 --source video.mp4
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Model directory (with `metadata.yaml`) or direct `.axm` path |
| `--source` | `0` | Video file, image, or camera index |
| `--task` | *(auto)* | Override: `detect` or `segment` |
| `--algo` | *(auto)* | Override: `yolo11`, `yolov8`, `yolo26`, `yolov9`, … |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--benchmark N` | off | Headless: run N frames and print FPS table |

### Controls (display mode)

| Key | Action |
|-----|--------|
| `q` or `ESC` | Quit |

---

## Benchmarking All Models — `benchmark.sh`

Runs `inference.py` headlessly for every compiled model and prints per-stage FPS tables.

```bash
# With a video file, 300 frames per model
./benchmark.sh video.mp4 300

# With webcam, default 300 frames
./benchmark.sh 0
```

### Available models

| Model | Task | Status |
|-------|------|--------|
| `yolo11n_axelera_model` | detect | ✅ compiled |
| `yolo11n-seg_axelera_model` | segment | ✅ compiled |
| `yolo26n_axelera_model` | detect (end-to-end NMS) | ✅ compiled |
| `yolov8n_axelera_model` | detect | ⚙️ compile first (see below) |
| `yolov8n-seg_axelera_model` | segment | ⚙️ compile first (see below) |

---

## Compiling Models

Use `axcompile` (included in the SDK) to compile ONNX models for Axelera hardware:

```bash
# Detection
axcompile yolo11n.onnx     --output yolo11n_axelera_model/
axcompile yolov8n.onnx     --output yolov8n_axelera_model/

# Segmentation
axcompile yolo11n-seg.onnx --output yolo11n-seg_axelera_model/
axcompile yolov8n-seg.onnx --output yolov8n-seg_axelera_model/
```

Pre-compiled model directories and raw `.pt` / `.onnx` weights are excluded from this
repository (see `.gitignore`). Export ONNX from Ultralytics if needed:

```bash
yolo export model=yolo11n.pt     format=onnx imgsz=640
yolo export model=yolov8n.pt     format=onnx imgsz=640
yolo export model=yolo11n-seg.pt format=onnx imgsz=640
yolo export model=yolov8n-seg.pt format=onnx imgsz=640
```

---

## FPS Measurement

`inference.py` times each stage independently using `time.perf_counter()`:

| Stage | What is timed |
|-------|--------------|
| **Preprocess** | `colorconvert + letterbox + totensor` |
| **Inference** | Model execution on Axelera AIPU |
| **Postprocess** | `decode + NMS + to_image_space` (+ mask generation for segmentation) |
| **End-to-end** | Wall-clock FPS across all stages (first frame excluded as warmup) |

A 30-frame rolling window is used for the live FPS overlay in display mode.
The final table averages over all frames processed.
