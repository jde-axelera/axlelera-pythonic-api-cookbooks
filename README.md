# Axelera Pythonic API Cookbooks

Example applications using the **Axelera Voyager SDK Pythonic API** (SDK 1.6+).
Inference runs on Axelera hardware via a composable pipeline API — no Ultralytics or heavy
frameworks required at runtime.

---

## Prerequisites

- Axelera Voyager SDK 1.6+ (provides `axelera-runtime`, `axelera-rt`)
- Python 3.8+
- OpenCV and NumPy (see [Installation](#installation))
- A compiled `.axm` model (download with `axdownloadmodel` or compile with `axcompile`)

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

## Getting Models

### Download pre-compiled `.axm` (fastest)

```bash
source pythonic_api/bin/activate

# List all available models
axdownloadmodel --list

# Download YOLO11n detection and segmentation
axdownloadmodel --axm yolo11n-coco-onnx yolo11nseg-coco-onnx
```

The `--axm` flag downloads and unpacks a ready-to-run `.axm` file in the current directory.

### Compile from ONNX

```bash
# Export ONNX from Ultralytics
yolo export model=yolo11n.pt     format=onnx imgsz=640
yolo export model=yolo11n-seg.pt format=onnx imgsz=640

# Compile for Axelera hardware (creates a model directory with metadata.yaml)
axcompile yolo11n.onnx     --output yolo11n_axelera_model/
axcompile yolo11n-seg.onnx --output yolo11n-seg_axelera_model/
```

Pre-compiled model directories and raw weights are excluded from this repository (see `.gitignore`).

---

## Unified Inference Script — `inference.py`

A single script handles **both detection and segmentation** for any COCO-compatible model.
Task and algorithm are **auto-detected** from the model's `metadata.yaml` when using a compiled
model directory. For standalone `.axm` files (e.g. from `axdownloadmodel --axm`), pass
`--task` and `--algo` explicitly.

### Quick start

```bash
# Detection — compiled model directory (auto-detect)
python inference.py --model yolo11n_axelera_model --source video.mp4

# Detection — standalone .axm (manual flags)
python inference.py --model yolo11n-coco-onnx.axm --task detect --algo yolo11 --source video.mp4

# Segmentation — standalone .axm
python inference.py --model yolo11nseg-coco-onnx.axm --task segment --algo yolo11 --source 0
```

### Headless benchmark (no display, prints per-stage FPS table)

```bash
python inference.py --model yolo11n-coco-onnx.axm --task detect --algo yolo11 \
    --source video.mp4 --benchmark 300
```

Output:

```
Model  : /yolo11n-coco-onnx.axm
Task   : detect
Algo   : yolo11   end2end=False   classes=80
Input  : 640×640

Benchmark: 300 frames in 8.5s

==========================================
Stage         Latency (us)   Effective FPS
==========================================
Preprocess      5,324            187.8
Inference      21,644             46.2
Postprocess     1,458            685.9
==========================================
End-to-end                        35.2
==========================================
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
./benchmark.sh video.mp4 300   # 300 frames per model
./benchmark.sh 0               # webcam, default 300 frames
```

---

## Performance Report

> **Hardware:** Axelera Metis (1 AIPU core, Omega generation)  
> **Source:** `traffic4_480p.mp4` (480p traffic video)  
> **Frames:** 300  
> **Models:** `yolo11n-coco-onnx` (detect) and `yolo11nseg-coco-onnx` (segment), both downloaded via `axdownloadmodel --axm`

### End-to-end throughput

| Model | Task | Native Voyager (fps) | Pythonic API (fps) | Ratio |
|-------|------|---------------------:|--------------------|------:|
| YOLO11n | Detection | **749.7** | 35.2 | 21× |
| YOLO11n | Segmentation | **573.7** | 9.6 | 60× |

### Per-stage latency breakdown — YOLO11n Detection

| Stage | Native Voyager | Pythonic API | Notes |
|-------|---------------:|-------------:|-------|
| Resize + pad | 1,059 µs | 5,324 µs | OpenCL GPU vs Python CPU |
| AIPU inference | 1,391 µs | 21,644 µs | Same chip — see note |
| Decode + NMS | 390 µs | 1,458 µs | C++ vs Python/NumPy |
| **Single-frame latency** | **32.3 ms** | **28.4 ms** | Comparable |
| **Pipelined throughput** | **749.7 fps** | **35.2 fps** | Native pipelines frames |

### Per-stage latency breakdown — YOLO11n Segmentation

| Stage | Native Voyager | Pythonic API | Notes |
|-------|---------------:|-------------:|-------|
| Resize + pad | 1,990 µs | 7,436 µs | OpenCL GPU vs Python CPU |
| AIPU inference | 2,009 µs | 37,077 µs | Same chip — see note |
| Decode + mask gen | 777 µs | 63,550 µs | C++ vs Python/NumPy |
| **Single-frame latency** | **42.6 ms** | **108.1 ms** | Mask gen is the bottleneck |
| **Pipelined throughput** | **573.7 fps** | **9.6 fps** | Native pipelines frames |

### Why the gap?

**1. Pipelining (biggest factor for throughput)**  
The native GStreamer pipeline runs preprocessing, AIPU inference, and postprocessing as concurrent
stages — while one frame is on the AIPU, the next is already being preprocessed in parallel.
The Pythonic API processes one frame at a time synchronously.

**2. Hardware-accelerated preprocessing**  
The native pipeline uses OpenCL (GPU) for resize and padding (~1 ms).
The Pythonic API uses Python/CPU (~5–7 ms) — 5× slower.

**3. AIPU inference time**  
The AIPU element latency reported by `--show-stats` (1.4–2 ms) is the *per-element* time inside
the GStreamer pipeline. The Pythonic API shows higher inference time because it includes Python
overhead for tensor allocation, DMA transfer, and synchronisation on every frame.

**4. Postprocessing — critical for segmentation**  
Native C++ libraries decode detections and generate masks in < 1 ms.  
The Pythonic API's `proto_to_mask` runs in Python/NumPy: ~64 ms per frame for the seg model —
the single largest bottleneck. Optimising this one stage would bring segmentation throughput
from ~10 fps to ~27 fps.

### When to use each

| | Native Voyager | Pythonic API |
|-|----------------|--------------|
| **Throughput target** | > 100 fps per stream | 10–50 fps per stream |
| **Latency target** | < 5 ms | < 110 ms |
| **Custom pipeline logic** | Requires GStreamer plugin | ✅ Pure Python |
| **Rapid prototyping** | Moderate (YAML config) | ✅ Fast |
| **Production deployment** | ✅ Optimised | Moderate |
| **Multi-stream scaling** | ✅ Native async | Sequential |

---

## FPS Measurement Details

`inference.py` times each stage independently with `time.perf_counter()`:

| Stage | What is timed |
|-------|--------------|
| **Preprocess** | `colorconvert + letterbox + totensor` |
| **Inference** | Model execution on Axelera AIPU |
| **Postprocess** | `decode + NMS + to_image_space` (+ mask generation for segmentation) |
| **End-to-end** | Wall-clock FPS (first frame excluded as warmup) |

A 30-frame rolling window is used for the live FPS overlay in display mode.
