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

> **Hardware:** Axelera Metis (Omega generation, 1 AIPU core)  
> **Source:** `traffic4_480p.mp4` (480p traffic video, 300 frames)  
> **Native:** Voyager SDK `inference.py` with GStreamer pipeline (`--no-display --frames 300 --show-stats`)  
> **Pythonic:** this repo's `inference.py` (`--benchmark 300`)

### End-to-end throughput — all models

| Model | Task | Native GST (fps) | AIPU latency | Pythonic API (fps) | Pre (µs) | Infer (µs) | Post (µs) | Gap |
|-------|------|:----------------:|:------------:|:-----------------:|:--------:|:----------:|:---------:|:---:|
| YOLO11n | Detection | **749.7** | 1,391 | 35.2 | 5,324 | 21,644 | 1,458 | **21×** |
| YOLO11n | Segmentation | **573.7** | 1,726 | 9.6 | 7,436 | 37,077 | 63,550 | **60×** |
| YOLO26n | Detection | **643.3** | 1,533 | 50.9 | 4,737 | 14,208 | 248 | **13×** |
| YOLOv8n | Detection | **839.9** | 1,173 | — ¹ | — | — | — | — |
| YOLOv8n | Segmentation | **~640** ² | 1,556 | — ¹ | — | — | — | — |
| YOLO26n | Segmentation | **507.6** | 1,950 | — ¹ | — | — | — | — |

> ¹ Pythonic API numbers not yet measured — compile models first:  
> `axcompile yolov8n.onnx --output yolov8n_axelera_model/`  
> `axcompile yolov8n-seg.onnx --output yolov8n-seg_axelera_model/`  
> `axcompile yolo26n-seg.onnx --output yolo26n-seg_axelera_model/`  
> ² Estimated from AIPU latency; pipelined throughput measurement pending (single-pass run too short to fill the GST pipeline).

### Per-stage latency — YOLO11n Detection

| Stage | Native GST | Pythonic API | Delta |
|-------|:----------:|:------------:|:-----:|
| Resize + pad | 1,059 µs (OpenCL) | 5,324 µs (Python CPU) | 5× |
| AIPU inference | 1,391 µs | 21,644 µs | 15× |
| Decode + NMS | 390 µs (C++) | 1,458 µs (NumPy) | 4× |
| **Single-frame latency** | **32.3 ms** | **28.4 ms** | ~1× |
| **Pipelined throughput** | **749.7 fps** | **35.2 fps** | **21×** |

### Per-stage latency — YOLO11n Segmentation

| Stage | Native GST | Pythonic API | Delta |
|-------|:----------:|:------------:|:-----:|
| Resize + pad | 1,990 µs (OpenCL) | 7,436 µs (Python CPU) | 4× |
| AIPU inference | 2,009 µs | 37,077 µs | 18× |
| Decode + mask gen | 777 µs (C++) | 63,550 µs (NumPy) | **82×** |
| **Single-frame latency** | **42.6 ms** | **108.1 ms** | 2.5× |
| **Pipelined throughput** | **573.7 fps** | **9.6 fps** | **60×** |

### Why the gap?

**1. Pipelining — biggest throughput factor**  
The native GStreamer pipeline overlaps preprocess, AIPU inference, and postprocessing across frames
concurrently. The Pythonic API is fully synchronous: one frame completes all stages before the next starts.

**2. Hardware-accelerated preprocessing (5× impact)**  
Native resize and letterbox use OpenCL (~1 ms). Pythonic uses Python/OpenCV on CPU (~5–7 ms),
capping throughput at ~140–190 fps even with instant inference.

**3. Python → AIPU call overhead (15–18× impact)**  
The native GStreamer element uses zero-copy DMA between OpenCL memory and the AIPU.
The Pythonic API incurs Python → C++ call overhead, NumPy tensor allocation, and synchronous
memory copy on every frame, multiplying the effective inference time by 15–18×.

**4. NumPy mask generation — segmentation bottleneck (82× impact)**  
`proto_to_mask` in Python/NumPy takes ~64 ms per frame vs < 1 ms in the native C++ library.
This alone is responsible for most of the 60× gap in segmentation.
Replacing it with a compiled extension would improve segmentation from ~10 fps to ~27 fps.

### When to use each

| | Native Voyager GST | Pythonic API |
|-|-------------------|--------------|
| **Throughput** | > 500 fps per stream | 10–50 fps per stream |
| **Single-frame latency** | 32–43 ms | 28–108 ms |
| **Custom pipeline logic** | Requires GStreamer plugin | ✅ Pure Python |
| **Rapid prototyping** | Moderate (YAML config) | ✅ Fast |
| **Production deployment** | ✅ Fully optimised | Moderate |
| **Multi-stream / async** | ✅ Native pipelining | Sequential only |

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
