#!/usr/bin/env python3
"""
Unified YOLO detection and segmentation inference for Axelera Voyager SDK.

Auto-detects task (detect/segment) and algorithm from the model's metadata.yaml.
Supports any COCO-compatible model compiled with axcompile. Per-stage timing is
reported in an SDK-style table at the end of each run.

Usage:
    python inference.py --model yolo11n_axelera_model --source video.mp4
    python inference.py --model yolo11n-seg_axelera_model --source 0
    python inference.py --model yolo26n_axelera_model --source video.mp4
    python inference.py --model yolo11n_axelera_model --source video.mp4 --benchmark 300
    python inference.py --model my_model.axm --task detect --algo yolov8 --source 0
"""

from __future__ import annotations

import argparse
import os
import re

# Set DISPLAY before cv2 is used. SSH sessions typically have no DISPLAY set,
# but the X server socket is present. Auto-detect the first available one.
if not os.environ.get('DISPLAY'):
    from pathlib import Path as _P
    _socks = sorted((_P('/tmp/.X11-unix')).glob('X*'))
    if _socks:
        os.environ['DISPLAY'] = f':{_socks[0].name[1:]}'
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from axelera.runtime import op


# ── COCO classes ──────────────────────────────────────────────────────────────

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

_rng = np.random.default_rng(42)
CLASS_COLORS = _rng.integers(50, 230, size=(len(COCO_NAMES), 3)).tolist()


# ── Model metadata ─────────────────────────────────────────────────────────────

_FAMILY_TO_ALGO: dict[str, str] = {
    "yolo11": "yolo11",
    "yolo26": "yolo26",
    "yolov10": "yolov10",
    "yolov9": "yolov9",
    "yolov8": "yolov8",
    "yolov7": "yolov7",
    "yolov5": "yolov5",
}


@dataclass
class ModelMeta:
    task: str           # 'detect' or 'segment'
    algo: str           # e.g. 'yolo11', 'yolov8', 'yolo26'
    end2end: bool       # True = model has baked-in NMS, skip explicit NMS op
    num_classes: int
    imgsz: tuple[int, int]
    axm_path: Path

    @classmethod
    def from_dir(cls, model_dir: Path, task_override: str | None = None,
                 algo_override: str | None = None) -> "ModelMeta":
        axm_files = sorted(model_dir.glob("*.axm"))
        if not axm_files:
            raise FileNotFoundError(f"No .axm file found in {model_dir}")

        meta_path = model_dir / "metadata.yaml"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.yaml not found in {model_dir}. Pass --task and --algo explicitly."
            )
        with open(meta_path) as f:
            meta = yaml.safe_load(f)

        task = task_override or meta.get("task", "detect")
        num_classes = len(meta.get("names", {})) or 80
        raw_imgsz = meta.get("imgsz", [640, 640])
        imgsz = (int(raw_imgsz[0]), int(raw_imgsz[1]))

        # end2end and family are more reliably read from compiler_config_final.toml
        end2end = bool(meta.get("end2end", False))
        family: str | None = None
        toml_path = model_dir / "compiler_config_final.toml"
        if toml_path.exists():
            toml_text = toml_path.read_text()
            if m := re.search(r'^end2end\s*=\s*(true|false)', toml_text, re.MULTILINE):
                end2end = m.group(1) == "true"
            if m := re.search(r'^family\s*=\s*"([^"]+)"', toml_text, re.MULTILINE):
                family = m.group(1)

        if algo_override:
            algo = algo_override
        else:
            algo = _FAMILY_TO_ALGO.get(family or "", None)
            if algo is None:
                desc = meta.get("description", "").lower()
                algo = next(
                    (v for k, v in _FAMILY_TO_ALGO.items() if k in desc), "yolo11"
                )

        return cls(
            task=task, algo=algo, end2end=end2end,
            num_classes=num_classes, imgsz=imgsz, axm_path=axm_files[0],
        )

    @classmethod
    def from_axm(cls, axm_path: Path, task: str, algo: str) -> "ModelMeta":
        return cls(
            task=task, algo=algo, end2end=False,
            num_classes=80, imgsz=(640, 640), axm_path=axm_path,
        )


# ── Pipeline builders ──────────────────────────────────────────────────────────

def build_pipelines(meta: ModelMeta, conf: float, iou: float):
    """Return (preprocess, infer, postprocess) as three independently callable ops."""
    h, w = meta.imgsz
    preprocess = op.seq(
        op.colorconvert("RGB", src="BGR"),
        op.letterbox(h, w),
        op.totensor(),
    )
    infer = op.load(str(meta.axm_path))

    if meta.task == "segment":
        postprocess = op.seq(
            op.decode_segmentation(
                algo=meta.algo,
                num_classes=meta.num_classes,
                num_mask_coeffs=32,
                confidence_threshold=conf,
            ),
            op.par(
                op.seq(op.itemgetter(0), op.nms(iou_threshold=iou, max_boxes=300)),
                op.itemgetter(1),
            ),
            op.par(
                op.seq(op.pack(), op.itemgetter(0), op.to_image_space()),
                op.proto_to_mask(),
            ),
        )
    elif meta.end2end:
        # Model has baked-in NMS; output is already (N, 6) boxes
        postprocess = op.seq(
            op.decode_detections(
                algo=meta.algo,
                num_classes=meta.num_classes,
                confidence_threshold=conf,
            ),
            op.to_image_space(),
        )
    else:
        postprocess = op.seq(
            op.decode_detections(
                algo=meta.algo,
                num_classes=meta.num_classes,
                confidence_threshold=conf,
            ),
            op.nms(iou_threshold=iou, max_boxes=300),
            op.to_image_space(),
        )

    return preprocess, infer, postprocess


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _draw_label(img: np.ndarray, x0: int, y0: int, text: str, color: list) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x0, y0 - th - 4), (x0 + tw, y0), color, -1)
    cv2.putText(img, text, (x0, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def _draw_fps(img: np.ndarray, fps: float) -> None:
    text = f"{fps:.1f} FPS"
    (fw, fh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img, (8, 8), (fw + 16, fh + 16), (0, 0, 0), -1)
    cv2.putText(img, text, (12, fh + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def draw_detections(img: np.ndarray, detections: np.ndarray, fps: float) -> np.ndarray:
    h, w = img.shape[:2]
    for det in detections:
        x0, y0, x1, y1, score, cid = (
            max(0, int(det[0])), max(0, int(det[1])),
            min(w, int(det[2])), min(h, int(det[3])),
            float(det[4]), int(det[5]),
        )
        color = CLASS_COLORS[cid % len(CLASS_COLORS)]
        name = COCO_NAMES[cid] if cid < len(COCO_NAMES) else str(cid)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        _draw_label(img, x0, y0, f"{name} {score:.2f}", color)
    _draw_fps(img, fps)
    return img


def draw_segmentation(img: np.ndarray, detections: np.ndarray, masks: list,
                      fps: float) -> np.ndarray:
    overlay = img.copy()
    h, w = img.shape[:2]
    for i, det in enumerate(detections):
        x0, y0, x1, y1, score, cid = (
            max(0, int(det[0])), max(0, int(det[1])),
            min(w, int(det[2])), min(h, int(det[3])),
            float(det[4]), int(det[5]),
        )
        color = CLASS_COLORS[cid % len(CLASS_COLORS)]
        name = COCO_NAMES[cid] if cid < len(COCO_NAMES) else str(cid)
        if i < len(masks):
            mask = masks[i]
            if mask.ndim == 2 and (x1 - x0) > 0 and (y1 - y0) > 0:
                resized = cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                overlay[y0:y1, x0:x1][resized > 127] = color
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        _draw_label(img, x0, y0, f"{name} {score:.2f}", color)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    _draw_fps(img, fps)
    return img


# ── SDK-style FPS table ────────────────────────────────────────────────────────

_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"


def _strip_ansi(s: str) -> str:
    return re.sub(r'\x1b\[[^m]*m', '', s)


def tabulate_fps(stage_avg_us: dict[str, float], end_to_end_fps: float) -> None:
    """Print per-stage latency and end-to-end FPS in SDK-style table."""
    def fps_str(us: float) -> str:
        return f"{1e6 / us:.1f}" if us > 0 else "—"

    header = ["Stage", "Latency (us)", "Effective FPS"]
    rows = [[name, f"{int(us):,}", fps_str(us)] for name, us in stage_avg_us.items()]
    end_row = [f"{_BOLD}End-to-end{_RESET}", "", f"{_BOLD}{end_to_end_fps:.1f}{_RESET}"]

    all_rows = [header] + rows + [end_row]
    widths = [max(len(_strip_ansi(r[i])) for r in all_rows) for i in range(3)]
    gap = 3
    line = "=" * (sum(widths) + gap * (len(widths) - 1))

    def fmt_row(row: list[str]) -> str:
        parts = []
        for i, cell in enumerate(row):
            pad = widths[i] - len(_strip_ansi(cell))
            parts.append(cell + " " * pad)
        return (" " * gap).join(parts).rstrip()

    print(f"\n{line}")
    print(fmt_row(header))
    print(line)
    for row in rows:
        print(fmt_row(row))
    print(line)
    print(fmt_row(end_row))
    print(line)


# ── Inference loop ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    if model_path.suffix == ".axm":
        if not args.task or not args.algo:
            raise SystemExit("When --model points to an .axm file, --task and --algo are required.")
        meta = ModelMeta.from_axm(model_path, args.task, args.algo)
    else:
        meta = ModelMeta.from_dir(model_path, args.task, args.algo)

    label = f"{meta.axm_path.parent.name}/{meta.axm_path.name}"
    print(f"\nModel  : {label}")
    print(f"Task   : {meta.task}")
    print(f"Algo   : {meta.algo}   end2end={meta.end2end}   classes={meta.num_classes}")
    print(f"Input  : {meta.imgsz[0]}×{meta.imgsz[1]}")

    preprocess, infer, postprocess = build_pipelines(meta, args.conf, args.iou)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source!r}")

    is_image = cap.get(cv2.CAP_PROP_FRAME_COUNT) == 1
    benchmark_n = args.benchmark
    win_title = f"Axelera — {meta.task}  [{meta.algo}]"

    WIN = 30
    pre_times:   deque[float] = deque(maxlen=WIN)
    infer_times: deque[float] = deque(maxlen=WIN)
    post_times:  deque[float] = deque(maxlen=WIN)
    frame_times: deque[float] = deque(maxlen=WIN)
    total_start: float | None = None
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if total_start is None:
                total_start = time.perf_counter()

            t0 = time.perf_counter()
            tensor = preprocess(frame)
            t1 = time.perf_counter()
            raw = infer(tensor)
            t2 = time.perf_counter()
            # Segmentation models output [det_tensor, proto_tensor]; unpack into
            # separate positional args so decode_segmentation receives both.
            if isinstance(raw, (list, tuple)):
                result = postprocess(*raw)
            else:
                result = postprocess(raw)
            t3 = time.perf_counter()

            pre_times.append(t1 - t0)
            infer_times.append(t2 - t1)
            post_times.append(t3 - t2)
            frame_times.append(t3 - t0)
            frame_count += 1

            fps = len(frame_times) / sum(frame_times)

            if not benchmark_n:
                vis = frame.copy()
                if meta.task == "segment":
                    detections, masks = result
                    vis = draw_segmentation(vis, detections, masks, fps)
                else:
                    vis = draw_detections(vis, result, fps)

                cv2.imshow(win_title, vis)
                key = cv2.waitKey(0 if is_image else 1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
            else:
                if frame_count % 50 == 0:
                    print(f"  {frame_count}/{benchmark_n} frames  {fps:.1f} fps", end="\r")
                if frame_count >= benchmark_n:
                    break

    finally:
        cap.release()
        if not benchmark_n:
            cv2.destroyAllWindows()

    if total_start is None or frame_count == 0:
        print("No frames processed.")
        return

    elapsed = time.perf_counter() - total_start
    # Drop first frame (warmup) from end-to-end FPS
    e2e_fps = max(frame_count - 1, 0) / elapsed

    def avg_us(q: deque) -> float:
        return sum(q) / len(q) * 1e6 if q else 0.0

    stage_avg_us = {
        "Preprocess":  avg_us(pre_times),
        "Inference":   avg_us(infer_times),
        "Postprocess": avg_us(post_times),
    }
    mode = "Benchmark" if benchmark_n else "Session"
    print(f"\n{mode}: {frame_count} frames in {elapsed:.1f}s")
    tabulate_fps(stage_avg_us, e2e_fps)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified YOLO detection + segmentation — Axelera Voyager SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python inference.py --model yolo11n_axelera_model --source video.mp4
  python inference.py --model yolo11n-seg_axelera_model --source 0
  python inference.py --model yolo26n_axelera_model --source video.mp4
  python inference.py --model yolo11n_axelera_model --source video.mp4 --benchmark 300
  python inference.py --model my_model.axm --task detect --algo yolov8 --source 0
        """,
    )
    parser.add_argument(
        "--model", required=True,
        help="Model directory containing metadata.yaml + *.axm, or direct path to .axm",
    )
    parser.add_argument(
        "--source", default="0",
        help="Video file, image path, or camera index (default: 0)",
    )
    parser.add_argument(
        "--task", choices=["detect", "segment"], default=None,
        help="Override auto-detected task",
    )
    parser.add_argument(
        "--algo", default=None,
        help="Override auto-detected algorithm (yolo11, yolov8, yolo26, yolov9, ...)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--benchmark", type=int, default=0, metavar="N",
        help="Headless benchmark: run exactly N frames then print FPS table",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
