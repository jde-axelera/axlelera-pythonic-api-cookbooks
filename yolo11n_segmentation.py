# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO11n Instance Segmentation with Axelera Voyager SDK.

Standalone example using the axelera-rt pipeline API.
No ultralytics dependency at runtime.

Usage:
    python yolo11n_segmentation.py --model yolo11n-seg_axelera_model/yolo11n-seg.axm --source video.mp4
    python yolo11n_segmentation.py --model yolo11n-seg_axelera_model/yolo11n-seg.axm --source 0
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import numpy as np
from axelera.runtime import op

# fmt: off
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
# fmt: on

_rng = np.random.default_rng(42)
CLASS_COLORS = _rng.integers(50, 230, size=(80, 3)).tolist()


def build_pipeline(model_path: str, conf: float = 0.25, iou: float = 0.45):
    """Build YOLO11n instance segmentation pipeline."""
    model_op = op.load(model_path)
    return op.seq(
        op.colorconvert("RGB", src="BGR"),
        op.letterbox(640, 640),
        op.totensor(),
        model_op,
        op.decode_segmentation(algo="yolo11", num_classes=80, num_mask_coeffs=32, confidence_threshold=conf),
        op.par(
            op.seq(op.itemgetter(0), op.nms(iou_threshold=iou, max_boxes=300)),
            op.itemgetter(1),
        ),
        op.par(
            op.seq(op.pack(), op.itemgetter(0), op.to_image_space()),
            op.proto_to_mask(),
        ),
    )


def draw_segmentation(image: np.ndarray, detections: np.ndarray, masks: list,
                      conf: float = 0.25, fps: float = 0.0, scale: float = 0.2) -> np.ndarray:
    """Draw instance segmentation results and FPS overlay on the image."""
    overlay = image.copy()
    h, w = image.shape[:2]

    for i, det in enumerate(detections):
        score = det[4]
        if score < conf:
            continue

        class_id = int(det[5])
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        name = COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else str(class_id)

        x0, y0, x1, y1 = map(int, det[:4])
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)

        if i < len(masks):
            mask = masks[i]
            if mask.ndim == 2 and (x1 - x0) > 0 and (y1 - y0) > 0:
                mask_resized = cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                overlay[y0:y1, x0:x1][mask_resized > 127] = color

        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

        label = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x0, y0 - th - 4), (x0 + tw, y0), color, -1)
        cv2.putText(image, label, (x0, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

    if fps > 0:
        fps_label = f"{fps:.1f} FPS"
        (fw, fh), _ = cv2.getTextSize(fps_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image, (8, 8), (fw + 16, fh + 16), (0, 0, 0), -1)
        cv2.putText(image, fps_label, (12, fh + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if scale != 1.0:
        dw, dh = int(w * scale), int(h * scale)
        image = cv2.resize(image, (dw, dh), interpolation=cv2.INTER_AREA)

    return image


def main():
    parser = argparse.ArgumentParser(description="YOLO11n Instance Segmentation -- Axelera Voyager SDK")
    parser.add_argument("--model", type=str, required=True, help="Path to compiled .axm model")
    parser.add_argument("--source", type=str, default="0", help="Video file path or camera index")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    args = parser.parse_args()

    pipeline = build_pipeline(args.model, args.conf, args.iou)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    is_image = cap.get(cv2.CAP_PROP_FRAME_COUNT) == 1
    frame_times: deque = deque(maxlen=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        detections, masks = pipeline(frame)
        frame_times.append(time.perf_counter() - t0)
        fps = len(frame_times) / sum(frame_times)
        print(f"{fps:.1f} fps")

        annotated = draw_segmentation(frame, detections, masks, args.conf, fps)
        cv2.imshow("YOLO11n Segmentation", annotated)
        if cv2.waitKey(0 if is_image else 1) & 0xFF in [ord("q"), ord("Q"), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
