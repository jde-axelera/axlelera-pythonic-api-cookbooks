#!/usr/bin/env bash
# Benchmark all compiled models and print per-stage FPS tables.
# Usage:  ./benchmark.sh [source] [frames]
#   source : video file path or camera index (default: 0)
#   frames : number of frames to run per model (default: 300)
#
# Note: YOLOv8n models are not yet compiled. Compile them first with:
#   axcompile yolov8n.onnx     --output yolov8n_axelera_model/
#   axcompile yolov8n-seg.onnx --output yolov8n-seg_axelera_model/

set -e
cd "$(dirname "$0")"
source pythonic_api/bin/activate

SOURCE="${1:-0}"
FRAMES="${2:-300}"

sep() { printf '\n\e[1m%s\e[0m\n' "━━━ $1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

echo "Source : $SOURCE   Frames per model : $FRAMES"

sep "YOLO11n   — Detection"
python inference.py --model yolo11n_axelera_model     --source "$SOURCE" --benchmark "$FRAMES"

sep "YOLO11n   — Segmentation"
python inference.py --model yolo11n-seg_axelera_model --source "$SOURCE" --benchmark "$FRAMES"

sep "YOLO26n   — Detection"
python inference.py --model yolo26n_axelera_model     --source "$SOURCE" --benchmark "$FRAMES"

# ── Placeholder entries for models not yet compiled ───────────────────────────
if [ -d yolov8n_axelera_model ]; then
    sep "YOLOv8n   — Detection"
    python inference.py --model yolov8n_axelera_model     --source "$SOURCE" --benchmark "$FRAMES"
else
    printf '\n\e[33mSkipping YOLOv8n detection — model not compiled.\e[0m\n'
    printf '  Run: axcompile yolov8n.onnx --output yolov8n_axelera_model/\n'
fi

if [ -d yolov8n-seg_axelera_model ]; then
    sep "YOLOv8n   — Segmentation"
    python inference.py --model yolov8n-seg_axelera_model --source "$SOURCE" --benchmark "$FRAMES"
else
    printf '\n\e[33mSkipping YOLOv8n segmentation — model not compiled.\e[0m\n'
    printf '  Run: axcompile yolov8n-seg.onnx --output yolov8n-seg_axelera_model/\n'
fi

echo
