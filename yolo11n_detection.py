from axelera.runtime import op, display
from collections import deque
from pathlib import Path
import cv2
import queue
import sys
import threading
import time


def video_reader(path):
    cap = cv2.VideoCapture(path)
    q = queue.Queue(maxsize=32)

    def decode():
        try:
            while (ret_img := cap.read())[0]:
                q.put(ret_img[1])
        finally:
            cap.release()
            q.put(None)

    t = threading.Thread(target=decode, daemon=True)
    t.start()
    try:
        while (img := q.get()) is not None:
            yield img
    finally:
        t.join()


def detection(path):
    pipeline = op.seq(
        op.colorconvert('BGR', 'RGB'),
        op.letterbox(640, 640),
        op.totensor(),
        op.load('yolo11n_axelera_model/yolo11n.axm'),
        op.decode_detections(algo='yolov11', num_classes=80, confidence_threshold=0.25),
        op.nms(iou_threshold=0.45, max_boxes=300),
        op.to_image_space(),
        op.axdetection(class_id_type=op.CocoClasses),
    )

    frame_times = deque(maxlen=30)

    with display.App(renderer='auto') as visualizer:
        wnd = visualizer.create_window('YOLO11n Detection', (900, 600))
        wnd.options(
            0,
            title=f"YOLO11n Detection - {Path(path).name}",
            title_anchor_x='center',
            title_position="50%, 0%",
        )
        for img in video_reader(path):
            t0 = time.perf_counter()
            detections = pipeline(img)
            frame_times.append(time.perf_counter() - t0)
            fps = len(frame_times) / sum(frame_times)
            print(f"{fps:.1f} fps")
            wnd(img, detections)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python yolo11n_detection.py <video_path_or_camera_index>")
        sys.exit(1)
    source = int(sys.argv[1]) if sys.argv[1].isdigit() else sys.argv[1]
    detection(source)
