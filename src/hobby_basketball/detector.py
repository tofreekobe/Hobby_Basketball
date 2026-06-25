from __future__ import annotations

from pathlib import Path

from hobby_basketball.models import MadeShotEvent
from hobby_basketball.trajectory import BallSample, RimCalibration, detect_made_shots


SPORTS_BALL_CLASS_ID = 32


def scan_video_for_made_shots(
    video_path: Path | str,
    rim: RimCalibration,
    *,
    sample_fps: float = 15.0,
    confidence: float = 0.15,
    model_name: str = "yolo11m.pt",
    device: str = "auto",
) -> list[MadeShotEvent]:
    try:
        import cv2
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - exercised when optional deps missing
        raise RuntimeError(
            "视频识别需要 OpenCV 和 Ultralytics。请先安装：python -m pip install -e .[vision]"
        ) from exc

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件：{video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(native_fps / sample_fps)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    x0, y0, x1, y1 = _rim_roi(rim, width, height)

    model = YOLO(model_name)
    device_arg = None if device == "auto" else device
    samples: list[BallSample] = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % stride != 0:
            continue
        t = frame_index / native_fps
        crop = frame[y0:y1, x0:x1]
        results = model.predict(
            crop,
            verbose=False,
            conf=confidence,
            classes=[SPORTS_BALL_CLASS_ID],
            imgsz=640,
            device=device_arg,
        )
        sample = _best_ball_sample(results, x0, y0, rim, t)
        if sample is not None:
            samples.append(sample)

    cap.release()
    return detect_made_shots(samples, rim, video_path=str(video_path))


def _rim_roi(rim: RimCalibration, width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, int(rim.center_x - 220))
    right = min(width, int(rim.center_x + 220))
    top = max(0, int(rim.center_y - 200))
    bottom = min(height, int(rim.center_y + 160))
    return left, top, right, bottom


def _best_ball_sample(results, x_offset: int, y_offset: int, rim: RimCalibration, t: float) -> BallSample | None:
    best: tuple[float, float, float, float] | None = None
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = x_offset + (x1 + x2) / 2.0
            cy = y_offset + (y1 + y2) / 2.0
            dist = (cx - rim.center_x) ** 2 + (cy - rim.center_y) ** 2
            if best is None or dist < best[0]:
                best = (dist, cx, cy, conf)
    if best is None:
        return None
    return BallSample(t=t, x=best[1], y=best[2], confidence=best[3])
