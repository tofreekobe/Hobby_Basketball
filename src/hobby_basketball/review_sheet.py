from __future__ import annotations

from pathlib import Path

from hobby_basketball.models import MadeShotEvent
from hobby_basketball.trajectory import RimCalibration


def build_candidate_review_sheet(
    video_path: Path | str,
    events: list[MadeShotEvent],
    rim: RimCalibration,
    output_path: Path | str,
    *,
    max_events: int = 24,
) -> Path:
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional vision dependency
        raise RuntimeError("Candidate review sheets require OpenCV. Install python -m pip install -e .[vision].") from exc

    if not events:
        raise ValueError("events must not be empty")

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    tiles = []
    try:
        for index, event in enumerate(events[:max_events], start=1):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, event.t_make) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            tile = _render_event_tile(frame, event, rim, index)
            tiles.append(tile)
    finally:
        cap.release()

    if not tiles:
        raise RuntimeError("No review frames could be read from the video")

    cols = min(3, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * 280, cols * 360, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, cols)
        sheet[row * 280 : (row + 1) * 280, col * 360 : (col + 1) * 360] = tile

    ok, encoded = cv2.imencode(".jpg", sheet)
    if not ok:
        raise RuntimeError(f"Unable to write review sheet: {output_path}")
    output_path.write_bytes(encoded.tobytes())
    return output_path


def _render_event_tile(frame, event: MadeShotEvent, rim: RimCalibration, index: int):
    import cv2

    height, width = frame.shape[:2]
    center_x = int(round(rim.center_x))
    center_y = int(round(rim.center_y))
    x0 = max(0, center_x - 260)
    x1 = min(width, center_x + 260)
    y0 = max(0, center_y - 240)
    y1 = min(height, center_y + 270)
    crop = frame[y0:y1, x0:x1].copy()
    if crop.size == 0:
        crop = frame.copy()
        x0 = 0
        y0 = 0

    left = int(round(rim.center_x - rim.half_width - x0))
    right = int(round(rim.center_x + rim.half_width - x0))
    top = int(round(rim.center_y - rim.half_height - y0))
    bottom = int(round(rim.center_y + rim.half_height - y0))
    cv2.rectangle(crop, (left, top), (right, bottom), (0, 0, 255), 2)

    label = f"{index:02d} {event.t_make:.2f}s {event.confidence:.2f} {event.notes or event.id}"
    cv2.putText(crop, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(crop, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.resize(crop, (360, 280), interpolation=cv2.INTER_AREA)
