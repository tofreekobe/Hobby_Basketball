from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hobby_basketball.models import MadeShotEvent
from hobby_basketball.trajectory import BallSample, RimCalibration, detect_made_shots


DEFAULT_SAMPLE_FPS = 12.0
DEFAULT_MODEL_NAME = "none"
SPORTS_BALL_CLASS_ID = 32
YOLO_IMAGE_SIZE = 416
YOLO_DISABLED_MODEL_NAMES = {"", "none", "off", "false", "0", "color", "fast"}
CUDA_KERNEL_ERROR_MARKERS = (
    "CUDA error: no kernel image is available",
    "no kernel image is available for execution on the device",
)


def scan_video_for_made_shots(
    video_path: Path | str,
    rim: RimCalibration,
    *,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    confidence: float = 0.15,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
) -> list[MadeShotEvent]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - exercised when optional deps missing
        raise RuntimeError(
            "视频识别需要 OpenCV。请先安装：python -m pip install -e .[vision]"
        ) from exc

    model = None
    if _yolo_enabled(model_name):
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - exercised when optional deps missing
            raise RuntimeError(
                "YOLO detection requires Ultralytics. Install python -m pip install -e .[vision], or set model_name to none for fast color/motion detection."
            ) from exc
        model = YOLO(model_name)

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件：{video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(native_fps / sample_fps)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    x0, y0, x1, y1 = _rim_roi(rim, width, height)

    device_state = _PredictionDeviceState(device)
    samples: list[BallSample] = []
    frame_index = 0
    previous_crop = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % stride != 0:
                continue
            t = frame_index / native_fps
            crop = frame[y0:y1, x0:x1]
            yolo_sample = None
            if model is not None:
                results = _predict_with_device_fallback(
                    model,
                    crop,
                    device_state=device_state,
                    confidence=confidence,
                )
                yolo_sample = _best_ball_sample(results, x0, y0, rim, t)
            color_sample = _best_color_ball_sample(crop, x0, y0, rim, t, previous_crop=previous_crop)
            sample = _best_observed_ball_sample([yolo_sample, color_sample], rim)
            if sample is not None:
                samples.append(sample)
            previous_crop = crop.copy()
    finally:
        cap.release()

    return detect_made_shots(samples, rim, video_path=str(video_path))


def _yolo_enabled(model_name: str) -> bool:
    return (model_name or "").strip().lower() not in YOLO_DISABLED_MODEL_NAMES


def _is_cuda_kernel_error(exc: BaseException) -> bool:
    message = str(exc)
    return any(marker in message for marker in CUDA_KERNEL_ERROR_MARKERS)


def _predict_with_cuda_fallback(model, crop, *, device_arg, confidence: float):
    device_state = _StaticPredictionDeviceState(device_arg)
    return _predict_with_device_fallback(model, crop, device_state=device_state, confidence=confidence)


def _predict_with_device_fallback(model, crop, *, device_state, confidence: float):
    try:
        return model.predict(
            crop,
            verbose=False,
            conf=confidence,
            classes=[SPORTS_BALL_CLASS_ID],
            imgsz=YOLO_IMAGE_SIZE,
            device=device_state.device_arg,
        )
    except Exception as exc:
        if device_state.can_fallback_to_cpu() and _is_cuda_kernel_error(exc):
            device_state.fallback_to_cpu()
            return model.predict(
                crop,
                verbose=False,
                conf=confidence,
                classes=[SPORTS_BALL_CLASS_ID],
                imgsz=YOLO_IMAGE_SIZE,
                device="cpu",
            )
        raise


@dataclass
class _PredictionDeviceState:
    requested_device: str
    torch_module: object | None = None

    def __post_init__(self) -> None:
        self.device_arg = _resolve_device_arg(self.requested_device, torch_module=self.torch_module)

    def can_fallback_to_cpu(self) -> bool:
        return _device_can_fallback_to_cpu(self.device_arg)

    def fallback_to_cpu(self) -> None:
        self.device_arg = "cpu"


@dataclass
class _StaticPredictionDeviceState:
    device_arg: object

    def can_fallback_to_cpu(self) -> bool:
        return _device_can_fallback_to_cpu(self.device_arg)

    def fallback_to_cpu(self) -> None:
        self.device_arg = "cpu"


def _resolve_device_arg(device: str, *, torch_module=None):
    normalized = (device or "cpu").strip().lower()
    device_arg = None if normalized == "auto" else normalized
    if _device_can_fallback_to_cpu(device_arg) and _torch_cuda_arch_is_unsupported(torch_module):
        return "cpu"
    return device_arg


def _torch_cuda_arch_is_unsupported(torch_module=None) -> bool:
    try:
        if torch_module is None:
            import torch as torch_module

        cuda = torch_module.cuda
        if not cuda.is_available():
            return False

        supported_arches = set(cuda.get_arch_list() or [])
        if not supported_arches:
            return False

        major, minor = cuda.get_device_capability(0)
        current_arch = f"sm_{major}{minor}"
        return current_arch not in supported_arches
    except Exception:
        return False


def _device_can_fallback_to_cpu(device_arg) -> bool:
    if device_arg is None:
        return True
    if isinstance(device_arg, int):
        return device_arg >= 0
    if isinstance(device_arg, str):
        normalized = device_arg.strip().lower()
        return normalized in {"auto", "cuda", "cuda:0", "0"}
    return False


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


def _best_color_ball_sample(
    crop,
    x_offset: int,
    y_offset: int,
    rim: RimCalibration,
    t: float,
    *,
    previous_crop=None,
) -> BallSample | None:
    try:
        import cv2
        import numpy as np
    except Exception:  # pragma: no cover - cv2 is already required by video scanning
        return None

    if crop is None or crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(hsv, np.array([0, 60, 70]), np.array([28, 255, 255]))

    if previous_crop is not None and previous_crop.shape == crop.shape:
        diff = cv2.absdiff(crop, previous_crop)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, motion_mask = cv2.threshold(gray_diff, 18, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.dilate(motion_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
        moving_orange = cv2.bitwise_and(orange_mask, motion_mask)
        if cv2.countNonZero(moving_orange) >= 12:
            orange_mask = moving_orange

    kernel = np.ones((3, 3), dtype=np.uint8)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: tuple[float, float, float] | None = None
    max_area = max(300.0, crop.shape[0] * crop.shape[1] * 0.08)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 12.0 or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 4 or h < 4 or w > 120 or h > 120:
            continue
        aspect = w / max(h, 1)
        if aspect < 0.35 or aspect > 2.5:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0 if perimeter <= 0 else min(1.0, 4.0 * 3.141592653589793 * area / (perimeter * perimeter))
        cx = x_offset + x + w / 2.0
        cy = y_offset + y + h / 2.0
        dist = ((cx - rim.center_x) ** 2 + (cy - rim.center_y) ** 2) ** 0.5
        proximity = max(0.0, 1.0 - dist / max(max(rim.half_width, rim.half_height) * 4.0, 1.0))
        area_score = min(1.0, area / 900.0)
        score = 0.18 + proximity * 0.36 + circularity * 0.24 + area_score * 0.22
        if best is None or score > best[0]:
            best = (score, cx, cy)

    if best is None:
        return None
    return BallSample(t=t, x=best[1], y=best[2], confidence=round(min(0.85, best[0]), 6))


def _best_observed_ball_sample(samples: list[BallSample | None], rim: RimCalibration) -> BallSample | None:
    observed = [sample for sample in samples if sample is not None]
    if not observed:
        return None
    return max(observed, key=lambda sample: _sample_observation_score(sample, rim))


def _sample_observation_score(sample: BallSample, rim: RimCalibration) -> float:
    dist = ((sample.x - rim.center_x) ** 2 + (sample.y - rim.center_y) ** 2) ** 0.5
    proximity = max(0.0, 1.0 - dist / max(max(rim.half_width, rim.half_height) * 5.0, 1.0))
    return sample.confidence * 0.7 + proximity * 0.3
