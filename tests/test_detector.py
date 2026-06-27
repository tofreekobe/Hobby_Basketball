import pytest

from hobby_basketball.detector import (
    _PredictionDeviceState,
    _best_color_ball_sample,
    inspect_cuda_runtime,
    _is_cuda_kernel_error,
    _predict_with_cuda_fallback,
    _predict_with_device_fallback,
    _resolve_device_arg,
    scan_video_for_made_shots,
)
from hobby_basketball.trajectory import RimCalibration


def test_cuda_kernel_error_is_detected():
    error = RuntimeError("CUDA error: no kernel image is available for execution on the device")

    assert _is_cuda_kernel_error(error)


def test_predict_falls_back_to_cpu_after_auto_cuda_kernel_error():
    model = FakeModel()

    result = _predict_with_cuda_fallback(model, "frame", device_arg=None, confidence=0.15)

    assert result == ["cpu-result"]
    assert model.devices == [None, "cpu"]


def test_predict_does_not_swallow_non_cuda_errors():
    model = FakeModel(error=RuntimeError("unexpected inference failure"))

    with pytest.raises(RuntimeError, match="unexpected inference failure"):
        _predict_with_cuda_fallback(model, "frame", device_arg=None, confidence=0.15)

    assert model.devices == [None]


def test_cuda_fallback_is_sticky_for_remaining_predictions():
    model = FakeModel()
    device_state = _PredictionDeviceState("cuda", torch_module=FakeSupportedTorch)

    first_result = _predict_with_device_fallback(model, "frame-1", device_state=device_state, confidence=0.15)
    second_result = _predict_with_device_fallback(model, "frame-2", device_state=device_state, confidence=0.15)

    assert first_result == ["cpu-result"]
    assert second_result == ["cpu-result"]
    assert model.devices == ["cuda", "cpu", "cpu"]


def test_unsupported_cuda_arch_resolves_to_cpu_before_prediction():
    assert _resolve_device_arg("cuda", torch_module=FakeUnsupportedTorch) == "cpu"
    assert _resolve_device_arg("auto", torch_module=FakeUnsupportedTorch) == "cpu"


def test_cuda_runtime_status_reports_unsupported_arch():
    status = inspect_cuda_runtime(torch_module=FakeUnsupportedTorch)

    assert status["cuda_available"] is True
    assert status["current_arch"] == "sm_120"
    assert status["cuda_supported"] is False
    assert status["default_device"] == "cpu"
    assert "sm_120" in status["message"]


def test_color_ball_sample_prefers_moving_orange_blob_near_rim():
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    rim = RimCalibration(center_x=100, center_y=80, half_width=25, half_height=25)
    previous = np.zeros((160, 220, 3), dtype=np.uint8)
    current = previous.copy()
    cv2.circle(previous, (100, 65), 8, (255, 255, 255), -1)
    cv2.circle(current, (105, 92), 10, (0, 95, 255), -1)

    sample = _best_color_ball_sample(current, 0, 0, rim, 2.0, previous_crop=previous)

    assert sample is not None
    assert 99 <= sample.x <= 111
    assert 86 <= sample.y <= 98
    assert sample.confidence >= 0.3


def test_color_ball_sample_ignores_static_orange_rim_with_motion_context():
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    rim = RimCalibration(center_x=100, center_y=80, half_width=25, half_height=25)
    previous = np.zeros((160, 220, 3), dtype=np.uint8)
    current = previous.copy()
    cv2.rectangle(previous, (72, 67), (128, 82), (0, 80, 255), -1)
    cv2.line(previous, (78, 82), (122, 130), (0, 80, 255), 3)
    current[:] = previous

    sample = _best_color_ball_sample(current, 0, 0, rim, 2.0, previous_crop=previous)

    assert sample is None


def test_scan_video_can_detect_make_without_yolo(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    video_path = tmp_path / "synthetic_make.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (200, 160),
    )
    assert writer.isOpened()
    ball_positions = {
        2: (100, 48),
        3: (100, 62),
        4: (100, 82),
        5: (100, 100),
    }
    for frame_index in range(10):
        frame = np.zeros((160, 200, 3), dtype=np.uint8)
        cv2.rectangle(frame, (78, 68), (122, 75), (240, 240, 240), 1)
        if frame_index in ball_positions:
            cv2.circle(frame, ball_positions[frame_index], 7, (0, 95, 255), -1)
        writer.write(frame)
    writer.release()

    rim = RimCalibration(center_x=100, center_y=70, half_width=20, half_height=10)

    makes = scan_video_for_made_shots(
        video_path,
        rim,
        sample_fps=10.0,
        model_name="none",
        device="cpu",
    )

    assert len(makes) == 1
    assert 0.25 <= makes[0].t_make <= 0.55


def test_scan_video_applies_confidence_to_final_fast_candidates(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    video_path = tmp_path / "synthetic_make.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (200, 160),
    )
    assert writer.isOpened()
    for frame_index, position in enumerate([(100, 48), (100, 62), (100, 82), (100, 100)]):
        frame = np.zeros((160, 200, 3), dtype=np.uint8)
        cv2.rectangle(frame, (78, 68), (122, 75), (240, 240, 240), 1)
        cv2.circle(frame, position, 7, (0, 95, 255), -1)
        writer.write(frame)
    writer.release()

    rim = RimCalibration(center_x=100, center_y=70, half_width=20, half_height=10)

    makes = scan_video_for_made_shots(
        video_path,
        rim,
        sample_fps=10.0,
        confidence=0.99,
        model_name="none",
        device="cpu",
    )

    assert makes == []


def test_scan_video_expands_thin_rim_line_to_net_region(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    video_path = tmp_path / "thin_rim_make.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.0,
        (544, 960),
    )
    assert writer.isOpened()
    ball_positions = {
        4: (251, 390),
        5: (251, 412),
        6: (251, 455),
    }
    for frame_index in range(16):
        frame = np.zeros((960, 544, 3), dtype=np.uint8)
        cv2.rectangle(frame, (221, 388), (281, 412), (230, 230, 230), 2)
        cv2.line(frame, (223, 412), (279, 470), (230, 230, 230), 1)
        if frame_index in ball_positions:
            cv2.circle(frame, ball_positions[frame_index], 9, (0, 95, 255), -1)
        writer.write(frame)
    writer.release()

    narrow_rim_line = RimCalibration(center_x=251, center_y=400, half_width=30, half_height=12)

    makes = scan_video_for_made_shots(
        video_path,
        narrow_rim_line,
        sample_fps=12.0,
        confidence=0.15,
        model_name="none",
        device="cpu",
    )

    assert len(makes) == 1
    assert makes[0].notes == "rim-net entry"


class FakeModel:
    def __init__(self, error: Exception | None = None):
        self.devices = []
        self.error = error or RuntimeError(
            "CUDA error: no kernel image is available for execution on the device"
        )

    def predict(self, *args, **kwargs):
        self.devices.append(kwargs.get("device"))
        if len(self.devices) == 1:
            raise self.error
        return ["cpu-result"]


class FakeUnsupportedCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device_capability(index=0):
        return (12, 0)

    @staticmethod
    def get_arch_list():
        return ["sm_50", "sm_60", "sm_90"]


class FakeUnsupportedTorch:
    cuda = FakeUnsupportedCuda


class FakeSupportedCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device_capability(index=0):
        return (12, 0)

    @staticmethod
    def get_arch_list():
        return ["sm_120"]


class FakeSupportedTorch:
    cuda = FakeSupportedCuda
