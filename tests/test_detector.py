import pytest

from hobby_basketball.detector import (
    _PredictionDeviceState,
    _best_color_ball_sample,
    _is_cuda_kernel_error,
    _predict_with_cuda_fallback,
    _predict_with_device_fallback,
    _resolve_device_arg,
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
