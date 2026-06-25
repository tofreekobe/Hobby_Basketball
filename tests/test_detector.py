import pytest

from hobby_basketball.detector import (
    _PredictionDeviceState,
    _is_cuda_kernel_error,
    _predict_with_cuda_fallback,
    _predict_with_device_fallback,
    _resolve_device_arg,
)


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
