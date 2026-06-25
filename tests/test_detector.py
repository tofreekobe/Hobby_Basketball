import pytest

from hobby_basketball.detector import _is_cuda_kernel_error, _predict_with_cuda_fallback


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
