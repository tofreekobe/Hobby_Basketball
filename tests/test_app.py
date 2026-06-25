from fastapi.testclient import TestClient

from hobby_basketball.models import MadeShotEvent
from hobby_basketball.app import app


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_plan_clips_endpoint_returns_default_window():
    client = TestClient(app)

    response = client.post(
        "/api/plan-clips",
        json={
            "video_path": "game.mp4",
            "events": [
                {"id": "make-1", "video_path": "game.mp4", "t_make": 10.0, "confidence": 0.8}
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["clips"][0]["start"] == 5.0
    assert response.json()["clips"][0]["end"] == 11.5


def test_index_contains_file_picker_and_export_controls():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert 'type="file"' in html
    assert 'id="device"' in html
    assert 'value="cpu" selected' in html
    assert 'id="outputFormat"' in html
    assert "执行识别并剪辑" in html


def test_upload_video_saves_file_and_returns_preview_url():
    client = TestClient(app)

    response = client.post(
        "/api/upload-video",
        files={"file": ("game.mp4", b"fake-video-bytes", "video/mp4")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["video_id"]
    assert data["filename"] == "game.mp4"
    assert data["preview_url"].startswith("/api/videos/")


def test_process_video_runs_detection_and_export(monkeypatch):
    client = TestClient(app)
    upload = client.post(
        "/api/upload-video",
        files={"file": ("game.mp4", b"fake-video-bytes", "video/mp4")},
    ).json()

    detector_kwargs = {}

    def fake_detector(video_path, rim, **kwargs):
        detector_kwargs.update(kwargs)
        return [
            MadeShotEvent(
                id="make-1",
                video_path=str(video_path),
                t_make=10.0,
                confidence=0.9,
            )
        ]

    def fake_export(clips, output_dir, final_path):
        final_path.write_bytes(b"rendered")
        return []

    monkeypatch.setattr("hobby_basketball.app.scan_video_for_made_shots", fake_detector)
    monkeypatch.setattr("hobby_basketball.app.export_reel", fake_export)

    response = client.post(
        "/api/process-video",
        json={
            "video_id": upload["video_id"],
            "rim": {"center_x": 100, "center_y": 100, "half_width": 20, "half_height": 10},
            "pre_seconds": 5.0,
            "post_seconds": 1.5,
            "output_format": "mp4",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["events"][0]["id"] == "make-1"
    assert data["clips"][0]["start"] == 5.0
    assert data["preview_url"].endswith(".mp4")
    assert detector_kwargs["device"] == "cpu"
