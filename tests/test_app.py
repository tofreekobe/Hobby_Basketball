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
    assert 'id="sampleFps"' in html
    assert 'id="sampleFps" type="number" step="0.5" min="1" max="15" value="12.0"' in html
    assert 'id="rimOverlay"' in html
    assert "startRimDrag" in html
    assert 'id="detectBtn"' in html
    assert 'id="exportSelectedBtn"' in html
    assert 'id="candidateList"' in html
    assert 'id="truthTimes"' in html
    assert "seekCandidate" in html
    assert "markCurrentTruthTime" in html
    assert "/api/evaluate-candidates" in html
    assert "target_precision" in html
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
    assert detector_kwargs["sample_fps"] == 12.0
    assert detector_kwargs["model_name"] == "none"


def test_detect_video_returns_reviewable_candidates_without_export(monkeypatch):
    client = TestClient(app)
    upload = client.post(
        "/api/upload-video",
        files={"file": ("game.mp4", b"fake-video-bytes", "video/mp4")},
    ).json()

    def fake_detector(video_path, rim, **kwargs):
        return [
            MadeShotEvent(
                id="make-1",
                video_path=str(video_path),
                t_make=18.53,
                confidence=0.86,
                notes="rim-net entry",
            )
        ]

    monkeypatch.setattr("hobby_basketball.app.scan_video_for_made_shots", fake_detector)

    response = client.post(
        "/api/detect-video",
        json={
            "video_id": upload["video_id"],
            "rim": {"center_x": 100, "center_y": 100, "half_width": 20, "half_height": 10},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["events"][0]["t_make"] == 18.53
    assert data["clips"][0]["start"] == 13.53
    assert data["events"][0]["notes"] == "rim-net entry"


def test_export_events_exports_only_kept_candidates(monkeypatch):
    client = TestClient(app)
    upload = client.post(
        "/api/upload-video",
        files={"file": ("game.mp4", b"fake-video-bytes", "video/mp4")},
    ).json()
    exported = {}

    def fake_export(clips, output_dir, final_path):
        exported["clips"] = clips
        final_path.write_bytes(b"rendered")
        return []

    monkeypatch.setattr("hobby_basketball.app.export_reel", fake_export)

    response = client.post(
        "/api/export-events",
        json={
            "video_id": upload["video_id"],
            "events": [
                {
                    "id": "make-1",
                    "video_path": "",
                    "t_make": 18.53,
                    "confidence": 0.86,
                    "kept": True,
                },
                {
                    "id": "false-1",
                    "video_path": "",
                    "t_make": 30.0,
                    "confidence": 0.4,
                    "kept": False,
                },
            ],
            "output_format": "mp4",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["clips"][0]["event_ids"] == ["make-1"]
    assert len(exported["clips"]) == 1


def test_evaluate_events_endpoint_returns_precision_recall_f1():
    client = TestClient(app)

    response = client.post(
        "/api/evaluate-events",
        json={
            "predicted_times": [18.53, 44.66, 70.0],
            "truth_times": [18.4, 44.7],
            "tolerance_sec": 1.0,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["precision"] == 0.666667
    assert data["recall"] == 1.0
    assert data["f1"] == 0.8


def test_evaluate_candidates_endpoint_recommends_confidence_threshold():
    client = TestClient(app)

    response = client.post(
        "/api/evaluate-candidates",
        json={
            "events": [
                {"id": "make-1", "video_path": "game.mp4", "t_make": 10.0, "confidence": 0.92},
                {"id": "make-2", "video_path": "game.mp4", "t_make": 20.0, "confidence": 0.84},
                {"id": "false-1", "video_path": "game.mp4", "t_make": 45.0, "confidence": 0.30},
            ],
            "truth_times": [10.2, 20.1],
            "tolerance_sec": 1.0,
            "target_precision": 0.95,
            "target_recall": 0.95,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["target_met"] is True
    assert data["recommended_threshold"] == 0.84
    assert data["recommended"]["precision"] == 1.0
    assert data["recommended"]["recall"] == 1.0
