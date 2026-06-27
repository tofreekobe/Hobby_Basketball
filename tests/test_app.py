import json

from fastapi.testclient import TestClient
import pytest

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
    assert 'id="rimHalfWidth" type="number" value="42"' in html
    assert 'id="rimHalfHeight" type="number" value="50"' in html
    assert "ensureUsableRimBox" in html
    assert "displayedVideoRect" in html
    assert "object-fit: contain" in html
    assert "startRimDrag" in html
    assert 'id="detectBtn"' in html
    assert 'id="exportSelectedBtn"' in html
    assert 'id="reviewSheetBtn"' in html
    assert 'id="saveCandidateReviewBtn"' in html
    assert 'id="saveEvaluationBtn"' in html
    assert 'id="candidateList"' in html
    assert 'id="truthTimes"' in html
    assert "seekCandidate" in html
    assert "markCurrentTruthTime" in html
    assert "/api/evaluate-candidates" in html
    assert "target_precision" in html
    assert "applyRecommendedThreshold" in html
    assert "/api/save-evaluation-run" in html
    assert "/api/evaluation-summary" in html
    assert 'id="evaluationSummaryBtn"' in html
    assert "/api/candidate-review-sheet" in html
    assert "/api/save-candidate-review" in html
    assert "/api/review-regression-summary" in html
    assert 'id="reviewRegressionSummaryBtn"' in html
    assert "/api/review-regression-sheet" in html
    assert 'id="reviewRegressionSheetBtn"' in html
    assert "generateReviewRegressionSheet" in html
    assert "sourcePreview" in html
    assert "/api/videos/${currentVideoId}" in html
    assert "/api/device-status" in html
    assert "refreshDeviceStatus" in html
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


def test_device_status_endpoint_reports_runtime(monkeypatch):
    client = TestClient(app)

    def fake_status():
        return {
            "cuda_available": True,
            "current_arch": "sm_120",
            "supported_arches": ["sm_50", "sm_90"],
            "cuda_supported": False,
            "default_device": "cpu",
            "message": "CUDA is not compatible with sm_120; using CPU.",
        }

    monkeypatch.setattr("hobby_basketball.app.inspect_cuda_runtime", fake_status)

    response = client.get("/api/device-status")

    assert response.status_code == 200
    data = response.json()
    assert data["current_arch"] == "sm_120"
    assert data["cuda_supported"] is False
    assert data["default_device"] == "cpu"


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


def test_candidate_review_sheet_persists_jpeg_for_uploaded_video(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    monkeypatch.setattr("hobby_basketball.workspace.UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr("hobby_basketball.app.REVIEW_DIR", tmp_path / "reviews")
    video_path = tmp_path / "game.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (200, 160),
    )
    assert writer.isOpened()
    for frame_index in range(10):
        frame = np.zeros((160, 200, 3), dtype=np.uint8)
        cv2.rectangle(frame, (80, 70), (120, 95), (240, 240, 240), 2)
        cv2.circle(frame, (100, 82 + frame_index), 7, (0, 95, 255), -1)
        writer.write(frame)
    writer.release()

    client = TestClient(app)
    with video_path.open("rb") as handle:
        upload = client.post(
            "/api/upload-video",
            files={"file": ("game.mp4", handle, "video/mp4")},
        ).json()

    response = client.post(
        "/api/candidate-review-sheet",
        json={
            "video_id": upload["video_id"],
            "events": [
                {"id": "make-1", "video_path": "", "t_make": 0.5, "confidence": 0.86, "notes": "rim-net entry"}
            ],
            "rim": {"center_x": 100, "center_y": 82, "half_width": 20, "half_height": 18},
        },
    )

    assert response.status_code == 200
    data = response.json()
    saved = tmp_path / "reviews" / f'{data["sheet_id"]}.jpg'
    assert data["review_path"] == str(saved)
    assert data["preview_url"] == f'/api/reviews/{data["sheet_id"]}.jpg'
    assert saved.exists()
    assert saved.read_bytes().startswith(b"\xff\xd8")
    preview = client.get(data["preview_url"])
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/jpeg"


def test_save_candidate_review_persists_precision_summary(tmp_path, monkeypatch):
    monkeypatch.setattr("hobby_basketball.app.REVIEW_DIR", tmp_path / "reviews")
    client = TestClient(app)

    response = client.post(
        "/api/save-candidate-review",
        json={
            "video_id": "video-1",
            "events": [
                {"id": "make-1", "video_path": "", "t_make": 10.0, "confidence": 0.9, "kept": True},
                {"id": "make-2", "video_path": "", "t_make": 20.0, "confidence": 0.8, "kept": True},
                {"id": "false-1", "video_path": "", "t_make": 30.0, "confidence": 0.7, "kept": False},
            ],
            "rim": {"center_x": 100, "center_y": 80, "half_width": 42, "half_height": 50},
            "reviewer": "manual",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["candidate_count"] == 3
    assert data["summary"]["accepted_count"] == 2
    assert data["summary"]["rejected_count"] == 1
    assert data["summary"]["review_precision"] == 0.666667
    saved = tmp_path / "reviews" / f'{data["review_id"]}.json'
    assert data["review_path"] == str(saved)
    assert saved.exists()
    content = saved.read_text(encoding="utf-8")
    assert '"review_precision":0.666667' in content


def test_review_regression_summary_reruns_detector_against_review_labels(tmp_path, monkeypatch):
    review_dir = tmp_path / "reviews"
    review_dir.mkdir()
    video_path = tmp_path / "game.mp4"
    video_path.write_bytes(b"fake-video")
    monkeypatch.setattr("hobby_basketball.app.REVIEW_DIR", review_dir)
    monkeypatch.setattr("hobby_basketball.app.VIDEO_REGISTRY", {})
    review_payload = {
        "review_id": "review-1",
        "video_id": "video-1",
        "rim": {"center_x": 100, "center_y": 80, "half_width": 42, "half_height": 50},
        "events": [
            {"id": "make-1", "video_path": str(video_path), "t_make": 10.0, "confidence": 0.9, "kept": True},
            {"id": "make-2", "video_path": str(video_path), "t_make": 20.0, "confidence": 0.8, "kept": True},
            {"id": "false-1", "video_path": str(video_path), "t_make": 30.0, "confidence": 0.7, "kept": False},
        ],
        "summary": {"candidate_count": 3, "accepted_count": 2, "rejected_count": 1, "review_precision": 0.666667},
    }
    (review_dir / "review-1.json").write_text(json.dumps(review_payload), encoding="utf-8")

    detector_calls = []

    def fake_detector(video_path_arg, rim, **kwargs):
        detector_calls.append((video_path_arg, rim, kwargs))
        return [
            MadeShotEvent(id="make-current-1", video_path="", t_make=10.2, confidence=0.9),
            MadeShotEvent(id="false-current-1", video_path="", t_make=30.1, confidence=0.7),
            MadeShotEvent(id="unreviewed-current-1", video_path="", t_make=99.0, confidence=0.6),
        ]

    monkeypatch.setattr("hobby_basketball.app.scan_video_for_made_shots", fake_detector)
    client = TestClient(app)

    response = client.get("/api/review-regression-summary")

    assert response.status_code == 200
    data = response.json()
    assert data["review_count"] == 1
    assert data["skipped_review_count"] == 0
    assert data["reviewed_candidate_count"] == 3
    assert data["accepted_label_count"] == 2
    assert data["rejected_label_count"] == 1
    assert data["accepted_preserved_count"] == 1
    assert data["missed_accepted_count"] == 1
    assert data["false_positive_recurrences"] == 1
    assert data["rejected_suppressed_count"] == 0
    assert data["unreviewed_prediction_count"] == 1
    assert data["reviewed_precision"] == 0.5
    assert data["accepted_recall"] == 0.5
    assert data["rejected_suppression_rate"] == 0.0
    assert data["target_met"] is False
    assert data["metrics_scope"] == "reviewed_candidate_labels_only"
    assert data["evaluated_review_ids"] == ["review-1"]
    assert detector_calls[0][0] == video_path


def test_review_regression_sheet_generates_unreviewed_candidate_sheet(tmp_path, monkeypatch):
    review_dir = tmp_path / "reviews"
    review_dir.mkdir()
    video_path = tmp_path / "game.mp4"
    video_path.write_bytes(b"fake-video")
    monkeypatch.setattr("hobby_basketball.app.REVIEW_DIR", review_dir)
    review_payload = {
        "review_id": "review-1",
        "video_id": "video-1",
        "rim": {"center_x": 100, "center_y": 80, "half_width": 42, "half_height": 50},
        "events": [
            {"id": "make-1", "video_path": str(video_path), "t_make": 10.0, "confidence": 0.9, "kept": True},
            {"id": "false-1", "video_path": str(video_path), "t_make": 30.0, "confidence": 0.7, "kept": False},
        ],
    }
    (review_dir / "review-1.json").write_text(json.dumps(review_payload), encoding="utf-8")

    def fake_detector(video_path_arg, rim, **kwargs):
        return [
            MadeShotEvent(id="make-current-1", video_path="", t_make=10.2, confidence=0.9),
            MadeShotEvent(id="false-current-1", video_path="", t_make=30.1, confidence=0.7),
            MadeShotEvent(id="unreviewed-current-1", video_path="", t_make=99.0, confidence=0.6),
        ]

    sheet_calls = []

    def fake_review_sheet(video_path_arg, events, rim, output_path):
        sheet_calls.append((video_path_arg, events, rim, output_path))
        output_path.write_bytes(b"\xff\xd8fake-jpeg")

    monkeypatch.setattr("hobby_basketball.app.scan_video_for_made_shots", fake_detector)
    monkeypatch.setattr("hobby_basketball.app.build_candidate_review_sheet", fake_review_sheet)
    client = TestClient(app)

    response = client.get("/api/review-regression-sheet")

    assert response.status_code == 200
    data = response.json()
    assert data["events"][0]["id"] == "unreviewed-current-1"
    assert data["events"][0]["t_make"] == 99.0
    assert data["video_id"] == "video-1"
    assert data["source_review_ids"] == ["review-1"]
    saved = review_dir / f'{data["sheet_id"]}.jpg'
    assert data["review_path"] == str(saved)
    assert data["preview_url"] == f'/api/reviews/{data["sheet_id"]}.jpg'
    assert saved.read_bytes().startswith(b"\xff\xd8")
    assert len(sheet_calls) == 1
    assert sheet_calls[0][0] == video_path
    assert [event.t_make for event in sheet_calls[0][1]] == [99.0]
    preview = client.get("/api/videos/video-1")
    assert preview.status_code == 200
    assert preview.content == b"fake-video"


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


def test_save_evaluation_run_persists_report_for_uploaded_video(tmp_path, monkeypatch):
    monkeypatch.setattr("hobby_basketball.workspace.WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr("hobby_basketball.workspace.UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr("hobby_basketball.workspace.EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setattr("hobby_basketball.app.EVALUATION_DIR", tmp_path / "evaluations")
    client = TestClient(app)
    upload = client.post(
        "/api/upload-video",
        files={"file": ("game.mp4", b"fake-video-bytes", "video/mp4")},
    ).json()

    response = client.post(
        "/api/save-evaluation-run",
        json={
            "video_id": upload["video_id"],
            "events": [
                {"id": "make-1", "video_path": "", "t_make": 10.0, "confidence": 0.92},
                {"id": "make-2", "video_path": "", "t_make": 20.0, "confidence": 0.84},
            ],
            "truth_times": [10.2, 20.1],
            "rim": {"center_x": 100, "center_y": 100, "half_width": 20, "half_height": 10},
            "sample_fps": 12.0,
            "confidence": 0.15,
            "model_name": "none",
            "device": "cpu",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["report"]["target_met"] is True
    saved_path = data["evaluation_path"]
    saved = tmp_path / "evaluations" / f"{data['run_id']}.json"
    assert saved_path == str(saved)
    assert saved.exists()
    content = saved.read_text(encoding="utf-8")
    assert '"truth_times":[10.2,20.1]' in content
    assert '"recommended_threshold":0.84' in content


def test_evaluation_summary_aggregates_saved_runs(tmp_path, monkeypatch):
    evaluation_dir = tmp_path / "evaluations"
    evaluation_dir.mkdir()
    monkeypatch.setattr("hobby_basketball.app.EVALUATION_DIR", evaluation_dir)
    (evaluation_dir / "run-1.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "report": {
                    "target_precision": 0.95,
                    "target_recall": 0.95,
                    "target_met": True,
                    "recommended": {
                        "predicted_count": 2,
                        "true_positives": 2,
                        "false_positives": 0,
                        "false_negatives": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (evaluation_dir / "run-2.json").write_text(
        json.dumps(
            {
                "run_id": "run-2",
                "report": {
                    "target_precision": 0.95,
                    "target_recall": 0.95,
                    "target_met": False,
                    "recommended": {
                        "predicted_count": 3,
                        "true_positives": 2,
                        "false_positives": 1,
                        "false_negatives": 1,
                        "precision": 0.666667,
                        "recall": 0.666667,
                        "f1": 0.666667,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(app)

    response = client.get("/api/evaluation-summary")

    assert response.status_code == 200
    data = response.json()
    assert data["run_count"] == 2
    assert data["target_met_count"] == 1
    assert data["target_precision"] == 0.95
    assert data["target_recall"] == 0.95
    assert data["target_met"] is False
    assert data["totals"] == {
        "predicted_count": 5,
        "true_positives": 4,
        "false_positives": 1,
        "false_negatives": 1,
    }
    assert data["micro_precision"] == 0.8
    assert data["micro_recall"] == 0.8
    assert data["micro_f1"] == 0.8
    assert data["macro_precision"] == 0.833334
    assert data["macro_recall"] == 0.833334
    assert data["macro_f1"] == 0.833334
    assert data["run_ids"] == ["run-1", "run-2"]


def test_evaluation_summary_counts_empty_recommendations_as_misses(tmp_path, monkeypatch):
    evaluation_dir = tmp_path / "evaluations"
    evaluation_dir.mkdir()
    monkeypatch.setattr("hobby_basketball.app.EVALUATION_DIR", evaluation_dir)
    (evaluation_dir / "empty.json").write_text(
        json.dumps(
            {
                "run_id": "empty",
                "truth_times": [12.0, 20.0],
                "report": {
                    "target_precision": 0.95,
                    "target_recall": 0.95,
                    "target_met": False,
                    "recommended": None,
                    "points": [],
                },
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(app)

    response = client.get("/api/evaluation-summary")

    assert response.status_code == 200
    data = response.json()
    assert data["run_count"] == 1
    assert data["target_met_count"] == 0
    assert data["target_met"] is False
    assert data["totals"] == {
        "predicted_count": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 2,
    }
    assert data["micro_precision"] == 0.0
    assert data["micro_recall"] == 0.0
    assert data["micro_f1"] == 0.0
    assert data["macro_precision"] == 0.0
    assert data["macro_recall"] == 0.0
    assert data["macro_f1"] == 0.0
    assert data["run_ids"] == ["empty"]
