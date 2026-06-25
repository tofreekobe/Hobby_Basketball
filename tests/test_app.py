from fastapi.testclient import TestClient

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
