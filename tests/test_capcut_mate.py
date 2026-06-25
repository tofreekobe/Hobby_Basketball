from hobby_basketball.capcut_mate import CapCutMateClient, clips_to_video_infos
from hobby_basketball.models import ClipInterval


def test_clips_to_video_infos_places_precut_clips_contiguously():
    clips = [
        ClipInterval(video_path="clips/clip_000.mp4", start=0.0, end=6.5, event_ids=["a"]),
        ClipInterval(video_path="clips/clip_001.mp4", start=0.0, end=4.0, event_ids=["b"]),
    ]

    infos = clips_to_video_infos(clips)

    assert infos[0]["start"] == 0
    assert infos[0]["end"] == 6_500_000
    assert infos[1]["start"] == 6_500_000
    assert infos[1]["end"] == 10_500_000


def test_client_calls_create_and_add_videos_with_json_string():
    calls = []

    class FakeHttp:
        def post(self, url, json, timeout):
            calls.append((url, json))

            class Response:
                def raise_for_status(self):
                    pass

                def json(self):
                    if url.endswith("/create_draft"):
                        return {"draft_url": "http://local/get_draft?draft_id=1"}
                    return {"draft_url": "http://local/get_draft?draft_id=1", "segment_ids": ["s1"]}

            return Response()

    client = CapCutMateClient("http://local/openapi/capcut-mate/v1", http=FakeHttp())

    draft_url = client.create_draft(width=1920, height=1080)
    client.add_videos(draft_url, [{"video_url": "clip.mp4", "start": 0, "end": 1_000_000}])

    assert calls[0][0] == "http://local/openapi/capcut-mate/v1/create_draft"
    assert calls[1][1]["video_infos"].startswith("[")
