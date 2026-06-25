from pathlib import Path

from hobby_basketball.ffmpeg_export import build_clip_command, build_concat_command
from hobby_basketball.models import ClipInterval


def test_build_clip_command_uses_source_start_and_duration():
    clip = ClipInterval(video_path="game.mp4", start=5.0, end=11.5, event_ids=["make-1"])

    cmd = build_clip_command(clip, Path("out/clip_000.mp4"))

    assert cmd[:5] == ["ffmpeg", "-y", "-ss", "5.000", "-i"]
    assert "-t" in cmd
    assert "6.500" in cmd


def test_build_concat_command_targets_final_mp4():
    cmd = build_concat_command(Path("clips.txt"), Path("final.mp4"))

    assert cmd == [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "clips.txt",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        "final.mp4",
    ]
