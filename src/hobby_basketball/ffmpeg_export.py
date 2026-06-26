from __future__ import annotations

import subprocess
from pathlib import Path

from hobby_basketball.models import ClipInterval


def build_clip_command(clip: ClipInterval, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        f"{clip.start:.3f}",
        "-i",
        clip.video_path,
        "-t",
        f"{clip.duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_path),
    ]


def build_concat_command(list_file: Path, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def export_reel(clips: list[ClipInterval], output_dir: Path, final_path: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: list[Path] = []
    for index, clip in enumerate(clips):
        clip_path = output_dir / f"clip_{index:03d}.mp4"
        subprocess.run(build_clip_command(clip, clip_path), check=True)
        clip_paths.append(clip_path)

    list_file = output_dir / "clips.txt"
    list_file.write_text(
        "".join(f"file '{path.name}'\n" for path in clip_paths),
        encoding="utf-8",
    )
    subprocess.run(build_concat_command(list_file, final_path), check=True)
    return clip_paths
