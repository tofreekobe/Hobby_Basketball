from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import UploadFile


WORKSPACE_ROOT = Path(os.getenv("HOBBY_BASKETBALL_WORKSPACE", Path.cwd() / "workspace"))
UPLOAD_DIR = WORKSPACE_ROOT / "uploads"
EXPORT_DIR = WORKSPACE_ROOT / "exports"
EVALUATION_DIR = WORKSPACE_ROOT / "evaluations"
REVIEW_DIR = WORKSPACE_ROOT / "reviews"


def save_upload(file: UploadFile) -> tuple[str, Path]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    video_id = uuid.uuid4().hex
    safe_name = f"{video_id}{suffix.lower()}"
    target = UPLOAD_DIR / safe_name
    with target.open("wb") as handle:
        while chunk := file.file.read(1024 * 1024):
            handle.write(chunk)
    return video_id, target


def export_path(video_id: str, output_format: str) -> Path:
    fmt = normalize_output_format(output_format)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return EXPORT_DIR / f"{video_id}_highlights.{fmt}"


def normalize_output_format(output_format: str) -> str:
    fmt = output_format.lower().strip().lstrip(".")
    if fmt not in {"mp4", "mov"}:
        raise ValueError("output_format must be mp4 or mov")
    return fmt
