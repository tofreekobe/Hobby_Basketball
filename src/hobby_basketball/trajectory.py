from __future__ import annotations

from pydantic import BaseModel, Field

from hobby_basketball.models import MadeShotEvent


class BallSample(BaseModel):
    t: float = Field(ge=0)
    x: float
    y: float
    confidence: float = Field(ge=0, le=1)


class RimCalibration(BaseModel):
    center_x: float
    center_y: float
    half_width: float = Field(gt=0)
    half_height: float = Field(gt=0)


def detect_made_shots(
    samples: list[BallSample],
    rim: RimCalibration,
    *,
    x_gate_margin: float = 0.45,
    up_margin_frac: float = 1.0,
    down_margin_frac: float = 1.5,
    max_cross_gap_sec: float = 0.6,
    min_descent_px_s: float = 60.0,
    bounce_up_window_sec: float = 0.5,
    min_spacing_sec: float = 3.0,
    video_path: str = "",
) -> list[MadeShotEvent]:
    ordered = sorted(samples, key=lambda item: item.t)
    x_lo = rim.center_x - rim.half_width * (1 + x_gate_margin)
    x_hi = rim.center_x + rim.half_width * (1 + x_gate_margin)
    y_above = rim.center_y - rim.half_height * up_margin_frac
    y_below = rim.center_y + rim.half_height * down_margin_frac

    gated = [sample for sample in ordered if x_lo <= sample.x <= x_hi]
    makes: list[MadeShotEvent] = []

    for above in gated:
        if above.y >= y_above:
            continue
        cross: BallSample | None = None
        for below in gated:
            if below.t <= above.t:
                continue
            gap = below.t - above.t
            if gap > max_cross_gap_sec:
                break
            if below.y <= y_below:
                continue
            descent = (below.y - above.y) / max(gap, 1e-3)
            if descent < min_descent_px_s:
                continue
            cross = below
            break
        if cross is None:
            continue
        if _bounced_back(gated, cross, y_above, bounce_up_window_sec):
            continue

        score = min(1.0, (above.confidence + cross.confidence) / 2.0 + 0.1)
        makes.append(
            MadeShotEvent(
                id=f"make-{len(makes) + 1:04d}",
                video_path=video_path,
                t_make=round((above.t + cross.t) / 2.0, 6),
                t_above=above.t,
                t_below=cross.t,
                confidence=round(score, 6),
                notes="rim-plane crossing",
            )
        )

    return _dedupe_makes(makes, min_spacing_sec)


def _bounced_back(
    samples: list[BallSample],
    cross: BallSample,
    y_above: float,
    window_sec: float,
) -> bool:
    for sample in samples:
        if sample.t <= cross.t:
            continue
        if sample.t - cross.t > window_sec:
            break
        if sample.y < y_above:
            return True
    return False


def _dedupe_makes(makes: list[MadeShotEvent], min_spacing_sec: float) -> list[MadeShotEvent]:
    deduped: list[MadeShotEvent] = []
    for make in sorted(makes, key=lambda item: item.t_make):
        if deduped and make.t_make - deduped[-1].t_make < min_spacing_sec:
            if make.confidence > deduped[-1].confidence:
                deduped[-1] = make
            continue
        deduped.append(make)
    return deduped
