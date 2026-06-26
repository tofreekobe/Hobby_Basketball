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

    makes.extend(
        _detect_rim_net_entries(
            ordered,
            rim,
            video_path=video_path,
            start_index=len(makes),
            min_spacing_sec=min_spacing_sec,
            bounce_up_window_sec=bounce_up_window_sec,
        )
    )
    return _dedupe_makes(makes, min_spacing_sec)


def _detect_rim_net_entries(
    samples: list[BallSample],
    rim: RimCalibration,
    *,
    video_path: str,
    start_index: int,
    min_spacing_sec: float,
    bounce_up_window_sec: float,
    x_margin_frac: float = 0.7,
    entry_top_frac: float = 0.45,
    exit_bottom_frac: float = 1.7,
    min_drop_frac: float = 0.55,
    max_event_gap_sec: float = 1.5,
) -> list[MadeShotEvent]:
    x_lo = rim.center_x - rim.half_width * (1 + x_margin_frac)
    x_hi = rim.center_x + rim.half_width * (1 + x_margin_frac)
    y_entry_top = rim.center_y - rim.half_height * entry_top_frac
    y_exit_bottom = rim.center_y + rim.half_height * exit_bottom_frac
    min_drop_px = max(8.0, rim.half_height * min_drop_frac)

    candidates = [
        sample
        for sample in samples
        if x_lo <= sample.x <= x_hi and y_entry_top <= sample.y <= y_exit_bottom
    ]
    makes: list[MadeShotEvent] = []
    for entry in candidates:
        exit_sample: BallSample | None = None
        for later in candidates:
            if later.t <= entry.t:
                continue
            gap = later.t - entry.t
            if gap > max_event_gap_sec:
                break
            if later.y - entry.y < min_drop_px:
                continue
            if later.y < rim.center_y:
                continue
            exit_sample = later
            break
        if exit_sample is None:
            continue
        if _bounced_back(candidates, exit_sample, y_entry_top, bounce_up_window_sec):
            continue

        drop_score = min(0.25, (exit_sample.y - entry.y) / max(rim.half_height * 4.0, 1.0))
        conf_score = (entry.confidence + exit_sample.confidence) / 2.0
        score = min(1.0, conf_score + drop_score)
        makes.append(
            MadeShotEvent(
                id=f"make-{start_index + len(makes) + 1:04d}",
                video_path=video_path,
                t_make=round((entry.t + exit_sample.t) / 2.0, 6),
                t_above=entry.t,
                t_below=exit_sample.t,
                confidence=round(score, 6),
                notes="rim-net entry",
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
