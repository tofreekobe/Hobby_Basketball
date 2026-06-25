from hobby_basketball.trajectory import BallSample, RimCalibration, detect_made_shots


def test_detects_descending_ball_crossing_rim_plane():
    rim = RimCalibration(center_x=100, center_y=100, half_width=20, half_height=10)
    samples = [
        BallSample(t=1.00, x=100, y=80, confidence=0.8),
        BallSample(t=1.25, x=101, y=118, confidence=0.8),
    ]

    makes = detect_made_shots(samples, rim)

    assert len(makes) == 1
    assert makes[0].t_make == 1.125


def test_rejects_ball_crossing_outside_horizontal_gate():
    rim = RimCalibration(center_x=100, center_y=100, half_width=20, half_height=10)
    samples = [
        BallSample(t=1.00, x=180, y=80, confidence=0.8),
        BallSample(t=1.25, x=180, y=118, confidence=0.8),
    ]

    assert detect_made_shots(samples, rim) == []


def test_rejects_bounce_back_above_rim():
    rim = RimCalibration(center_x=100, center_y=100, half_width=20, half_height=10)
    samples = [
        BallSample(t=1.00, x=100, y=80, confidence=0.8),
        BallSample(t=1.25, x=101, y=118, confidence=0.8),
        BallSample(t=1.40, x=100, y=79, confidence=0.8),
    ]

    assert detect_made_shots(samples, rim) == []
