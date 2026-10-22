import math
import random

import pytest

from attention_tracker.calibration.fitting import Quality
from attention_tracker.calibration.procedure import (
    CalibrationProcedure,
    HeadPose,
    RetryReason,
    StepKind,
    StepState,
)
from attention_tracker.calibration.profile import CalibrationProfile
from attention_tracker.config import CalibrationConfig
from attention_tracker.models import FrameFeatures

CFG = CalibrationConfig()
W, H = 1280, 720
FPS = 30.0
YAW_B, PITCH_B = 4.0, -6.0  # як користувач сидить відносно камери

# Поза голови для кожного повороту: ΔYaw, ΔPitch (вниз = +, як у моделі обличчя)
POSES = {HeadPose.LEFT: (18.0, 0.0), HeadPose.RIGHT: (-18.0, 0.0),
         HeadPose.UP: (0.0, -14.0), HeadPose.DOWN: (0.0, 14.0)}


class SimUser:
    """Дивиться на ціль; очі компенсують поворот голови."""

    def __init__(self, noise: float = 0.002, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.noise = noise

    def frame(self, ts, target=None, d_yaw=0.0, d_pitch=0.0, ear=0.30, valid=True, iris=130.0):
        if not valid:
            return FrameFeatures.invalid(ts)
        sx, sy = target if target is not None else (0.5, 0.5)
        n = lambda: self.rng.gauss(0.0, self.noise)  # noqa: E731
        return FrameFeatures(
            ts=ts, valid=True, ear=ear + n(),
            yaw=YAW_B + d_yaw + 10 * n(), pitch=PITCH_B + d_pitch + 10 * n(), roll=0.0,
            gaze_h=0.5 + 0.25 * (sx - 0.5) - 0.006 * d_yaw + n(),
            gaze_v=0.15 * (sy - 0.5) - 0.004 * d_pitch + n(),
            iris_distance_px=iris,
        )


def frames_for(step, user: SimUser, t0: float, seconds: float):
    """Кадри, які «користувач» дає на кроці."""
    d_yaw, d_pitch = POSES.get(step.pose, (0.0, 0.0))
    ear = 0.30
    if step.kind is StepKind.BOTTOM:
        d_pitch, ear = 3.0, 0.26
    elif step.kind is StepKind.PHONE:
        d_pitch, ear = 20.0, 0.18
    for i in range(int(seconds * FPS)):
        yield user.frame(round(t0 + i / FPS, 6), step.target, d_yaw, d_pitch, ear)


def run_step(proc, user, t0):
    """Запускає і проходить поточний крок; повертає час після кроку."""
    proc.start_step(t0)
    t = t0
    for f in frames_for(proc.current, user, t0, 3.0):
        t = f.ts
        if proc.feed(f) in (StepState.DONE, StepState.RETRY):
            break
    return t + 1.0


def run_all(proc, user) -> None:
    t = 0.0
    while not proc.finished:
        t = run_step(proc, user, t)
        assert proc.state is StepState.DONE, (proc.current, proc.retry_reason)
        proc.next()


@pytest.fixture
def proc():
    return CalibrationProcedure(CFG, ear_freeze_factor=0.7, frame_w=W, frame_h=H, rng=random.Random(1))


# --- структура ----------------------------------------------------------------


def test_steps_layout(proc):
    kinds = [s.kind for s in proc.steps]
    assert kinds.count(StepKind.GRID) == 8
    assert kinds.count(StepKind.HEAD) == 12
    assert kinds.count(StepKind.CHECK) == 4
    assert kinds[:2] == [StepKind.PRECHECK, StepKind.CENTER]
    assert kinds[-6:-4] == [StepKind.BOTTOM, StepKind.PHONE]
    training = 1 + 8 + 12
    assert sum(k in (StepKind.CENTER, StepKind.GRID, StepKind.HEAD) for k in kinds) == training


def test_check_points_away_from_training_targets(proc):
    training = [s.target for s in proc.steps if s.kind in (StepKind.CENTER, StepKind.GRID, StepKind.HEAD)]
    for step in proc.steps:
        if step.kind is StepKind.CHECK:
            lo, hi = CFG.check_point_range
            assert lo <= step.target[0] <= hi and lo <= step.target[1] <= hi
            assert min(math.dist(step.target, t) for t in training) >= CFG.check_point_min_dist


# --- повний прохід -------------------------------------------------------------


def test_full_calibration_gives_accurate_profile(proc):
    run_all(proc, SimUser())
    assert proc.quality is Quality.OK
    assert proc.calib_error < 0.05

    profile = proc.result(user_id=1, monitor_id="mon", camera_id="index:0", created_at="2026-10-22T10:00:00")
    assert profile.yaw_baseline == pytest.approx(YAW_B, abs=0.5)
    assert profile.pitch_baseline == pytest.approx(PITCH_B, abs=0.5)
    assert profile.s_pitch == 1
    assert profile.ear_enabled is True
    assert profile.ear_threshold == pytest.approx(0.22, abs=0.01)
    assert profile.pd_bottom == pytest.approx(3.0, abs=0.5)
    assert profile.pitch_threshold == pytest.approx(11.5, abs=0.5)
    assert (profile.frame_w, profile.frame_h) == (W, H)


def test_profile_json_roundtrip_and_validity(proc):
    run_all(proc, SimUser())
    profile = proc.result(1, "mon", "index:0", "2026-10-22T10:00:00")
    restored = CalibrationProfile.from_json(profile.to_json())
    assert restored.regressor_x.predict([0.5, 0.0, 0.25, 0.0]) == profile.regressor_x.predict([0.5, 0.0, 0.25, 0.0])
    assert restored.pitch_threshold == profile.pitch_threshold
    assert restored.is_valid_for("mon", "index:0", W, H)
    assert not restored.is_valid_for("other", "index:0", W, H)
    assert not restored.is_valid_for("mon", "index:1", W, H)
    assert not restored.is_valid_for("mon", "index:0", 640, 480)


def test_result_before_finish_raises(proc):
    with pytest.raises(RuntimeError):
        proc.result(1, "mon", "index:0", "2026-10-22T10:00:00")


# --- перевірка перед стартом ----------------------------------------------


def precheck(proc, user, **frame_kwargs):
    proc.start_step(0.0)
    for i in range(int(2.5 * FPS)):
        f = user.frame(round(i / FPS, 6), **frame_kwargs)
        if proc.feed(f) in (StepState.DONE, StepState.RETRY):
            break
    return proc.state, proc.retry_reason


def test_precheck_ok(proc):
    assert precheck(proc, SimUser()) == (StepState.DONE, None)


@pytest.mark.parametrize("iris, reason", [(80.0, RetryReason.TOO_FAR), (420.0, RetryReason.TOO_CLOSE)])
def test_precheck_distance(proc, iris, reason):
    assert precheck(proc, SimUser(), iris=iris) == (StepState.RETRY, reason)


def test_precheck_face_ratio(proc):
    user = SimUser()
    proc.start_step(0.0)
    for i in range(int(2.5 * FPS)):
        proc.feed(user.frame(round(i / FPS, 6), valid=(i % 5 != 0)))  # 80 % кадрів з обличчям
    assert (proc.state, proc.retry_reason) == (StepState.RETRY, RetryReason.FACE_RATIO)


def test_precheck_low_fps(proc):
    user = SimUser()
    proc.start_step(0.0)
    for i in range(30):
        proc.feed(user.frame(i * 0.1))  # 10 FPS
    assert (proc.state, proc.retry_reason) == (StepState.RETRY, RetryReason.LOW_FPS)


# --- збір на кроці ---------------------------------------------------------------


def advance_to(proc, user, kind):
    t = 0.0
    while proc.current.kind is not kind:
        t = run_step(proc, user, t)
        proc.next()
    return t


def test_low_valid_ratio_retries_then_succeeds(proc):
    user = SimUser()
    t = advance_to(proc, user, StepKind.CENTER)
    proc.start_step(t)
    for i in range(int(2.5 * FPS)):
        if proc.feed(user.frame(round(t + i / FPS, 6), valid=(i % 2 == 0))) is StepState.RETRY:
            break
    assert proc.retry_reason is RetryReason.LOW_VALID

    run_step(proc, user, t + 5.0)  # повтор того ж кроку
    assert proc.state is StepState.DONE


def test_settle_frames_are_ignored(proc):
    user = SimUser(noise=0.0)
    t = advance_to(proc, user, StepKind.CENTER)
    proc.start_step(t)
    for i in range(int(2.5 * FPS)):
        ts = round(t + i / FPS, 6)
        d_yaw = 30.0 if ts - t < CFG.t_settle else 0.0  # спершу голова ще не на місці
        if proc.feed(user.frame(ts, d_yaw=d_yaw)) is StepState.DONE:
            break
    assert proc._yaw_baseline == pytest.approx(YAW_B)


def test_progress_during_collection(proc):
    user = SimUser()
    t = advance_to(proc, user, StepKind.CENTER)
    proc.start_step(t)
    assert proc.progress(t + 1.0) == pytest.approx(0.5)
    assert proc.progress(t + 5.0) == pytest.approx(1.0)


# --- повороти голови ----------------------------------------------------------


def test_head_step_waits_for_pose(proc):
    user = SimUser()
    t = advance_to(proc, user, StepKind.HEAD)
    assert proc.current.pose is HeadPose.LEFT
    proc.start_step(t)
    for i in range(30):  # голова прямо — збір не починається
        assert proc.feed(user.frame(round(t + i / FPS, 6))) is StepState.WAITING_POSE
    assert proc.pose_ok(user.frame(t + 1.0, d_yaw=5.0)) is False
    assert proc.pose_ok(user.frame(t + 1.0, d_yaw=18.0)) is True
    assert proc.feed(user.frame(t + 1.1, d_yaw=18.0)) is StepState.COLLECTING


def test_right_pose_requires_opposite_sign(proc):
    user = SimUser()
    t = advance_to(proc, user, StepKind.HEAD)
    while proc.current.pose is HeadPose.LEFT:  # LEFT зафіксував знак +
        t = run_step(proc, user, t)
        proc.next()
    assert proc.current.pose is HeadPose.RIGHT
    assert proc.pose_ok(user.frame(t, d_yaw=18.0)) is False
    assert proc.pose_ok(user.frame(t, d_yaw=-18.0)) is True


def test_pose_ok_is_none_outside_head_steps(proc):
    assert proc.pose_ok(SimUser().frame(0.0)) is None


def test_next_before_done_raises(proc):
    with pytest.raises(RuntimeError):
        proc.next()
