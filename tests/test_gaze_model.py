import numpy as np
import pytest

from attention_tracker.vision.gaze_model import GazeRegressor, phi_x, phi_y


def synthetic(n: int = 60, seed: int = 0):
    """Відома залежність sx від Gaze_h і ΔYaw (з компенсацією повороту голови)."""
    rng = np.random.default_rng(seed)
    gaze_h = rng.uniform(0.35, 0.65, n)
    d_yaw = rng.uniform(-20, 20, n)
    sx = 0.5 + 3.0 * (gaze_h - 0.5) + 0.03 * d_yaw
    phi = np.array([phi_x(g, y) for g, y in zip(gaze_h, d_yaw)])
    return phi, sx


def test_fit_recovers_known_relation():
    phi, sx = synthetic()
    reg = GazeRegressor.fit(phi, sx, lam=0.05, sigma_min=1e-6)
    phi_test, sx_test = synthetic(n=30, seed=1)
    error = np.abs(reg.predict(phi_test) - sx_test).max()
    assert error < 0.02


def test_predict_single_point_returns_float():
    phi, sx = synthetic()
    reg = GazeRegressor.fit(phi, sx, lam=0.05, sigma_min=1e-6)
    value = reg.predict(phi[0])
    assert isinstance(value, float)
    assert value == pytest.approx(sx[0], abs=0.02)


def test_constant_feature_gets_unit_sigma():
    phi = np.array([phi_y(0.1 * i, 0.0) for i in range(10)])  # ΔPitch = 0 → σ = 0
    reg = GazeRegressor.fit(phi, np.linspace(0, 1, 10), lam=0.05, sigma_min=1e-6)
    assert reg.sigma[1] == 1.0
    assert np.isfinite(reg.predict(phi)).all()


def test_serialization_roundtrip_gives_identical_prediction():
    phi, sx = synthetic()
    reg = GazeRegressor.fit(phi, sx, lam=0.05, sigma_min=1e-6)
    restored = GazeRegressor.from_dict(reg.to_dict())
    assert np.array_equal(restored.predict(phi), reg.predict(phi))


def test_fit_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        GazeRegressor.fit(np.zeros((5, 3)), np.zeros(5), lam=0.05, sigma_min=1e-6)
