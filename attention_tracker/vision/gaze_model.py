"""Регресія точки погляду на екрані (ridge).

Окремо для sx (ознаки φx) і sy (ознаки φy):
    φx = [Gaze_h, ΔYaw,   Gaze_h², Gaze_h·ΔYaw]
    φy = [Gaze_v, ΔPitch, Gaze_v², Gaze_v·ΔPitch]
    φ̃ = (φ − μ) / σ                (σ_k < sigma_min → 1)
    навчання: c₀ = mean(s),  c = (Φ̃ᵀΦ̃ + λI)⁻¹ Φ̃ᵀ (s − c₀)
    прогноз:  s = c₀ + φ̃ · c
"""

from __future__ import annotations

import numpy as np

N_FEATURES = 4


def phi_x(gaze_h: float, d_yaw: float) -> np.ndarray:
    return np.array([gaze_h, d_yaw, gaze_h * gaze_h, gaze_h * d_yaw], dtype=np.float64)


def phi_y(gaze_v: float, d_pitch: float) -> np.ndarray:
    return np.array([gaze_v, d_pitch, gaze_v * gaze_v, gaze_v * d_pitch], dtype=np.float64)


class GazeRegressor:
    """Ridge-регресія однієї координати екрана за 4 ознаками."""

    def __init__(self, c0: float, c: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> None:
        self.c0 = float(c0)
        self.c = np.asarray(c, dtype=np.float64)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)

    @classmethod
    def fit(
        cls, phi: np.ndarray, targets: np.ndarray, lam: float, sigma_min: float
    ) -> GazeRegressor:
        phi = np.asarray(phi, dtype=np.float64)
        s = np.asarray(targets, dtype=np.float64)
        if phi.ndim != 2 or phi.shape[1] != N_FEATURES or phi.shape[0] != s.shape[0]:
            raise ValueError(f"очікується Φ (N×{N_FEATURES}) і s (N), отримано {phi.shape}, {s.shape}")
        mu = phi.mean(axis=0)
        sigma = phi.std(axis=0)
        sigma = np.where(sigma < sigma_min, 1.0, sigma)
        phi_t = (phi - mu) / sigma
        c0 = float(s.mean())
        c = np.linalg.solve(phi_t.T @ phi_t + lam * np.eye(N_FEATURES), phi_t.T @ (s - c0))
        return cls(c0, c, mu, sigma)

    def predict(self, phi: np.ndarray) -> float | np.ndarray:
        """Одна точка (4,) → float; кілька (N, 4) → масив (N,)."""
        phi = np.asarray(phi, dtype=np.float64)
        value = self.c0 + ((phi - self.mu) / self.sigma) @ self.c
        return float(value) if np.ndim(value) == 0 else value

    def to_dict(self) -> dict:
        return {
            "c0": self.c0,
            "c": self.c.tolist(),
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> GazeRegressor:
        return cls(data["c0"], data["c"], data["mu"], data["sigma"])
