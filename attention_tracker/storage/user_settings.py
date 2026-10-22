"""Налаштування користувача: значення, межі, збереження в таблиці settings.

Збережені значення накладаються на `AppConfig` через `apply_settings()`.
Некоректне або відсутнє значення в БД замінюється стандартним.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from attention_tracker.config import AppConfig
from attention_tracker.storage.storage import Storage

_SECONDS_PER_MINUTE = 60.0

# Ключі таблиці settings
KEY_N = "focus.n"
KEY_Y = "focus.y"
KEY_WORK = "pomodoro.work_minutes"
KEY_SHORT_BREAK = "pomodoro.short_break_minutes"
KEY_LONG_BREAK = "pomodoro.long_break_minutes"
KEY_BEFORE_LONG = "pomodoro.before_long_break"
KEY_SOUND = "pomodoro.sound"
KEY_CAMERA_INDEX = "camera.index"
KEY_CAMERA_RESOLUTION = "camera.resolution"  # "WxH"


@dataclass(frozen=True)
class UserSettings:
    n: float
    y: float
    work_minutes: int
    short_break_minutes: int
    long_break_minutes: int
    pomodoros_before_long_break: int
    sound: bool
    camera_index: int
    camera_resolution: tuple[int, int]

    @classmethod
    def defaults(cls, cfg: AppConfig) -> UserSettings:
        return cls(
            n=cfg.focus.n,
            y=cfg.focus.y,
            work_minutes=round(cfg.pomodoro.work / _SECONDS_PER_MINUTE),
            short_break_minutes=round(cfg.pomodoro.short_break / _SECONDS_PER_MINUTE),
            long_break_minutes=round(cfg.pomodoro.long_break / _SECONDS_PER_MINUTE),
            pomodoros_before_long_break=cfg.pomodoro.pomodoros_before_long_break,
            sound=cfg.pomodoro.sound,
            camera_index=cfg.camera.index,
            camera_resolution=(cfg.camera.width, cfg.camera.height),
        )


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(max(value, bounds[0]), bounds[1])


def validate(settings: UserSettings, cfg: AppConfig) -> UserSettings:
    """Значення в допустимих межах; невідома роздільна здатність — стандартна."""
    lim = cfg.limits
    resolution = settings.camera_resolution
    if resolution not in lim.camera_resolutions:
        resolution = (cfg.camera.width, cfg.camera.height)
    return dataclasses.replace(
        settings,
        n=_clamp(settings.n, (cfg.focus.n_min, cfg.focus.n_max)),
        y=_clamp(settings.y, (cfg.focus.y_min, cfg.focus.y_max)),
        work_minutes=int(_clamp(settings.work_minutes, lim.work_minutes)),
        short_break_minutes=int(_clamp(settings.short_break_minutes, lim.break_minutes)),
        long_break_minutes=int(_clamp(settings.long_break_minutes, lim.break_minutes)),
        pomodoros_before_long_break=int(
            _clamp(settings.pomodoros_before_long_break, lim.pomodoros_before_long_break)
        ),
        camera_index=int(_clamp(settings.camera_index, lim.camera_index)),
        camera_resolution=resolution,
    )


def _parse_resolution(text: str) -> tuple[int, int]:
    w, _, h = text.partition("x")
    return int(w), int(h)


def load_settings(storage: Storage, cfg: AppConfig) -> UserSettings:
    d = UserSettings.defaults(cfg)

    def read(key: str, parse, default):
        text = storage.get_setting(key)
        if text is None:
            return default
        try:
            return parse(text)
        except ValueError:
            return default

    loaded = UserSettings(
        n=read(KEY_N, float, d.n),
        y=read(KEY_Y, float, d.y),
        work_minutes=read(KEY_WORK, int, d.work_minutes),
        short_break_minutes=read(KEY_SHORT_BREAK, int, d.short_break_minutes),
        long_break_minutes=read(KEY_LONG_BREAK, int, d.long_break_minutes),
        pomodoros_before_long_break=read(KEY_BEFORE_LONG, int, d.pomodoros_before_long_break),
        sound=read(KEY_SOUND, lambda t: t == "1", d.sound),
        camera_index=read(KEY_CAMERA_INDEX, int, d.camera_index),
        camera_resolution=read(KEY_CAMERA_RESOLUTION, _parse_resolution, d.camera_resolution),
    )
    return validate(loaded, cfg)


def save_settings(storage: Storage, settings: UserSettings) -> None:
    w, h = settings.camera_resolution
    values = {
        KEY_N: f"{settings.n:g}",
        KEY_Y: f"{settings.y:g}",
        KEY_WORK: str(settings.work_minutes),
        KEY_SHORT_BREAK: str(settings.short_break_minutes),
        KEY_LONG_BREAK: str(settings.long_break_minutes),
        KEY_BEFORE_LONG: str(settings.pomodoros_before_long_break),
        KEY_SOUND: "1" if settings.sound else "0",
        KEY_CAMERA_INDEX: str(settings.camera_index),
        KEY_CAMERA_RESOLUTION: f"{w}x{h}",
    }
    for key, value in values.items():
        storage.set_setting(key, value)


def apply_settings(cfg: AppConfig, settings: UserSettings) -> AppConfig:
    """Конфігурація з урахуванням налаштувань користувача."""
    w, h = settings.camera_resolution
    return dataclasses.replace(
        cfg,
        focus=dataclasses.replace(cfg.focus, n=settings.n, y=settings.y),
        pomodoro=dataclasses.replace(
            cfg.pomodoro,
            work=settings.work_minutes * _SECONDS_PER_MINUTE,
            short_break=settings.short_break_minutes * _SECONDS_PER_MINUTE,
            long_break=settings.long_break_minutes * _SECONDS_PER_MINUTE,
            pomodoros_before_long_break=settings.pomodoros_before_long_break,
            sound=settings.sound,
        ),
        camera=dataclasses.replace(cfg.camera, index=settings.camera_index, width=w, height=h),
    )
