"""Storage — єдина точка доступу до SQLite.

Зʼєднання створюється і використовується тільки в головному потоці
(sqlite3 сам перевіряє це через check_same_thread). Зміни схеми — через
PRAGMA user_version і послідовні міграції.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from attention_tracker.calibration.profile import CalibrationProfile

DEFAULT_USER = "Default"
_SCHEMA_V1 = Path(__file__).with_name("schema.sql")
_MIGRATIONS: tuple[Path, ...] = (_SCHEMA_V1,)  # індекс + 1 = версія схеми після міграції
SCHEMA_VERSION = len(_MIGRATIONS)


@dataclass(frozen=True)
class ProfileInfo:
    """Рядок списку профілів без розбору JSON."""

    id: int
    monitor_id: str
    monitor_name: str | None
    camera_id: str
    frame_w: int
    frame_h: int
    calib_error: float
    created_at: str


class Storage:
    def __init__(self, path: str | Path) -> None:
        """`path` — файл БД або ":memory:" для тестів."""
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Storage:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        return self._db.execute("PRAGMA user_version").fetchone()[0]

    def _migrate(self) -> None:
        version = self.schema_version
        for target, script in enumerate(_MIGRATIONS[version:], start=version + 1):
            with self._db:
                self._db.executescript(script.read_text(encoding="utf-8"))
                self._db.execute(f"PRAGMA user_version = {target}")

    # --- користувачі -----------------------------------------------------------

    def ensure_user(self, name: str, created_at: str) -> int:
        """id користувача; створює, якщо його ще немає."""
        with self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO users (name, created_at) VALUES (?, ?)", (name, created_at)
            )
        return self._db.execute("SELECT id FROM users WHERE name = ?", (name,)).fetchone()[0]

    # --- профілі калібрування ------------------------------------------------

    def save_profile(self, profile: CalibrationProfile, monitor_name: str | None) -> int:
        """Зберігає профіль; профіль з тим самим (user, monitor, camera) замінюється."""
        with self._db:
            self._db.execute(
                """
                INSERT INTO calibration_profiles (
                    user_id, monitor_id, monitor_name, camera_id, frame_w, frame_h,
                    data_json, calib_error, schema_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (user_id, monitor_id, camera_id) DO UPDATE SET
                    monitor_name = excluded.monitor_name,
                    frame_w = excluded.frame_w,
                    frame_h = excluded.frame_h,
                    data_json = excluded.data_json,
                    calib_error = excluded.calib_error,
                    schema_version = excluded.schema_version,
                    created_at = excluded.created_at
                """,
                (
                    profile.user_id, profile.monitor_id, monitor_name, profile.camera_id,
                    profile.frame_w, profile.frame_h, profile.to_json(), profile.calib_error,
                    profile.schema_version, profile.created_at,
                ),
            )
        row = self._db.execute(
            "SELECT id FROM calibration_profiles WHERE user_id = ? AND monitor_id = ? AND camera_id = ?",
            (profile.user_id, profile.monitor_id, profile.camera_id),
        ).fetchone()
        return row[0]

    def load_profile(self, user_id: int, monitor_id: str, camera_id: str) -> CalibrationProfile | None:
        row = self._db.execute(
            "SELECT data_json FROM calibration_profiles "
            "WHERE user_id = ? AND monitor_id = ? AND camera_id = ?",
            (user_id, monitor_id, camera_id),
        ).fetchone()
        return CalibrationProfile.from_json(row[0]) if row else None

    def list_profiles(self, user_id: int) -> list[ProfileInfo]:
        rows = self._db.execute(
            "SELECT id, monitor_id, monitor_name, camera_id, frame_w, frame_h, calib_error, created_at "
            "FROM calibration_profiles WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [ProfileInfo(*row) for row in rows]

    def delete_profile(self, profile_id: int) -> None:
        with self._db:
            self._db.execute("DELETE FROM calibration_profiles WHERE id = ?", (profile_id,))

    # --- налаштування ----------------------------------------------------------

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        row = self._db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
