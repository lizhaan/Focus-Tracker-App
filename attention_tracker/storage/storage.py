"""Storage — єдина точка доступу до SQLite.

Зʼєднання створюється і використовується тільки в головному потоці
(sqlite3 сам перевіряє це через check_same_thread). Зміни схеми — через
PRAGMA user_version і послідовні міграції.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from attention_tracker.calibration.profile import CalibrationProfile
from attention_tracker.focus.metrics import FocusEventRow, PomodoroEventRow, SessionMetrics
from attention_tracker.focus.pomodoro import PomodoroEvent
from attention_tracker.models import FocusState, PomodoroPhase, Reason

DEFAULT_USER = "Default"
_MIGRATIONS: tuple[Path, ...] = (  # індекс + 1 = версія схеми після міграції
    Path(__file__).with_name("schema.sql"),
    Path(__file__).with_name("schema_v2.sql"),
)
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


@dataclass(frozen=True)
class SessionInfo:
    """Рядок історії сесій; metrics — None, поки сесію не завершено."""

    id: int
    started_at: str
    ended_at: str | None
    settings: dict
    monitor_id: str
    profile_id: int | None
    metrics: SessionMetrics | None


_METRIC_COLUMNS = (
    "t_focused", "t_not_focused", "t_lost", "focus_ratio", "n_lost_episodes",
    "mean_lost_duration", "longest_focus", "n_pomodoros",
)


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

    def profile_id(self, user_id: int, monitor_id: str, camera_id: str) -> int | None:
        row = self._db.execute(
            "SELECT id FROM calibration_profiles WHERE user_id = ? AND monitor_id = ? AND camera_id = ?",
            (user_id, monitor_id, camera_id),
        ).fetchone()
        return row[0] if row else None

    # --- сесії ---------------------------------------------------------------

    def create_session(
        self, user_id: int, started_at: str, settings: dict, monitor_id: str, profile_id: int | None
    ) -> int:
        with self._db:
            cur = self._db.execute(
                "INSERT INTO sessions (user_id, started_at, settings_json, monitor_id, profile_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, started_at, json.dumps(settings), monitor_id, profile_id),
            )
        return cur.lastrowid

    def finish_session(self, session_id: int, ended_at: str, metrics: SessionMetrics) -> None:
        values = [getattr(metrics, c) for c in _METRIC_COLUMNS]
        with self._db:
            self._db.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (ended_at, session_id))
            self._db.execute(
                f"INSERT OR REPLACE INTO session_metrics (session_id, {', '.join(_METRIC_COLUMNS)}, "
                f"lost_by_reason_json) VALUES (?, {', '.join('?' * len(_METRIC_COLUMNS))}, ?)",
                (session_id, *values, json.dumps(metrics.lost_by_reason)),
            )

    def add_focus_event(self, session_id: int, event: FocusEventRow) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO focus_events (session_id, t_offset, from_state, to_state, reason) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, event.t, event.old.value, event.new.value, event.reason.value),
            )

    def add_pomodoro_event(self, session_id: int, event: PomodoroEventRow) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO pomodoro_events (session_id, t_offset, event, phase) VALUES (?, ?, ?, ?)",
                (session_id, event.t, event.event.value, event.phase.value),
            )

    def focus_events(self, session_id: int) -> list[FocusEventRow]:
        rows = self._db.execute(
            "SELECT t_offset, from_state, to_state, reason FROM focus_events "
            "WHERE session_id = ? ORDER BY t_offset, id",
            (session_id,),
        ).fetchall()
        return [FocusEventRow(t, FocusState(old), FocusState(new), Reason(r)) for t, old, new, r in rows]

    def pomodoro_events(self, session_id: int) -> list[PomodoroEventRow]:
        rows = self._db.execute(
            "SELECT t_offset, event, phase FROM pomodoro_events WHERE session_id = ? ORDER BY t_offset, id",
            (session_id,),
        ).fetchall()
        return [PomodoroEventRow(t, PomodoroEvent(e), PomodoroPhase(p)) for t, e, p in rows]

    def last_event_offset(self, session_id: int) -> float:
        """Час останньої записаної події (для сесій, що обірвались аварійно)."""
        row = self._db.execute(
            "SELECT MAX(t) FROM (SELECT t_offset AS t FROM focus_events WHERE session_id = ? "
            "UNION ALL SELECT t_offset FROM pomodoro_events WHERE session_id = ?)",
            (session_id, session_id),
        ).fetchone()
        return row[0] or 0.0

    def unfinished_sessions(self) -> list[tuple[int, str]]:
        """(id, started_at) сесій без ended_at."""
        return self._db.execute("SELECT id, started_at FROM sessions WHERE ended_at IS NULL").fetchall()

    def list_sessions(self, user_id: int) -> list[SessionInfo]:
        """Сесії користувача, найновіші першими, разом з метриками."""
        rows = self._db.execute(
            f"SELECT s.id, s.started_at, s.ended_at, s.settings_json, s.monitor_id, s.profile_id, "
            f"m.session_id, {', '.join('m.' + c for c in _METRIC_COLUMNS)}, m.lost_by_reason_json "
            "FROM sessions s LEFT JOIN session_metrics m ON m.session_id = s.id "
            "WHERE s.user_id = ? ORDER BY s.started_at DESC, s.id DESC",
            (user_id,),
        ).fetchall()
        return [self._session_info(row) for row in rows]

    def delete_session(self, session_id: int) -> None:
        with self._db:
            self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    @staticmethod
    def _session_info(row: tuple) -> SessionInfo:
        sid, started, ended, settings_json, monitor_id, profile_id, metrics_id, *rest = row
        metrics = None
        if metrics_id is not None:
            values, reasons_json = rest[:-1], rest[-1]
            metrics = SessionMetrics(
                **dict(zip(_METRIC_COLUMNS, values)),
                lost_by_reason=json.loads(reasons_json or "{}"),
            )
        return SessionInfo(sid, started, ended, json.loads(settings_json), monitor_id, profile_id, metrics)

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
