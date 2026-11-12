-- Версія 2: сесії, переходи станів, події Pomodoro, кеш метрик.
-- Видалення профілю не видаляє історію: profile_id стає NULL.

CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    started_at TEXT NOT NULL,         -- ISO 8601, локальний час
    ended_at TEXT,
    settings_json TEXT NOT NULL,      -- n, y, тривалості на момент старту
    monitor_id TEXT NOT NULL,         -- робочий монітор сесії
    profile_id INTEGER REFERENCES calibration_profiles(id) ON DELETE SET NULL
);

CREATE TABLE focus_events (           -- тільки переходи станів, НЕ кожен кадр
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t_offset REAL NOT NULL,           -- секунди від початку сесії (monotonic)
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    reason TEXT NOT NULL
);

CREATE TABLE pomodoro_events (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t_offset REAL NOT NULL,
    event TEXT NOT NULL,              -- phase_start, phase_end, auto_pause, auto_resume, manual_pause, manual_resume, stop
    phase TEXT NOT NULL
);

CREATE TABLE session_metrics (        -- кеш, рахується при завершенні сесії
    session_id INTEGER PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    t_focused REAL, t_not_focused REAL, t_lost REAL,
    focus_ratio REAL, n_lost_episodes INTEGER,
    mean_lost_duration REAL, longest_focus REAL,
    n_pomodoros INTEGER,
    lost_by_reason_json TEXT
);

CREATE INDEX idx_focus_events_session ON focus_events(session_id);
CREATE INDEX idx_pomodoro_events_session ON pomodoro_events(session_id);
CREATE INDEX idx_sessions_user ON sessions(user_id, started_at);
