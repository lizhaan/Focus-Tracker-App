-- Версія 1: користувачі, профілі калібрування, налаштування.
-- Таблиці сесій і метрик додаються пізніше окремою міграцією.

CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE calibration_profiles (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    monitor_id TEXT NOT NULL,
    monitor_name TEXT,
    camera_id TEXT NOT NULL,
    frame_w INTEGER NOT NULL,
    frame_h INTEGER NOT NULL,
    data_json TEXT NOT NULL,          -- усі поля CalibrationProfile
    calib_error REAL NOT NULL,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (user_id, monitor_id, camera_id)   -- нове калібрування замінює старе
);

CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
