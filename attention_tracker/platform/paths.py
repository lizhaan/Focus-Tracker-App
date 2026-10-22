"""Шляхи до даних застосунку."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QStandardPaths

APP_NAME = "AttentionTracker"
DB_FILENAME = "attention_tracker.db"


def default_db_path() -> Path:
    """Файл БД у теці даних користувача (QStandardPaths.AppDataLocation).

    Тека залежить від імені застосунку — main() задає його через APP_NAME.
    """
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    return Path(base) / DB_FILENAME
