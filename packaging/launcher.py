"""Точка входу для PyInstaller (запуск пакета без `-m`)."""

import sys

from attention_tracker.main import main

if __name__ == "__main__":
    sys.exit(main())
