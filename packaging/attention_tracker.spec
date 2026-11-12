# Збірка Attention Tracker у теку з AttentionTracker.exe:
#     pyinstaller packaging/attention_tracker.spec --noconfirm
# Результат: dist/AttentionTracker/AttentionTracker.exe
# Перевірка збірки: dist/AttentionTracker/AttentionTracker.exe --self-test report.txt
# Налагоджувальна збірка з консоллю (видно помилки запуску): AT_CONSOLE=1 перед командою.

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent  # noqa: F821 — SPECPATH задає PyInstaller
PKG = ROOT / "attention_tracker"
NAME = "AttentionTracker"

datas = [
    (str(PKG / "assets" / "face_landmarker.task"), "attention_tracker/assets"),
    (str(PKG / "assets" / "*.svg"), "attention_tracker/assets"),
    (str(PKG / "assets" / "app_icon.png"), "attention_tracker/assets"),
    (str(PKG / "storage" / "*.sql"), "attention_tracker/storage"),
]
# libmediapipe.dll MediaPipe завантажує вручну (ctypes) — PyInstaller сам її не знайде
binaries = collect_dynamic_libs("mediapipe")

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=collect_submodules("attention_tracker"),
    excludes=["pytest", "IPython", "jupyter", "notebook"],
    runtime_hooks=[str(ROOT / "packaging" / "rthook_six.py")],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    console=os.environ.get("AT_CONSOLE") == "1",  # звичайна збірка — без вікна консолі
    icon=str(ROOT / "packaging" / "app.ico"),
)
coll = COLLECT(exe, a.binaries, a.datas, name=NAME)
