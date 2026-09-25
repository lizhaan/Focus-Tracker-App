# Focus-Tracker-App

Навчальний проєкт з дисципліни «Парадигма об'єктно-орієнтованого програмування» (2025-2026)

Десктоп-застосунок для трекінгу уваги користувача через веб-камеру під час
роботи/навчання

## Стек технологій
 
- **Python 3.11+**
- **MediaPipe Face Mesh** (refine_landmarks=True) — детекція 468 landmark-ів обличчя + iris
- **OpenCV** — захоплення відео, допоміжні обчислення (в т.ч. `solvePnP`)
- **PySide6 (Qt)** — GUI, багатопотоковість через сигнали/слоти
- **pyqtgraph** — графіки концентрації в реальному часі
- **SQLite** — локальне сховище калібрувань і історії сесій
- Пакування: **PyInstaller** / **Nuitka**

## Зовнішні бібліотеки
 
| Бібліотека | Версія | Для чого |
|---|---|---|
| **mediapipe** | `0.10.35` | Face Mesh — детекція 478 3D-landmark-ів обличчя (`refine_landmarks=True` дає додатково точки райдужки, потрібні для `compute_gaze_raw`). Використовується в `core/capture.py`. |
| **opencv-python** | latest | Захоплення відео з камери (`cv2.VideoCapture`), конвертація кольорових просторів, розв'язання PnP-задачі (`cv2.solvePnP`, `cv2.Rodrigues`) для head pose в `core/features.py`. |
| **numpy** | latest | Векторні обчислення — відстані між landmark-ами, медіана у згладжуванні, побудова масивів координат. |
 
Тести (`tests/test_features.py`) додатково використовують `unittest`
(стандартна бібліотека) та ті самі `opencv-python`/`numpy` — окремих
тестових залежностей у проєкті немає.

## Запуск
 
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
