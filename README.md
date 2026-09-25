# Focus-Tracker-App

Навчальний проєкт з дисципліни «Парадигма об'єктно-орієнтованого програмування» (2025-2026)

Десктоп-застосунок для трекінгу уваги користувача через веб-камеру під час
роботи/навчання

## Структура
 
| Каталог | Зміст |
|---|---|
| `core/features.py` | EAR, Yaw/Pitch/Roll, Gaze_raw, EMA-згладжування — **готово** |
| `core/capture.py` | камера + MediaPipe Face Mesh — **готово** |
| `tests/` | unit-тести (`unittest`) для `core/features.py` |


## Стек технологій
 
- **Python 3.11+**
- **MediaPipe Face Mesh** (refine_landmarks=True) — детекція 468 landmark-ів обличчя + iris
- **OpenCV** — захоплення відео, допоміжні обчислення (в т.ч. `solvePnP`)
- **PySide6 (Qt)** — GUI, багатопотоковість через сигнали/слоти
- **pyqtgraph** — графіки концентрації в реальному часі
- **SQLite** — локальне сховище калібрувань і історії сесій
- Пакування: **PyInstaller** / **Nuitka**

## Зовнішні бібліотеки
 
| Бібліотека | Для чого |
|---|---|
| [mediapipe](https://github.com/google/mediapipe) `0.10.35` | Face Mesh — 478 landmark-ів обличчя + райдужка |
| [opencv-python](https://github.com/opencv/opencv-python) | захоплення відео, `solvePnP`/`Rodrigues` для head pose |
| [numpy](https://numpy.org/) | векторні обчислення (відстані, медіана, масиви) |

## Запуск
 
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Використання ШІ
 
Під час розробки використовується Claude (Anthropic) як асистент.
Історія запитів зберігається; весь код переглядається й розуміється
автором.
