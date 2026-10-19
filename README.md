# Focus-Tracker-App

Навчальний проєкт з дисципліни «Парадигма об'єктно-орієнтованого програмування» (2025-2026)

Десктоп-застосунок для трекінгу уваги користувача через веб-камеру під час
роботи/навчання

## Документація

- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) — ідея, вимоги, архітектура, класи, етапи
- [`docs/MATH.md`](docs/MATH.md) — формули, пороги, калібрування, машина станів (джерело правди)
- [`docs/CLAUDE.md`](docs/CLAUDE.md) — правила розробки

## Структура

Пакет `attention_tracker/`.
Логіка (`vision/`, `focus/`, `calibration/`, `storage/`) — без Qt; Qt тільки в `workers/` і `ui/`.

| Шлях | Зміст |
|---|---|
| `attention_tracker/main.py` | точка входу (зараз відкриває режим налагодження) |
| `attention_tracker/config.py` | `AppConfig` — усі числові параметри |
| `attention_tracker/models.py` | енуми `FocusState`, `Reason`, `PomodoroPhase` і dataclass-и `FrameFeatures`, `DistractionResult`, `StateChange`, `FrameResult` |
| `attention_tracker/assets/face_landmarker.task` | модель MediaPipe FaceLandmarker |
| `attention_tracker/capture/camera.py` | `Camera` — вебкамера, мітки часу `time.monotonic()` |
| `attention_tracker/capture/landmarker.py` | `FaceLandmarkerWrapper` — 478 landmark-ів (Tasks API) |
| `attention_tracker/vision/landmarks.py` | індекси landmark-ів, 3D-модель обличчя |
| `attention_tracker/vision/features.py` | EAR, Yaw/Pitch/Roll, Gaze_h/Gaze_v; `FeatureExtractor` |
| `attention_tracker/vision/filters.py` | median + EMA: `MedianEmaFilter`, `SignalFilter` |
| `attention_tracker/workers/pipeline_worker.py` | `PipelineWorker` (QThread): камера → landmarks → ознаки → фільтр |
| `attention_tracker/ui/debug_view.py` | превʼю камери з landmark-ами та значеннями сигналів |
| `attention_tracker/{calibration,focus,storage,session,platform}/` | заготовки для наступних кроків |
| `tests/` | unit-тести (`pytest`), без камери |

## Стек технологій

- **Python 3.12** (підтримується 3.11)
- **MediaPipe FaceLandmarker** (Tasks API, `running_mode=VIDEO`) — 478 landmark-ів обличчя + райдужки
- **OpenCV** — захоплення відео, `solvePnP` для пози голови
- **NumPy** — обчислення
- **PySide6 (Qt)** — GUI, робочий потік (`QThread`), сигнали/слоти
- **pyqtgraph** — графіки концентрації
- **SQLite** — локальне сховище калібрувань і історії сесій
- **pytest** — тести
- Пакування: **PyInstaller**

## Зовнішні бібліотеки

Точні версії — у `requirements.txt`.

| Бібліотека | Для чого |
|---|---|
| [mediapipe](https://github.com/google/mediapipe) `0.10.35` | FaceLandmarker — landmark-и обличчя й райдужки |
| [opencv-contrib-python](https://github.com/opencv/opencv-python) `5.0.0.93` | захоплення відео, `solvePnP`/`Rodrigues` для пози голови |
| [numpy](https://numpy.org/) `2.5.3` | векторні обчислення |
| [PySide6](https://doc.qt.io/qtforpython-6/) `6.12.0` | інтерфейс і потоки |
| [pyqtgraph](https://www.pyqtgraph.org/) `0.14.0` | графіки |
| [pytest](https://docs.pytest.org/) `9.1.1` | тести |

## Запуск

```bash
python -m venv venv
venv\Scripts\activate            # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python -m attention_tracker.main # запуск
pytest                           # тести
```

Відео й кадри не зберігаються і нікуди не передаються — усе працює локально.

## Використання ШІ
 
Під час розробки використовується Claude Code (Anthropic).
Історія запитів зберігається; весь код переглядається й розуміється
автором.
