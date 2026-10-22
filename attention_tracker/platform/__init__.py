# mediapipe має імпортуватися раніше за PySide6: інакше хук імпорту shiboken
# падає на six.moves (mediapipe → matplotlib → dateutil → six).
import mediapipe  # noqa: F401
