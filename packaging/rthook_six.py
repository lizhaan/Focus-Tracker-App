"""Runtime-хук збірки: сумісність хука імпорту PySide6 з six.moves.

У зібраній програмі PyInstaller імпортує PySide6 ще до нашого коду, тож
трюк «mediapipe раніше за PySide6» не діє. Хук імпорту shiboken при імпорті
six.moves (mediapipe → matplotlib → dateutil → six) звертається до
`loader._path`, якого в six._SixMetaPathImporter немає, і програма падає.
Порожній `_path` прибирає падіння за будь-якого порядку імпортів.
"""

import six

if not hasattr(six._SixMetaPathImporter, "_path"):
    six._SixMetaPathImporter._path = []
