from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QWidget

from .base import CardFrame


def _format_timer(seconds: float) -> str:
    value = max(0.0, float(seconds))
    minutes = int(value // 60)
    seconds_whole = int(value % 60)
    centiseconds = int((value - int(value)) * 100)
    return f"{minutes:02d}:{seconds_whole:02d}:{centiseconds:02d}"


class TimerPanel(CardFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Timers", parent)
        self.main_timer = QLabel("00:00:00", self)
        self.main_timer.setObjectName("TimerMain")
        self.main_timer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.secondary_timer = QLabel("00:00:00", self)
        self.secondary_timer.setObjectName("TimerSecondary")
        self.secondary_timer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.content_layout.addWidget(self.main_timer)
        self.content_layout.addWidget(self.secondary_timer)
        self.content_layout.addStretch(1)

    def set_elapsed(self, main_seconds: float, secondary_seconds: float) -> None:
        self.main_timer.setText(_format_timer(main_seconds))
        self.secondary_timer.setText(_format_timer(secondary_seconds))

