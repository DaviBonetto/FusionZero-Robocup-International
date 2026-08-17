from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QWidget

from .base import CardFrame


class TransitionLogPanel(CardFrame):
    def __init__(self, parent: QWidget | None = None, max_lines: int = 150) -> None:
        super().__init__("Transitions", parent)
        self._max_lines = max(30, int(max_lines))
        self.list_widget = QListWidget(self)
        self.list_widget.setObjectName("TransitionList")
        self.list_widget.setAlternatingRowColors(False)
        self.content_layout.addWidget(self.list_widget, 1)

    def append_line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        item = QListWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.list_widget.addItem(item)
        while self.list_widget.count() > self._max_lines:
            self.list_widget.takeItem(0)
        self.list_widget.scrollToBottom()

