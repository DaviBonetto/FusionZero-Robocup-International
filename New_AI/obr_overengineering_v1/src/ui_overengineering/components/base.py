from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget


class CardFrame(QFrame):
    def __init__(self, title: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CardFrame")
        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(12, 10, 12, 12)
        self._root_layout.setSpacing(8)

        if title:
            self._title_label = QLabel(title, self)
            self._title_label.setObjectName("CardTitle")
            self._title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._root_layout.addWidget(self._title_label)
        else:
            self._title_label = None

        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        self._root_layout.addLayout(self.content_layout, 1)


class MetricPill(QFrame):
    def __init__(self, name: str, value: str = "--", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MetricPill")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(6)

        self.name_label = QLabel(name, self)
        self.name_label.setObjectName("MetricKey")
        self.value_label = QLabel(value, self)
        self.value_label.setObjectName("MetricValue")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self.name_label)
        layout.addWidget(self.value_label, 1)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class StatusBadge(QFrame):
    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatusBadge")
        self.setProperty("active", False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        self.name_label = QLabel(name, self)
        self.name_label.setObjectName("StatusName")
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.value_label = QLabel("OFF", self)
        self.value_label.setObjectName("StatusValue")
        self.value_label.setProperty("active", False)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.name_label)
        layout.addWidget(self.value_label)

    def set_active(self, active: bool) -> None:
        self.value_label.setText("ON" if active else "OFF")
        self.setProperty("active", active)
        self.value_label.setProperty("active", active)
        self._refresh_style()

    def _refresh_style(self) -> None:
        self.style().unpolish(self)
        self.style().polish(self)
        self.style().unpolish(self.value_label)
        self.style().polish(self.value_label)

