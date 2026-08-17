from __future__ import annotations

from typing import Mapping

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from .base import CardFrame


ROW_ORDER: tuple[tuple[str, str], ...] = (
    ("camera", "Camera"),
    ("serial", "Serial"),
    ("link", "Link"),
    ("power", "Power"),
    ("profile", "Profile"),
    ("recording", "Recording"),
)


class OpsHealthPanel(CardFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Ops Health", parent)
        self._scroll_area = QScrollArea(self)
        self._scroll_area.setObjectName("CardScrollArea")
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll_content = QWidget(self._scroll_area)
        self._scroll_content.setObjectName("CardScrollContent")
        self._scroll_content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._scroll_area.viewport().setStyleSheet("background: transparent;")
        self._scroll_content.setStyleSheet("background: transparent;")
        content_layout = QGridLayout(self._scroll_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setHorizontalSpacing(8)
        content_layout.setVerticalSpacing(8)
        self._scroll_area.setWidget(self._scroll_content)
        self.content_layout.addWidget(self._scroll_area, 1)

        self._value_labels: dict[str, QLabel] = {}
        self._detail_labels: dict[str, QLabel] = {}
        for index, (key, title) in enumerate(ROW_ORDER):
            row_widget = QWidget(self._scroll_content)
            row_widget.setObjectName("HealthTile")
            row_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            row_layout = QVBoxLayout(row_widget)
            row_layout.setContentsMargins(7, 4, 7, 4)
            row_layout.setSpacing(2)

            header_layout = QHBoxLayout()
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(8)

            name_label = QLabel(title, row_widget)
            name_label.setObjectName("HealthKey")
            name_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            name_label.setMinimumWidth(0)
            name_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

            value_label = QLabel("--", row_widget)
            value_label.setObjectName("HealthValue")
            value_label.setProperty("level", "neutral")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
            value_label.setMinimumWidth(88)
            value_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)

            detail_label = QLabel("", row_widget)
            detail_label.setObjectName("HealthDetail")
            detail_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            detail_label.setWordWrap(False)
            detail_label.setTextFormat(Qt.TextFormat.PlainText)
            detail_label.setMinimumWidth(0)
            detail_label.setMinimumHeight(16)
            detail_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

            header_layout.addWidget(name_label, 1)
            header_layout.addWidget(value_label, 0, Qt.AlignmentFlag.AlignRight)
            row_layout.addLayout(header_layout)
            row_layout.addWidget(detail_label)
            content_layout.addWidget(row_widget, index // 2, index % 2)
            self._value_labels[key] = value_label
            self._detail_labels[key] = detail_label
        content_layout.setColumnStretch(0, 1)
        content_layout.setColumnStretch(1, 1)

    def update_rows(self, rows: Mapping[str, Mapping[str, str]]) -> None:
        for key, _ in ROW_ORDER:
            payload = rows.get(key, {})
            value = str(payload.get("value", "--")).strip() or "--"
            detail = str(payload.get("detail", "")).strip()
            level = str(payload.get("level", "neutral")).strip().lower() or "neutral"
            self._set_row(key, value=value, detail=detail, level=level)

    def _set_row(self, key: str, *, value: str, detail: str, level: str) -> None:
        value_label = self._value_labels.get(key)
        detail_label = self._detail_labels.get(key)
        if value_label is None or detail_label is None:
            return
        value_label.setText(value)
        value_label.setProperty("level", level)
        detail_label.setText(detail)
        detail_label.setToolTip(detail)
        value_label.updateGeometry()
        detail_label.updateGeometry()
        self._refresh_label(value_label)

    @staticmethod
    def _refresh_label(label: QLabel) -> None:
        label.style().unpolish(label)
        label.style().polish(label)
