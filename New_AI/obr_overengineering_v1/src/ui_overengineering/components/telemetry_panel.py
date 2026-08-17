from __future__ import annotations

from typing import Mapping

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QLabel, QWidget

from .base import CardFrame


TELEMETRY_KEYS: tuple[tuple[str, str], ...] = (
    ("front", "Front"),
    ("left", "Left"),
    ("right", "Right"),
    ("back", "Back"),
    ("yaw", "Yaw"),
    ("roll", "Roll"),
    ("pitch", "Pitch"),
    ("gripper", "Gripper"),
)


class TelemetryPanel(CardFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Telemetry", parent)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        self.content_layout.addLayout(grid)

        self._value_labels: dict[str, QLabel] = {}
        for row, (key, title) in enumerate(TELEMETRY_KEYS):
            label_name = QLabel(f"{title}:", self)
            label_name.setObjectName("CardTitle")
            label_name.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            label_value = QLabel("--", self)
            label_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            label_value.setObjectName("MetricValue")

            grid.addWidget(label_name, row, 0)
            grid.addWidget(label_value, row, 1)
            self._value_labels[key] = label_value

        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)

    def update_values(self, values: Mapping[str, float | int | str | None]) -> None:
        for key, _ in TELEMETRY_KEYS:
            if key not in values:
                continue
            self._set_value(key, values[key])

    def _set_value(self, key: str, value: float | int | str | None) -> None:
        label = self._value_labels.get(key)
        if label is None:
            return
        if value is None:
            label.setText("--")
            return
        if isinstance(value, str):
            label.setText(value)
            return

        suffix = " mm"
        if key in {"yaw", "roll", "pitch"}:
            suffix = " deg"
        if key == "gripper":
            suffix = " mm"
        label.setText(f"{float(value):.2f}{suffix}" if key in {"yaw", "roll", "pitch"} else f"{int(value)}{suffix}")

