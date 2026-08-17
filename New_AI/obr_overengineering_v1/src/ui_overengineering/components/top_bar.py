from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QWidget

from .base import MetricPill


class TopMetricsBar(QWidget):
    METRIC_ORDER: tuple[tuple[str, str], ...] = (
        ("CPU", "CPU"),
        ("MEM", "MEM"),
        ("CAP", "CAP FPS"),
        ("PROC", "PROC FPS"),
        ("NET", "NET"),
        ("QUEUE", "QUEUE"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addStretch(1)

        self._metrics: dict[str, MetricPill] = {}
        for key, label in self.METRIC_ORDER:
            pill = MetricPill(label, "--", self)
            pill.setMinimumWidth(104)
            layout.addWidget(pill)
            self._metrics[key] = pill
        layout.addStretch(1)

    def set_metric(self, key: str, value: str) -> None:
        item = self._metrics.get(key.upper())
        if item is not None:
            item.set_value(value)

    def update_metrics(self, metrics: dict[str, str]) -> None:
        for key, _ in self.METRIC_ORDER:
            self.set_metric(key, metrics.get(key, "--"))

    def set_scale(self, scale: float) -> None:
        factor = max(0.8, min(1.2, float(scale)))
        for pill in self._metrics.values():
            pill.setMinimumWidth(round(104 * factor))
