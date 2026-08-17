from __future__ import annotations

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from .base import CardFrame


class RobotSilhouette(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(170, 140)

    def paintEvent(self, _event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect().adjusted(8, 8, -8, -8)
        painter.fillRect(rect, QColor("#1A1D22"))

        body = rect.adjusted(30, 30, -30, -38)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ECEEF2"))
        painter.drawRoundedRect(body, 8, 8)

        painter.setBrush(QColor("#2A2F36"))
        wheel_w = int(body.width() * 0.2)
        wheel_h = int(body.height() * 0.32)
        painter.drawEllipse(body.left() - wheel_w // 2, body.bottom() - wheel_h // 2, wheel_w, wheel_h)
        painter.drawEllipse(body.right() - wheel_w // 2, body.bottom() - wheel_h // 2, wheel_w, wheel_h)

        painter.setBrush(QColor("#D9DDE5"))
        top = body.adjusted(18, -18, -18, -body.height() + 20)
        painter.drawRoundedRect(top, 6, 6)

        pen = QPen(QColor("#B7BECA"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(QPointF(body.left() + 10, body.center().y()), QPointF(body.right() - 10, body.center().y()))
        painter.drawLine(QPointF(body.center().x(), body.top() + 10), QPointF(body.center().x(), body.bottom() - 10))


class RobotPlaceholderPanel(CardFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Robot", parent)

        self.silhouette = RobotSilhouette(self)
        self.content_layout.addWidget(self.silhouette, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(6)
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._dots: list[QLabel] = []
        for _ in range(3):
            dot = QLabel("●", self)
            dot.setObjectName("CardTitle")
            footer.addWidget(dot)
            self._dots.append(dot)
        self.content_layout.addLayout(footer)
        self.set_step(0)

    def set_step(self, step: int) -> None:
        for index, dot in enumerate(self._dots):
            dot.setText("●")
            dot.setStyleSheet("color: #2E343D;" if index != step % len(self._dots) else "color: #E8EAF0;")

