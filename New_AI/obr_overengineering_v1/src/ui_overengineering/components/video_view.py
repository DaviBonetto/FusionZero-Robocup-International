from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .base import CardFrame


def _frame_to_pixmap(frame: np.ndarray | None) -> QPixmap | None:
    if frame is None or frame.size == 0:
        return None
    if frame.ndim == 2:
        rgb = np.repeat(frame[:, :, None], 3, axis=2)
    else:
        rgb = frame[:, :, :3][:, :, ::-1]
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]
    image = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class VideoView(CardFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.video_label = QLabel("No camera connected", self)
        self.video_label.setObjectName("VideoLabel")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(280, 180)
        self.video_label.setScaledContents(False)
        self.content_layout.addWidget(self.video_label, 1)

        footer = QVBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(4)
        self.corner_label = QLabel("", self)
        self.corner_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.corner_label.setObjectName("CardTitle")
        self.overlay_label = QLabel("", self)
        self.overlay_label.setObjectName("OverlayText")
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.setVisible(False)
        footer.addWidget(self.corner_label)
        footer.addWidget(self.overlay_label)
        self.content_layout.addLayout(footer)

    def set_frame(self, frame: np.ndarray | None, *, fallback_text: str = "No signal") -> None:
        pixmap = _frame_to_pixmap(frame)
        if pixmap is None:
            self.video_label.setPixmap(QPixmap())
            self.video_label.setText(fallback_text)
            return

        self.video_label.setText("")
        target = self.video_label.size()
        if target.width() > 0 and target.height() > 0:
            pixmap = pixmap.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.video_label.setPixmap(pixmap)

    def set_overlay(self, text: str) -> None:
        text = text.strip()
        self.overlay_label.setText(text)
        self.overlay_label.setVisible(bool(text))

    def set_corner(self, text: str) -> None:
        self.corner_label.setText(text)

    def resizeEvent(self, event: Any) -> None:  # noqa: ANN401
        super().resizeEvent(event)
        pixmap = self.video_label.pixmap()
        if pixmap is not None and not pixmap.isNull():
            self.video_label.setPixmap(
                pixmap.scaled(
                    self.video_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
