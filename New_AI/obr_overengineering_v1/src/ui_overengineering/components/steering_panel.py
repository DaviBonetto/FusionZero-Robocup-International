from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class SteeringPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SteeringPanel")
        self.setFixedWidth(46)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.mode_label = QLabel("n", self)
        self.mode_label.setObjectName("SteeringChip")
        self.mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.mode_label)

        self.current_label = QLabel("0.0 deg", self)
        self.current_label.setObjectName("SteeringText")
        self.current_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.current_label)

        self.arrow_label = QLabel("^", self)
        self.arrow_label.setObjectName("SteeringArrow")
        self.arrow_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.arrow_label)

        self.target_label = QLabel("0 deg", self)
        self.target_label.setObjectName("SteeringText")
        self.target_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.target_label)

        layout.addStretch(1)

    def update_command(
        self,
        *,
        mode: str,
        current_angle_deg: float,
        target_angle_deg: float,
        arrow_symbol: str,
    ) -> None:
        self.mode_label.setText(mode if mode else "n")
        self.current_label.setText(f"{float(current_angle_deg):.1f} deg")
        self.arrow_label.setText(arrow_symbol if arrow_symbol else "^")
        self.target_label.setText(f"{float(target_angle_deg):.0f} deg")
