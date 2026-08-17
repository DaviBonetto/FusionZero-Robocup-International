from __future__ import annotations

from typing import Mapping

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from .base import CardFrame, StatusBadge


STATE_SUMMARY: dict[str, str] = {
    "SEARCHING_LINE": "Searching line markers",
    "FOLLOWING_LINE": "Tracking black line",
    "VALIDATING_GAP": "Validating line gap",
    "CROSSING_GAP": "Crossing gap section",
    "VICTIM_FOUND": "Victim routine active",
    "RESCUE_ZONE_DETECTED": "Rescue zone routine",
}


STATUS_KEYS = (
    "LINE",
    "SILVER",
    "GREEN",
    "RED",
    "GREEN CORNER",
    "RED CORNER",
    "SILVER BALL",
    "BLACK BALL",
)


class StatusPanel(CardFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("State", parent)

        self.state_label = QLabel("SEARCHING_LINE", self)
        self.state_label.setObjectName("StateHeadline")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.summary_label = QLabel("Waiting for state update", self)
        self.summary_label.setObjectName("StateSummary")
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.content_layout.addWidget(self.state_label)
        self.content_layout.addWidget(self.summary_label)

        badge_grid = QGridLayout()
        badge_grid.setContentsMargins(0, 0, 0, 0)
        badge_grid.setHorizontalSpacing(6)
        badge_grid.setVerticalSpacing(6)
        self._badges: dict[str, StatusBadge] = {}

        for idx, key in enumerate(STATUS_KEYS):
            badge = StatusBadge(key, self)
            badge_grid.addWidget(badge, idx // 3, idx % 3)
            self._badges[key] = badge
        self.content_layout.addLayout(badge_grid)
        self.content_layout.addStretch(1)

    def update_state(self, state: str) -> None:
        cleaned = state.strip().upper() if state else "SEARCHING_LINE"
        self.state_label.setText(cleaned)
        self.summary_label.setText(STATE_SUMMARY.get(cleaned, "State update received"))

    def update_statuses(self, statuses: Mapping[str, bool]) -> None:
        for key, badge in self._badges.items():
            badge.set_active(bool(statuses.get(key, False)))
