from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PyQt6.QtCore import QObject, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

if __package__ in (None, ""):
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from core.event_bus import EventBus, EventTopic, FrameEvent, HealthEvent, LogEvent, StateSnapshotEvent, Subscription, UICommandEvent, VisionDetectionEvent  # type: ignore
    from ui_overengineering.components.base import CardFrame, StatusBadge  # type: ignore
    from ui_overengineering.components.theme import build_dashboard_stylesheet  # type: ignore
    from ui_overengineering.components.video_view import VideoView  # type: ignore
else:
    from ..core.event_bus import EventBus, EventTopic, FrameEvent, HealthEvent, LogEvent, StateSnapshotEvent, Subscription, UICommandEvent, VisionDetectionEvent
    from .components.base import CardFrame, StatusBadge
    from .components.theme import build_dashboard_stylesheet
    from .components.video_view import VideoView


def _frame_from_event(event: FrameEvent) -> np.ndarray | None:
    encoding = str(event.encoding or "bgr8").strip().lower()
    if encoding in {"jpeg", "jpg"}:
        if cv2 is None or not event.data:
            return None
        encoded = np.frombuffer(event.data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return None
        return np.ascontiguousarray(frame)

    if event.width <= 0 or event.height <= 0:
        return None
    raw = np.frombuffer(event.data, dtype=np.uint8)
    expected = event.width * event.height * 3
    if raw.size < expected:
        return None
    frame = raw[:expected].reshape((event.height, event.width, 3))
    if encoding.startswith("rgb"):
        frame = frame[:, :, ::-1]
    return np.ascontiguousarray(frame)


def _normalize_token(value: Any, fallback: str = "--") -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return fallback
    return text.replace(" ", "_").upper()


def _format_float(value: Any, digits: int = 3, fallback: str = "--") -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return fallback


def _format_int(value: Any, fallback: str = "--") -> str:
    try:
        return str(int(value))
    except Exception:
        return fallback


def _bool_text(value: bool) -> str:
    return "YES" if bool(value) else "NO"


def _safe_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


class CommUiBridge(QObject):
    processed_frame_event = pyqtSignal(object)
    detection_event = pyqtSignal(object)
    health_event = pyqtSignal(object)
    log_event = pyqtSignal(object)
    state_event = pyqtSignal(object)


class CommDashboardWindow(QMainWindow):
    def __init__(self, event_bus: EventBus, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._event_bus = event_bus
        self._bridge = CommUiBridge()
        self._subscriptions: list[Subscription] = []
        self._last_health: HealthEvent | None = None
        self._last_serial: dict[str, Any] = {}
        self._last_network: dict[str, Any] = {}
        self._last_detection: VisionDetectionEvent | None = None
        self._last_state = "WAITING"
        self._relay_override_state = ""

        self.connection_values: dict[str, QLabel] = {}
        self.vision_values: dict[str, QLabel] = {}
        self.telemetry_values: dict[str, QLabel] = {}
        self.scenario_badges: dict[str, StatusBadge] = {}
        self.command_buttons: dict[str, QPushButton] = {}

        self.setWindowTitle("FusionZero Pi <-> Arduino Comm Test")
        self.resize(1360, 840)

        self._build_ui()
        self._connect_bridge()
        self._subscribe_to_bus()
        self._refresh_connection_panel()
        self._refresh_vision_panel()
        self._refresh_telemetry_panel()
        self._refresh_badges()

    def _build_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("DashboardRoot")
        root.setStyleSheet(build_dashboard_stylesheet())
        self.setCentralWidget(root)

        outer = QHBoxLayout(root)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(14)

        splitter = QSplitter(Qt.Orientation.Horizontal, root)
        outer.addWidget(splitter, 1)

        left_column = QWidget(splitter)
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        self.video_view = VideoView("Processed Video", left_column)
        self.video_view.set_corner("Pi camera -> vision -> relay -> dashboard")
        self.video_view.set_overlay("WAITING")
        left_layout.addWidget(self.video_view, 3)

        self.log_card = CardFrame("Logs / Events", left_column)
        self.log_list = QListWidget(self.log_card)
        self.log_list.setObjectName("TransitionList")
        self.log_card.content_layout.addWidget(self.log_list, 1)
        left_layout.addWidget(self.log_card, 2)

        right_column = QWidget(splitter)
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        self.connection_card = CardFrame("Connections / Commands", right_column)
        self._build_connection_card(self.connection_card)
        right_layout.addWidget(self.connection_card, 2)

        self.vision_card = CardFrame("Vision", right_column)
        self._build_vision_card(self.vision_card)
        right_layout.addWidget(self.vision_card, 2)

        self.telemetry_card = CardFrame("Arduino Telemetry", right_column)
        self._build_telemetry_card(self.telemetry_card)
        right_layout.addWidget(self.telemetry_card, 2)

        self.badges_card = CardFrame("Scenario Badges", right_column)
        self._build_badges_card(self.badges_card)
        right_layout.addWidget(self.badges_card, 1)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

    def _build_connection_card(self, card: CardFrame) -> None:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.connection_values["pi_relay"] = self._add_value_row(grid, 0, "Pi relay")
        self.connection_values["arduino_serial"] = self._add_value_row(grid, 1, "Arduino serial")
        self.connection_values["heartbeat"] = self._add_value_row(grid, 2, "Heartbeat")
        card.content_layout.addLayout(grid)

        commands = QGridLayout()
        commands.setContentsMargins(0, 6, 0, 0)
        commands.setHorizontalSpacing(8)
        commands.setVerticalSpacing(8)

        self._add_command_button(commands, 0, 0, "forward_test", "Forward test", "robot.forward_test", {"duration_ms": 5000})
        self._add_command_button(commands, 0, 1, "reverse_test", "Reverse test", "robot.reverse_test", {"duration_ms": 5000})
        self._add_command_button(commands, 1, 0, "stop", "STOP", "robot.stop")
        self._add_command_button(commands, 1, 1, "estop", "ESTOP", "robot.force_stop")
        self._add_command_button(commands, 2, 0, "clear_estop", "Clear ESTOP", "robot.clear_estop")
        self._add_command_button(commands, 2, 1, "obstacle_test", "Obstacle test", "robot.obstacle_test")
        self._add_command_button(commands, 3, 0, "clear_obstacle", "Clear obstacle", "robot.clear_obstacle")
        card.content_layout.addLayout(commands)

    def _build_vision_card(self, card: CardFrame) -> None:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.vision_values["line"] = self._add_value_row(grid, 0, "line")
        self.vision_values["green"] = self._add_value_row(grid, 1, "green")
        self.vision_values["green_instruction"] = self._add_value_row(grid, 2, "green_instruction")
        self.vision_values["line_offset_norm"] = self._add_value_row(grid, 3, "line_offset_norm")
        self.vision_values["line_angle_deg"] = self._add_value_row(grid, 4, "line_angle_deg")
        card.content_layout.addLayout(grid)

    def _build_telemetry_card(self, card: CardFrame) -> None:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.telemetry_values["mode"] = self._add_value_row(grid, 0, "mode")
        self.telemetry_values["assist_kind"] = self._add_value_row(grid, 1, "assist_kind")
        self.telemetry_values["line_error"] = self._add_value_row(grid, 2, "line_error")
        self.telemetry_values["pid_output"] = self._add_value_row(grid, 3, "pid_output")
        self.telemetry_values["green_instruction"] = self._add_value_row(grid, 4, "green_instruction")
        self.telemetry_values["obstacle_state"] = self._add_value_row(grid, 5, "obstacle_state")
        self.telemetry_values["failsafe"] = self._add_value_row(grid, 6, "failsafe")
        card.content_layout.addLayout(grid)

    def _build_badges_card(self, card: CardFrame) -> None:
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.scenario_badges["line_detected"] = self._add_badge(layout, 0, 0, "linha detectada")
        self.scenario_badges["line_assist"] = self._add_badge(layout, 0, 1, "assistencia LINE enviada")
        self.scenario_badges["green_half_turn"] = self._add_badge(layout, 1, 0, "duplo verde -> VERDE_MEIA_VOLTA")
        self.scenario_badges["arduino_green"] = self._add_badge(layout, 1, 1, "Arduino em GREEN")
        self.scenario_badges["failsafe"] = self._add_badge(layout, 2, 0, "failsafe ativo")
        card.content_layout.addLayout(layout)

    def _connect_bridge(self) -> None:
        self._bridge.processed_frame_event.connect(self._on_processed_frame_event)
        self._bridge.detection_event.connect(self._on_detection_event)
        self._bridge.health_event.connect(self._on_health_event)
        self._bridge.log_event.connect(self._on_log_event)
        self._bridge.state_event.connect(self._on_state_event)

    def _subscribe_to_bus(self) -> None:
        self._subscriptions = [
            self._event_bus.subscribe(EventTopic.VISION_PROCESSED_FRAME, lambda event: self._bridge.processed_frame_event.emit(event)),
            self._event_bus.subscribe(EventTopic.VISION_DETECTIONS, lambda event: self._bridge.detection_event.emit(event)),
            self._event_bus.subscribe(EventTopic.SYSTEM_HEALTH, lambda event: self._bridge.health_event.emit(event)),
            self._event_bus.subscribe(EventTopic.SYSTEM_LOG, lambda event: self._bridge.log_event.emit(event)),
            self._event_bus.subscribe(EventTopic.FSM_STATE, lambda event: self._bridge.state_event.emit(event)),
        ]

    def _add_value_row(self, layout: QGridLayout, row: int, key: str) -> QLabel:
        key_label = QLabel(key, self)
        key_label.setObjectName("HealthKey")
        value_label = QLabel("--", self)
        value_label.setObjectName("HealthValue")
        value_label.setWordWrap(True)
        value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(key_label, row, 0)
        layout.addWidget(value_label, row, 1)
        return value_label

    def _add_badge(self, layout: QGridLayout, row: int, column: int, title: str) -> StatusBadge:
        badge = StatusBadge(title, self)
        badge.set_active(False)
        layout.addWidget(badge, row, column)
        return badge

    def _add_command_button(
        self,
        layout: QGridLayout,
        row: int,
        column: int,
        key: str,
        label: str,
        command: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        button = QPushButton(label, self)
        button.clicked.connect(lambda _checked=False, cmd=command, payload=dict(params or {}): self._publish_ui_command(cmd, payload))
        layout.addWidget(button, row, column)
        self.command_buttons[key] = button

    def _publish_ui_command(self, command: str, params: dict[str, Any] | None = None) -> None:
        try:
            self._event_bus.publish(
                EventTopic.UI_COMMAND,
                UICommandEvent(timestamp=time.time(), command=command, params=dict(params or {})),
            )
        except Exception:
            return

    @pyqtSlot(object)
    def _on_processed_frame_event(self, event: object) -> None:
        if not isinstance(event, FrameEvent):
            return
        frame = _frame_from_event(event)
        self.video_view.set_frame(frame)

    @pyqtSlot(object)
    def _on_detection_event(self, event: object) -> None:
        if not isinstance(event, VisionDetectionEvent):
            return
        self._last_detection = event
        self._last_state = str(event.state or self._last_state or "WAITING").strip().upper() or "WAITING"
        metadata = _safe_mapping(event.metadata)
        serial = _safe_mapping(metadata.get("serial"))
        if serial:
            self._last_serial = serial
        self.video_view.set_overlay(self._last_state)
        self.vision_values["line"].setText(_bool_text(event.line))
        self.vision_values["green"].setText(_bool_text(event.green))
        self.vision_values["green_instruction"].setText(_normalize_token(metadata.get("green_instruction"), "NO_GREEN"))
        self.vision_values["line_offset_norm"].setText(_format_float(metadata.get("line_offset_norm"), digits=3))
        self.vision_values["line_angle_deg"].setText(_format_float(metadata.get("line_angle_deg"), digits=1))
        self._refresh_connection_panel()
        self._refresh_telemetry_panel()
        self._refresh_badges()

    @pyqtSlot(object)
    def _on_health_event(self, event: object) -> None:
        if not isinstance(event, HealthEvent):
            return
        self._last_health = event
        metadata = _safe_mapping(event.metadata)
        self._last_network = _safe_mapping(metadata.get("network"))
        serial = _safe_mapping(metadata.get("serial"))
        if serial:
            self._last_serial = serial
        self._refresh_connection_panel()
        self._refresh_telemetry_panel()
        self._refresh_badges()

    @pyqtSlot(object)
    def _on_log_event(self, event: object) -> None:
        if not isinstance(event, LogEvent):
            return
        message = str(event.message or "").strip()
        source = str(event.source or "").strip()
        level = str(event.level or "INFO").strip().upper()
        prefix = f"[{level}]"
        if source:
            prefix += f" {source}"
        self.log_list.insertItem(0, f"{prefix} {message}")
        while self.log_list.count() > 150:
            self.log_list.takeItem(self.log_list.count() - 1)

        lowered = message.lower()
        if "connected to raspberry" in lowered or "session ready" in lowered:
            self._relay_override_state = "connected"
        elif "disconnected" in lowered:
            self._relay_override_state = "disconnected"
        self._refresh_connection_panel()

    @pyqtSlot(object)
    def _on_state_event(self, event: object) -> None:
        if not isinstance(event, StateSnapshotEvent):
            return
        self._last_state = str(event.state or self._last_state or "WAITING").strip().upper() or "WAITING"
        self.video_view.set_overlay(self._last_state)

    def _refresh_connection_panel(self) -> None:
        relay_state = _normalize_token(self._last_network.get("state"), "WAITING")
        if self._relay_override_state == "disconnected":
            relay_state = "DISCONNECTED"
        elif self._relay_override_state == "connected" and relay_state in {"WAITING", "--"}:
            relay_state = "CONNECTED"
        relay_peer = str(self._last_network.get("peer", "")).strip()
        relay_latency = self._last_network.get("latency_ms", self._last_network.get("latency"))
        relay_parts = [relay_state]
        if relay_peer:
            relay_parts.append(relay_peer)
        if relay_latency not in (None, ""):
            relay_parts.append(f"{_format_float(relay_latency, digits=1)} ms")
        self._set_status_text(self.connection_values["pi_relay"], " | ".join(relay_parts), relay_state)

        serial_state = _normalize_token(self._last_serial.get("state"), "WAITING")
        serial_port = str(self._last_serial.get("port", "")).strip()
        assist_kind = _normalize_token(self._last_serial.get("assist_kind"), "NONE")
        serial_parts = [serial_state]
        if serial_port:
            serial_parts.append(serial_port)
        if assist_kind not in {"--", "NONE"}:
            serial_parts.append(f"assist {assist_kind}")
        self._set_status_text(self.connection_values["arduino_serial"], " | ".join(serial_parts), serial_state)

        heartbeat_ok = bool(self._last_serial.get("heartbeat_ok", False))
        heartbeat_age = self._last_serial.get("heartbeat_age_ms")
        telemetry_age = self._last_serial.get("telemetry_age_ms")
        heartbeat_parts = ["OK" if heartbeat_ok else "WAITING"]
        if heartbeat_age is not None:
            heartbeat_parts.append(f"{_format_int(heartbeat_age)} ms")
        if telemetry_age is not None:
            heartbeat_parts.append(f"telem {_format_int(telemetry_age)} ms")
        self._set_status_text(self.connection_values["heartbeat"], " | ".join(heartbeat_parts), "OK" if heartbeat_ok else "WAITING")

    def _refresh_vision_panel(self) -> None:
        self.vision_values["line"].setText("NO")
        self.vision_values["green"].setText("NO")
        self.vision_values["green_instruction"].setText("NO_GREEN")
        self.vision_values["line_offset_norm"].setText("--")
        self.vision_values["line_angle_deg"].setText("--")

    def _refresh_telemetry_panel(self) -> None:
        metadata = _safe_mapping(getattr(self._last_detection, "metadata", {}))
        serial = self._last_serial

        mode = _normalize_token(metadata.get("control_mode", serial.get("control_mode")), "--")
        assist_kind = _normalize_token(metadata.get("assist_kind", serial.get("assist_kind")), "--")
        line_error = metadata.get("line_error", serial.get("line_error"))
        pid_output = metadata.get("pid_output", serial.get("pid_output"))
        green_instruction = _normalize_token(metadata.get("green_instruction", serial.get("green_instruction")), "NO_GREEN")
        obstacle_state = _normalize_token(metadata.get("obstacle_state", serial.get("obstacle_state")), "CLEAR")
        failsafe = bool(metadata.get("failsafe", serial.get("failsafe", False)))

        self.telemetry_values["mode"].setText(mode)
        self.telemetry_values["assist_kind"].setText(assist_kind)
        self.telemetry_values["line_error"].setText(_format_float(line_error, digits=3))
        self.telemetry_values["pid_output"].setText(_format_float(pid_output, digits=3))
        self.telemetry_values["green_instruction"].setText(green_instruction)
        self.telemetry_values["obstacle_state"].setText(obstacle_state)
        self.telemetry_values["failsafe"].setText("ACTIVE" if failsafe else "CLEAR")
        self._set_status_text(self.telemetry_values["failsafe"], self.telemetry_values["failsafe"].text(), "ERROR" if failsafe else "OK")

    def _refresh_badges(self) -> None:
        event = self._last_detection
        metadata = _safe_mapping(getattr(event, "metadata", {}))
        serial = self._last_serial
        instruction = _normalize_token(metadata.get("green_instruction", serial.get("green_instruction")), "NO_GREEN")
        mode = _normalize_token(metadata.get("control_mode", serial.get("control_mode")), "--")
        assist_kind = _normalize_token(metadata.get("assist_kind", serial.get("assist_kind")), "NONE")
        failsafe = bool(metadata.get("failsafe", serial.get("failsafe", False)))

        self.scenario_badges["line_detected"].set_active(bool(getattr(event, "line", False)))
        self.scenario_badges["line_assist"].set_active(assist_kind == "LINE")
        self.scenario_badges["green_half_turn"].set_active(instruction == "VERDE_MEIA_VOLTA")
        self.scenario_badges["arduino_green"].set_active(mode == "GREEN")
        self.scenario_badges["failsafe"].set_active(failsafe)

    def _set_status_text(self, label: QLabel, text: str, state: str) -> None:
        label.setText(text)
        normalized = _normalize_token(state, "WAITING")
        if normalized in {"CONNECTED", "OK", "GREEN", "CLEAR"}:
            level = "ok"
        elif normalized in {"DISCONNECTED", "ERROR", "ESTOP", "FAILSAFE"}:
            level = "error"
        else:
            level = "warn"
        label.setProperty("level", level)
        label.style().unpolish(label)
        label.style().polish(label)

    def closeEvent(self, event: Any) -> None:  # noqa: ANN401
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions.clear()
        super().closeEvent(event)


def run_comm_dashboard(event_bus: EventBus) -> int:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)
    window = CommDashboardWindow(event_bus=event_bus)
    window.show()
    return app.exec() if owns_app else 0
