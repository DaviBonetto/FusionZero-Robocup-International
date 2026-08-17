from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QScrollArea,
    QSplitter,
    QToolButton,
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

    from core.event_bus import (  # type: ignore
        EventBus,
        EventTopic,
        FrameEvent,
        HealthEvent,
        LogEvent,
        PathEvent,
        StateSnapshotEvent,
        StateTransitionEvent,
        UICommandEvent,
        VisionDetectionEvent,
    )
    from ui_overengineering.components import (  # type: ignore
        ControlCenterPanel,
        OpsHealthPanel,
        RobotPlaceholderPanel,
        StatusPanel,
        SteeringPanel,
        TelemetryPanel,
        TimerPanel,
        TopMetricsBar,
        TransitionLogPanel,
        TuningPanel,
        VideoView,
        build_dashboard_stylesheet,
    )
    from ops_profiles import load_ops_profile_catalog  # type: ignore
else:
    from ..core.event_bus import (
        EventBus,
        EventTopic,
        FrameEvent,
        HealthEvent,
        LogEvent,
        PathEvent,
        StateSnapshotEvent,
        StateTransitionEvent,
        UICommandEvent,
        VisionDetectionEvent,
    )
    from .components import (
        ControlCenterPanel,
        OpsHealthPanel,
        RobotPlaceholderPanel,
        StatusPanel,
        SteeringPanel,
        TelemetryPanel,
        TimerPanel,
        TopMetricsBar,
        TransitionLogPanel,
        TuningPanel,
        VideoView,
        build_dashboard_stylesheet,
    )
    from ..ops_profiles import load_ops_profile_catalog


STATUS_TEMPLATE: dict[str, bool] = {
    "LINE": False,
    "SILVER": False,
    "GREEN": False,
    "RED": False,
    "GREEN CORNER": False,
    "RED CORNER": False,
    "SILVER BALL": False,
    "BLACK BALL": False,
}


STATE_SEQUENCE: tuple[str, ...] = (
    "SEARCHING_LINE",
    "FOLLOWING_LINE",
    "VALIDATING_GAP",
    "CROSSING_GAP",
    "VICTIM_FOUND",
    "RESCUE_ZONE_DETECTED",
)


def _iso_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="milliseconds")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


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


class UiBridge(QObject):
    raw_frame_event = pyqtSignal(object)
    processed_frame_event = pyqtSignal(object)
    detection_event = pyqtSignal(object)
    state_event = pyqtSignal(object)
    transition_event = pyqtSignal(object)
    health_event = pyqtSignal(object)
    log_event = pyqtSignal(object)
    path_event = pyqtSignal(object)
    camera_frame = pyqtSignal(object)
    camera_status = pyqtSignal(str)


class CameraCaptureWorker(QObject):
    frame_ready = pyqtSignal(object)
    status = pyqtSignal(str)

    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        super().__init__()
        self._index = int(index)
        self._width = int(width)
        self._height = int(height)
        self._fps = max(1, int(fps))
        self._timer: QTimer | None = None
        self._capture: Any = None
        self._running = False
        self._last_fail_status_ts = 0.0

    @pyqtSlot()
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._ensure_timer()
        self._open_camera()
        self._timer.start(max(8, int(1000 / max(1, self._fps))))

    @pyqtSlot()
    def stop(self) -> None:
        self._running = False
        if self._timer is not None:
            self._timer.stop()
        self._release_camera()

    @pyqtSlot(int, int, int, int)
    def set_config(self, index: int, width: int, height: int, fps: int) -> None:
        self._index = int(index)
        self._width = int(width)
        self._height = int(height)
        self._fps = max(1, int(fps))
        if self._timer is not None:
            self._timer.setInterval(max(8, int(1000 / max(1, self._fps))))
        if self._running:
            self._open_camera()

    @pyqtSlot()
    def reconnect(self) -> None:
        if self._running:
            self._open_camera()

    def _ensure_timer(self) -> None:
        if self._timer is not None:
            return
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def _open_camera(self) -> None:
        self._release_camera()
        if cv2 is None:
            self.status.emit("camera unavailable: opencv not installed")
            return

        cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(self._index)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self._index)

        if cap is None or not cap.isOpened():
            self.status.emit(f"camera {self._index} not available")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._capture = cap
        self.status.emit(f"camera {self._index} connected ({self._width}x{self._height}@{self._fps})")

    def _release_camera(self) -> None:
        if self._capture is None:
            return
        try:
            self._capture.release()
        except Exception:
            pass
        self._capture = None

    def _tick(self) -> None:
        if not self._running:
            return
        if self._capture is None:
            return
        ok, frame = self._capture.read()
        if not ok or frame is None:
            now = time.monotonic()
            if now - self._last_fail_status_ts >= 2.0:
                self._last_fail_status_ts = now
                self.status.emit(f"camera {self._index} frame read failed")
            return
        self.frame_ready.emit(np.ascontiguousarray(frame))


class MockSource(QObject):
    raw_frame = pyqtSignal(object)
    processed_frame = pyqtSignal(object)
    detection_event = pyqtSignal(object)
    state_event = pyqtSignal(object)
    transition_event = pyqtSignal(object)
    health_event = pyqtSignal(object)
    log_event = pyqtSignal(object)

    def __init__(self, fps: int = 14) -> None:
        super().__init__()
        self._fps = max(2, int(fps))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._frame_id = 0
        self._state_index = 0
        self._state_change_ts = time.monotonic()
        self._current_state = STATE_SEQUENCE[self._state_index]

    def start(self) -> None:
        if self._timer.isActive():
            return
        self._timer.start(int(1000 / self._fps))

    def stop(self) -> None:
        self._timer.stop()

    def is_running(self) -> bool:
        return self._timer.isActive()

    def _tick(self) -> None:
        self._frame_id += 1
        now_wall = time.time()
        now_mono = time.monotonic()
        phase = self._frame_id / 11.0

        if now_mono - self._state_change_ts >= 5.0:
            old_state = self._current_state
            self._state_index = (self._state_index + 1) % len(STATE_SEQUENCE)
            self._current_state = STATE_SEQUENCE[self._state_index]
            self._state_change_ts = now_mono
            transition = StateTransitionEvent(
                timestamp=now_wall,
                old_state=old_state,
                new_state=self._current_state,
                trigger="ON_TIMEOUT",
                reason="mock progression",
            )
            self.transition_event.emit(transition)
            self.state_event.emit(StateSnapshotEvent(timestamp=now_wall, state=self._current_state))
            self.log_event.emit(
                LogEvent(
                    timestamp=now_wall,
                    level="INFO",
                    message=f"{old_state} --ON_TIMEOUT--> {self._current_state} | mock progression",
                    source="ui.mock",
                    state=self._current_state,
                )
            )

        line_on = self._current_state != "SEARCHING_LINE" or (math.sin(phase) > 0.1)
        silver_on = self._current_state == "VALIDATING_GAP" and math.sin(phase * 1.8) > 0.0
        green_on = self._current_state in {"FOLLOWING_LINE", "VALIDATING_GAP"} and math.cos(phase * 1.3) > 0.45
        red_on = self._current_state == "RESCUE_ZONE_DETECTED" or (
            self._current_state == "FOLLOWING_LINE" and math.sin(phase * 0.7) > 0.95
        )
        silver_ball_on = self._current_state == "VICTIM_FOUND" and math.sin(phase * 1.1) > -0.1
        black_ball_on = self._current_state == "VICTIM_FOUND" and math.cos(phase * 1.1) > 0.65
        latency_ms = 12.0 + 8.0 * abs(math.sin(phase * 0.85))

        telemetry = {
            "front": int(600 + 350 * math.sin(phase * 0.8)),
            "left": int(900 + 500 * math.cos(phase * 0.7)),
            "right": int(350 + 260 * math.sin(phase * 1.15)),
            "back": int(700 + 400 * math.cos(phase * 0.6)),
            "yaw": round((math.degrees(math.sin(phase * 0.35)) * 2.5), 2),
            "roll": round(5.0 * math.cos(phase * 0.4), 2),
            "pitch": round(4.0 * math.sin(phase * 0.45), 2),
            "gripper": int(90 + 20 * math.sin(phase * 0.5)),
        }
        metadata: dict[str, Any] = {
            "telemetry": telemetry,
            "silver_line": {"found": silver_on, "confidence": 0.96 if silver_on else 0.32},
            "dead_victim": {"found": black_ball_on, "confidence": 0.91 if black_ball_on else 0.22},
            "mock": True,
        }

        detection = VisionDetectionEvent(
            timestamp=now_wall,
            state=self._current_state,
            line=line_on,
            balls=1 if silver_ball_on else 0,
            green=green_on,
            red=red_on,
            victims=1 if black_ball_on else 0,
            latency_ms=latency_ms,
            metadata=metadata,
        )

        raw = self._build_frame(phase, self._current_state, detection, processed=False)
        processed = self._build_frame(phase, self._current_state, detection, processed=True)
        self.raw_frame.emit(raw)
        self.processed_frame.emit(processed)
        self.detection_event.emit(detection)
        self.health_event.emit(
            HealthEvent(
                timestamp=now_wall,
                cpu_percent=45.0 + 22.0 * abs(math.sin(phase * 0.38)),
                fps_capture=float(self._fps),
                fps_process=1000.0 / max(latency_ms, 1e-3),
                fps_ui=float(self._fps),
                queue_depth=int(1 + abs(math.sin(phase * 0.4)) * 3),
                metadata={
                    "memory_percent": 33.0 + 8.0 * abs(math.cos(phase * 0.32)),
                    "network_latency_ms": 0.0,
                    "network": {"latency_ms": 0.0, "state": "local", "peer": "mock"},
                    "camera": {
                        "state": "online",
                        "index": 0,
                        "width": 640,
                        "height": 360,
                        "fps": self._fps,
                        "backend": "mock",
                    },
                    "serial": {"state": "dry-run", "connected": False, "port": "mock"},
                    "power": {
                        "available": False,
                        "status": "neutral",
                        "summary": "mock power",
                        "raw_value": "",
                    },
                    "profiles": {
                        "active": "lab_pc",
                        "description": "mock dashboard profile",
                        "available": [],
                    },
                    "recording": {
                        "enabled": False,
                        "event_count": 0,
                        "frame_count": 0,
                        "options": {"include_raw": False, "include_processed": True, "every_n_frames": 20},
                    },
                },
            )
        )

    def _build_frame(self, phase: float, state: str, detection: VisionDetectionEvent, *, processed: bool) -> np.ndarray:
        height, width = 360, 640
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        base = 34 if not processed else 28
        frame[:, :] = (base + 8, base + 9, base + 12)
        gradient = np.linspace(0, 22, width, dtype=np.uint8)
        frame[:, :, 1] = np.clip(frame[:, :, 1] + gradient, 0, 255)

        line_x = int(width * (0.5 + 0.3 * math.sin(phase * 0.55)))
        if cv2 is not None:
            if detection.line:
                cv2.line(frame, (line_x, height), (line_x - 90, 0), (10, 10, 10), 38)
            if detection.green:
                cv2.rectangle(frame, (line_x - 70, height - 120), (line_x + 10, height - 40), (50, 205, 95), 2)
            if detection.red:
                cv2.rectangle(frame, (int(width * 0.6), 28), (width - 15, 88), (60, 25, 220), 3)
            if detection.balls > 0:
                cv2.circle(frame, (int(width * 0.75), int(height * 0.65)), 18, (180, 180, 180), 3)
            if detection.victims > 0:
                cv2.circle(frame, (int(width * 0.8), int(height * 0.55)), 15, (20, 20, 20), 3)
            cv2.putText(
                frame,
                state,
                (16, height - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (210, 220, 230),
                2,
                cv2.LINE_AA,
            )
            if processed:
                cv2.putText(
                    frame,
                    "mock processed",
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (80, 220, 140),
                    2,
                    cv2.LINE_AA,
                )
        return frame


class DashboardWindow(QMainWindow):
    camera_set_config_signal = pyqtSignal(int, int, int, int)
    camera_reconnect_signal = pyqtSignal()
    camera_start_signal = pyqtSignal()
    camera_stop_signal = pyqtSignal()

    def __init__(
        self,
        event_bus: EventBus | None = None,
        *,
        camera_index: int = 0,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        start_mock: bool = False,
        config_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._event_bus = event_bus
        self._profile_catalog = load_ops_profile_catalog(config_path)
        self._profile_payloads: dict[str, dict[str, Any]] = {
            str(item.get("name", "")).strip().lower(): dict(item)
            for item in self._profile_catalog.available_payload()
            if isinstance(item, Mapping) and str(item.get("name", "")).strip()
        }
        self._subscriptions: list[Any] = []
        self._bridge = UiBridge()
        self._frame_lock = threading.Lock()
        self._latest_raw_frame: np.ndarray | None = None
        self._latest_processed_frame: np.ndarray | None = None
        self._raw_generation = 0
        self._processed_generation = 0
        self._rendered_raw_generation = -1
        self._rendered_processed_generation = -1
        self._latest_bus_frame_ts = 0.0

        self._cpu_percent: float | None = None
        self._memory_percent: float | None = None
        self._fps_capture: float | None = None
        self._fps_process: float | None = None
        self._queue_depth: int | None = None
        self._ips: float | None = None
        self._network_latency_ms: float | None = None
        self._health_metadata: dict[str, Any] = {}
        self._fps_ui = 0.0
        self._last_ui_tick = time.monotonic()

        self._state = "SEARCHING_LINE"
        self._state_started_ts = time.monotonic()
        self._app_started_ts = time.monotonic()
        self._status_flags = dict(STATUS_TEMPLATE)
        self._mock_enabled = False
        self._corner_window = 5
        self._corner_on_votes = 3
        self._corner_off_votes = 1
        self._green_corner_votes: deque[int] = deque(maxlen=self._corner_window)
        self._red_corner_votes: deque[int] = deque(maxlen=self._corner_window)
        self._green_corner_conf: deque[float] = deque(maxlen=self._corner_window)
        self._red_corner_conf: deque[float] = deque(maxlen=self._corner_window)
        self._green_corner_on = False
        self._red_corner_on = False
        self._steering_current_deg = 0.0
        self._steering_target_deg = 0.0
        self._calibration_view_mode = "raw"
        self._freeze_enabled = False
        self._freeze_stamp = 0
        self._frozen_raw_frame: np.ndarray | None = None
        self._frozen_processed_frame: np.ndarray | None = None
        self._camera_status_text = ""
        self._processed_overlay_text = ""
        self._processed_corner_text = ""
        self._latest_detection_metadata: dict[str, Any] = {}
        self._active_profile_name = "custom"
        self._active_profile_description = "manual base config"
        self._ui_zoom_percent = 100

        self.setWindowTitle("FusionZero Overengineering Dashboard")
        # Keep the window usable on a normal 1365x768 workstation display;
        # both the dashboard workspace and operations panel can scroll.
        self.setMinimumSize(900, 600)
        self._build_ui()
        self._fit_to_available_screen()
        self._apply_ui_zoom(self._ui_zoom_percent)
        self._wire_signals()
        self._prime_ops_controls()
        self._setup_camera_worker(camera_index, camera_width, camera_height, camera_fps)
        self._setup_mock_source()
        self._subscribe_event_bus()

        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._render_tick)
        self._render_timer.start(33)

        self._append_transition_log("dashboard boot complete")
        if start_mock:
            self._set_mock_mode(True)
        else:
            self._set_mock_mode(False)
            if event_bus is None:
                self.camera_start_signal.emit()

    def _fit_to_available_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1200, 700)
            return

        available = screen.availableGeometry()
        width = max(900, min(1360, available.width() - 16))
        height = max(600, min(820, available.height() - 16))
        self.resize(width, height)

    def _build_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("DashboardRoot")
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(14, 12, 14, 12)
        root_layout.setSpacing(10)
        self._root_layout = root_layout

        splitter = QSplitter(Qt.Orientation.Horizontal, root)
        splitter.setChildrenCollapsible(False)
        self._main_splitter = splitter
        root_layout.addWidget(splitter, 1)

        dashboard_host = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_host)
        dashboard_layout.setContentsMargins(0, 0, 0, 0)
        dashboard_layout.setSpacing(10)
        self._dashboard_layout = dashboard_layout

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)
        self.top_bar = TopMetricsBar(dashboard_host)
        header_row.addWidget(self.top_bar, 1)
        self.tuning_toggle_button = QToolButton(dashboard_host)
        self.tuning_toggle_button.setObjectName("AdvancedToggleButton")
        self.tuning_toggle_button.setText("AJUSTES")
        self.tuning_toggle_button.setArrowType(Qt.ArrowType.LeftArrow)
        self.tuning_toggle_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.tuning_toggle_button.setToolTip("Mostrar ou guardar os ajustes avancados")
        self.tuning_toggle_button.setAccessibleName("Mostrar ou guardar ajustes avancados")
        header_row.addWidget(self.tuning_toggle_button)
        dashboard_layout.addLayout(header_row)

        frame_row = QHBoxLayout()
        frame_row.setContentsMargins(0, 0, 0, 0)
        frame_row.setSpacing(10)
        self.raw_view = VideoView("Raw View", dashboard_host)
        self.processed_view = VideoView("Processed View", dashboard_host)
        frame_row.addWidget(self.raw_view, 1)
        frame_row.addWidget(self.processed_view, 1)
        dashboard_layout.addLayout(frame_row, 5)

        self.primary_controls = ControlCenterPanel(dashboard_host)
        dashboard_layout.addWidget(self.primary_controls)

        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        info_row.setSpacing(10)
        self.status_panel = StatusPanel(dashboard_host)
        self.telemetry_panel = TelemetryPanel(dashboard_host)
        self.health_panel = OpsHealthPanel(dashboard_host)
        self.log_panel = TransitionLogPanel(dashboard_host)
        self.status_panel.setMinimumHeight(220)
        self.health_panel.setMinimumHeight(220)
        self.log_panel.setMinimumHeight(220)
        info_row.addWidget(self.status_panel, 3)
        info_row.addWidget(self.health_panel, 4)
        info_row.addWidget(self.log_panel, 5)
        dashboard_layout.addLayout(info_row, 4)

        # Retain the data sinks used by the event pipeline and compatibility
        # tests, but keep the obsolete dashboard visuals completely hidden.
        self.steering_panel = SteeringPanel(dashboard_host)
        self.timer_panel = TimerPanel(dashboard_host)
        self.robot_panel = RobotPlaceholderPanel(dashboard_host)
        for legacy_panel in (
            self.telemetry_panel,
            self.steering_panel,
            self.timer_panel,
            self.robot_panel,
        ):
            legacy_panel.hide()

        dashboard_scroll = QScrollArea(splitter)
        dashboard_scroll.setWidgetResizable(True)
        dashboard_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        dashboard_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        dashboard_scroll.setWidget(dashboard_host)
        self.dashboard_scroll = dashboard_scroll

        self.tuning_panel = TuningPanel(splitter, show_primary_controls=False)
        self.tuning_panel.setMinimumWidth(360)
        self.tuning_panel.setMaximumWidth(440)

        splitter.addWidget(dashboard_scroll)
        splitter.addWidget(self.tuning_panel)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([1080, 400])

    def _wire_signals(self) -> None:
        self._bridge.raw_frame_event.connect(self._on_raw_frame_event)
        self._bridge.processed_frame_event.connect(self._on_processed_frame_event)
        self._bridge.detection_event.connect(self._on_detection_event)
        self._bridge.state_event.connect(self._on_state_event)
        self._bridge.transition_event.connect(self._on_transition_event)
        self._bridge.health_event.connect(self._on_health_event)
        self._bridge.log_event.connect(self._on_log_event)
        self._bridge.path_event.connect(self._on_path_event)
        self._bridge.camera_frame.connect(self._on_camera_frame)
        self._bridge.camera_status.connect(self._on_camera_status)

        self.tuning_panel.parameter_changed.connect(self._on_parameter_changed)
        self.tuning_panel.camera_reconnect_requested.connect(self._on_camera_reconnect_requested)
        self.tuning_panel.mock_mode_changed.connect(self._set_mock_mode)
        self.tuning_panel.mode_switch_requested.connect(self._on_mode_switch_requested)
        self.tuning_panel.robot_command_requested.connect(self._on_robot_command_requested)
        self.tuning_panel.profile_apply_requested.connect(self._on_profile_apply_requested)
        self.tuning_panel.recording_command_requested.connect(self._on_recording_command_requested)
        self.tuning_panel.calibration_changed.connect(self._on_calibration_changed)
        self.tuning_panel.calibration_snapshot_requested.connect(self._on_calibration_snapshot_requested)
        self.tuning_panel.corner_timing_apply_requested.connect(
            self._on_corner_timing_apply_requested
        )
        self.tuning_panel.green_half_turn_apply_requested.connect(
            self._on_green_half_turn_apply_requested
        )
        self.primary_controls.robot_command_requested.connect(
            self._on_robot_command_requested
        )
        self.primary_controls.recording_command_requested.connect(
            self._on_recording_command_requested
        )
        self.primary_controls.zoom_requested.connect(self._on_zoom_requested)
        self.tuning_toggle_button.clicked.connect(self._toggle_tuning_panel)

    def _on_zoom_requested(self, delta: int) -> None:
        self._apply_ui_zoom(self._ui_zoom_percent + int(delta))

    def _apply_ui_zoom(self, percent: int) -> None:
        value = max(80, min(120, int(percent)))
        self._ui_zoom_percent = value
        scale = value / 100.0

        self.setStyleSheet(build_dashboard_stylesheet(scale))
        self.primary_controls.set_zoom_percent(value)
        self.primary_controls.setMinimumHeight(round(154 * scale))
        self.top_bar.set_scale(scale)

        for view in (self.raw_view, self.processed_view):
            view.video_label.setMinimumSize(round(280 * scale), round(180 * scale))
        for panel in (self.status_panel, self.health_panel, self.log_panel):
            panel.setMinimumHeight(round(220 * scale))

        self.tuning_panel.setMinimumWidth(round(360 * scale))
        self.tuning_panel.setMaximumWidth(round(440 * scale))
        self._root_layout.setContentsMargins(
            round(14 * scale),
            round(12 * scale),
            round(14 * scale),
            round(12 * scale),
        )
        self._root_layout.setSpacing(round(10 * scale))
        self._dashboard_layout.setSpacing(round(10 * scale))
        self.centralWidget().updateGeometry()

    def _prime_ops_controls(self) -> None:
        self.tuning_panel.apply_values(self._profile_catalog.default_tuning, emit_changes=False)
        self.tuning_panel.set_camera_values(self._profile_catalog.default_camera)
        self.tuning_panel.set_profiles(self._profile_catalog.available_payload(), self._profile_catalog.default_profile_name)
        self.tuning_panel.set_active_profile(self._active_profile_name, self._active_profile_description)
        self.tuning_panel.update_recording_status(
            {
                "enabled": False,
                "options": self._profile_catalog.default_recording,
            }
        )
        self.primary_controls.update_recording_status(
            {
                "enabled": False,
                "options": self._profile_catalog.default_recording,
            }
        )
        self.health_panel.update_rows(self._build_health_rows({}))

    def _toggle_tuning_panel(self) -> None:
        show_panel = not self.tuning_panel.isVisible()
        self.tuning_panel.setVisible(show_panel)
        self.tuning_toggle_button.setArrowType(
            Qt.ArrowType.LeftArrow if show_panel else Qt.ArrowType.RightArrow
        )
        self.tuning_toggle_button.setText("AJUSTES" if show_panel else "MOSTRAR AJUSTES")
        if show_panel:
            total_width = max(1, self._main_splitter.width())
            sidebar_width = min(420, max(360, total_width // 3))
            self._main_splitter.setSizes([total_width - sidebar_width, sidebar_width])

    def _setup_camera_worker(self, index: int, width: int, height: int, fps: int) -> None:
        self._camera_thread = QThread(self)
        self._camera_worker = CameraCaptureWorker(index=index, width=width, height=height, fps=fps)
        self._camera_worker.moveToThread(self._camera_thread)
        self._camera_worker.frame_ready.connect(self._bridge.camera_frame)
        self._camera_worker.status.connect(self._bridge.camera_status)

        self.camera_set_config_signal.connect(self._camera_worker.set_config)
        self.camera_reconnect_signal.connect(self._camera_worker.reconnect)
        self.camera_start_signal.connect(self._camera_worker.start)
        self.camera_stop_signal.connect(self._camera_worker.stop)
        self._camera_thread.start()

    def _setup_mock_source(self) -> None:
        self._mock_source = MockSource()
        self._mock_source.raw_frame.connect(self._store_raw_frame)
        self._mock_source.processed_frame.connect(self._store_processed_frame)
        self._mock_source.detection_event.connect(self._on_detection_event)
        self._mock_source.state_event.connect(self._on_state_event)
        self._mock_source.transition_event.connect(self._on_transition_event)
        self._mock_source.health_event.connect(self._on_health_event)
        self._mock_source.log_event.connect(self._on_log_event)

    def _subscribe_event_bus(self) -> None:
        if self._event_bus is None:
            return
        self._subscriptions.append(
            self._event_bus.subscribe(EventTopic.VISION_RAW_FRAME, lambda event: self._bridge.raw_frame_event.emit(event))
        )
        self._subscriptions.append(
            self._event_bus.subscribe(
                EventTopic.VISION_PROCESSED_FRAME,
                lambda event: self._bridge.processed_frame_event.emit(event),
            )
        )
        self._subscriptions.append(
            self._event_bus.subscribe(EventTopic.VISION_DETECTIONS, lambda event: self._bridge.detection_event.emit(event))
        )
        self._subscriptions.append(
            self._event_bus.subscribe(EventTopic.FSM_STATE, lambda event: self._bridge.state_event.emit(event))
        )
        self._subscriptions.append(
            self._event_bus.subscribe(
                EventTopic.FSM_TRANSITION,
                lambda event: self._bridge.transition_event.emit(event),
            )
        )
        self._subscriptions.append(
            self._event_bus.subscribe(EventTopic.SYSTEM_HEALTH, lambda event: self._bridge.health_event.emit(event))
        )
        self._subscriptions.append(
            self._event_bus.subscribe(EventTopic.SYSTEM_LOG, lambda event: self._bridge.log_event.emit(event))
        )
        self._subscriptions.append(
            self._event_bus.subscribe(EventTopic.NAV_PATH, lambda event: self._bridge.path_event.emit(event))
        )
        self._append_transition_log("event bus subscribers attached")

    def _set_mock_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._mock_enabled:
            return
        self._mock_enabled = enabled
        self.tuning_panel.set_mock_mode(enabled)

        if enabled:
            self.camera_stop_signal.emit()
            self._mock_source.start()
            self._append_transition_log("mock mode enabled")
        else:
            self._mock_source.stop()
            if self._event_bus is None:
                self.camera_start_signal.emit()
            self._append_transition_log("mock mode disabled")

    def _on_parameter_changed(self, key: str, value: object) -> None:
        payload = {"key": key, "value": value}
        self._publish_ui_command("tuning.update", payload)

    def _on_corner_timing_apply_requested(self, payload: Mapping[str, Any]) -> None:
        advance_ms = int(payload.get("approach_left_min_ms", 550))
        pivot_ms = int(payload.get("pivot_left_ms", 2100))
        self._append_transition_log(
            f"left corner timing requested: advance {advance_ms} ms | pivot {pivot_ms} ms"
        )
        self._publish_ui_command(
            "control.left_corner_timing.apply",
            {"approach_min_ms": advance_ms, "pivot_ms": pivot_ms},
        )

    def _on_green_half_turn_apply_requested(self, payload: Mapping[str, Any]) -> None:
        first_ms = int(payload.get("first_ms", 1900))
        reverse_ms = int(payload.get("reverse_ms", 200))
        second_ms = int(payload.get("second_ms", 2200))
        self._append_transition_log(
            "green half-turn timing requested: "
            f"first {first_ms} ms | reverse {reverse_ms} ms | second {second_ms} ms"
        )
        self._publish_ui_command(
            "control.green_half_turn.apply",
            {
                "first_ms": first_ms,
                "reverse_ms": reverse_ms,
                "second_ms": second_ms,
            },
        )

    def _on_camera_reconnect_requested(self, config: Mapping[str, Any]) -> None:
        index = int(config.get("index", 0))
        width = int(config.get("width", 640))
        height = int(config.get("height", 480))
        fps = int(config.get("fps", 30))

        if self._event_bus is None:
            self.camera_set_config_signal.emit(index, width, height, fps)
            self.camera_reconnect_signal.emit()
            if not self._mock_enabled:
                self.camera_start_signal.emit()

        self._append_transition_log(f"camera reconnect requested idx={index} {width}x{height}@{fps}")
        self._publish_ui_command(
            "camera.reconnect",
            {"index": index, "width": width, "height": height, "fps": fps},
        )

    def _on_mode_switch_requested(self, mode: str) -> None:
        normalized = str(mode or "").strip().lower()
        if normalized not in {"line", "rescue"}:
            return
        self._append_transition_log(f"mode switch requested: {normalized}")
        self._publish_ui_command("fsm.force_mode", {"mode": normalized})

    def _on_robot_command_requested(self, command: str, payload: Mapping[str, Any]) -> None:
        normalized = str(command or "").strip().lower()
        if normalized not in {
            "system.start",
            "system.stop",
            "robot.start",
            "robot.forward_test",
            "robot.stop",
            "robot.force_stop",
            "robot.clear_estop",
            "robot.obstacle_test",
            "robot.obstacle_clear",
            "leds.on",
            "leds.off",
            "leds.toggle",
            "led1.on",
            "led1.off",
            "led1.toggle",
            "led2.on",
            "led2.off",
            "led2.toggle",
            "leds.set",
        }:
            return
        detail = dict(payload or {})
        if normalized == "system.start":
            self._append_transition_log("master START requested")
        elif normalized == "system.stop":
            self._append_transition_log("master STOP requested")
        elif normalized == "robot.start":
            self._append_transition_log("robot START requested")
        elif normalized == "robot.forward_test":
            duration_ms = int(detail.get("duration_ms", 1200))
            detail["duration_ms"] = duration_ms
            self._append_transition_log(f"robot forward test requested: {duration_ms} ms")
        elif normalized == "robot.stop":
            self._append_transition_log("robot STOP requested")
        elif normalized == "robot.force_stop":
            self._append_transition_log("robot Force STOP requested")
        elif normalized == "robot.obstacle_test":
            self._append_transition_log("robot obstacle TEST requested")
        elif normalized == "robot.obstacle_clear":
            self._append_transition_log("robot obstacle CLEAR requested")
        elif normalized.startswith("led"):
            self._append_transition_log(f"{normalized} requested")
        else:
            self._append_transition_log("robot Clear ESTOP requested")
        self._publish_ui_command(normalized, detail)

    def _on_profile_apply_requested(self, profile_name: str) -> None:
        normalized = str(profile_name or "").strip().lower()
        payload = self._profile_payloads.get(normalized, {})
        if not payload:
            return
        camera = payload.get("camera", {})
        tuning = payload.get("tuning", {})
        description = str(payload.get("description", normalized.replace("_", " "))).strip()
        if isinstance(camera, Mapping):
            self.tuning_panel.set_camera_values(camera)
        if isinstance(tuning, Mapping):
            self.tuning_panel.apply_values(tuning, emit_changes=False)
        if isinstance(payload.get("recording"), Mapping):
            recording_payload = {
                "enabled": False,
                "options": payload.get("recording", {}),
            }
            self.tuning_panel.update_recording_status(recording_payload)
            self.primary_controls.update_recording_status(recording_payload)
        self.tuning_panel.set_active_profile(normalized, description)
        self._active_profile_name = normalized
        self._active_profile_description = description
        self._append_transition_log(f"profile requested: {normalized}")
        if self._event_bus is None and isinstance(camera, Mapping):
            self._on_camera_reconnect_requested(camera)
            return
        self._publish_ui_command("config.load_profile", {"profile": normalized})

    def _on_recording_command_requested(self, action: str, payload: Mapping[str, Any]) -> None:
        normalized = str(action or "").strip().lower()
        if normalized not in {"configure", "start", "stop"}:
            return
        self._append_transition_log(f"session recording {normalized} requested")
        self._publish_ui_command(f"session.recording.{normalized}", dict(payload))

    def _on_calibration_changed(self, payload: Mapping[str, Any]) -> None:
        next_mode = str(payload.get("view_mode", "raw")).strip().lower() or "raw"
        next_freeze = bool(payload.get("freeze", False))
        if next_freeze != self._freeze_enabled:
            self._freeze_enabled = next_freeze
            self._freeze_stamp += 1
            if next_freeze:
                self._capture_frozen_frames()
            else:
                self._frozen_raw_frame = None
                self._frozen_processed_frame = None
        self._calibration_view_mode = next_mode
        self._publish_ui_command("calibration.view", {"view_mode": next_mode, "freeze": self._freeze_enabled})

    def _on_calibration_snapshot_requested(self, payload: Mapping[str, Any]) -> None:
        snapshot = {
            "view_mode": str(payload.get("view_mode", self._calibration_view_mode)),
            "freeze": bool(payload.get("freeze", self._freeze_enabled)),
            "state": self._state,
            "profile": self._active_profile_name,
            "tuning": self.tuning_panel.control_snapshot(),
        }
        if self._latest_detection_metadata:
            snapshot["vision"] = dict(self._latest_detection_metadata)
        self._append_transition_log("calibration snapshot requested")
        self._publish_ui_command("calibration.snapshot", snapshot)

    def _publish_ui_command(self, command: str, params: Mapping[str, Any]) -> None:
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish(
                EventTopic.UI_COMMAND,
                UICommandEvent(timestamp=time.time(), command=command, params=dict(params)),
            )
        except Exception as exc:
            self._append_transition_log(f"ui.command publish failed: {exc}")

    def _on_raw_frame_event(self, event: object) -> None:
        if not isinstance(event, FrameEvent):
            return
        frame = _frame_from_event(event)
        if frame is None:
            return
        self._latest_bus_frame_ts = time.monotonic()
        self._store_raw_frame(frame)

    def _on_processed_frame_event(self, event: object) -> None:
        if not isinstance(event, FrameEvent):
            return
        frame = _frame_from_event(event)
        if frame is None:
            return
        self._store_processed_frame(frame)

    def _on_camera_frame(self, frame: object) -> None:
        if self._mock_enabled:
            return
        if not isinstance(frame, np.ndarray):
            return
        live_bus = (time.monotonic() - self._latest_bus_frame_ts) < 1.0
        if live_bus:
            return
        self._store_raw_frame(frame)
        with self._frame_lock:
            has_processed = self._latest_processed_frame is not None
        if not has_processed:
            self._store_processed_frame(self._build_processed_fallback(frame))

    def _on_camera_status(self, text: str) -> None:
        self._append_transition_log(text)
        self._camera_status_text = text if "connected" not in text.lower() else ""
        lowered = text.lower()
        if (
            self._event_bus is None
            and not self._mock_enabled
            and ("not available" in lowered or "unavailable" in lowered)
        ):
            self._append_transition_log("camera unavailable, enabling mock fallback")
            self._set_mock_mode(True)

    def _on_detection_event(self, event: object) -> None:
        if not isinstance(event, VisionDetectionEvent):
            return

        self._latest_detection_metadata = dict(event.metadata or {})
        self._state = event.state or self._state
        self.status_panel.update_state(self._state)

        metadata = event.metadata if isinstance(event.metadata, Mapping) else {}
        self._update_corner_runtime_config(metadata)

        silver_line = (
            bool(metadata.get("silver_line", {}).get("found", False)) if isinstance(metadata.get("silver_line"), Mapping) else False
        )
        silver_ball = bool(metadata.get("silver_ball_found", False)) or bool(event.balls > 0)
        black_ball = bool(metadata.get("black_ball_found", False))
        if not black_ball and isinstance(metadata.get("dead_victim"), Mapping):
            black_ball = bool(metadata.get("dead_victim", {}).get("found", False))
        if event.victims > 0:
            black_ball = True

        raw_green_corner = bool(metadata.get("green_corner_found", False))
        raw_red_corner = bool(metadata.get("red_corner_found", False))
        green_corner_conf = _safe_float(metadata.get("green_corner_confidence"))
        red_corner_conf = _safe_float(metadata.get("red_corner_confidence"))
        self._green_corner_on = self._apply_corner_hysteresis(
            queue=self._green_corner_votes,
            conf_queue=self._green_corner_conf,
            detected=raw_green_corner,
            confidence=green_corner_conf if green_corner_conf is not None else 0.0,
            current=self._green_corner_on,
        )
        self._red_corner_on = self._apply_corner_hysteresis(
            queue=self._red_corner_votes,
            conf_queue=self._red_corner_conf,
            detected=raw_red_corner,
            confidence=red_corner_conf if red_corner_conf is not None else 0.0,
            current=self._red_corner_on,
        )

        self._status_flags = {
            "LINE": bool(event.line),
            "SILVER": silver_line,
            "GREEN": bool(event.green),
            "RED": bool(event.red),
            "GREEN CORNER": bool(self._green_corner_on),
            "RED CORNER": bool(self._red_corner_on),
            "SILVER BALL": bool(silver_ball),
            "BLACK BALL": black_ball,
        }
        self.status_panel.update_statuses(self._status_flags)

        if event.latency_ms > 0:
            self._ips = 1000.0 / float(event.latency_ms)

        telemetry = self._extract_telemetry(metadata)
        if telemetry:
            self.telemetry_panel.update_values(telemetry)

        status_text = metadata.get("status_text") if isinstance(metadata, Mapping) else None
        if isinstance(status_text, str) and status_text.strip():
            active_text = status_text.strip()
        else:
            active = [key for key, enabled in self._status_flags.items() if enabled]
            if not active:
                active_text = "No markers"
            else:
                active_text = " | ".join(active[:3]) + (" ..." if len(active) > 3 else "")
        corner_parts: list[str] = []
        gc_conf = self._corner_confidence(self._green_corner_conf)
        rc_conf = self._corner_confidence(self._red_corner_conf)
        if self._green_corner_on or gc_conf > 0.01:
            corner_parts.append(f"GC {int(round(gc_conf * 100.0)):02d}%")
        if self._red_corner_on or rc_conf > 0.01:
            corner_parts.append(f"RC {int(round(rc_conf * 100.0)):02d}%")
        if corner_parts:
            active_text = f"{active_text} | {' | '.join(corner_parts)}"
        control_meta = metadata.get("control") if isinstance(metadata, Mapping) else None
        control_parts: list[str] = []
        if isinstance(control_meta, Mapping):
            control_mode = str(control_meta.get("control_mode", "")).strip().upper()
            obstacle_state = str(control_meta.get("obstacle_state", "")).strip().upper()
            green_instruction = str(control_meta.get("green_instruction", "")).strip().upper().replace(" ", "_")
            green_route_decision = str(
                control_meta.get("green_route_decision", "")
            ).strip().upper().replace(" ", "_")
            vision_green_instruction = str(metadata.get("green_instruction", "")).strip().upper().replace(" ", "_")
            vision_confidence = _safe_float(control_meta.get("vision_confidence"))
            line_error = _safe_float(control_meta.get("line_error"))
            pid_output = _safe_float(control_meta.get("pid_output"))
            if control_mode:
                control_parts.append(control_mode)
            if vision_confidence is not None and vision_confidence > 0.0:
                control_parts.append(f"VC {int(round(vision_confidence * 100.0)):02d}%")
            if line_error is not None:
                control_parts.append(f"E {line_error:+.2f}")
            if pid_output is not None:
                control_parts.append(f"PID {pid_output:+.1f}")
            if obstacle_state and obstacle_state not in {"CLEAR", "NONE"}:
                control_parts.append(f"OBS {obstacle_state}")
            route_labels = {
                "BEFORE_LEFT": "G_VERDE_ANTES_ESQUERDA",
                "BEFORE_RIGHT": "G_VERDE_ANTES_DIREITA",
                "AFTER_STRAIGHT": "G_VERDE_DEPOIS_RETO",
                "HALF_TURN": "G_2_VERDES_180",
            }
            route_label = route_labels.get(green_route_decision, "")
            if route_label:
                control_parts.append(route_label)
            elif not bool(event.green):
                control_parts.append("G_VERDE_NADA")
            else:
                visible_green = vision_green_instruction or green_instruction
                if visible_green and visible_green not in {"NO_GREEN", "NONE"}:
                    control_parts.append(f"G {visible_green}")
            if bool(control_meta.get("failsafe", False)):
                control_parts.append("FAILSAFE")
        if control_parts:
            active_text = f"{active_text} | {' | '.join(control_parts[:6])}"
        self._processed_overlay_text = active_text
        self._processed_corner_text = f"{event.latency_ms:.1f} ms"
        self._update_steering_from_detection(event, metadata)
        self.robot_panel.set_step(int(time.monotonic() * 1.5))

    def _on_state_event(self, event: object) -> None:
        if not isinstance(event, StateSnapshotEvent):
            return
        next_state = event.state.strip().upper() if event.state else self._state
        if next_state != self._state:
            self._state_started_ts = time.monotonic()
        self._state = next_state
        self.status_panel.update_state(self._state)

    def _on_transition_event(self, event: object) -> None:
        if not isinstance(event, StateTransitionEvent):
            return
        self._state = event.new_state or self._state
        self._state_started_ts = time.monotonic()
        line = (
            f"{_iso_timestamp(event.timestamp)} [{event.new_state}] "
            f"{event.old_state} --{event.trigger}--> {event.new_state} | {event.reason}"
        )
        self._append_transition_log(line)
        self.status_panel.update_state(self._state)

    def _on_health_event(self, event: object) -> None:
        if not isinstance(event, HealthEvent):
            return
        network = event.metadata.get("network", {}) if isinstance(event.metadata, Mapping) else {}
        if self._event_bus is not None and isinstance(network, Mapping):
            if str(network.get("peer", "")).strip().lower() == "mock":
                return
        self._cpu_percent = float(event.cpu_percent)
        self._fps_capture = float(event.fps_capture) if event.fps_capture > 0 else None
        self._fps_process = float(event.fps_process) if event.fps_process > 0 else None
        self._queue_depth = int(event.queue_depth)
        self._health_metadata = dict(event.metadata or {})
        self._memory_percent = _safe_float(self._health_metadata.get("memory_percent"))
        self._network_latency_ms = _safe_float(self._health_metadata.get("network_latency_ms"))
        if event.fps_ui > 0:
            self._fps_ui = float(event.fps_ui)
        elif event.fps_process > 0:
            self._fps_ui = float(event.fps_process)
        self._update_ops_from_health()

    def _on_log_event(self, event: object) -> None:
        if not isinstance(event, LogEvent):
            return
        state = event.state.strip().upper() if event.state else self._state
        line = f"{_iso_timestamp(event.timestamp)} [{state}] {event.source}: {event.message}"
        self._append_transition_log(line)

    def _on_path_event(self, event: object) -> None:
        if not isinstance(event, PathEvent):
            return
        poses = event.poses if isinstance(event.poses, list) else []
        if not poses:
            return
        last_pose = poses[-1]
        yaw = _safe_float(getattr(last_pose, "theta", None))
        if yaw is not None:
            self.telemetry_panel.update_values({"yaw": math.degrees(yaw)})
        self.robot_panel.set_step(len(poses) % 3)

    def _store_raw_frame(self, frame: object) -> None:
        if not isinstance(frame, np.ndarray):
            return
        with self._frame_lock:
            self._latest_raw_frame = np.ascontiguousarray(frame)
            self._raw_generation += 1

    def _store_processed_frame(self, frame: object) -> None:
        if not isinstance(frame, np.ndarray):
            return
        with self._frame_lock:
            self._latest_processed_frame = np.ascontiguousarray(frame)
            self._processed_generation += 1

    def _extract_telemetry(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        keys = ("front", "left", "right", "back", "yaw", "roll", "pitch", "gripper")
        out: dict[str, Any] = {}
        for candidate_key in ("telemetry", "sensors", "distances"):
            candidate = metadata.get(candidate_key)
            if not isinstance(candidate, Mapping):
                continue
            for key in keys:
                if key in candidate:
                    out[key] = candidate[key]
            if "yaw" not in out and "heading_deg" in candidate:
                out["yaw"] = candidate["heading_deg"]
        return out

    def _update_ops_from_health(self) -> None:
        profiles = self._health_metadata.get("profiles")
        if isinstance(profiles, Mapping):
            available = profiles.get("available", [])
            if isinstance(available, list) and available:
                self._profile_payloads = {
                    str(item.get("name", "")).strip().lower(): dict(item)
                    for item in available
                    if isinstance(item, Mapping) and str(item.get("name", "")).strip()
                }
                self.tuning_panel.set_profiles(available, str(profiles.get("active", "")).strip().lower() or None)
            self._active_profile_name = str(profiles.get("active", self._active_profile_name)).strip().lower() or "custom"
            self._active_profile_description = (
                str(profiles.get("description", self._active_profile_description)).strip()
                or self._active_profile_description
            )
            self.tuning_panel.set_active_profile(self._active_profile_name, self._active_profile_description)

        recording = self._health_metadata.get("recording")
        if isinstance(recording, Mapping):
            self.tuning_panel.update_recording_status(recording)
            self.primary_controls.update_recording_status(recording)

        serial = self._health_metadata.get("serial")
        if isinstance(serial, Mapping):
            primary_status = dict(serial)
            master_switch = self._health_metadata.get("master_switch")
            if isinstance(master_switch, Mapping):
                primary_status["master_switch"] = dict(master_switch)
            self.primary_controls.update_robot_status(primary_status)
            self.tuning_panel.update_corner_timing_status(serial)
            self.tuning_panel.update_green_half_turn_status(serial)
            telemetry = self._extract_telemetry({"telemetry": serial.get("telemetry", {})})
            if telemetry:
                self.telemetry_panel.update_values(telemetry)
            pid_values: dict[str, Any] = {}
            serial_to_control = {
                "pid_kp_us": "control.pid.kp_us",
                "pid_ki_us": "control.pid.ki_us",
                "pid_kd_us": "control.pid.kd_us",
                "pid_integral_limit": "control.pid.integral_limit",
                "pid_derivative_filter": "control.pid.derivative_filter",
                "max_output_us": "control.pid.max_output_us",
                "line_hold_ms": "control.line_hold_ms",
                "left_base_throttle_us": "control.base.left_us",
                "right_base_throttle_us": "control.base.right_us",
                "line_error_deadband": "control.line.deadband",
            }
            for remote_key, control_key in serial_to_control.items():
                if remote_key in serial and serial.get(remote_key) is not None:
                    pid_values[control_key] = serial[remote_key]
            if pid_values:
                self.tuning_panel.apply_values(pid_values, emit_changes=False)

        self.health_panel.update_rows(self._build_health_rows(self._health_metadata))

    def _build_health_rows(self, metadata: Mapping[str, Any]) -> dict[str, dict[str, str]]:
        camera = metadata.get("camera", {})
        serial = metadata.get("serial", {})
        network = metadata.get("network", {})
        power = metadata.get("power", {})
        recording = metadata.get("recording", {})
        profiles = metadata.get("profiles", {})

        camera_state = str(camera.get("state", "unknown")).strip().lower() if isinstance(camera, Mapping) else "unknown"
        camera_detail = ""
        if isinstance(camera, Mapping):
            camera_detail = (
                f"idx {int(camera.get('index', 0))} | "
                f"{int(camera.get('width', 0))}x{int(camera.get('height', 0))}@{int(camera.get('fps', 0))}"
            )
            backend = str(camera.get("backend", "")).strip()
            if backend:
                camera_detail = f"{camera_detail} | {backend}"
            camera_detail = self._compact_health_text(camera_detail, 34)

        serial_state = str(serial.get("state", "unknown")).strip().lower() if isinstance(serial, Mapping) else "unknown"
        serial_detail_lines: list[str] = []
        if isinstance(serial, Mapping):
            serial_port = self._compact_health_text(str(serial.get("port", "")).strip(), 28)
            if serial_port:
                serial_detail_lines.append(serial_port)
            hb_age = serial.get("heartbeat_age_ms")
            tlm_age = serial.get("telemetry_age_ms")
            age_parts: list[str] = []
            if hb_age is not None:
                age_parts.append(f"hb {int(hb_age)}ms")
            if tlm_age is not None:
                age_parts.append(f"tlm {int(tlm_age)}ms")
            if age_parts:
                serial_detail_lines.append(" | ".join(age_parts))
            control_parts: list[str] = []
            control_mode = str(serial.get("control_mode", "")).strip().upper()
            obstacle_state = str(serial.get("obstacle_state", "")).strip().upper()
            green_instruction = str(serial.get("green_instruction", "")).strip().upper()
            green_route_decision = str(
                serial.get("green_route_decision", "")
            ).strip().upper()
            if "motor_armed" in serial:
                control_parts.append(
                    "ARMED" if bool(serial.get("motor_armed", False)) else "DISARMED"
                )
            if control_mode:
                control_parts.append(control_mode)
            if obstacle_state and obstacle_state not in {"CLEAR", "NONE"}:
                control_parts.append(f"OBS {obstacle_state}")
            if green_instruction and green_instruction not in {"NO_GREEN", "NONE"}:
                control_parts.append(f"G {green_instruction}")
            if green_route_decision:
                control_parts.append(f"ROUTE {green_route_decision}")
            if bool(serial.get("failsafe", False)):
                control_parts.append("FAILSAFE")
            if control_parts:
                serial_detail_lines.append(self._compact_health_text(" | ".join(control_parts), 34))
            line_error = _safe_float(serial.get("line_error"))
            pid_output = _safe_float(serial.get("pid_output"))
            pid_parts: list[str] = []
            if line_error is not None:
                pid_parts.append(f"e {line_error:+.2f}")
            if pid_output is not None:
                pid_parts.append(f"pid {pid_output:+.1f}")
            if pid_parts:
                serial_detail_lines.append(" | ".join(pid_parts))
            serial_telemetry = serial.get("telemetry", {})
            if not isinstance(serial_telemetry, Mapping):
                serial_telemetry = {}
            left_pwm = serial.get("left_pwm", serial_telemetry.get("left_pwm"))
            right_pwm = serial.get("right_pwm", serial_telemetry.get("right_pwm"))
            if left_pwm is not None and right_pwm is not None:
                try:
                    serial_detail_lines.append(f"PWM L {int(left_pwm)} | R {int(right_pwm)} us")
                except (TypeError, ValueError):
                    pass
        serial_detail = "\n".join(serial_detail_lines)
        network_state = str(network.get("state", "local")).strip().lower() if isinstance(network, Mapping) else "local"
        network_peer = self._compact_health_text(str(network.get("peer", "local")).strip(), 28) if isinstance(network, Mapping) else "local"
        network_latency = _safe_float(network.get("latency_ms")) if isinstance(network, Mapping) else None

        power_available = bool(power.get("available", False)) if isinstance(power, Mapping) else False
        power_status = str(power.get("status", "unknown")).strip().lower() if isinstance(power, Mapping) else "unknown"
        power_summary = str(power.get("summary", "power unknown")).strip() if isinstance(power, Mapping) else "power unknown"
        power_raw = str(power.get("raw_value", "")).strip() if isinstance(power, Mapping) else ""

        profile_active = self._active_profile_name
        profile_detail = self._active_profile_description
        if isinstance(profiles, Mapping):
            profile_active = str(profiles.get("active", profile_active)).strip().lower() or profile_active
            profile_detail = str(profiles.get("description", profile_detail)).strip() or profile_detail

        recording_enabled = bool(recording.get("enabled", False)) if isinstance(recording, Mapping) else False
        recording_detail = "idle"
        if isinstance(recording, Mapping):
            options = recording.get("options", {})
            stride = int(options.get("every_n_frames", 0)) if isinstance(options, Mapping) else 0
            recording_detail = f"{int(recording.get('event_count', 0))} ev | {int(recording.get('frame_count', 0))} fr"
            if stride > 0:
                recording_detail = f"{recording_detail} | every {stride}f"
            session_dir = str(recording.get("session_dir", "")).strip()
            if session_dir:
                recording_detail = f"{recording_detail}\n{self._compact_health_path(session_dir)}"

        if profile_detail:
            profile_detail = self._compact_health_text(profile_detail, 46)

        if power_summary:
            compact_power = power_summary
            if power_raw:
                compact_power = f"{compact_power} ({power_raw})"
        else:
            compact_power = power_raw
        compact_power = self._compact_health_text(compact_power, 34)

        return {
            "camera": {
                "value": camera_state.upper() if camera_state else "--",
                "detail": camera_detail,
                "level": self._level_from_state(camera_state, ok={"online"}, warn={"degraded"}, error={"offline"}),
            },
            "serial": {
                "value": serial_state.upper() if serial_state else "--",
                "detail": serial_detail,
                "level": self._level_from_state(
                    serial_state,
                    ok={"connected"},
                    warn={"waiting", "dry-run", "degraded"},
                    error={"offline", "error"},
                ),
            },
            "link": {
                "value": (
                    f"{network_latency:.1f} ms"
                    if network_state == "connected" and network_latency is not None
                    else network_state.upper()
                ),
                "detail": network_peer,
                "level": self._level_from_state(network_state, ok={"connected"}, warn={"local"}, error={"timeout", "disconnected"}),
            },
            "power": {
                "value": power_status.upper() if power_available else "N/A",
                "detail": compact_power,
                "level": power_status if power_available else "neutral",
            },
            "profile": {
                "value": profile_active.upper(),
                "detail": profile_detail,
                "level": "neutral",
            },
            "recording": {
                "value": "REC ON" if recording_enabled else "OFF",
                "detail": recording_detail,
                "level": "ok" if recording_enabled else "neutral",
            },
        }

    @staticmethod
    def _level_from_state(state: str, *, ok: set[str], warn: set[str], error: set[str]) -> str:
        normalized = str(state or "").strip().lower()
        if normalized in ok:
            return "ok"
        if normalized in warn:
            return "warn"
        if normalized in error:
            return "error"
        return "neutral"

    @staticmethod
    def _compact_health_text(value: str, limit: int) -> str:
        text = " ".join(str(value or "").strip().split())
        if not text or len(text) <= limit:
            return text
        return f"{text[: max(0, limit - 3)]}..."

    @staticmethod
    def _compact_health_path(value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            return ""
        try:
            candidate = Path(cleaned)
            parts = [part for part in candidate.parts if part and part != candidate.anchor]
        except Exception:
            return DashboardWindow._compact_health_text(cleaned, 28)
        if not parts:
            return DashboardWindow._compact_health_text(cleaned, 28)
        parent = "/".join(parts[-2:-1])
        if len(parts) > 2 and parent:
            parent = f".../{parent}"
        parent = DashboardWindow._compact_health_text(parent, 24)
        leaf = DashboardWindow._compact_health_text(parts[-1], 28)
        if parent:
            return f"{parent}\n{leaf}"
        return leaf

    def _capture_frozen_frames(self) -> None:
        with self._frame_lock:
            self._frozen_raw_frame = (
                None if self._latest_raw_frame is None else np.ascontiguousarray(self._latest_raw_frame.copy())
            )
            if self._latest_processed_frame is not None:
                self._frozen_processed_frame = np.ascontiguousarray(self._latest_processed_frame.copy())
            elif self._latest_raw_frame is not None:
                self._frozen_processed_frame = self._build_processed_fallback(self._latest_raw_frame)
            else:
                self._frozen_processed_frame = None

    def _current_frame_sources(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._frame_lock:
            raw = self._latest_raw_frame
            processed = self._latest_processed_frame
        if self._freeze_enabled:
            raw = self._frozen_raw_frame
            processed = self._frozen_processed_frame
        return raw, processed

    def _build_calibration_preview(
        self,
        raw: np.ndarray | None,
        processed: np.ndarray | None,
    ) -> np.ndarray | None:
        mode = self._calibration_view_mode
        if mode == "processed":
            return processed if processed is not None else raw
        if raw is None:
            return processed
        if mode == "raw":
            return raw
        if cv2 is None:
            return self._build_processed_fallback(raw)

        hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
        line_mask = self._line_mask(hsv)
        green_mask = self._green_mask(hsv)
        red_mask = self._red_mask(hsv)
        victim_mask = self._victim_mask(raw)

        if mode == "line_mask":
            return self._mask_to_frame(line_mask, tint=(220, 220, 220))
        if mode == "green_mask":
            return self._mask_to_frame(green_mask, tint=(40, 220, 90))
        if mode == "red_mask":
            return self._mask_to_frame(red_mask, tint=(70, 70, 240))
        if mode == "victim_mask":
            return self._mask_to_frame(victim_mask, tint=(30, 180, 255))

        composite = raw.copy()
        composite = self._blend_mask(composite, line_mask, (220, 220, 220))
        composite = self._blend_mask(composite, green_mask, (40, 220, 90))
        composite = self._blend_mask(composite, red_mask, (70, 70, 240))
        composite = self._blend_mask(composite, victim_mask, (30, 180, 255))
        return composite

    def _line_mask(self, hsv: np.ndarray) -> np.ndarray:
        mask = cv2.inRange(
            hsv,
            (0, 0, 0),
            (
                180,
                int(self.tuning_panel.value("line.black_s_max", 255)),
                int(self.tuning_panel.value("line.black_v_max", 70)),
            ),
        )
        erode_iter = max(0, int(self.tuning_panel.value("line.erode_iter", 0)))
        dilate_iter = max(0, int(self.tuning_panel.value("line.dilate_iter", 0)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        if erode_iter > 0:
            mask = cv2.erode(mask, kernel, iterations=erode_iter)
        if dilate_iter > 0:
            mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
        return mask

    def _green_mask(self, hsv: np.ndarray) -> np.ndarray:
        return cv2.inRange(
            hsv,
            (
                int(self.tuning_panel.value("green.h_min", 35)),
                int(self.tuning_panel.value("green.s_min", 70)),
                int(self.tuning_panel.value("green.v_min", 50)),
            ),
            (
                int(self.tuning_panel.value("green.h_max", 90)),
                255,
                255,
            ),
        )

    def _red_mask(self, hsv: np.ndarray) -> np.ndarray:
        lower = cv2.inRange(
            hsv,
            (
                int(self.tuning_panel.value("red.h1_min", 0)),
                int(self.tuning_panel.value("red.s_min", 120)),
                int(self.tuning_panel.value("red.v_min", 80)),
            ),
            (
                int(self.tuning_panel.value("red.h1_max", 12)),
                255,
                255,
            ),
        )
        upper = cv2.inRange(
            hsv,
            (
                int(self.tuning_panel.value("red.h2_min", 165)),
                int(self.tuning_panel.value("red.s_min", 120)),
                int(self.tuning_panel.value("red.v_min", 80)),
            ),
            (
                int(self.tuning_panel.value("red.h2_max", 179)),
                255,
                255,
            ),
        )
        return cv2.bitwise_or(lower, upper)

    def _victim_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold = max(0, min(255, int(self.tuning_panel.value("dead.black_v_max", 60))))
        mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        return cv2.GaussianBlur(mask, (5, 5), 0)

    @staticmethod
    def _mask_to_frame(mask: np.ndarray, *, tint: tuple[int, int, int]) -> np.ndarray:
        output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        output[mask > 0] = tint
        return output

    @staticmethod
    def _blend_mask(base: np.ndarray, mask: np.ndarray, tint: tuple[int, int, int]) -> np.ndarray:
        colored = np.zeros_like(base)
        colored[mask > 0] = tint
        return cv2.addWeighted(base, 0.82, colored, 0.55, 0.0)

    def _calibration_label(self) -> str:
        labels = {
            "raw": "RAW",
            "processed": "PROCESSED",
            "line_mask": "LINE MASK",
            "green_mask": "GREEN MASK",
            "red_mask": "RED MASK",
            "victim_mask": "VICTIM MASK",
            "composite": "COMPOSITE",
        }
        return labels.get(self._calibration_view_mode, self._calibration_view_mode.upper())

    def _update_corner_runtime_config(self, metadata: Mapping[str, Any]) -> None:
        runtime = metadata.get("runtime")
        if not isinstance(runtime, Mapping):
            return
        window = int(runtime.get("corner_stability_window", self._corner_window))
        on_votes = int(runtime.get("corner_on_votes", self._corner_on_votes))
        off_votes = int(runtime.get("corner_off_votes", self._corner_off_votes))

        window = max(3, min(15, window))
        on_votes = max(1, min(window, on_votes))
        off_votes = max(0, min(on_votes, off_votes))

        if window == self._corner_window and on_votes == self._corner_on_votes and off_votes == self._corner_off_votes:
            return
        self._corner_window = window
        self._corner_on_votes = on_votes
        self._corner_off_votes = off_votes
        self._green_corner_votes = deque(self._green_corner_votes, maxlen=window)
        self._red_corner_votes = deque(self._red_corner_votes, maxlen=window)
        self._green_corner_conf = deque(self._green_corner_conf, maxlen=window)
        self._red_corner_conf = deque(self._red_corner_conf, maxlen=window)

    def _apply_corner_hysteresis(
        self,
        *,
        queue: deque[int],
        conf_queue: deque[float],
        detected: bool,
        confidence: float,
        current: bool,
    ) -> bool:
        queue.append(1 if detected else 0)
        conf_queue.append(max(0.0, min(1.0, float(confidence))))
        votes = int(sum(queue))
        if not current:
            return votes >= self._corner_on_votes
        return votes > self._corner_off_votes

    @staticmethod
    def _corner_confidence(conf_queue: deque[float]) -> float:
        if not conf_queue:
            return 0.0
        return float(sum(conf_queue) / len(conf_queue))

    def _build_processed_fallback(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame
        if cv2 is None:
            return np.ascontiguousarray(np.clip(frame * 0.85 + 10, 0, 255).astype(np.uint8))

        output = frame.copy()
        cv2.rectangle(output, (10, 10), (output.shape[1] - 10, output.shape[0] - 10), (120, 130, 140), 1)
        cv2.putText(
            output,
            "processed fallback",
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (80, 220, 120),
            2,
            cv2.LINE_AA,
        )
        return output

    def _append_transition_log(self, text: str) -> None:
        if not text:
            return
        self.log_panel.append_line(text)

    def _render_tick(self) -> None:
        now_mono = time.monotonic()
        dt = now_mono - self._last_ui_tick
        self._last_ui_tick = now_mono
        if dt > 0:
            instant = 1.0 / dt
            if self._fps_ui <= 0:
                self._fps_ui = instant
            else:
                self._fps_ui = 0.9 * self._fps_ui + 0.1 * instant

        raw, processed = self._current_frame_sources()
        raw_display = self._build_calibration_preview(raw, processed)
        processed_display = processed
        if processed_display is None and isinstance(raw, np.ndarray):
            processed_display = self._build_processed_fallback(raw)

        raw_source_generation = self._processed_generation if self._calibration_view_mode == "processed" else self._raw_generation
        raw_token = (
            "frozen" if self._freeze_enabled else "live",
            self._freeze_stamp if self._freeze_enabled else raw_source_generation,
            self._calibration_view_mode,
            self._camera_status_text,
        )
        if raw_token != self._rendered_raw_generation:
            self._rendered_raw_generation = raw_token
            self.raw_view.set_frame(raw_display, fallback_text="No camera connected")
            if isinstance(raw_display, np.ndarray):
                raw_corner = f"{raw_display.shape[1]}x{raw_display.shape[0]} | {self._calibration_label()}"
                if self._freeze_enabled:
                    raw_corner = f"{raw_corner} | FROZEN"
                self.raw_view.set_corner(raw_corner)
            raw_overlay_parts: list[str] = []
            if self._camera_status_text:
                raw_overlay_parts.append(self._camera_status_text)
            if self._calibration_view_mode != "raw":
                raw_overlay_parts.append(f"Calibration {self._calibration_label()}")
            if self._freeze_enabled:
                raw_overlay_parts.append("FROZEN")
            self.raw_view.set_overlay(" | ".join(raw_overlay_parts))

        processed_source_generation = self._processed_generation if processed is not None else self._raw_generation
        processed_token = (
            "frozen" if self._freeze_enabled else "live",
            self._freeze_stamp if self._freeze_enabled else processed_source_generation,
            self._processed_overlay_text,
            self._processed_corner_text,
        )
        if processed_token != self._rendered_processed_generation:
            self._rendered_processed_generation = processed_token
            processed_fallback = (
                "No camera connected"
                if raw is None and processed is None
                else "No processed stream"
            )
            self.processed_view.set_frame(
                processed_display,
                fallback_text=processed_fallback,
            )
            overlay = self._processed_overlay_text.strip()
            if self._freeze_enabled:
                overlay = f"{overlay} | FROZEN".strip(" |")
            self.processed_view.set_overlay(overlay)
            corner = self._processed_corner_text.strip()
            if not corner and isinstance(processed_display, np.ndarray):
                corner = f"{processed_display.shape[1]}x{processed_display.shape[0]}"
            self.processed_view.set_corner(corner)

        self.timer_panel.set_elapsed(
            now_mono - self._app_started_ts,
            now_mono - self._state_started_ts,
        )
        self.top_bar.update_metrics(
            {
                "CPU": f"{self._cpu_percent:.1f}%" if self._cpu_percent is not None else "--",
                "MEM": f"{self._memory_percent:.1f}%" if self._memory_percent is not None else "--",
                "CAP": f"{self._fps_capture:.1f}" if self._fps_capture is not None else "--",
                "PROC": f"{self._fps_process:.1f}" if self._fps_process is not None else "--",
                "NET": f"{self._network_latency_ms:.1f} ms" if self._network_latency_ms is not None else "--",
                "QUEUE": str(self._queue_depth) if self._queue_depth is not None else "--",
            }
        )

    def _update_steering_from_detection(self, event: VisionDetectionEvent, metadata: Mapping[str, Any]) -> None:
        state = (event.state or self._state or "SEARCHING_LINE").strip().upper()
        green_side = str(metadata.get("green_side", "NONE")).strip().upper()
        green_instruction = str(metadata.get("green_instruction", "")).strip().upper()
        line_angle = _safe_float(metadata.get("line_angle_deg"))
        if line_angle is None or (not event.line and state == "SEARCHING_LINE"):
            current_turn = 0.0
        else:
            current_turn = max(-180.0, min(180.0, float(line_angle) - 90.0))
        target_turn = 0.0
        mode = "n"
        arrow = "^"

        if ("MEIA VOLTA" in green_instruction) or green_side == "BOTH":
            mode = "u"
            arrow = "UT"
            target_turn = 180.0
        elif green_side == "LEFT":
            mode = "g"
            arrow = "<"
            target_turn = 90.0
        elif green_side == "RIGHT":
            mode = "g"
            arrow = ">"
            target_turn = 90.0
        elif state == "FOLLOWING_LINE" and event.line:
            mode = "n"
            arrow = "^"
            target_turn = 0.0
        elif state == "SEARCHING_LINE":
            mode = "n"
            arrow = "~"
            target_turn = 0.0
            current_turn = 0.0
        else:
            if current_turn > 8.0:
                arrow = "<"
                target_turn = abs(current_turn)
            elif current_turn < -8.0:
                arrow = ">"
                target_turn = abs(current_turn)
            else:
                arrow = "^"
                target_turn = 0.0

        self._steering_current_deg = current_turn
        self._steering_target_deg = target_turn
        self.steering_panel.update_command(
            mode=mode,
            current_angle_deg=current_turn,
            target_angle_deg=target_turn,
            arrow_symbol=arrow,
        )

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self._render_timer.stop()
        self._mock_source.stop()
        self.camera_stop_signal.emit()
        try:
            self._camera_thread.quit()
            self._camera_thread.wait(1200)
        except Exception:
            pass
        for sub in self._subscriptions:
            try:
                sub.unsubscribe()
            except Exception:
                pass
        super().closeEvent(event)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FusionZero PyQt6 dashboard")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index (for example 0 or 1)")
    parser.add_argument("--camera-width", type=int, default=640, help="Capture width")
    parser.add_argument("--camera-height", type=int, default=480, help="Capture height")
    parser.add_argument("--camera-fps", type=int, default=30, help="Capture fps")
    parser.add_argument("--mock", action="store_true", help="Start in mock mode")
    parser.add_argument(
        "--config",
        type=str,
        default="New_AI/obr_overengineering_v1/configs/vision_config.json",
        help="Vision config path used to load dashboard profiles",
    )
    return parser


def run_dashboard(
    *,
    event_bus: EventBus | None = None,
    camera_index: int = 0,
    camera_width: int = 640,
    camera_height: int = 480,
    camera_fps: int = 30,
    mock: bool = False,
    config_path: str | Path | None = None,
) -> int:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)
    window = DashboardWindow(
        event_bus=event_bus,
        camera_index=camera_index,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fps=camera_fps,
        start_mock=mock,
        config_path=config_path,
    )
    window.show()
    if owns_app:
        return app.exec()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_dashboard(
        camera_index=args.camera_index,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        mock=args.mock,
        config_path=args.config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
