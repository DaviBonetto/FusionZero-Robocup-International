from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from src.core.event_bus import EventBus, EventTopic, FrameEvent, HealthEvent, StateSnapshotEvent, UICommandEvent, VisionDetectionEvent
from src.ui_overengineering.dashboard import DashboardWindow


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _spin_events(app: QApplication, duration_s: float = 0.4) -> None:
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)


def _wait_until(app: QApplication, condition, timeout_s: float = 1.5) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return bool(condition())


def test_ui_handles_continuous_event_stream_without_freeze(qapp) -> None:
    bus = EventBus(max_queue_size=4096, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=False)
    frame = np.full((120, 160, 3), 180, dtype=np.uint8)

    try:
        for idx in range(240):
            ts = time.time()
            payload = frame.tobytes()
            bus.publish(
                EventTopic.VISION_RAW_FRAME,
                FrameEvent(timestamp=ts, frame_id=idx, width=160, height=120, encoding="bgr8", data=payload),
            )
            bus.publish(
                EventTopic.VISION_PROCESSED_FRAME,
                FrameEvent(timestamp=ts, frame_id=idx, width=160, height=120, encoding="bgr8", data=payload),
            )
            bus.publish(
                EventTopic.VISION_DETECTIONS,
                VisionDetectionEvent(
                    timestamp=ts,
                    state="FOLLOWING_LINE",
                    line=True,
                    balls=0,
                    green=(idx % 3 == 0),
                    red=(idx % 9 == 0),
                    victims=0,
                    latency_ms=16.0,
                    metadata={"telemetry": {"front": 600, "left": 500, "right": 450, "back": 700}},
                ),
            )
            bus.publish(
                EventTopic.SYSTEM_HEALTH,
                HealthEvent(
                    timestamp=ts,
                    cpu_percent=55.0,
                    fps_capture=30.0,
                    fps_process=62.0,
                    fps_ui=30.0,
                    queue_depth=min(idx, 20),
                ),
            )
            if idx % 25 == 0:
                bus.publish(EventTopic.FSM_STATE, StateSnapshotEvent(timestamp=ts, state="FOLLOWING_LINE"))
            qapp.processEvents()

        ts = time.time()
        bus.publish(EventTopic.FSM_STATE, StateSnapshotEvent(timestamp=ts, state="FOLLOWING_LINE"))
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                timestamp=ts,
                state="FOLLOWING_LINE",
                line=True,
                balls=0,
                green=True,
                red=False,
                victims=0,
                latency_ms=12.0,
                metadata={"telemetry": {"front": 600, "left": 500, "right": 450, "back": 700}},
            ),
        )
        _spin_events(qapp, 0.5)
        assert window._render_timer.isActive()
        assert window._fps_ui > 0.0
        assert _wait_until(qapp, lambda: window.status_panel.state_label.text() == "FOLLOWING_LINE", timeout_s=1.0)
        assert window.status_panel.state_label.text() == "FOLLOWING_LINE"
        assert window.processed_view.overlay_label.text().strip() != ""
    finally:
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_fallback_enables_mock_mode_without_camera(qapp) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=False)
    try:
        # Avoid real camera open in test environment while still exercising fallback branch.
        window._event_bus = None
        window._on_camera_status("camera 0 not available")
        _spin_events(qapp, 0.1)
        assert window._mock_enabled is True
        assert window.tuning_panel.mock_checkbox.isChecked() is True
    finally:
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_steering_panel_maps_green_commands(qapp) -> None:
    bus = EventBus(max_queue_size=128, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    try:
        base = {
            "timestamp": time.time(),
            "state": "FOLLOWING_LINE",
            "line": True,
            "balls": 0,
            "green": True,
            "red": False,
            "victims": 0,
            "latency_ms": 12.0,
        }

        window._on_detection_event(
            VisionDetectionEvent(
                **base,
                metadata={"line_angle_deg": 90, "green_side": "LEFT", "green_instruction": "VERDE DEPOIS"},
            )
        )
        assert window.steering_panel.arrow_label.text() == "<"
        assert window.steering_panel.target_label.text() == "90 deg"

        window._on_detection_event(
            VisionDetectionEvent(
                **base,
                metadata={"line_angle_deg": 90, "green_side": "RIGHT", "green_instruction": "VERDE DEPOIS"},
            )
        )
        assert window.steering_panel.arrow_label.text() == ">"
        assert window.steering_panel.target_label.text() == "90 deg"

        window._on_detection_event(
            VisionDetectionEvent(
                **base,
                metadata={"line_angle_deg": 90, "green_side": "BOTH", "green_instruction": "VERDE MEIA VOLTA"},
            )
        )
        assert window.steering_panel.arrow_label.text() == "UT"
        assert window.steering_panel.target_label.text() == "180 deg"

        last = dict(base)
        last["green"] = False
        window._on_detection_event(
            VisionDetectionEvent(
                **last,
                metadata={"line_angle_deg": 90, "green_side": "NONE", "green_instruction": "NO GREEN"},
            )
        )
        assert window.steering_panel.arrow_label.text() == "^"
        assert window.steering_panel.target_label.text() == "0 deg"
    finally:
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_corner_badges_use_hysteresis_window(qapp) -> None:
    bus = EventBus(max_queue_size=128, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    base = {
        "timestamp": time.time(),
        "state": "FOLLOWING_LINE",
        "line": True,
        "balls": 0,
        "green": False,
        "red": False,
        "victims": 0,
        "latency_ms": 10.0,
    }
    try:
        for idx in range(5):
            detect_green = idx in (0, 2, 4)
            window._on_detection_event(
                VisionDetectionEvent(
                    **base,
                    metadata={
                        "green_corner_found": detect_green,
                        "green_corner_confidence": 0.92 if detect_green else 0.05,
                        "runtime": {
                            "corner_stability_window": 5,
                            "corner_on_votes": 3,
                            "corner_off_votes": 1,
                        },
                    },
                )
            )
        assert window._status_flags["GREEN CORNER"] is True

        for idx in range(5):
            detect_green = idx == 0
            window._on_detection_event(
                VisionDetectionEvent(
                    **base,
                    metadata={
                        "green_corner_found": detect_green,
                        "green_corner_confidence": 0.40 if detect_green else 0.0,
                    },
                )
            )
        assert window._status_flags["GREEN CORNER"] is False
    finally:
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_mode_switch_publishes_force_mode_command(qapp) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    received: list[object] = []
    subscription = bus.subscribe(EventTopic.UI_COMMAND, lambda event: received.append(event))
    try:
        window._on_mode_switch_requested("rescue")
        assert _wait_until(qapp, lambda: any(getattr(evt, "command", "") == "fsm.force_mode" for evt in received), timeout_s=1.0)
        force_events = [evt for evt in received if getattr(evt, "command", "") == "fsm.force_mode"]
        assert force_events
        assert force_events[-1].params.get("mode") == "rescue"
    finally:
        subscription.unsubscribe()
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_updates_ops_health_and_profile_controls(qapp) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    try:
        window._on_health_event(
            HealthEvent(
                timestamp=time.time(),
                cpu_percent=48.0,
                fps_capture=20.0,
                fps_process=18.0,
                fps_ui=18.0,
                queue_depth=4,
                metadata={
                    "memory_percent": 41.0,
                    "network_latency_ms": 12.5,
                    "network": {"state": "connected", "peer": "127.0.0.1:8765", "latency_ms": 12.5},
                    "camera": {"state": "online", "index": 0, "width": 640, "height": 480, "fps": 20, "backend": "v4l2"},
                    "serial": {
                        "state": "connected",
                        "port": "/dev/i2c-1",
                        "pid_kp_us": 275,
                        "pid_ki_us": 0,
                        "pid_kd_us": 14,
                        "pid_integral_limit": 0.15,
                        "pid_derivative_filter": 0.6,
                        "max_output_us": 240,
                        "line_hold_ms": 90,
                    },
                    "power": {"available": True, "status": "warn", "summary": "power issue detected earlier", "raw_value": "0x50000"},
                    "profiles": {
                        "active": "pi3_field",
                        "description": "Preset de campo",
                        "available": [
                            {"name": "lab_pc", "description": "Lab"},
                            {"name": "pi3_field", "description": "Preset de campo"},
                        ],
                    },
                    "recording": {
                        "enabled": True,
                        "event_count": 12,
                        "frame_count": 3,
                        "options": {"include_raw": False, "include_processed": True, "every_n_frames": 10},
                    },
                },
            )
        )
        _spin_events(qapp, 0.1)
        window._render_tick()
        assert window.health_panel._value_labels["camera"].text() == "ONLINE"
        assert window.health_panel._value_labels["power"].text() == "WARN"
        assert "pi3_field" in window.tuning_panel.profile_status_label.text().lower()
        assert "REC ON" in window.tuning_panel.recording_status_label.text()
        assert window.top_bar._metrics["MEM"].value_label.text() == "41.0%"
        assert window.tuning_panel.value("control.pid.kp_us") == 275.0
        assert window.tuning_panel.value("control.pid.kd_us") == 14.0
        assert window.tuning_panel.value("control.line_hold_ms") == 90
    finally:
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_profile_recording_and_calibration_commands_publish(qapp) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    received: list[UICommandEvent] = []
    subscription = bus.subscribe(EventTopic.UI_COMMAND, lambda event: received.append(event))  # type: ignore[arg-type]
    try:
        window._on_profile_apply_requested("pi3_field")
        window._on_recording_command_requested("configure", {"include_processed": True, "every_n_frames": 8})
        window._on_recording_command_requested("start", {"include_processed": True, "every_n_frames": 8})
        window._on_recording_command_requested("stop", {})
        window._on_calibration_snapshot_requested({"view_mode": "line_mask", "freeze": True})
        assert _wait_until(qapp, lambda: len(received) >= 5, timeout_s=1.0)
        commands = [event.command for event in received]
        assert "config.load_profile" in commands
        assert "session.recording.configure" in commands
        assert "session.recording.start" in commands
        assert "session.recording.stop" in commands
        assert "calibration.snapshot" in commands
    finally:
        subscription.unsubscribe()
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_recording_status_keeps_last_session_path_and_buttons(qapp, tmp_path: Path) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    session_dir = tmp_path / "session_001"
    session_dir.mkdir(parents=True)
    try:
        window.tuning_panel.update_recording_status(
            {
                "enabled": True,
                "session_dir": str(session_dir),
                "event_count": 5,
                "frame_count": 2,
                "dropped_records": 1,
                "options": {"include_raw": False, "include_processed": True, "every_n_frames": 12},
            }
        )
        assert window.tuning_panel.recording_stop_button.isEnabled() is True
        assert window.tuning_panel.recording_open_button.isEnabled() is True
        assert session_dir.name not in window.tuning_panel.recording_status_label.text().lower()
        assert session_dir.name in window.tuning_panel.recording_path_label.text()
        assert str(session_dir) != window.tuning_panel.recording_path_label.text()
        assert window.tuning_panel.recording_path_label.toolTip() == str(session_dir)

        window.tuning_panel.update_recording_status(
            {
                "enabled": False,
                "session_dir": str(session_dir),
                "event_count": 5,
                "frame_count": 2,
                "options": {"include_raw": False, "include_processed": True, "every_n_frames": 12},
            }
        )
        assert "last session available" in window.tuning_panel.recording_status_label.text().lower()
        assert window.tuning_panel.recording_start_button.isEnabled() is True
        assert window.tuning_panel.recording_stop_button.isEnabled() is False
        assert window.tuning_panel.recording_open_button.isEnabled() is True
    finally:
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_capture_controls_are_explicit_and_ready_for_four_samples_per_second(qapp) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    try:
        panel = window.tuning_panel
        assert panel.recording_start_button.text() == "INICIAR CAPTURA"
        assert panel.recording_stop_button.text() == "PARAR CAPTURA"
        assert panel.recording_start_button.objectName() == "CaptureStartButton"
        assert panel.recording_stop_button.objectName() == "CaptureStopButton"
        options = panel.current_recording_options()
        assert options["include_raw"] is True
        assert options["include_processed"] is True
        assert options["every_n_frames"] == 5
        primary_options = window.primary_controls.current_recording_options()
        assert primary_options["include_raw"] is True
        assert primary_options["include_processed"] is True
        assert primary_options["every_n_frames"] == 5
    finally:
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_recording_panel_does_not_overflow_sidebar_width(qapp, tmp_path: Path) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    session_dir = (
        tmp_path
        / "FusionZero-Robocup-International"
        / "New_AI"
        / "obr_overengineering_v1"
        / "artifacts"
        / "session_recordings"
        / "session_20260310_131658_894"
    )
    session_dir.mkdir(parents=True)
    try:
        window.resize(1400, 900)
        window.show()
        _spin_events(qapp, 0.1)
        window.tuning_panel.update_recording_status(
            {
                "enabled": True,
                "session_dir": str(session_dir),
                "event_count": 7,
                "frame_count": 4,
                "options": {"include_raw": False, "include_processed": True, "every_n_frames": 20},
            }
        )
        _spin_events(qapp, 0.1)
        scroll = window.tuning_panel.scroll_area
        assert scroll.horizontalScrollBar().maximum() == 0
        assert scroll.widget() is not None
        assert scroll.widget().width() == scroll.viewport().width()
        assert "session_recordings" in window.tuning_panel.recording_path_label.text()
        assert window.tuning_panel.recording_path_label.text() != str(session_dir)
    finally:
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_ops_health_compacts_long_detail_text(qapp) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    try:
        rows = window._build_health_rows(
            {
                "network": {
                    "state": "connected",
                    "peer": "192.168.0.155:8765/dashboard-relay/with-extra-debug-info",
                    "latency_ms": 4.2,
                },
                "profiles": {
                    "active": "pi3_field",
                    "description": "Preset de campo para Raspberry Pi 3 com runner headless, dashboard remoto e watchdog serial estendido",
                },
                "recording": {
                    "enabled": True,
                    "event_count": 3,
                    "frame_count": 1,
                    "session_dir": (
                        "C:/Users/Davib/OneDrive/Area de Trabalho/OBR - Arquivos/"
                        "FusionZero-Robocup-International/New_AI/obr_overengineering_v1/"
                        "artifacts/session_recordings/session_20260310_131658_894"
                    ),
                    "options": {"every_n_frames": 20},
                },
            }
        )
        assert rows["profile"]["detail"].endswith("...")
        assert len(rows["link"]["detail"]) <= 28
        assert "session_20260310_131658_894" in rows["recording"]["detail"]
        assert "FusionZero-Robocup-International" not in rows["recording"]["detail"]

        window.health_panel.update_rows(rows)
        _spin_events(qapp, 0.1)
        assert window.health_panel._detail_labels["profile"].wordWrap() is False
        assert window.health_panel._detail_labels["profile"].toolTip() == rows["profile"]["detail"]
        assert window.health_panel._detail_labels["recording"].wordWrap() is False
    finally:
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_ops_health_does_not_clip_when_maximized(qapp) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    try:
        window._health_metadata = {
            "camera": {"state": "online", "index": 0, "width": 640, "height": 480, "fps": 30, "backend": "usb"},
            "serial": {"state": "dry-run", "port": "COM5"},
            "network": {
                "state": "connected",
                "latency_ms": 3.4,
                "peer": "192.168.0.55:8765/dashboard-relay/with-extra-debug-info",
            },
            "power": {"available": False, "status": "neutral", "summary": "power unavailable", "raw_value": ""},
            "profiles": {
                "active": "pi3_field",
                "description": "Preset de campo para Raspberry Pi 3 com runner headless e dashboard remoto",
                "available": [{"name": "pi3_field", "description": "Preset de campo para Raspberry Pi 3 com runner headless e dashboard remoto"}],
            },
            "recording": {
                "enabled": True,
                "event_count": 7,
                "frame_count": 4,
                "session_dir": (
                    "C:/Users/Davib/OneDrive/Area de Trabalho/OBR - Arquivos/"
                    "FusionZero-Robocup-International/New_AI/obr_overengineering_v1/"
                    "artifacts/session_recordings/session_20260310_131658_894"
                ),
                "options": {"include_raw": False, "include_processed": True, "every_n_frames": 20},
            },
        }
        window._update_ops_from_health()
        window.showMaximized()
        _spin_events(qapp, 0.1)
        assert window.health_panel.height() >= 220
        assert window.health_panel._scroll_area.verticalScrollBar().maximum() == 0
        assert window.health_panel._detail_labels["recording"].geometry().height() >= 16
    finally:
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_clean_workspace_hides_legacy_panels_and_collapses_advanced(qapp) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    try:
        window.show()
        _spin_events(qapp, 0.1)
        assert window.primary_controls.isVisible() is True
        assert window.telemetry_panel.isVisible() is False
        assert window.timer_panel.isVisible() is False
        assert window.robot_panel.isVisible() is False
        assert window.steering_panel.isVisible() is False
        assert window.tuning_panel.recording_group.isVisible() is False
        assert window.tuning_panel.robot_group.isVisible() is False

        window._toggle_tuning_panel()
        _spin_events(qapp, 0.05)
        assert window.tuning_panel.isVisible() is False
        assert window.tuning_toggle_button.text() == "MOSTRAR AJUSTES"

        window._toggle_tuning_panel()
        _spin_events(qapp, 0.05)
        assert window.tuning_panel.isVisible() is True
        assert window.tuning_toggle_button.text() == "AJUSTES"
    finally:
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_disconnected_camera_and_full_robot_runtime_are_explicit(qapp) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=False)
    try:
        window._render_tick()
        assert window.raw_view.video_label.text() == "No camera connected"
        assert window.processed_view.video_label.text() == "No camera connected"
        assert window.primary_controls.maximumHeight() > 1000

        window.primary_controls.update_robot_status(
            {
                "state": "connected",
                "motor_armed": False,
                "control_mode": "ESTOP",
                "green_instruction": "VERDE_ANTES_ESQUERDA",
                "green_route_decision": "LEFT",
                "line_error": -0.12,
                "pid_output": -26.6,
                "left_pwm": 1600,
                "right_pwm": 1600,
            }
        )
        detail = window.primary_controls.robot_detail_label.text()
        assert "DISARMED | ESTOP" in detail
        assert "G VERDE_ANTES_ESQUERDA" in detail
        assert "ROUTE LEFT" in detail
        assert "E -0.12" in detail
        assert "PID -26.6" in detail
        assert "PWM L 1600 | R 1600 us" in detail
    finally:
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_robot_commands_publish_expected_payloads(qapp) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    received: list[UICommandEvent] = []
    subscription = bus.subscribe(EventTopic.UI_COMMAND, lambda event: received.append(event))  # type: ignore[arg-type]
    try:
        window.tuning_panel.robot_forward_ms_spin.setValue(900)
        window.tuning_panel._emit_robot_start()
        window.tuning_panel._emit_robot_forward_test()
        window.tuning_panel._emit_robot_stop()
        window.tuning_panel._emit_robot_force_stop()
        window.tuning_panel._emit_robot_clear_estop()
        window.tuning_panel._emit_robot_obstacle_test()
        window.tuning_panel._emit_robot_obstacle_clear()
        assert _wait_until(qapp, lambda: len(received) >= 7, timeout_s=1.0)
        commands = [(event.command, dict(event.params)) for event in received]
        assert window.tuning_panel.robot_start_button.text() == "START MOTORES"
        assert "#b91c1c" in window.tuning_panel.robot_start_button.styleSheet()
        assert ("robot.start", {}) in commands
        assert ("robot.forward_test", {"duration_ms": 900}) in commands
        assert ("robot.stop", {}) in commands
        assert ("robot.force_stop", {}) in commands
        assert ("robot.clear_estop", {}) in commands
        assert ("robot.obstacle_test", {}) in commands
        assert ("robot.obstacle_clear", {}) in commands
    finally:
        subscription.unsubscribe()
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_primary_led_controls_publish_expected_commands(qapp) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    received: list[UICommandEvent] = []
    subscription = bus.subscribe(EventTopic.UI_COMMAND, lambda event: received.append(event))  # type: ignore[arg-type]
    try:
        window.primary_controls.leds_on_button.click()
        window.primary_controls.leds_off_button.click()

        assert _wait_until(qapp, lambda: len(received) >= 2, timeout_s=1.0)
        commands = [(event.command, dict(event.params)) for event in received]
        assert ("leds.on", {}) in commands
        assert ("leds.off", {}) in commands
    finally:
        subscription.unsubscribe()
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_primary_master_controls_publish_system_commands(qapp) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    received: list[UICommandEvent] = []
    subscription = bus.subscribe(EventTopic.UI_COMMAND, lambda event: received.append(event))  # type: ignore[arg-type]
    try:
        assert window.primary_controls.robot_start_button.text() == "START ROBÔ"
        window.primary_controls.robot_start_button.click()
        window.primary_controls.robot_stop_button.click()

        assert _wait_until(qapp, lambda: len(received) >= 2, timeout_s=1.0)
        commands = [(event.command, dict(event.params)) for event in received]
        assert ("system.start", {}) in commands
        assert ("system.stop", {}) in commands
    finally:
        subscription.unsubscribe()
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_zoom_controls_scale_dashboard_with_safe_limits(qapp) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    try:
        assert window._ui_zoom_percent == 100
        assert window.primary_controls.zoom_value_label.text() == "100%"

        window.primary_controls.zoom_out_button.click()
        _spin_events(qapp, 0.05)
        assert window._ui_zoom_percent == 90
        assert window.primary_controls.zoom_value_label.text() == "90%"
        assert window.raw_view.video_label.minimumHeight() == 162
        assert window.status_panel.minimumHeight() == 198

        window._apply_ui_zoom(1)
        assert window._ui_zoom_percent == 80
        assert window.primary_controls.zoom_out_button.isEnabled() is False
        assert window.primary_controls.zoom_in_button.isEnabled() is True

        window._apply_ui_zoom(999)
        assert window._ui_zoom_percent == 120
        assert window.primary_controls.zoom_out_button.isEnabled() is True
        assert window.primary_controls.zoom_in_button.isEnabled() is False
    finally:
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_left_corner_timing_button_waits_for_pi_confirmation(qapp) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    received: list[UICommandEvent] = []
    subscription = bus.subscribe(EventTopic.UI_COMMAND, lambda event: received.append(event))  # type: ignore[arg-type]
    try:
        window.tuning_panel.corner_advance_ms_spin.setValue(550)
        window.tuning_panel.corner_pivot_ms_spin.setValue(1900)
        window.tuning_panel._emit_corner_timing_apply()
        assert _wait_until(qapp, lambda: len(received) >= 1, timeout_s=1.0)
        commands = [(event.command, dict(event.params)) for event in received]
        assert (
            "control.left_corner_timing.apply",
            {"approach_min_ms": 550, "pivot_ms": 1900},
        ) in commands
        assert "Enviando" in window.tuning_panel.corner_timing_status_label.text()

        window.tuning_panel.update_corner_timing_status(
            {
                "corner_approach_left_min_ms": 500,
                "corner_pivot_right_ms": 1900,
                "corner_pivot_left_ms": 1900,
            }
        )
        assert "Aguardando confirmacao" in window.tuning_panel.corner_timing_status_label.text()

        window.tuning_panel.update_corner_timing_status(
            {
                "corner_approach_left_min_ms": 550,
                "corner_pivot_right_ms": 1900,
                "corner_pivot_left_ms": 1900,
            }
        )
        assert "APLICADO" in window.tuning_panel.corner_timing_status_label.text()
    finally:
        subscription.unsubscribe()
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_green_half_turn_button_waits_for_pi_confirmation(qapp) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    received: list[UICommandEvent] = []
    subscription = bus.subscribe(EventTopic.UI_COMMAND, lambda event: received.append(event))  # type: ignore[arg-type]
    try:
        window.tuning_panel.green_half_turn_first_ms_spin.setValue(1900)
        window.tuning_panel.green_half_turn_reverse_ms_spin.setValue(200)
        window.tuning_panel.green_half_turn_second_ms_spin.setValue(2200)
        window.tuning_panel._emit_green_half_turn_apply()
        assert _wait_until(qapp, lambda: len(received) >= 1, timeout_s=1.0)
        commands = [(event.command, dict(event.params)) for event in received]
        assert (
            "control.green_half_turn.apply",
            {"first_ms": 1900, "reverse_ms": 200, "second_ms": 2200},
        ) in commands
        assert "Enviando" in window.tuning_panel.green_half_turn_status_label.text()

        window.tuning_panel.update_green_half_turn_status(
            {"green_half_turn_ms": 4000}
        )
        assert "Aguardando confirmacao" in window.tuning_panel.green_half_turn_status_label.text()

        window.tuning_panel.update_green_half_turn_status(
            {
                "green_half_turn_ms": 4100,
                "green_half_turn_left_us": 300,
                "green_half_turn_right_us": 300,
                "green_half_turn_first_ms": 1900,
                "green_half_turn_second_ms": 2200,
                "green_half_turn_reverse_ms": 200,
            }
        )
        assert "APLICADO NO PI" in window.tuning_panel.green_half_turn_status_label.text()
        assert "pivo L 300 / R 300 us" in window.tuning_panel.green_half_turn_status_label.text()
        applied_text = window.tuning_panel.green_half_turn_status_label.text()
        assert "giro 1 1900 ms" in applied_text
        assert "re 200 ms" in applied_text
        assert "giro 2 2200 ms" in applied_text
    finally:
        subscription.unsubscribe()
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_calibration_preview_supports_freeze(qapp) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    frame = np.full((80, 120, 3), 160, dtype=np.uint8)
    try:
        window._store_raw_frame(frame)
        window._store_processed_frame(frame)
        window._on_calibration_changed({"view_mode": "line_mask", "freeze": False})
        window._render_tick()
        assert "LINE MASK" in window.raw_view.corner_label.text()

        window._on_calibration_changed({"view_mode": "line_mask", "freeze": True})
        window._render_tick()
        assert "FROZEN" in window.raw_view.corner_label.text()
    finally:
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_ui_overlay_surfaces_control_debug_fields(qapp) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    window = DashboardWindow(event_bus=bus, start_mock=True)
    try:
        window._on_detection_event(
            VisionDetectionEvent(
                timestamp=time.time(),
                state="FOLLOWING_LINE",
                line=True,
                balls=0,
                green=False,
                red=False,
                victims=0,
                latency_ms=11.5,
                metadata={
                    "status_text": "... Following Line ...",
                    "line_angle_deg": 93,
                    "control": {
                        "control_mode": "FOLLOW_LINE",
                        "vision_confidence": 0.82,
                        "line_error": -0.08,
                        "pid_output": 14.5,
                        "obstacle_state": "CLEAR",
                        "green_instruction": "NO_GREEN",
                        "failsafe": False,
                    },
                },
            )
        )
        assert "FOLLOW_LINE" in window._processed_overlay_text
        assert "VC 82%" in window._processed_overlay_text
        assert "PID +14.5" in window._processed_overlay_text
    finally:
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()
