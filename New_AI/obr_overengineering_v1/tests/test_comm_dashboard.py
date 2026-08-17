from __future__ import annotations

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from src.core.event_bus import EventBus, EventTopic, FrameEvent, HealthEvent, LogEvent, UICommandEvent, VisionDetectionEvent
from src.ui_overengineering.comm_dashboard import CommDashboardWindow


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


def test_comm_dashboard_updates_remote_status_video_and_badges(qapp) -> None:
    bus = EventBus(max_queue_size=256, drop_oldest=False)
    window = CommDashboardWindow(event_bus=bus)
    frame = np.full((120, 160, 3), 180, dtype=np.uint8)

    try:
        bus.publish(
            EventTopic.SYSTEM_HEALTH,
            HealthEvent(
                timestamp=time.time(),
                cpu_percent=35.0,
                fps_capture=20.0,
                fps_process=18.0,
                fps_ui=18.0,
                queue_depth=2,
                metadata={
                    "network": {
                        "state": "connected",
                        "peer": "192.168.0.55:8765",
                        "latency_ms": 22.5,
                    },
                    "serial": {
                        "state": "connected",
                        "port": "/dev/ttyACM0",
                        "heartbeat_ok": True,
                        "heartbeat_age_ms": 120,
                        "assist_kind": "line",
                        "control_mode": "GREEN",
                        "line_error": -0.12,
                        "pid_output": 14.4,
                        "green_instruction": "VERDE_MEIA_VOLTA",
                        "obstacle_state": "CLEAR",
                        "failsafe": True,
                    },
                },
            ),
        )
        bus.publish(
            EventTopic.VISION_PROCESSED_FRAME,
            FrameEvent(
                timestamp=time.time(),
                frame_id=8,
                width=160,
                height=120,
                encoding="bgr8",
                data=frame.tobytes(),
            ),
        )
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                timestamp=time.time(),
                state="FOLLOWING_LINE",
                line=True,
                balls=0,
                green=True,
                red=False,
                victims=0,
                latency_ms=12.4,
                metadata={
                    "line_offset_norm": -0.21,
                    "line_angle_deg": 104.0,
                    "green_instruction": "VERDE MEIA VOLTA",
                    "green_side": "BOTH",
                    "assist_kind": "line",
                    "control_mode": "GREEN",
                    "line_error": -0.12,
                    "pid_output": 14.4,
                    "obstacle_state": "CLEAR",
                    "failsafe": True,
                },
            ),
        )
        bus.publish(
            EventTopic.SYSTEM_LOG,
            LogEvent(
                timestamp=time.time(),
                level="INFO",
                message="remote dashboard session ready session=4",
                source="remote_client",
                state="FOLLOWING_LINE",
            ),
        )

        assert _wait_until(qapp, lambda: window.vision_values["green_instruction"].text() == "VERDE_MEIA_VOLTA")
        assert "CONNECTED" in window.connection_values["pi_relay"].text().upper()
        assert "/DEV/TTYACM0" in window.connection_values["arduino_serial"].text().upper()
        assert "OK" in window.connection_values["heartbeat"].text().upper()
        assert window.vision_values["line"].text() == "YES"
        assert window.vision_values["green"].text() == "YES"
        assert window.vision_values["line_offset_norm"].text() == "-0.210"
        assert window.vision_values["line_angle_deg"].text() == "104.0"
        assert window.telemetry_values["mode"].text() == "GREEN"
        assert window.telemetry_values["assist_kind"].text() == "LINE"
        assert window.telemetry_values["failsafe"].text() == "ACTIVE"
        assert window.scenario_badges["line_detected"].property("active") is True
        assert window.scenario_badges["line_assist"].property("active") is True
        assert window.scenario_badges["green_half_turn"].property("active") is True
        assert window.scenario_badges["arduino_green"].property("active") is True
        assert window.scenario_badges["failsafe"].property("active") is True
        assert window.video_view.overlay_label.text() == "FOLLOWING_LINE"
        assert window.log_list.count() >= 1
        assert window.video_view.video_label.pixmap() is not None
    finally:
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_comm_dashboard_buttons_publish_expected_ui_commands(qapp) -> None:
    bus = EventBus(max_queue_size=128, drop_oldest=False)
    window = CommDashboardWindow(event_bus=bus)
    received: list[str] = []
    subscription = bus.subscribe(EventTopic.UI_COMMAND, lambda event: received.append(getattr(event, "command", "")))

    try:
        window.command_buttons["forward_test"].click()
        window.command_buttons["stop"].click()
        window.command_buttons["estop"].click()
        window.command_buttons["clear_estop"].click()
        window.command_buttons["obstacle_test"].click()
        window.command_buttons["clear_obstacle"].click()

        assert _wait_until(qapp, lambda: len(received) == 6, timeout_s=1.0)
        assert received == [
            "robot.forward_test",
            "robot.stop",
            "robot.force_stop",
            "robot.clear_estop",
            "robot.obstacle_test",
            "robot.clear_obstacle",
        ]
    finally:
        subscription.unsubscribe()
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()


def test_comm_dashboard_tracks_remote_disconnect_logs(qapp) -> None:
    bus = EventBus(max_queue_size=128, drop_oldest=False)
    window = CommDashboardWindow(event_bus=bus)

    try:
        bus.publish(
            EventTopic.SYSTEM_LOG,
            LogEvent(
                timestamp=time.time(),
                level="WARNING",
                message="remote dashboard disconnected from 192.168.0.55:8765",
                source="remote_client",
                state="",
            ),
        )

        assert _wait_until(qapp, lambda: "DISCONNECTED" in window.connection_values["pi_relay"].text().upper())
    finally:
        window.close()
        _spin_events(qapp, 0.1)
        bus.stop()
