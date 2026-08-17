from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.core.event_bus import EventTopic, HealthEvent, UICommandEvent, VisionDetectionEvent
from src.live_dashboard_runner import LiveDashboardRunner


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


class DummyVisionNode:
    def __init__(self, *args, **kwargs) -> None:
        self.fps = 0.0
        self.closed = False
        self._pipeline = type(
            "Pipeline",
            (),
            {
                "line_detector": type("Line", (), {"black_v_max": 70, "black_s_max": 255, "min_black_area": 50, "erode_iter": 3, "dilate_iter": 4})(),
                "color_detector": type(
                    "Color",
                    (),
                    {
                        "green_h_min": 35,
                        "green_h_max": 90,
                        "green_s_min": 70,
                        "green_v_min": 50,
                        "green_min_area": 180.0,
                        "red_h1_min": 0,
                        "red_h1_max": 12,
                        "red_h2_min": 165,
                        "red_h2_max": 179,
                        "red_s_min": 120,
                        "red_v_min": 80,
                        "red_min_area": 300.0,
                    },
                )(),
                "ball_detector": type("Ball", (), {"silver_blur": 7, "dead_black_threshold": (60, 60, 60)})(),
                "silver": type("Silver", (), {"threshold": 0.95})(),
            },
        )()

    def process_frame(self, frame, *, frame_id: int, timestamp: float) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class DummyRobotAdapter:
    def __init__(self, *args, **kwargs) -> None:
        self.config = kwargs.get("config")
        self.started = False
        self.stopped = False
        self.calls: list[tuple[str, str, str, bool]] = []
        self.line_control_updates: list[dict[str, float | int]] = []
        self.corner_timing_updates: list[dict[str, int]] = []
        self.left_corner_timing_updates: list[dict[str, int]] = []
        self.green_half_turn_updates: list[dict[str, int]] = []
        self.status = {
            "state": "connected",
            "connected": True,
            "port": "/dev/ttyACM0",
            "control_mode": "FOLLOW_LINE",
            "line_error": -0.08,
            "pid_output": 14.5,
            "obstacle_state": "CLEAR",
            "green_instruction": "NO_GREEN",
            "failsafe": False,
            "assist_kind": "line",
            "telemetry": {
                "front": 320,
                "left": 450,
                "right": 420,
                "back": 900,
                "yaw": 2.1,
                "roll": 0.4,
                "pitch": -0.2,
                "gripper": 90,
            },
        }

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def request_safe_stop(self, *, reason: str, state: str, emergency: bool) -> None:
        self.calls.append(("request_safe_stop", reason, state, emergency))

    def send_stop(self, *, reason: str, state: str) -> None:
        self.calls.append(("send_stop", reason, state, False))

    def send_estop(self, *, reason: str, state: str) -> None:
        self.calls.append(("send_estop", reason, state, True))

    def status_payload(self) -> dict:
        return dict(self.status)

    def update_line_control(self, **values):
        self.line_control_updates.append(dict(values))
        return dict(values)

    def update_corner_timing(self, **values):
        update = {key: int(value) for key, value in values.items()}
        self.corner_timing_updates.append(update)
        pivot_ms = int(update.get("pivot_ms", 1900))
        return {
            "approach_min_ms": int(update.get("approach_min_ms", 350)),
            "pivot_right_ms": pivot_ms,
            "pivot_left_ms": pivot_ms,
        }

    def update_left_corner_timing(self, **values):
        update = {key: int(value) for key, value in values.items()}
        self.left_corner_timing_updates.append(update)
        return {
            "approach_left_min_ms": int(update.get("approach_min_ms", 550)),
            "pivot_left_ms": int(update.get("pivot_ms", 1900)),
        }

    def update_green_half_turn_timing(self, **values):
        update = {key: int(value) for key, value in values.items()}
        self.green_half_turn_updates.append(update)
        first_ms = int(update.get("first_ms", 1900))
        second_ms = int(update.get("second_ms", 2200))
        return {
            "green_half_turn_ms": first_ms + second_ms,
            "green_half_turn_first_ms": first_ms,
            "green_half_turn_second_ms": second_ms,
            "green_half_turn_reverse_ms": int(update.get("reverse_ms", 200)),
        }


class DummyPcaRobotAdapter(DummyRobotAdapter):
    pass


def _make_runner(monkeypatch, clock: FakeClock) -> LiveDashboardRunner:
    monkeypatch.setattr("src.live_dashboard_runner.VisionNode", DummyVisionNode)
    monkeypatch.setattr("src.live_dashboard_runner.SerialRobotAdapter", DummyRobotAdapter)
    return LiveDashboardRunner(
        camera_index=0,
        camera_width=640,
        camera_height=480,
        camera_fps=30,
        monotonic=clock,
    )


def test_runner_can_select_pca9685_robot_backend(monkeypatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr("src.live_dashboard_runner.VisionNode", DummyVisionNode)
    monkeypatch.setattr("src.live_dashboard_runner.SerialRobotAdapter", DummyRobotAdapter)
    monkeypatch.setattr("src.live_dashboard_runner.Pca9685RobotAdapter", DummyPcaRobotAdapter)
    runner = LiveDashboardRunner(
        camera_index=0,
        camera_width=640,
        camera_height=480,
        camera_fps=30,
        robot_backend="pca9685",
        robot_dry_run=True,
        line_only=True,
        pca9685_green_half_turn_us=300,
        pca9685_green_half_turn_ms=3800,
        monotonic=clock,
    )
    try:
        assert isinstance(runner.robot, DummyPcaRobotAdapter)
        assert runner.robot.config.enable_green_maneuvers is True
        assert runner.robot.config.green_half_turn_us == 300
        assert runner.robot.config.green_half_turn_ms == 3800
    finally:
        runner.bus.stop()


def test_runner_master_start_bridges_leds_and_authorized_motor_start(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    received: list[UICommandEvent] = []
    subscription = runner.bus.subscribe(EventTopic.UI_COMMAND, lambda event: received.append(event))  # type: ignore[arg-type]
    try:
        runner._on_ui_command(UICommandEvent(command="system.start", params={}))
        runner.bus._queue.join()

        commands = [event.command for event in received]
        assert "leds.on" in commands
        assert "robot.start" in commands
        robot_start = next(event for event in received if event.command == "robot.start")
        assert robot_start.params["master_authorized"] is True
    finally:
        subscription.unsubscribe()
        runner.bus.stop()


def test_runner_master_interlock_blocks_start_while_switch_open(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    runner._master_switch_required = True
    runner.master_switch = type("OpenMasterSwitch", (), {"start_permitted": False})()
    received: list[UICommandEvent] = []
    subscription = runner.bus.subscribe(EventTopic.UI_COMMAND, lambda event: received.append(event))  # type: ignore[arg-type]
    try:
        runner._on_ui_command(UICommandEvent(command="system.start", params={}))
        runner.bus._queue.join()

        commands = [event.command for event in received]
        assert "robot.start" not in commands
        assert "robot.stop" in commands
        assert "leds.off" in commands
    finally:
        subscription.unsubscribe()
        runner.bus.stop()


def test_runner_rotates_sideways_usb_camera_before_vision(monkeypatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr("src.live_dashboard_runner.VisionNode", DummyVisionNode)
    monkeypatch.setattr("src.live_dashboard_runner.SerialRobotAdapter", DummyRobotAdapter)
    runner = LiveDashboardRunner(
        camera_index=0,
        camera_width=640,
        camera_height=480,
        camera_fps=30,
        camera_backend="opencv",
        camera_rotation=90,
        monotonic=clock,
    )
    try:
        frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((2, 3, 3))
        oriented = runner._orient_camera_frame(frame)

        assert np.array_equal(oriented, cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))
        assert oriented.shape == (3, 2, 3)
        assert runner._camera_status_payload()["rotation"] == 90
        assert runner._session_context()["camera"]["rotation"] == 90
    finally:
        runner.bus.stop()


def test_runner_rejects_invalid_camera_rotation(monkeypatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr("src.live_dashboard_runner.VisionNode", DummyVisionNode)
    monkeypatch.setattr("src.live_dashboard_runner.SerialRobotAdapter", DummyRobotAdapter)

    try:
        LiveDashboardRunner(
            camera_index=0,
            camera_width=640,
            camera_height=480,
            camera_fps=30,
            camera_rotation=45,
            monotonic=clock,
        )
    except ValueError as exc:
        assert "camera_rotation" in str(exc)
    else:
        raise AssertionError("invalid camera rotation must be rejected")


def test_runner_line_only_rejects_rescue_force_mode(monkeypatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr("src.live_dashboard_runner.VisionNode", DummyVisionNode)
    monkeypatch.setattr("src.live_dashboard_runner.SerialRobotAdapter", DummyRobotAdapter)
    runner = LiveDashboardRunner(
        camera_index=0,
        camera_width=640,
        camera_height=480,
        camera_fps=30,
        line_only=True,
        monotonic=clock,
    )
    try:
        assert runner.fsm.state.value == "SEARCHING_LINE"
        runner._on_ui_command(UICommandEvent(command="fsm.force_mode", params={"mode": "rescue"}))
        assert runner.fsm.state.value == "SEARCHING_LINE"
        runner._on_detection(
            VisionDetectionEvent(
                state="SEARCHING_LINE",
                line=False,
                red=True,
                metadata={"red_corner": True},
            )
        )
        assert runner.fsm.state.value == "SEARCHING_LINE"
    finally:
        runner.bus.stop()


def test_runner_camera_reconnect_resets_runtime_state(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    opened: list[int] = []
    monkeypatch.setattr(runner, "_open_camera", lambda: opened.append(runner.camera_index))
    try:
        runner._frame_id = 42
        runner._fps_capture = 18.0
        runner._camera_fault_active = True
        runner._fsm_fault_active = True
        runner._camera_read_fail_streak = 5
        runner._line_lost_streak = 3
        clock.advance(2.0)

        runner._on_ui_command(UICommandEvent(command="camera.reconnect", params={"index": 1, "width": 320, "height": 240, "fps": 15}))

        assert opened == [1]
        assert runner.camera_index == 1
        assert runner.camera_width == 320
        assert runner.camera_height == 240
        assert runner.camera_fps == 15
        assert runner._frame_id == 0
        assert runner._fps_capture == 0.0
        assert runner._camera_fault_active is False
        assert runner._fsm_fault_active is False
        assert runner._camera_read_fail_streak == 0
        assert runner._line_lost_streak == 0
    finally:
        runner.bus.stop()


def test_runner_control_overlay_falls_back_to_vision_line_after_reconnect(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    try:
        runner.robot.status.update(
            {
                "control_mode": "STOPPED",
                "line_error": 0.0,
                "pid_output": 0.0,
            }
        )
        event = VisionDetectionEvent(
            state="FOLLOWING_LINE",
            line=True,
            metadata={"line_offset_norm": 0.35, "line_confidence": 0.82},
        )
        runner._inject_telemetry(event)

        assert event.metadata["control"]["line_error"] == 0.35
        assert event.metadata["control"]["pid_output"] == 77.0
    finally:
        runner.bus.stop()


def test_runner_camera_timeout_requests_safe_stop(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    try:
        runner._last_vision_ok_at = clock.value - 2.0
        runner._evaluate_safety()
        assert runner.robot.calls[-1] == ("request_safe_stop", "camera_timeout", "SEARCHING_LINE", False)
    finally:
        runner.bus.stop()


def test_runner_fsm_stall_requests_estop(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    try:
        runner._frame_id = 5
        runner._last_vision_ok_at = clock.value
        runner._last_detection_at = clock.value - 2.0
        runner._evaluate_safety()
        assert runner.robot.calls[-1] == ("request_safe_stop", "fsm_stall", "SEARCHING_LINE", True)
    finally:
        runner.bus.stop()


def test_runner_does_not_interrupt_bounded_green_half_turn(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    try:
        runner.robot.status.update(
            {
                "green_half_turn_phase": "PIVOT_1",
                "green_maneuver_remaining_ms": 4200,
                "motor_armed": True,
                "failsafe": False,
            }
        )
        runner._frame_id = 5
        runner._last_vision_ok_at = clock.value - 2.0
        runner._last_detection_at = clock.value - 2.0

        runner._evaluate_safety()

        assert runner.robot.calls == []
    finally:
        runner.bus.stop()


def test_runner_invalid_detection_state_requests_estop(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    try:
        runner._on_detection(
            VisionDetectionEvent(
                state="BROKEN_STATE",
                line=True,
                green=False,
                red=False,
                victims=0,
                metadata={},
            )
        )
        assert runner.robot.calls[-1] == ("request_safe_stop", "invalid_detection_state", "SEARCHING_LINE", True)
    finally:
        runner.bus.stop()


def test_runner_fsm_exception_requests_estop(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    try:
        def _boom(*args, **kwargs):
            raise RuntimeError("fsm exploded")

        runner.fsm.handle = _boom  # type: ignore[method-assign]
        runner._on_detection(
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=False,
                green=False,
                red=True,
                victims=0,
                metadata={},
            )
        )
        assert runner.robot.calls[-1] == ("request_safe_stop", "fsm_handle_failure", "SEARCHING_LINE", True)
    finally:
        runner.bus.stop()


def test_runner_stop_requests_safe_stop_before_robot_shutdown(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    runner.stop()
    assert runner.robot.calls[0] == ("request_safe_stop", "runner_shutdown", "SEARCHING_LINE", False)
    assert runner.robot.stopped is True


def test_runner_reopens_camera_after_read_failure_streak(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    released: list[str] = []
    monkeypatch.setattr(runner, "_release_camera", lambda: released.append("released"))
    try:
        for _ in range(runner._camera_read_fail_threshold - 1):
            assert runner._handle_capture_read_failure() is False
        assert released == []
        assert runner._handle_capture_read_failure() is True
        assert released == ["released"]
        assert runner._camera_read_fail_streak == 0
    finally:
        runner.bus.stop()


def test_runner_capture_success_clears_failure_streak(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    try:
        assert runner._handle_capture_read_failure() is False
        assert runner._camera_read_fail_streak == 1
        runner._handle_capture_read_success()
        assert runner._camera_read_fail_streak == 0
    finally:
        runner.bus.stop()


def test_runner_profile_command_applies_named_profile(monkeypatch, tmp_path: Path) -> None:
    clock = FakeClock()
    monkeypatch.setattr("src.live_dashboard_runner.VisionNode", DummyVisionNode)
    monkeypatch.setattr("src.live_dashboard_runner.SerialRobotAdapter", DummyRobotAdapter)
    runner = LiveDashboardRunner(
        camera_index=0,
        camera_width=640,
        camera_height=480,
        camera_fps=30,
        recordings_root=tmp_path,
        monotonic=clock,
    )
    try:
        runner._on_ui_command(UICommandEvent(command="config.load_profile", params={"profile": "pi3_field"}))
        assert runner.camera_fps == 20
        assert runner._active_profile_name == "pi3_field"
        assert runner.session_recorder.status_payload()["options"]["every_n_frames"] == 30
        assert runner.vision._pipeline.color_detector.green_s_min == 78
    finally:
        runner.stop()


def test_runner_session_recording_commands_create_session(monkeypatch, tmp_path: Path) -> None:
    clock = FakeClock()
    monkeypatch.setattr("src.live_dashboard_runner.VisionNode", DummyVisionNode)
    monkeypatch.setattr("src.live_dashboard_runner.SerialRobotAdapter", DummyRobotAdapter)
    runner = LiveDashboardRunner(
        camera_index=0,
        camera_width=640,
        camera_height=480,
        camera_fps=30,
        recordings_root=tmp_path,
        monotonic=clock,
    )
    try:
        runner._on_ui_command(
            UICommandEvent(
                command="session.recording.start",
                params={"include_processed": True, "include_raw": False, "every_n_frames": 5},
            )
        )
        status = runner.session_recorder.status_payload()
        assert status["enabled"] is True
        assert Path(status["session_dir"]).exists()

        runner._on_ui_command(UICommandEvent(command="session.recording.stop", params={}))
        assert runner.session_recorder.status_payload()["enabled"] is False
    finally:
        runner.stop()


def test_runner_health_event_includes_ops_metadata(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    received: list[HealthEvent] = []
    subscription = runner.bus.subscribe(EventTopic.SYSTEM_HEALTH, lambda event: received.append(event))  # type: ignore[arg-type]
    try:
        runner._publish_health_if_due()
        runner.bus._queue.join()
        assert received
        metadata = received[-1].metadata
        assert "camera" in metadata
        assert "serial" in metadata
        assert "profiles" in metadata
        assert metadata["network"]["state"] == "local"
    finally:
        subscription.unsubscribe()
        runner.stop()


def test_runner_applies_live_pid_tuning_command(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    try:
        runner._on_ui_command(
            UICommandEvent(
                command="tuning.update",
                params={"key": "control.pid.kp_us", "value": 275.0},
            )
        )
        runner._on_ui_command(
            UICommandEvent(
                command="tuning.update",
                params={"key": "control.line_hold_ms", "value": 90},
            )
        )
        assert runner.robot.line_control_updates == [
            {"pid_kp_us": 275.0},
            {"line_hold_ms": 90},
        ]
    finally:
        runner.stop()


def test_runner_applies_live_straight_base_and_deadband(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    try:
        runner._on_ui_command(
            UICommandEvent(
                command="tuning.update",
                params={"key": "control.base.right_us", "value": 210},
            )
        )
        runner._on_ui_command(
            UICommandEvent(
                command="tuning.update",
                params={"key": "control.line.deadband", "value": 0.03},
            )
        )
        assert runner.robot.line_control_updates == [
            {"right_base_throttle_us": 210},
            {"line_error_deadband": 0.03},
        ]
    finally:
        runner.stop()


def test_runner_applies_corner_timing_as_one_confirmed_update(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    try:
        runner._on_ui_command(
            UICommandEvent(
                command="control.corner_timing.apply",
                params={"approach_min_ms": 350, "pivot_ms": 1900},
            )
        )
        assert runner.robot.corner_timing_updates == [
            {"approach_min_ms": 350, "pivot_ms": 1900}
        ]
    finally:
        runner.stop()


def test_runner_applies_only_left_corner_timing(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    try:
        runner._on_ui_command(
            UICommandEvent(
                command="control.left_corner_timing.apply",
                params={"approach_min_ms": 600, "pivot_ms": 2000},
            )
        )
        assert runner.robot.left_corner_timing_updates == [
            {"approach_min_ms": 600, "pivot_ms": 2000}
        ]
        assert runner.robot.corner_timing_updates == []
    finally:
        runner.stop()


def test_runner_applies_green_half_turn_timing_as_one_confirmed_update(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    try:
        runner._on_ui_command(
            UICommandEvent(
                command="control.green_half_turn.apply",
                params={"first_ms": 1900, "reverse_ms": 200, "second_ms": 2200},
            )
        )
        assert runner.robot.green_half_turn_updates == [
            {"first_ms": 1900, "reverse_ms": 200, "second_ms": 2200}
        ]
    finally:
        runner.stop()


def test_runner_injects_robot_telemetry_into_detection(monkeypatch) -> None:
    clock = FakeClock()
    runner = _make_runner(monkeypatch, clock)
    try:
        event = VisionDetectionEvent(
            state="FOLLOWING_LINE",
            line=True,
            green=False,
            red=False,
            victims=0,
            metadata={"line_angle_deg": 94, "line_confidence": 0.72},
        )
        runner._inject_telemetry(event)
        assert event.metadata["telemetry"]["front"] == 320
        assert event.metadata["control"]["control_mode"] == "FOLLOW_LINE"
        assert event.metadata["control"]["pid_output"] == 14.5
        assert event.metadata["serial"]["port"] == "/dev/ttyACM0"
    finally:
        runner.stop()
