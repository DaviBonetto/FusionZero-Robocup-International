from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import psutil

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SRC_ROOT = Path(__file__).resolve().parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from src.core.event_bus import EventBus, EventTopic, HealthEvent, LogEvent, PoseEvent, UICommandEvent, VisionDetectionEvent
    from src.core.state_machine import RobotEvent, RobotState, StateMachine
    from src.remote_dashboard import RemoteDashboardServer
    from src.camera_capture import Picamera2Capture
    from src.modules.control import (
        GpioLedController,
        GpioMasterSwitchController,
        Pca9685RobotAdapter,
        Pca9685RobotConfig,
        RobotSerialConfig,
        SerialRobotAdapter,
    )
    from src.modules.vision.vision_node import VisionNode
    from src.ops_profiles import OpsProfile, OpsProfileCatalog, load_ops_profile_catalog
    from src.session_recording import SessionRecorder
else:
    from .core.event_bus import EventBus, EventTopic, HealthEvent, LogEvent, PoseEvent, UICommandEvent, VisionDetectionEvent
    from .core.state_machine import RobotEvent, RobotState, StateMachine
    from .remote_dashboard import RemoteDashboardServer
    from .camera_capture import Picamera2Capture
    from .modules.control import (
        GpioLedController,
        GpioMasterSwitchController,
        Pca9685RobotAdapter,
        Pca9685RobotConfig,
        RobotSerialConfig,
        SerialRobotAdapter,
    )
    from .modules.vision.vision_node import VisionNode
    from .ops_profiles import OpsProfile, OpsProfileCatalog, load_ops_profile_catalog
    from .session_recording import SessionRecorder


class LiveDashboardRunner:
    def __init__(
        self,
        *,
        camera_index: int,
        camera_width: int,
        camera_height: int,
        camera_fps: int,
        camera_backend: str = "auto",
        camera_rotation: int = 0,
        config_path: str | Path | None = None,
        robot_serial_port: str | None = None,
        robot_baud: int = 115200,
        robot_green_forward_ms: int = 5000,
        robot_green_streak: int = 2,
        robot_green_confirm_hold_ms: int = 180,
        robot_green_cooldown_ms: int = 6000,
        robot_green_hold_ms: int = 900,
        robot_obstacle_hold_ms: int = 1200,
        robot_dry_run: bool = False,
        robot_backend: str = "serial",
        pca9685_i2c_address: int = 0x40,
        pca9685_frequency_hz: int = 50,
        pca9685_left_channel: int = 4,
        pca9685_right_channel: int = 0,
        pca9685_left_channels: tuple[int, ...] = (),
        pca9685_right_channels: tuple[int, ...] = (),
        pca9685_min_us: int = 1000,
        pca9685_neutral_us: int = 1600,
        pca9685_max_us: int = 2000,
        pca9685_base_throttle_us: int | None = None,
        pca9685_left_base_throttle_us: int = 300,
        pca9685_right_base_throttle_us: int = 200,
        pca9685_turn_gain_us: int = 220,
        pca9685_pid_kp_us: float | None = 320.0,
        pca9685_pid_ki_us: float = 0.0,
        pca9685_pid_kd_us: float = 12.0,
        pca9685_pid_integral_limit: float = 0.15,
        pca9685_pid_derivative_filter: float = 0.60,
        pca9685_line_hold_ms: int = 120,
        pca9685_gap_crossing_enabled: bool = False,
        pca9685_gap_straight_confirm_frames: int = 3,
        pca9685_gap_reacquire_confirm_frames: int = 2,
        pca9685_gap_crossing_timeout_ms: int = 2200,
        pca9685_gap_max_entry_error: float = 0.22,
        pca9685_gap_max_entry_bend: float = 0.18,
        pca9685_max_output_us: int = 240,
        pca9685_line_error_deadband: float = 0.025,
        pca9685_simple_line_follow: bool = False,
        pca9685_basic_line_follow: bool = False,
        pca9685_line_steering_inverted: bool = False,
        pca9685_sharp_corner_maneuver_enabled: bool = True,
        pca9685_corner_sequence_enabled: bool = False,
        pca9685_corner_confirm_frames: int = 2,
        pca9685_corner_min_elbow_row_contrast: float = 1.25,
        pca9685_corner_min_wide_row_occupancy: float = 0.68,
        pca9685_corner_approach_stop_row_ratio: float = 0.40,
        pca9685_corner_approach_throttle_scale: float = 0.72,
        pca9685_corner_approach_min_ms: int = 350,
        pca9685_corner_approach_left_min_ms: int = 550,
        pca9685_corner_brake_ms: int = 500,
        pca9685_corner_pivot_speed_us: int = 300,
        pca9685_corner_pivot_right_ms: int = 1900,
        pca9685_corner_pivot_left_ms: int = 2100,
        pca9685_corner_reacquire_speed_us: int = 130,
        pca9685_corner_reacquire_timeout_ms: int = 1200,
        pca9685_curve_lookahead_gain: float = 0.0,
        pca9685_curve_throttle_scale: float = 1.0,
        pca9685_ordinary_sharp_hold_ms: int = 900,
        pca9685_sharp_curve_threshold: float = 0.55,
        pca9685_sharp_curve_hold_ms: int = 900,
        pca9685_sharp_curve_outer_scale: float = 1.0,
        pca9685_sharp_curve_inner_reverse_scale: float = 1.0,
        pca9685_sharp_curve_entry_min_row_ratio: float = 0.0,
        pca9685_sharp_curve_visible_commit_ms: int = 2600,
        pca9685_sharp_curve_finish_inner_reverse_scale: float = 1.0,
        pca9685_sharp_curve_exit_settle_ms: int = 320,
        pca9685_sharp_curve_exit_throttle_scale: float = 1.0,
        pca9685_sharp_curve_exit_max_correction_us: float = 0.0,
        pca9685_sharp_curve_recovery_ms: int = 520,
        pca9685_sharp_curve_recovery_max_correction_us: float = 45.0,
        pca9685_green_turn_us: int = 260,
        pca9685_green_half_turn_us: int = 300,
        pca9685_green_half_turn_ms: int = 4000,
        pca9685_green_half_turn_first_ms: int | None = None,
        pca9685_green_half_turn_second_ms: int | None = None,
        pca9685_green_half_turn_reverse_ms: int = 550,
        pca9685_green_maneuvers_enabled: bool = True,
        pca9685_start_disarmed: bool = False,
        pca9685_left_inverted: bool = False,
        pca9685_right_inverted: bool = True,
        line_only: bool = False,
        enable_leds: bool = False,
        led1_gpio: int = 18,
        led2_gpio: int = 23,
        enable_master_switch: bool = False,
        master_switch_gpio: int = 17,
        master_switch_debounce_ms: int = 20,
        profile_name: str | None = None,
        recordings_root: str | Path | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.camera_index = int(camera_index)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.camera_fps = max(1, int(camera_fps))
        self.camera_backend = str(camera_backend or "auto").strip().lower()
        if self.camera_backend not in {"auto", "opencv", "picamera2"}:
            raise ValueError("camera_backend must be auto, opencv, or picamera2")
        self.camera_rotation = int(camera_rotation)
        if self.camera_rotation not in {0, 90, 180, 270}:
            raise ValueError("camera_rotation must be 0, 90, 180, or 270 degrees clockwise")
        self.config_path = config_path
        self._monotonic = monotonic
        self._profile_catalog: OpsProfileCatalog = load_ops_profile_catalog(config_path)
        self._active_profile_name = "custom"
        self._active_profile: OpsProfile | None = None
        self._recording_auto_start = False
        self._last_detection_latency_ms = 0.0
        self._camera_backend_name = ""
        self._display_turn_gain_us = float(pca9685_turn_gain_us)
        self._line_only = bool(line_only)
        self._last_power_sample_at = 0.0
        self._power_status_cache: dict[str, Any] = {
            "available": False,
            "status": "unknown",
            "summary": "vcgencmd unavailable",
            "undervoltage_now": False,
            "undervoltage_occurred": False,
            "throttled_now": False,
            "throttled_occurred": False,
            "raw_value": "",
        }

        self.bus = EventBus(max_queue_size=2048, drop_oldest=True)
        self.fsm = StateMachine(event_bus=self.bus)
        self.vision = VisionNode(
            self.bus,
            config=config_path,
            publish_raw_frame=True,
            publish_processed_frame=True,
        )
        self._robot_backend = str(robot_backend or "serial").strip().lower().replace("-", "_")
        if self._robot_backend == "pca9685":
            self.robot = Pca9685RobotAdapter(
                self.bus,
                config=Pca9685RobotConfig(
                    i2c_address=int(pca9685_i2c_address),
                    frequency_hz=int(pca9685_frequency_hz),
                    left_channel=int(pca9685_left_channel),
                    right_channel=int(pca9685_right_channel),
                    left_channels=tuple(int(channel) for channel in pca9685_left_channels),
                    right_channels=tuple(int(channel) for channel in pca9685_right_channels),
                    min_us=int(pca9685_min_us),
                    neutral_us=int(pca9685_neutral_us),
                    max_us=int(pca9685_max_us),
                    base_throttle_us=None if pca9685_base_throttle_us is None else int(pca9685_base_throttle_us),
                    left_base_throttle_us=int(pca9685_left_base_throttle_us),
                    right_base_throttle_us=int(pca9685_right_base_throttle_us),
                    turn_gain_us=int(pca9685_turn_gain_us),
                    pid_kp_us=None if pca9685_pid_kp_us is None else float(pca9685_pid_kp_us),
                    pid_ki_us=float(pca9685_pid_ki_us),
                    pid_kd_us=float(pca9685_pid_kd_us),
                    pid_integral_limit=float(pca9685_pid_integral_limit),
                    pid_derivative_filter=float(pca9685_pid_derivative_filter),
                    line_hold_ms=int(pca9685_line_hold_ms),
                    gap_crossing_enabled=bool(pca9685_gap_crossing_enabled),
                    gap_straight_confirm_frames=int(
                        pca9685_gap_straight_confirm_frames
                    ),
                    gap_reacquire_confirm_frames=int(
                        pca9685_gap_reacquire_confirm_frames
                    ),
                    gap_crossing_timeout_ms=int(pca9685_gap_crossing_timeout_ms),
                    gap_max_entry_error=float(pca9685_gap_max_entry_error),
                    gap_max_entry_bend=float(pca9685_gap_max_entry_bend),
                    max_output_us=int(pca9685_max_output_us),
                    line_error_deadband=float(pca9685_line_error_deadband),
                    simple_line_follow=bool(pca9685_simple_line_follow),
                    basic_line_follow=bool(pca9685_basic_line_follow),
                    line_steering_inverted=bool(pca9685_line_steering_inverted),
                    sharp_corner_maneuver_enabled=bool(
                        pca9685_sharp_corner_maneuver_enabled
                    ),
                    corner_sequence_enabled=bool(pca9685_corner_sequence_enabled),
                    corner_confirm_frames=int(pca9685_corner_confirm_frames),
                    corner_min_elbow_row_contrast=float(
                        pca9685_corner_min_elbow_row_contrast
                    ),
                    corner_min_wide_row_occupancy=float(
                        pca9685_corner_min_wide_row_occupancy
                    ),
                    corner_approach_stop_row_ratio=float(
                        pca9685_corner_approach_stop_row_ratio
                    ),
                    corner_approach_throttle_scale=float(
                        pca9685_corner_approach_throttle_scale
                    ),
                    corner_approach_min_ms=int(pca9685_corner_approach_min_ms),
                    corner_approach_left_min_ms=int(
                        pca9685_corner_approach_left_min_ms
                    ),
                    corner_brake_ms=int(pca9685_corner_brake_ms),
                    corner_pivot_speed_us=int(pca9685_corner_pivot_speed_us),
                    corner_pivot_right_ms=int(pca9685_corner_pivot_right_ms),
                    corner_pivot_left_ms=int(pca9685_corner_pivot_left_ms),
                    corner_reacquire_speed_us=int(
                        pca9685_corner_reacquire_speed_us
                    ),
                    corner_reacquire_timeout_ms=int(
                        pca9685_corner_reacquire_timeout_ms
                    ),
                    curve_lookahead_gain=float(pca9685_curve_lookahead_gain),
                    curve_throttle_scale=float(pca9685_curve_throttle_scale),
                    ordinary_sharp_hold_ms=int(pca9685_ordinary_sharp_hold_ms),
                    sharp_curve_threshold=float(pca9685_sharp_curve_threshold),
                    sharp_curve_hold_ms=int(pca9685_sharp_curve_hold_ms),
                    sharp_curve_outer_scale=float(pca9685_sharp_curve_outer_scale),
                    sharp_curve_inner_reverse_scale=float(
                        pca9685_sharp_curve_inner_reverse_scale
                    ),
                    sharp_curve_entry_min_row_ratio=float(
                        pca9685_sharp_curve_entry_min_row_ratio
                    ),
                    sharp_curve_visible_commit_ms=int(
                        pca9685_sharp_curve_visible_commit_ms
                    ),
                    sharp_curve_finish_inner_reverse_scale=float(
                        pca9685_sharp_curve_finish_inner_reverse_scale
                    ),
                    sharp_curve_exit_settle_ms=int(
                        pca9685_sharp_curve_exit_settle_ms
                    ),
                    sharp_curve_exit_throttle_scale=float(
                        pca9685_sharp_curve_exit_throttle_scale
                    ),
                    sharp_curve_exit_max_correction_us=float(
                        pca9685_sharp_curve_exit_max_correction_us
                    ),
                    sharp_curve_recovery_ms=int(pca9685_sharp_curve_recovery_ms),
                    sharp_curve_recovery_max_correction_us=float(
                        pca9685_sharp_curve_recovery_max_correction_us
                    ),
                    green_turn_us=int(pca9685_green_turn_us),
                    green_half_turn_us=int(pca9685_green_half_turn_us),
                    green_half_turn_ms=int(pca9685_green_half_turn_ms),
                    green_half_turn_first_ms=(
                        None
                        if pca9685_green_half_turn_first_ms is None
                        else int(pca9685_green_half_turn_first_ms)
                    ),
                    green_half_turn_second_ms=(
                        None
                        if pca9685_green_half_turn_second_ms is None
                        else int(pca9685_green_half_turn_second_ms)
                    ),
                    green_half_turn_reverse_ms=int(pca9685_green_half_turn_reverse_ms),
                    green_hold_ms=int(robot_green_hold_ms),
                    green_trigger_streak=int(robot_green_streak),
                    green_single_confirm_hold_ms=int(robot_green_confirm_hold_ms),
                    green_cooldown_ms=int(robot_green_cooldown_ms),
                    left_inverted=bool(pca9685_left_inverted),
                    right_inverted=bool(pca9685_right_inverted),
                    # Green markers are part of line navigation. line_only
                    # keeps the FSM out of rescue states but must not disable
                    # the PCA9685 green maneuver itself.
                    enable_green_maneuvers=bool(pca9685_green_maneuvers_enabled),
                    start_disarmed=bool(pca9685_start_disarmed),
                    dry_run=bool(robot_dry_run),
                ),
                monotonic=monotonic,
            )
        else:
            self._robot_backend = "serial"
            self.robot = SerialRobotAdapter(
                self.bus,
                config=RobotSerialConfig(
                    port=(str(robot_serial_port).strip() if robot_serial_port else None),
                    baud_rate=int(robot_baud),
                    green_forward_ms=int(robot_green_forward_ms),
                    green_trigger_streak=int(robot_green_streak),
                    green_cooldown_ms=int(robot_green_cooldown_ms),
                    green_hold_ms=int(robot_green_hold_ms),
                    obstacle_hold_ms=int(robot_obstacle_hold_ms),
                    dry_run=bool(robot_dry_run),
                ),
            )
        self.leds = (
            GpioLedController(self.bus, led1_gpio=int(led1_gpio), led2_gpio=int(led2_gpio))
            if bool(enable_leds)
            else None
        )
        self._master_switch_required = bool(enable_master_switch)
        self.master_switch = (
            GpioMasterSwitchController(
                self.bus,
                gpio=int(master_switch_gpio),
                debounce_ms=int(master_switch_debounce_ms),
            )
            if self._master_switch_required
            else None
        )
        # Subscribe the runner before the recorder so detections are enriched
        # with the post-control PID/PWM telemetry before they are persisted.
        # The robot adapter is subscribed earlier and applies the command first.
        self._subs = [
            self.bus.subscribe(EventTopic.VISION_DETECTIONS, self._on_detection),
            self.bus.subscribe(EventTopic.UI_COMMAND, self._on_ui_command),
        ]
        self.session_recorder = SessionRecorder(
            self.bus,
            output_root=recordings_root,
            default_options=self._profile_catalog.default_recording,
        )

        self._capture_lock = threading.RLock()
        self._capture: Any = None
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._safety_worker: threading.Thread | None = None

        self._frame_id = 0
        self._last_capture_ts = 0.0
        self._fps_capture = 0.0
        self._last_health_ts = 0.0
        self._last_no_camera_log = 0.0
        self._last_camera_ok_at = self._monotonic()
        self._last_vision_ok_at = self._monotonic()
        self._last_detection_at = self._monotonic()
        self._camera_timeout_s = 1.0
        self._fsm_stall_timeout_s = 1.2
        self._vision_fresh_window_s = 0.5
        self._camera_fault_active = False
        self._fsm_fault_active = False
        self._camera_read_fail_streak = 0
        self._camera_read_fail_threshold = 8
        self._camera_reopen_backoff_s = 0.25

        self._line_lost_streak = 0
        self._line_found_streak = 0
        self._pose_x = 0.0
        self._pose_y = 0.0
        self._pose_theta = 0.0

        if profile_name:
            self._apply_profile(profile_name, source="startup")

    def start(self) -> None:
        self._stop_event.clear()
        now = self._monotonic()
        self._last_camera_ok_at = now
        self._last_vision_ok_at = now
        self._last_detection_at = now
        self._camera_fault_active = False
        self._fsm_fault_active = False
        self._camera_read_fail_streak = 0
        self._open_camera()
        self.robot.start()
        if self._line_only:
            try:
                next_state = self.fsm.handle(
                    RobotEvent.ON_LINE_FOUND,
                    payload={"reason": "line_only_startup"},
                )
                if next_state != RobotState.FOLLOWING_LINE:
                    raise RuntimeError(f"line-only startup entered {next_state!r}")
            except Exception as exc:
                self._publish_log("ERROR", f"line-only startup failed: {exc}")
                self._request_safe_stop("line_only_startup_failure", emergency=True)
                raise
        if self.leds is not None:
            try:
                self.leds.start()
            except Exception as exc:
                self._publish_log("ERROR", f"LED outputs unavailable: {exc}")
                self.leds = None
        if self.master_switch is not None:
            try:
                self.master_switch.start()
            except Exception as exc:
                # Keep the controller object and the required flag intact: a
                # missing switch must block START instead of silently bypassing
                # the physical safety gate.
                self._publish_log("ERROR", f"master switch unavailable; START blocked: {exc}")
        if self._recording_auto_start and not self.session_recorder.status_payload()["enabled"]:
            self.session_recorder.start(reason="profile_auto_start", context=self._session_context())
        self._worker = threading.Thread(target=self._capture_loop, name="live-dashboard-capture", daemon=True)
        self._worker.start()
        self._safety_worker = threading.Thread(target=self._safety_loop, name="live-dashboard-safety", daemon=True)
        self._safety_worker.start()
        self._publish_log(
            "INFO",
            (
                f"live runner started cam={self.camera_index} "
                f"{self.camera_width}x{self.camera_height}@{self.camera_fps} "
                f"rotation={self.camera_rotation} "
                f"profile={self._active_profile_name} line_only={self._line_only}"
            ),
        )

    def stop(self) -> None:
        self._stop_event.set()
        self._request_safe_stop("runner_shutdown", emergency=False)
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self._worker = None
        if self._safety_worker is not None:
            self._safety_worker.join(timeout=2.0)
        self._safety_worker = None

        if self.master_switch is not None:
            self.master_switch.stop()

        for sub in self._subs:
            try:
                sub.unsubscribe()
            except Exception:
                pass
        self._subs.clear()

        self._release_camera()
        self.robot.stop()
        if self.leds is not None:
            self.leds.stop()
        self.session_recorder.close()
        self.vision.close()
        self.bus.stop()

    def _capture_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                cap = self._get_capture()
                if cap is None:
                    now = self._monotonic()
                    if now - self._last_no_camera_log >= 2.0:
                        self._last_no_camera_log = now
                        self._publish_log("WARNING", f"camera {self.camera_index} unavailable, retrying")
                    self._open_camera()
                    time.sleep(0.2)
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    reopened = self._handle_capture_read_failure()
                    time.sleep(self._camera_reopen_backoff_s if reopened else 0.03)
                    continue

                frame = self._orient_camera_frame(frame)
                self._handle_capture_read_success()
                self._last_camera_ok_at = self._monotonic()
                ts = time.time()
                self._frame_id += 1
                try:
                    self.vision.process_frame(frame, frame_id=self._frame_id, timestamp=ts)
                    self._last_vision_ok_at = self._monotonic()
                except Exception as exc:
                    self._publish_log("ERROR", f"vision process failed: {exc}")
                    time.sleep(0.01)
                    continue

                self._update_capture_fps()
                self._publish_health_if_due()
        except Exception as exc:
            self._publish_log("ERROR", f"capture loop crashed: {exc}")
            self._request_safe_stop("capture_loop_crash", emergency=True)
            self._stop_event.set()

    def _orient_camera_frame(self, frame):
        if self.camera_rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.camera_rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.camera_rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _safety_loop(self) -> None:
        while not self._stop_event.wait(0.1):
            try:
                self._evaluate_safety()
            except Exception as exc:
                self._publish_log("ERROR", f"safety monitor failed: {exc}")
                self._request_safe_stop("safety_monitor_failure", emergency=True)
                self._stop_event.set()
                return

    def _evaluate_safety(self) -> None:
        now = self._monotonic()
        green_half_turn_active = self._green_half_turn_active()

        if not green_half_turn_active and (now - self._last_vision_ok_at) > self._camera_timeout_s:
            if not self._camera_fault_active:
                self._camera_fault_active = True
                self._publish_log("ERROR", "camera/vision heartbeat lost; issuing safe STOP")
                self._request_safe_stop("camera_timeout", emergency=False)
        elif self._camera_fault_active and not green_half_turn_active:
            self._camera_fault_active = False
            self._publish_log("INFO", "camera/vision heartbeat restored")

        detection_stale = (now - self._last_detection_at) > self._fsm_stall_timeout_s
        vision_fresh = (now - self._last_vision_ok_at) <= self._vision_fresh_window_s
        if not green_half_turn_active and self._frame_id > 0 and vision_fresh and detection_stale:
            if not self._fsm_fault_active:
                self._fsm_fault_active = True
                self._publish_log("ERROR", "fsm/detection pipeline stalled; issuing ESTOP")
                self._request_safe_stop("fsm_stall", emergency=True)
        elif self._fsm_fault_active and not detection_stale and not green_half_turn_active:
            self._fsm_fault_active = False
            self._publish_log("INFO", "fsm/detection pipeline recovered")

        if not self._is_valid_state_object(self.fsm.state):
            self._publish_log("ERROR", f"invalid fsm state object: {self.fsm.state!r}")
            self._request_safe_stop("invalid_fsm_state", emergency=True)

    def _green_half_turn_active(self) -> bool:
        try:
            status = self.robot.status_payload()
        except Exception:
            return False
        phase = str(status.get("green_half_turn_phase", "IDLE")).strip().upper()
        remaining_ms = int(status.get("green_maneuver_remaining_ms", 0) or 0)
        return bool(
            phase not in {"", "IDLE"}
            and remaining_ms > 0
            and bool(status.get("motor_armed", False))
            and not bool(status.get("failsafe", False))
        )

    def _on_detection(self, event: VisionDetectionEvent) -> None:
        if not isinstance(event, VisionDetectionEvent):
            return
        self._last_detection_at = self._monotonic()
        self._last_detection_latency_ms = max(0.0, float(event.latency_ms))
        if self._fsm_fault_active:
            self._fsm_fault_active = False
            self._publish_log("INFO", "fsm/detection pipeline recovered")
        if not self._is_valid_state_name(event.state):
            self._publish_log("ERROR", f"invalid detection state received: {event.state!r}")
            self._request_safe_stop("invalid_detection_state", emergency=True)
            return
        if not self._is_valid_state_object(self.fsm.state):
            self._publish_log("ERROR", f"invalid fsm state object: {self.fsm.state!r}")
            self._request_safe_stop("invalid_fsm_state", emergency=True)
            return

        self._inject_telemetry(event)
        self._publish_pose(event)

        trigger: RobotEvent | None = None
        reason = "vision_live"

        if self._line_only:
            # In this integration mode only line following is active. Rescue,
            # victim, gap, and intersection events must not change the FSM.
            if event.line:
                self._line_found_streak += 1
                self._line_lost_streak = 0
                if self.fsm.state != RobotState.FOLLOWING_LINE:
                    trigger = RobotEvent.ON_LINE_FOUND
                    reason = "line_only_recover"
            else:
                self._line_lost_streak += 1
                self._line_found_streak = 0
                return
        elif event.red:
            trigger = RobotEvent.ON_RESCUE_RED_DETECTED
        elif event.victims > 0:
            trigger = RobotEvent.ON_VICTIM_DETECTED
        elif event.line:
            self._line_found_streak += 1
            self._line_lost_streak = 0
            if self.fsm.state in {
                RobotState.SEARCHING_LINE,
                RobotState.VALIDATING_GAP,
                RobotState.CROSSING_GAP,
            }:
                trigger = RobotEvent.ON_LINE_FOUND
            elif self.fsm.state == RobotState.FOLLOWING_LINE and event.green:
                trigger = RobotEvent.ON_INTERSECTION
        else:
            self._line_lost_streak += 1
            self._line_found_streak = 0
            if self.fsm.state == RobotState.FOLLOWING_LINE and self._line_lost_streak == 1:
                trigger = RobotEvent.ON_GAP
                reason = "line_gap_suspected"
            elif self._line_lost_streak >= 3:
                trigger = RobotEvent.ON_LINE_LOST
                reason = "line_lost_streak"

        if trigger is None:
            return
        try:
            next_state = self.fsm.handle(trigger, payload={"reason": reason})
            if not isinstance(next_state, RobotState):
                self._publish_log("ERROR", f"invalid fsm state returned: {next_state!r}")
                self._request_safe_stop("invalid_fsm_state", emergency=True)
        except Exception as exc:
            self._publish_log("ERROR", f"fsm handle failed: {exc}")
            self._request_safe_stop("fsm_handle_failure", emergency=True)

    def _on_ui_command(self, event: UICommandEvent) -> None:
        if not isinstance(event, UICommandEvent):
            return
        command = str(event.command or "").strip().lower()
        params = event.params if isinstance(event.params, Mapping) else {}

        if command in {"system.start", "system.run"}:
            if not self._master_start_permitted():
                self._publish_runtime_command(
                    "robot.stop",
                    source="master_interlock",
                    reason="master_switch_open_or_locked",
                    master_authorized=True,
                )
                self._publish_runtime_command(
                    "leds.off",
                    source="master_interlock",
                    reason="master_switch_open_or_locked",
                    master_authorized=True,
                )
                self._publish_log(
                    "WARNING",
                    "master START blocked: open the GPIO switch, then close it",
                )
                return
            source = str(params.get("source", "dashboard_master_start"))
            self._publish_runtime_command(
                "leds.on",
                source=source,
                reason="master_start",
                master_authorized=True,
            )
            self._publish_runtime_command(
                "robot.start",
                source=source,
                reason="master_start",
                master_authorized=True,
            )
            self._publish_log("INFO", f"master START accepted source={source}")
            return

        if command in {"system.stop", "system.halt"}:
            source = str(params.get("source", "dashboard_master_stop"))
            self._publish_runtime_command(
                "robot.stop",
                source=source,
                reason="master_stop",
                master_authorized=True,
            )
            self._publish_runtime_command(
                "leds.off",
                source=source,
                reason="master_stop",
                master_authorized=True,
            )
            self._publish_log("INFO", f"master STOP accepted source={source}")
            return

        if command in {"robot.start", "robot.arm"}:
            if not bool(params.get("master_authorized", False)) and not self._master_start_permitted():
                self._publish_runtime_command(
                    "robot.stop",
                    source="master_interlock",
                    reason="legacy_start_blocked",
                    master_authorized=True,
                )
                self._publish_runtime_command(
                    "leds.off",
                    source="master_interlock",
                    reason="legacy_start_blocked",
                    master_authorized=True,
                )
                self._publish_log("WARNING", f"{command} blocked by open master switch")
                return
            self._publish_runtime_command(
                "leds.on",
                source=str(params.get("source", "robot_start_bridge")),
                reason="robot_start_bridge",
                master_authorized=True,
            )
            return

        if command in {"robot.stop", "robot.force_stop", "robot.estop"}:
            self._publish_runtime_command(
                "leds.off",
                source=str(params.get("source", "robot_stop_bridge")),
                reason="robot_stop_bridge",
                master_authorized=True,
            )
            return

        if command == "control.green_half_turn.apply":
            updater = getattr(self.robot, "update_green_half_turn_timing", None)
            if not callable(updater):
                self._publish_log(
                    "WARNING",
                    "live green half-turn timing is available only for the PCA9685 backend",
                )
                return
            try:
                applied = updater(
                    first_ms=int(params.get("first_ms", 1900)),
                    second_ms=int(params.get("second_ms", 2100)),
                    reverse_ms=int(params.get("reverse_ms", 550)),
                )
                self._publish_log(
                    "INFO",
                    "green half-turn timing applied "
                    f"first={applied['green_half_turn_first_ms']} ms "
                    f"reverse={applied['green_half_turn_reverse_ms']} ms "
                    f"second={applied['green_half_turn_second_ms']} ms",
                )
            except Exception as exc:
                self._publish_log("ERROR", f"failed green half-turn timing update: {exc}")
            return

        if command == "control.corner_timing.apply":
            updater = getattr(self.robot, "update_corner_timing", None)
            if not callable(updater):
                self._publish_log(
                    "WARNING",
                    "live corner timing is available only for the PCA9685 backend",
                )
                return
            try:
                applied = updater(
                    approach_min_ms=int(params.get("approach_min_ms", 350)),
                    pivot_ms=int(params.get("pivot_ms", 1900)),
                )
                self._publish_log(
                    "INFO",
                    "corner timing applied "
                    f"advance={applied['approach_min_ms']} ms "
                    f"pivot={applied['pivot_right_ms']} ms",
                )
            except Exception as exc:
                self._publish_log("ERROR", f"failed corner timing update: {exc}")
            return

        if command == "control.left_corner_timing.apply":
            updater = getattr(self.robot, "update_left_corner_timing", None)
            if not callable(updater):
                self._publish_log(
                    "WARNING",
                    "live left-corner timing is available only for the PCA9685 backend",
                )
                return
            try:
                applied = updater(
                    approach_min_ms=int(params.get("approach_min_ms", 550)),
                    pivot_ms=int(params.get("pivot_ms", 2100)),
                )
                self._publish_log(
                    "INFO",
                    "left corner timing applied "
                    f"advance={applied['approach_left_min_ms']} ms "
                    f"pivot={applied['pivot_left_ms']} ms",
                )
            except Exception as exc:
                self._publish_log(
                    "ERROR",
                    f"failed left-corner timing update: {exc}",
                )
            return

        if command == "tuning.update":
            key = str(params.get("key", "")).strip()
            value = params.get("value")
            if key:
                self._apply_tuning(key, value)
            return

        if command == "config.load_profile":
            profile_name = str(params.get("profile", "")).strip().lower()
            if profile_name:
                self._apply_profile(profile_name, source="ui")
            return

        if command == "camera.reconnect":
            self.camera_index = int(params.get("index", self.camera_index))
            self.camera_width = int(params.get("width", self.camera_width))
            self.camera_height = int(params.get("height", self.camera_height))
            self.camera_fps = max(1, int(params.get("fps", self.camera_fps)))
            self._reset_camera_runtime_state()
            self._open_camera()
            self._publish_log(
                "INFO",
                f"camera reconfigured idx={self.camera_index} {self.camera_width}x{self.camera_height}@{self.camera_fps}",
            )
            return

        if command == "fsm.force_mode":
            mode = str(params.get("mode", "")).strip().lower()
            if self._line_only and mode != "line":
                self._publish_log("WARNING", "rescue mode ignored while --line-only is active")
                return
            if mode == "line":
                trigger = RobotEvent.ON_LINE_FOUND
            elif mode == "rescue":
                trigger = RobotEvent.ON_RESCUE_RED_DETECTED
            else:
                self._publish_log("WARNING", f"unknown mode request: {mode!r}")
                return
            try:
                next_state = self.fsm.handle(trigger, payload={"reason": f"ui_force_mode:{mode}"})
                if not isinstance(next_state, RobotState):
                    self._publish_log("ERROR", f"invalid fsm state returned: {next_state!r}")
                    self._request_safe_stop("invalid_fsm_state", emergency=True)
                    return
                self._publish_log("INFO", f"fsm forced to {self._state_value()} via {mode}")
            except Exception as exc:
                self._publish_log("ERROR", f"fsm force mode failed: {exc}")
                self._request_safe_stop("fsm_force_mode_failure", emergency=True)
            return

        if command == "session.recording.configure":
            options = self.session_recorder.configure(params)
            self._publish_log("INFO", f"session recording configured {options.to_payload()}")
            return

        if command == "session.recording.start":
            self.session_recorder.configure(params)
            session_dir = self.session_recorder.start(reason="ui", context=self._session_context())
            self._publish_log("INFO", f"session recording started dir={session_dir}")
            return

        if command == "session.recording.stop":
            self.session_recorder.stop(reason="ui")
            self._publish_log("INFO", "session recording stopped")

    def _master_start_permitted(self) -> bool:
        if not self._master_switch_required:
            return True
        return bool(self.master_switch is not None and self.master_switch.start_permitted)

    def _publish_runtime_command(self, command: str, **params: Any) -> None:
        try:
            self.bus.publish(
                EventTopic.UI_COMMAND,
                UICommandEvent(command=str(command), params=dict(params)),
            )
        except Exception as exc:
            self._publish_log("ERROR", f"failed runtime command {command}: {exc}")

    def _apply_tuning(self, key: str, value: object) -> None:
        control_fields = {
            "control.pid.kp_us": "pid_kp_us",
            "control.pid.ki_us": "pid_ki_us",
            "control.pid.kd_us": "pid_kd_us",
            "control.pid.integral_limit": "pid_integral_limit",
            "control.pid.derivative_filter": "pid_derivative_filter",
            "control.pid.max_output_us": "max_output_us",
            "control.line_hold_ms": "line_hold_ms",
            "control.base.left_us": "left_base_throttle_us",
            "control.base.right_us": "right_base_throttle_us",
            "control.line.deadband": "line_error_deadband",
        }
        control_field = control_fields.get(key)
        if control_field is not None:
            updater = getattr(self.robot, "update_line_control", None)
            if not callable(updater):
                self._publish_log("WARNING", "live line PID tuning is available only for the PCA9685 backend")
                return
            try:
                update_value: float | int
                if control_field in {
                    "line_hold_ms",
                    "max_output_us",
                    "left_base_throttle_us",
                    "right_base_throttle_us",
                }:
                    update_value = int(value)
                else:
                    update_value = float(value)
                applied = updater(**{control_field: update_value})
                shown = applied.get(control_field, update_value) if isinstance(applied, Mapping) else update_value
                self._publish_log("INFO", f"line control tuning applied {key}={shown}")
            except Exception as exc:
                self._publish_log("ERROR", f"failed line control tuning {key}: {exc}")
            return

        manager = self.vision._pipeline
        line = manager.line_detector
        color = manager.color_detector
        ball = manager.ball_detector

        try:
            if key == "line.black_v_max":
                line.black_v_max = int(value)
            elif key == "line.black_s_max":
                line.black_s_max = int(value)
            elif key == "line.min_area":
                line.min_black_area = int(value)
            elif key == "line.erode_iter":
                line.erode_iter = int(value)
            elif key == "line.dilate_iter":
                line.dilate_iter = int(value)
            elif key == "green.h_min":
                color.green_h_min = int(value)
            elif key == "green.h_max":
                color.green_h_max = int(value)
            elif key == "green.s_min":
                color.green_s_min = int(value)
            elif key == "green.v_min":
                color.green_v_min = int(value)
            elif key == "green.min_area":
                color.green_min_area = float(value)
            elif key == "red.h1_min":
                color.red_h1_min = int(value)
            elif key == "red.h1_max":
                color.red_h1_max = int(value)
            elif key == "red.h2_min":
                color.red_h2_min = int(value)
            elif key == "red.h2_max":
                color.red_h2_max = int(value)
            elif key == "red.s_min":
                color.red_s_min = int(value)
            elif key == "red.v_min":
                color.red_v_min = int(value)
            elif key == "red.min_area":
                color.red_min_area = float(value)
            elif key == "silver.conf":
                manager.silver.threshold = float(value)
            elif key == "silver.blur":
                blur = max(3, int(value))
                ball.silver_blur = blur if blur % 2 == 1 else blur + 1
            elif key == "dead.black_v_max":
                v = max(0, min(255, int(value)))
                ball.dead_black_threshold = (v, v, v)
            else:
                return
            self._publish_log("INFO", f"tuning applied {key}={value}")
        except Exception as exc:
            self._publish_log("ERROR", f"failed tuning {key}: {exc}")

    def _inject_telemetry(self, event: VisionDetectionEvent) -> None:
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        serial_status = self._serial_status_payload()
        telemetry = serial_status.get("telemetry", {}) if isinstance(serial_status, Mapping) else {}
        if not isinstance(telemetry, Mapping) or not telemetry:
            angle = float(metadata.get("line_angle_deg", 90.0))
            front = max(120, int(1200 - abs(angle - 90.0) * 8.0))
            left = max(100, int(900 + (angle - 90.0) * 9.0))
            right = max(100, int(900 - (angle - 90.0) * 9.0))
            back = 1400 if event.line else max(250, int(500 + self._line_lost_streak * 120))
            telemetry = {
                "front": front,
                "left": left,
                "right": right,
                "back": back,
                "yaw": round(self._pose_theta * 180.0 / math.pi, 2),
                "roll": round(math.sin(time.monotonic() * 0.7) * 4.0, 2),
                "pitch": round(math.cos(time.monotonic() * 0.6) * 4.0, 2),
                "gripper": 90 if event.victims == 0 else 1400,
            }
        else:
            telemetry = dict(telemetry)
            if "heading_deg" in telemetry and "yaw" not in telemetry:
                telemetry["yaw"] = telemetry["heading_deg"]
        metadata["telemetry"] = telemetry
        line_error, pid_output = self._display_line_control(event, metadata, serial_status)
        metadata["control"] = {
            "control_mode": str(serial_status.get("control_mode", "")).strip(),
            "line_error": line_error,
            "pid_output": pid_output,
            "vision_line_offset": serial_status.get("vision_line_offset"),
            "line_offset_source": str(serial_status.get("line_offset_source", "none")).strip(),
            "line_candidate_count": int(serial_status.get("line_candidate_count", 0) or 0),
            "steering_decision": str(serial_status.get("steering_decision", "")).strip(),
            "requested_left_speed_us": serial_status.get("requested_left_speed_us"),
            "requested_right_speed_us": serial_status.get("requested_right_speed_us"),
            "left_pwm": telemetry.get("left_pwm"),
            "right_pwm": telemetry.get("right_pwm"),
            "obstacle_state": str(serial_status.get("obstacle_state", "")).strip(),
            "green_instruction": str(serial_status.get("green_instruction", "")).strip(),
            "green_route_decision": str(
                serial_status.get("green_route_decision", "")
            ).strip(),
            "green_marker_count": int(serial_status.get("green_marker_count", 0) or 0),
            "green_half_turn_phase": str(
                serial_status.get("green_half_turn_phase", "IDLE")
            ).strip(),
            "green_maneuver_remaining_ms": int(
                serial_status.get("green_maneuver_remaining_ms", 0) or 0
            ),
            "motor_armed": bool(serial_status.get("motor_armed", False)),
            "failsafe": bool(serial_status.get("failsafe", False)),
            "assist_kind": str(serial_status.get("assist_kind", "none")).strip(),
            "sharp_curve_active": bool(serial_status.get("sharp_curve_active", False)),
            "sharp_curve_threshold": serial_status.get("sharp_curve_threshold"),
            "sharp_curve_hold_ms": serial_status.get("sharp_curve_hold_ms"),
            "sharp_curve_inner_reverse_scale": serial_status.get(
                "sharp_curve_inner_reverse_scale"
            ),
            "sharp_curve_reverse_inner": bool(
                serial_status.get("sharp_curve_reverse_inner", False)
            ),
            "corner_sequence_enabled": bool(
                serial_status.get("corner_sequence_enabled", False)
            ),
            "corner_phase": str(serial_status.get("corner_phase", "IDLE")),
            "corner_direction": str(serial_status.get("corner_direction", "NONE")),
            "corner_phase_elapsed_ms": int(
                serial_status.get("corner_phase_elapsed_ms", 0) or 0
            ),
            "corner_confirm_streak": int(
                serial_status.get("corner_confirm_streak", 0) or 0
            ),
            "corner_elbow_row_contrast": serial_status.get(
                "corner_elbow_row_contrast"
            ),
            "corner_geometry_fallback_active": bool(
                serial_status.get("corner_geometry_fallback_active", False)
            ),
            "corner_pivot_right_ms": serial_status.get("corner_pivot_right_ms"),
            "corner_pivot_left_ms": serial_status.get("corner_pivot_left_ms"),
            "corner_pivot_left_speed_us": serial_status.get(
                "corner_pivot_left_speed_us"
            ),
            "corner_pivot_line_lost_seen": bool(
                serial_status.get("corner_pivot_line_lost_seen", False)
            ),
            "corner_approach_min_ms": serial_status.get("corner_approach_min_ms"),
            "corner_approach_left_min_ms": serial_status.get(
                "corner_approach_left_min_ms"
            ),
            "corner_reacquire_streak": int(
                serial_status.get("corner_reacquire_streak", 0) or 0
            ),
            "corner_exit_active": bool(serial_status.get("corner_exit_active", False)),
            "corner_recovery_active": bool(
                serial_status.get("corner_recovery_active", False)
            ),
            "curve_signal": serial_status.get("curve_signal"),
            "vision_confidence": float(metadata.get("line_confidence", 0.0) or 0.0),
        }
        metadata["serial"] = serial_status
        metadata["robot"] = serial_status
        event.metadata = metadata

    def _display_line_control(
        self,
        event: VisionDetectionEvent,
        metadata: Mapping[str, Any],
        robot_status: Mapping[str, Any],
    ) -> tuple[object, object]:
        line_error = robot_status.get("line_error")
        pid_output = robot_status.get("pid_output")
        mode = str(robot_status.get("control_mode", "")).strip().upper()
        confidence = float(metadata.get("line_confidence", 0.0) or 0.0)
        if bool(event.line) and confidence > 0.0 and mode not in {"FOLLOW_LINE", "GREEN", "MANUAL"}:
            offset = float(metadata.get("line_offset_norm", 0.0) or 0.0)
            line_error = round(max(-1.0, min(1.0, offset)), 4)
            pid_output = round(float(line_error) * float(self._display_turn_gain_us), 2)
        return line_error, pid_output

    def _reset_camera_runtime_state(self) -> None:
        now = self._monotonic()
        self._frame_id = 0
        self._last_capture_ts = 0.0
        self._fps_capture = 0.0
        self._last_camera_ok_at = now
        self._last_vision_ok_at = now
        self._last_detection_at = now
        self._camera_fault_active = False
        self._fsm_fault_active = False
        self._camera_read_fail_streak = 0
        self._line_lost_streak = 0
        self._line_found_streak = 0

    def _publish_pose(self, event: VisionDetectionEvent) -> None:
        angle_deg = float(event.metadata.get("line_angle_deg", 90.0)) if isinstance(event.metadata, Mapping) else 90.0
        self._pose_theta = math.radians(max(0.0, min(180.0, angle_deg)) - 90.0)
        step = 0.012 if event.line else 0.004
        self._pose_x += math.cos(self._pose_theta) * step
        self._pose_y += math.sin(self._pose_theta) * step

        try:
            self.bus.publish(
                EventTopic.NAV_POSE,
                PoseEvent(timestamp=time.time(), x=self._pose_x, y=self._pose_y, theta=self._pose_theta),
            )
        except Exception:
            pass

    def _update_capture_fps(self) -> None:
        now = time.perf_counter()
        if self._last_capture_ts <= 0:
            self._last_capture_ts = now
            return
        dt = now - self._last_capture_ts
        self._last_capture_ts = now
        if dt <= 0:
            return
        instant = 1.0 / dt
        self._fps_capture = instant if self._fps_capture <= 0 else (0.9 * self._fps_capture + 0.1 * instant)

    def _publish_health_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_health_ts < 0.2:
            return
        self._last_health_ts = now
        memory = psutil.virtual_memory()
        self._power_status_cache = self._read_power_status(now)
        try:
            self.bus.publish(
                EventTopic.SYSTEM_HEALTH,
                HealthEvent(
                    timestamp=time.time(),
                    cpu_percent=float(psutil.cpu_percent(interval=None)),
                    fps_capture=float(self._fps_capture),
                    fps_process=float(self.vision.fps),
                    fps_ui=float(self.vision.fps),
                    queue_depth=int(self.bus.queue_depth),
                    metadata={
                        "memory_percent": float(memory.percent),
                        "memory_available_mb": round(float(memory.available) / (1024.0 * 1024.0), 1),
                        "network_latency_ms": 0.0,
                        "network": {
                            "latency_ms": 0.0,
                            "state": "local",
                            "peer": "local",
                        },
                        "camera": self._camera_status_payload(),
                        "serial": self._serial_status_payload(),
                        "power": dict(self._power_status_cache),
                        "profiles": self._profile_status_payload(),
                        "recording": self.session_recorder.status_payload(),
                        "master_switch": (
                            self.master_switch.status_payload()
                            if self.master_switch is not None
                            else {"enabled": False, "state": "DISABLED"}
                        ),
                        "vision": {
                            "latency_ms": float(self._last_detection_latency_ms),
                            "state": self._state_value(),
                        },
                    },
                ),
            )
        except Exception:
            pass

    def _open_camera(self) -> None:
        with self._capture_lock:
            self._release_camera_locked()
            if self.camera_backend in {"auto", "picamera2"} and self.camera_index == 0:
                try:
                    capture = Picamera2Capture(
                        self.camera_index,
                        self.camera_width,
                        self.camera_height,
                        self.camera_fps,
                    )
                    self._capture = capture
                    self._camera_backend_name = "picamera2"
                    self._camera_read_fail_streak = 0
                    self._publish_log(
                        "INFO",
                        (
                            f"camera {self.camera_index} connected backend=picamera2 "
                            f"{self.camera_width}x{self.camera_height}@{self.camera_fps}"
                        ),
                    )
                    return
                except Exception as exc:
                    if self.camera_backend == "picamera2":
                        self._publish_log("WARNING", f"Picamera2 camera open failed: {exc}")
                        self._capture = None
                        self._camera_backend_name = ""
                        return

            for backend_name, backend in self._preferred_camera_backends():
                cap = cv2.VideoCapture(self.camera_index, backend)
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                    continue

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
                self._capture = cap
                self._camera_backend_name = backend_name
                self._camera_read_fail_streak = 0
                self._publish_log(
                    "INFO",
                    (
                        f"camera {self.camera_index} connected backend={backend_name} "
                        f"{self.camera_width}x{self.camera_height}@{self.camera_fps}"
                    ),
                )
                return

            self._capture = None
            self._camera_backend_name = ""

    def _get_capture(self) -> Any:
        with self._capture_lock:
            return self._capture

    def _release_camera(self) -> None:
        with self._capture_lock:
            self._release_camera_locked()

    def _release_camera_locked(self) -> None:
        cap = self._capture
        self._capture = None
        self._camera_backend_name = ""
        if cap is None:
            return
        try:
            cap.release()
        except Exception:
            pass

    def _preferred_camera_backends(self) -> list[tuple[str, int]]:
        if self.camera_backend == "picamera2":
            return []
        backends: list[tuple[str, int]] = []
        if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
            backends.append(("dshow", int(cv2.CAP_DSHOW)))
        elif hasattr(cv2, "CAP_V4L2"):
            backends.append(("v4l2", int(cv2.CAP_V4L2)))
        if hasattr(cv2, "CAP_ANY"):
            any_backend = int(cv2.CAP_ANY)
            if not any(value == any_backend for _, value in backends):
                backends.append(("any", any_backend))
        if not backends:
            backends.append(("default", 0))
        return backends

    def _handle_capture_read_failure(self) -> bool:
        self._camera_read_fail_streak += 1
        if self._camera_read_fail_streak == 1:
            self._publish_log("WARNING", f"camera {self.camera_index} frame read failed")
            return False
        if self._camera_read_fail_streak < self._camera_read_fail_threshold:
            return False

        streak = self._camera_read_fail_streak
        self._camera_read_fail_streak = 0
        self._publish_log(
            "WARNING",
            f"camera {self.camera_index} read failure streak={streak}; reopening capture",
        )
        self._release_camera()
        return True

    def _handle_capture_read_success(self) -> None:
        if self._camera_read_fail_streak <= 0:
            return
        streak = self._camera_read_fail_streak
        self._camera_read_fail_streak = 0
        self._publish_log("INFO", f"camera {self.camera_index} frame read recovered after {streak} failure(s)")

    def _request_safe_stop(self, reason: str, *, emergency: bool) -> None:
        state = self._state_value()
        try:
            if hasattr(self.robot, "request_safe_stop"):
                self.robot.request_safe_stop(reason=reason, state=state, emergency=emergency)
                return
            if emergency and hasattr(self.robot, "send_estop"):
                self.robot.send_estop(reason=reason, state=state)
                return
            if hasattr(self.robot, "send_stop"):
                self.robot.send_stop(reason=reason, state=state)
        except Exception as exc:
            self._publish_log("ERROR", f"robot failsafe command failed: {exc}")

    def _state_value(self) -> str:
        state = self.fsm.state
        return state.value if isinstance(state, RobotState) else str(state)

    def _apply_profile(self, profile_name: str, *, source: str) -> bool:
        profile = self._profile_catalog.get(profile_name)
        if profile is None:
            self._publish_log("WARNING", f"unknown config profile: {profile_name!r}")
            return False

        self._active_profile_name = profile.name
        self._active_profile = profile
        self._recording_auto_start = bool(profile.recording.get("auto_start", False))

        camera = profile.camera
        self.camera_index = int(camera.get("index", self.camera_index))
        self.camera_width = int(camera.get("width", self.camera_width))
        self.camera_height = int(camera.get("height", self.camera_height))
        self.camera_fps = max(1, int(camera.get("fps", self.camera_fps)))
        if self._get_capture() is not None:
            self._open_camera()

        for key, value in profile.tuning.items():
            self._apply_tuning(key, value)

        self.session_recorder.configure(profile.recording)
        should_start_recording = self._recording_auto_start and source != "startup"
        if should_start_recording and not self.session_recorder.status_payload()["enabled"]:
            self.session_recorder.start(reason=f"profile:{profile.name}", context=self._session_context())

        self._publish_log(
            "INFO",
            (
                f"profile applied name={profile.name} source={source} "
                f"camera={self.camera_index}:{self.camera_width}x{self.camera_height}@{self.camera_fps} "
                f"rotation={self.camera_rotation}"
            ),
        )
        return True

    def _profile_status_payload(self) -> dict[str, Any]:
        active_description = self._active_profile.description if self._active_profile is not None else "manual base config"
        return {
            "active": self._active_profile_name,
            "description": active_description,
            "available": self._profile_catalog.available_payload(),
        }

    def _session_context(self) -> dict[str, Any]:
        return {
            "profile": self._active_profile_name,
            "robot_backend": self._robot_backend,
            "camera": {
                "index": self.camera_index,
                "width": self.camera_width,
                "height": self.camera_height,
                "fps": self.camera_fps,
                "rotation": self.camera_rotation,
            },
            "config_path": str(self.config_path or self._profile_catalog.config_path),
        }

    def _camera_status_payload(self) -> dict[str, Any]:
        if self._get_capture() is None:
            state = "offline"
        elif self._camera_fault_active or self._camera_read_fail_streak > 0:
            state = "degraded"
        else:
            state = "online"
        return {
            "state": state,
            "index": int(self.camera_index),
            "width": int(self.camera_width),
            "height": int(self.camera_height),
            "fps": int(self.camera_fps),
            "backend": self._camera_backend_name,
            "rotation": int(self.camera_rotation),
            "read_fail_streak": int(self._camera_read_fail_streak),
        }

    def _serial_status_payload(self) -> dict[str, Any]:
        payload_fn = getattr(self.robot, "status_payload", None)
        if callable(payload_fn):
            try:
                payload = payload_fn()
                if isinstance(payload, Mapping):
                    return dict(payload)
            except Exception as exc:
                self._publish_log("ERROR", f"robot status payload failed: {exc}")

        connected = bool(getattr(self.robot, "connected", False))
        dry_run = bool(getattr(getattr(self.robot, "_config", None), "dry_run", False))
        if dry_run:
            state = "dry-run"
        elif connected:
            state = "connected"
        elif bool(getattr(self.robot, "enabled", False)):
            state = "waiting"
        else:
            state = "disabled"
        return {
            "state": state,
            "connected": connected,
            "port": str(getattr(self.robot, "resolved_port", "") or ""),
            "backend": self._robot_backend,
            "telemetry": {},
            "control_mode": "",
            "line_error": None,
            "pid_output": None,
            "obstacle_state": "",
            "green_instruction": "",
            "green_route_decision": "",
            "motor_armed": bool(dry_run),
            "failsafe": False,
            "assist_kind": "none",
        }

    def _read_power_status(self, now: float) -> dict[str, Any]:
        if (now - self._last_power_sample_at) < 2.0:
            return dict(self._power_status_cache)
        self._last_power_sample_at = now
        binary = shutil.which("vcgencmd")
        if not binary:
            return {
                "available": False,
                "status": "unknown",
                "summary": "vcgencmd unavailable",
                "undervoltage_now": False,
                "undervoltage_occurred": False,
                "throttled_now": False,
                "throttled_occurred": False,
                "raw_value": "",
            }
        try:
            completed = subprocess.run(
                [binary, "get_throttled"],
                capture_output=True,
                text=True,
                timeout=0.4,
                check=False,
            )
        except Exception as exc:
            return {
                "available": False,
                "status": "unknown",
                "summary": f"power status failed: {exc}",
                "undervoltage_now": False,
                "undervoltage_occurred": False,
                "throttled_now": False,
                "throttled_occurred": False,
                "raw_value": "",
            }
        return self._parse_power_status(completed.stdout.strip() or completed.stderr.strip())

    @staticmethod
    def _parse_power_status(raw_text: str) -> dict[str, Any]:
        text = str(raw_text or "").strip().lower()
        if "0x" not in text:
            return {
                "available": False,
                "status": "unknown",
                "summary": raw_text or "vcgencmd returned no value",
                "undervoltage_now": False,
                "undervoltage_occurred": False,
                "throttled_now": False,
                "throttled_occurred": False,
                "raw_value": raw_text,
            }
        try:
            raw_value = int(text.split("0x", 1)[1], 16)
        except Exception:
            return {
                "available": False,
                "status": "unknown",
                "summary": raw_text,
                "undervoltage_now": False,
                "undervoltage_occurred": False,
                "throttled_now": False,
                "throttled_occurred": False,
                "raw_value": raw_text,
            }
        undervoltage_now = bool(raw_value & (1 << 0))
        throttled_now = bool(raw_value & (1 << 2))
        undervoltage_occurred = bool(raw_value & (1 << 16))
        throttled_occurred = bool(raw_value & (1 << 18))
        if undervoltage_now or throttled_now:
            status = "error"
            summary = "undervoltage/throttling active"
        elif undervoltage_occurred or throttled_occurred:
            status = "warn"
            summary = "power issue detected earlier"
        else:
            status = "ok"
            summary = "power stable"
        return {
            "available": True,
            "status": status,
            "summary": summary,
            "undervoltage_now": undervoltage_now,
            "undervoltage_occurred": undervoltage_occurred,
            "throttled_now": throttled_now,
            "throttled_occurred": throttled_occurred,
            "raw_value": f"0x{raw_value:x}",
        }

    @staticmethod
    def _is_valid_state_name(state_name: object) -> bool:
        return str(state_name).strip().upper() in RobotState._value2member_map_

    @staticmethod
    def _is_valid_state_object(state: object) -> bool:
        return isinstance(state, RobotState)

    def _publish_log(self, level: str, message: str) -> None:
        try:
            self.bus.publish(
                EventTopic.SYSTEM_LOG,
                LogEvent(
                    timestamp=time.time(),
                    level=level,
                    message=message,
                    source="live_runner",
                    state=self._state_value(),
                ),
            )
        except Exception:
            pass


def _parse_channel_group(value: str) -> tuple[int, ...]:
    raw = str(value or "").strip()
    if not raw:
        return ()
    try:
        return tuple(int(item.strip(), 0) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("channels must be comma-separated integers, e.g. 0,2") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live camera + AI runner for overengineering dashboard")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index")
    parser.add_argument("--camera-width", type=int, default=640, help="Camera capture width")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera capture height")
    parser.add_argument("--camera-fps", type=int, default=30, help="Camera capture FPS target")
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "picamera2", "opencv"),
        default="auto",
        help="Capture backend; auto prefers Picamera2 for the CSI camera and falls back to OpenCV",
    )
    parser.add_argument(
        "--camera-rotation",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help="Clockwise frame rotation applied before vision processing",
    )
    parser.add_argument("--headless", action="store_true", help="Run only the AI + remote relay server, without opening a local dashboard")
    parser.add_argument(
        "--config",
        type=str,
        default="New_AI/obr_overengineering_v1/configs/vision_config.json",
        help="Vision config path",
    )
    parser.add_argument("--remote-bind", type=str, default="0.0.0.0", help="Bind address for the remote dashboard relay")
    parser.add_argument("--remote-port", type=int, default=8765, help="TCP port for the remote dashboard relay")
    parser.add_argument("--remote-stream-fps", type=float, default=8.0, help="Max frame rate sent to the remote dashboard")
    parser.add_argument("--remote-jpeg-quality", type=int, default=70, help="JPEG quality used for remote frame streaming")
    parser.add_argument("--robot-serial-port", type=str, default="", help="Arduino serial port on the Raspberry, for example /dev/ttyACM0")
    parser.add_argument("--robot-baud", type=int, default=115200, help="Arduino serial baud rate")
    parser.add_argument("--robot-backend", choices=("serial", "pca9685"), default="serial", help="Robot control backend")
    parser.add_argument("--robot-green-forward-ms", type=int, default=5000, help="Milliseconds sent on GREEN integration test")
    parser.add_argument("--robot-green-streak", type=int, default=2, help="Matching live single-GREEN detections required before triggering; one short dropout remains stopped, while validated double-GREEN remains immediate")
    parser.add_argument("--robot-green-confirm-hold-ms", type=int, default=180, help="Stationary confirmation time for one green marker before choosing LEFT, RIGHT, or STRAIGHT")
    parser.add_argument("--robot-green-cooldown-ms", type=int, default=6000, help="Cooldown between GREEN motor commands")
    parser.add_argument("--robot-green-hold-ms", type=int, default=900, help="How long a GREEN assist stays valid on the Arduino")
    parser.add_argument("--robot-obstacle-hold-ms", type=int, default=1200, help="How long an obstacle assist stays valid on the Arduino")
    parser.add_argument("--robot-dry-run", action="store_true", help="Log robot commands without writing to serial")
    parser.add_argument(
        "--line-only",
        action="store_true",
        help="Keep the FSM in FOLLOWING_LINE and ignore rescue/victim/gap/intersection transitions",
    )
    parser.add_argument("--pca9685-i2c-address", type=lambda value: int(str(value), 0), default=0x40, help="PCA9685 I2C address")
    parser.add_argument("--pca9685-frequency-hz", type=int, default=50, help="PCA9685 PWM frequency")
    parser.add_argument("--pca9685-left-channel", type=int, default=4, help="PCA9685 channel for the left servo/motor signal")
    parser.add_argument("--pca9685-right-channel", type=int, default=0, help="PCA9685 channel for the right servo/motor signal")
    parser.add_argument(
        "--pca9685-left-channels",
        type=_parse_channel_group,
        default=(),
        help="Comma-separated PCA9685 channels for the left side, e.g. 0,2",
    )
    parser.add_argument(
        "--pca9685-right-channels",
        type=_parse_channel_group,
        default=(),
        help="Comma-separated PCA9685 channels for the right side, e.g. 1,3",
    )
    parser.add_argument("--pca9685-min-us", type=int, default=1000, help="Minimum servo pulse width in microseconds")
    parser.add_argument("--pca9685-neutral-us", type=int, default=1600, help="Calibrated neutral/stop pulse width in microseconds")
    parser.add_argument("--pca9685-max-us", type=int, default=2000, help="Maximum servo pulse width in microseconds")
    parser.add_argument("--pca9685-base-throttle-us", type=int, default=None, help="Legacy symmetric pulse offset fallback")
    parser.add_argument("--pca9685-left-base-throttle-us", type=int, default=300, help="Calibrated left-side forward pulse offset")
    parser.add_argument("--pca9685-right-base-throttle-us", type=int, default=200, help="Calibrated right-side forward pulse offset")
    parser.add_argument("--pca9685-turn-gain-us", type=int, default=220, help="Turn correction gain in microseconds per normalized offset")
    parser.add_argument("--pca9685-pid-kp-us", type=float, default=320.0, help="Line PID proportional gain")
    parser.add_argument("--pca9685-pid-ki-us", type=float, default=0.0, help="Line PID integral gain")
    parser.add_argument("--pca9685-pid-kd-us", type=float, default=12.0, help="Line PID derivative gain")
    parser.add_argument("--pca9685-pid-integral-limit", type=float, default=0.15, help="Maximum normalized line-error integral")
    parser.add_argument("--pca9685-pid-derivative-filter", type=float, default=0.60, help="Previous-derivative weight, 0..0.99")
    parser.add_argument("--pca9685-line-hold-ms", type=int, default=120, help="Milliseconds to hold the last safe line command across a brief detector gap")
    parser.add_argument(
        "--pca9685-gap-crossing-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue exactly straight across a white break after a confirmed straight line",
    )
    parser.add_argument("--pca9685-gap-straight-confirm-frames", type=int, default=3, help="Stable straight frames required before a gap may start")
    parser.add_argument("--pca9685-gap-reacquire-confirm-frames", type=int, default=2, help="Line frames required to finish a gap crossing")
    parser.add_argument("--pca9685-gap-crossing-timeout-ms", type=int, default=2200, help="Maximum bounded straight gap crossing duration")
    parser.add_argument("--pca9685-gap-max-entry-error", type=float, default=0.22, help="Maximum line offset allowed before entering a gap")
    parser.add_argument("--pca9685-gap-max-entry-bend", type=float, default=0.18, help="Maximum line bend allowed before entering a gap")
    parser.add_argument("--pca9685-max-output-us", type=int, default=240, help="Maximum signed pulse offset from neutral")
    parser.add_argument("--pca9685-line-error-deadband", type=float, default=0.025, help="Ignore small normalized line offsets around image center")
    parser.add_argument(
        "--pca9685-simple-line-follow",
        action="store_true",
        help="Bench test: use only calibrated straight base while a confident line is visible; no PID or line hold",
    )
    parser.add_argument(
        "--pca9685-basic-line-follow",
        action="store_true",
        help="Basic line follower: current line offset gives proportional steering; no integral, derivative, or hold",
    )
    parser.add_argument(
        "--pca9685-line-steering-inverted",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Invert only the physical left/right steering response for the installed camera orientation",
    )
    parser.add_argument(
        "--pca9685-sharp-corner-maneuver-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow detector-confirmed 90-degree corners to start the closed pivot maneuver",
    )
    parser.add_argument(
        "--pca9685-corner-sequence-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the calibrated approach, brake, fixed pivot, reacquire, and exit sequence for a confirmed L",
    )
    parser.add_argument(
        "--pca9685-corner-confirm-frames",
        type=int,
        default=2,
        help="Consecutive structural L frames required before the 90-degree sequence owns control",
    )
    parser.add_argument(
        "--pca9685-corner-min-elbow-row-contrast",
        type=float,
        default=1.25,
        help="Minimum widest-row to median-row ratio that separates a hard L from an open curve",
    )
    parser.add_argument(
        "--pca9685-corner-min-wide-row-occupancy",
        type=float,
        default=0.68,
        help="Minimum normalized width of the L elbow's widest image row",
    )
    parser.add_argument(
        "--pca9685-corner-approach-stop-row-ratio",
        type=float,
        default=0.40,
        help="Near-field elbow row that triggers neutral braking before the fixed pivot",
    )
    parser.add_argument(
        "--pca9685-corner-approach-throttle-scale",
        type=float,
        default=0.72,
        help="Throttle scale while approaching a confirmed L before braking",
    )
    parser.add_argument(
        "--pca9685-corner-approach-min-ms",
        type=int,
        default=350,
        help="Straight placement creep after confirming an L and before braking",
    )
    parser.add_argument(
        "--pca9685-corner-approach-left-min-ms",
        type=int,
        default=550,
        help="Left-only straight placement creep after confirming an L",
    )
    parser.add_argument(
        "--pca9685-corner-brake-ms",
        type=int,
        default=500,
        help="Neutral pause over the L before the calibrated pivot",
    )
    parser.add_argument(
        "--pca9685-corner-pivot-speed-us",
        type=int,
        default=300,
        help="Equal signed pulse distance from neutral on both sides during a fixed 90-degree pivot",
    )
    parser.add_argument(
        "--pca9685-corner-pivot-right-ms",
        type=int,
        default=1900,
        help="Physically calibrated right-pivot duration",
    )
    parser.add_argument(
        "--pca9685-corner-pivot-left-ms",
        type=int,
        default=2100,
        help="Left-pivot duration; independently configurable from the right side",
    )
    parser.add_argument(
        "--pca9685-corner-reacquire-speed-us",
        type=int,
        default=130,
        help="Slow same-direction pivot speed while centering the outgoing line",
    )
    parser.add_argument(
        "--pca9685-corner-reacquire-timeout-ms",
        type=int,
        default=1200,
        help="Bounded outgoing-line search; expiry stops at neutral without reversing",
    )
    parser.add_argument(
        "--pca9685-curve-lookahead-gain",
        type=float,
        default=0.0,
        help="Blend upcoming path bend into ordinary line steering, 0..1",
    )
    parser.add_argument(
        "--pca9685-curve-throttle-scale",
        type=float,
        default=1.0,
        help="Forward throttle scale used only while open-curve geometry is visible",
    )
    parser.add_argument(
        "--pca9685-ordinary-sharp-hold-ms",
        type=int,
        default=900,
        help="Short last-visible hold for ordinary sharp curves; closed 90-degree turns use their own hold",
    )
    parser.add_argument(
        "--pca9685-sharp-curve-threshold",
        type=float,
        default=0.55,
        help="Normalized error that switches the basic follower to an inside-wheel-stop pivot",
    )
    parser.add_argument(
        "--pca9685-sharp-curve-hold-ms",
        type=int,
        default=900,
        help="Minimum strong-pivot interval and ordinary sharp hold while the line leaves the camera",
    )
    parser.add_argument(
        "--pca9685-sharp-curve-outer-scale",
        type=float,
        default=1.0,
        help="Forward scale applied only to the outside side during a sharp pivot",
    )
    parser.add_argument(
        "--pca9685-sharp-curve-inner-reverse-scale",
        type=float,
        default=1.0,
        help="Reverse scale for the inside side during a detector-confirmed 90-degree in-place pivot",
    )
    parser.add_argument(
        "--pca9685-sharp-curve-entry-min-row-ratio",
        type=float,
        default=0.0,
        help="Delay a confirmed 90-degree pivot until the elbow reaches this normalized ROI row",
    )
    parser.add_argument(
        "--pca9685-sharp-curve-visible-commit-ms",
        type=int,
        default=2600,
        help="Absolute corner commitment bound while waiting for the outgoing straight, including blind time",
    )
    parser.add_argument(
        "--pca9685-sharp-curve-finish-inner-reverse-scale",
        type=float,
        default=1.0,
        help="Inside reverse retained after the blind hold so a 90-degree turn stays closed",
    )
    parser.add_argument(
        "--pca9685-sharp-curve-exit-settle-ms",
        type=int,
        default=320,
        help="Anti-zig-zag stabilization window after the outgoing straight is reacquired",
    )
    parser.add_argument(
        "--pca9685-sharp-curve-exit-throttle-scale",
        type=float,
        default=1.0,
        help="Forward throttle scale during post-corner stabilization",
    )
    parser.add_argument(
        "--pca9685-sharp-curve-exit-max-correction-us",
        type=float,
        default=0.0,
        help="Maximum same-direction correction during post-corner stabilization",
    )
    parser.add_argument(
        "--pca9685-sharp-curve-recovery-ms",
        type=int,
        default=520,
        help="Gentle proportional recovery window after calibrated-straight stabilization",
    )
    parser.add_argument(
        "--pca9685-sharp-curve-recovery-max-correction-us",
        type=float,
        default=45.0,
        help="Maximum non-pivot correction during post-corner recovery",
    )
    parser.add_argument("--pca9685-green-turn-us", type=int, default=260, help="Signed pulse offset used for green maneuvers")
    parser.add_argument("--pca9685-green-half-turn-us", type=int, default=300, help="Calibrated signed pulse offset for two-green 180-degree pivot")
    parser.add_argument("--pca9685-green-half-turn-ms", type=int, default=4000, help="Total powered pivot time for the calibrated 1900 ms first turn plus second-turn trim")
    parser.add_argument("--pca9685-green-half-turn-first-ms", type=int, default=1900, help="First calibrated 90-degree segment for the two-green maneuver")
    parser.add_argument("--pca9685-green-half-turn-second-ms", type=int, default=2100, help="Second calibrated turn segment for the two-green maneuver")
    parser.add_argument("--pca9685-green-half-turn-reverse-ms", type=int, default=550, help="Short calibrated reverse between the two green turn segments")
    parser.add_argument(
        "--pca9685-green-maneuvers-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow green navigation maneuvers while the FSM remains in line-only mode",
    )
    parser.add_argument(
        "--pca9685-start-disarmed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep real PCA9685 outputs neutral until the dashboard START command arms motion",
    )
    parser.add_argument("--pca9685-left-inverted", action=argparse.BooleanOptionalAction, default=False, help="Invert left servo/motor signal")
    parser.add_argument("--pca9685-right-inverted", action=argparse.BooleanOptionalAction, default=True, help="Invert right servo/motor signal")
    parser.add_argument("--enable-leds", action="store_true", help="Enable the two optional GPIO status LED outputs")
    parser.add_argument("--led1-gpio", type=int, default=18, help="GPIO number for LED1 (physical pin 12 by default)")
    parser.add_argument("--led2-gpio", type=int, default=23, help="GPIO number for LED2 (physical pin 16 by default)")
    parser.add_argument(
        "--enable-master-switch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the pull-up master switch between a GPIO and ground",
    )
    parser.add_argument(
        "--master-switch-gpio",
        type=int,
        default=17,
        help="BCM GPIO for the master switch (GPIO17 is physical pin 11)",
    )
    parser.add_argument(
        "--master-switch-debounce-ms",
        type=int,
        default=80,
        help="Hardware switch debounce interval",
    )
    parser.add_argument("--profile", type=str, default="", help="Named ops profile from configs/vision_config.json")
    parser.add_argument("--recordings-root", type=str, default="", help="Output directory for session recordings")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = LiveDashboardRunner(
        camera_index=args.camera_index,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        camera_backend=args.camera_backend,
        camera_rotation=args.camera_rotation,
        config_path=args.config,
        robot_serial_port=args.robot_serial_port,
        robot_baud=args.robot_baud,
        robot_green_forward_ms=args.robot_green_forward_ms,
        robot_green_streak=args.robot_green_streak,
        robot_green_confirm_hold_ms=args.robot_green_confirm_hold_ms,
        robot_green_cooldown_ms=args.robot_green_cooldown_ms,
        robot_green_hold_ms=args.robot_green_hold_ms,
        robot_obstacle_hold_ms=args.robot_obstacle_hold_ms,
        robot_dry_run=args.robot_dry_run,
        robot_backend=args.robot_backend,
        pca9685_i2c_address=args.pca9685_i2c_address,
        pca9685_frequency_hz=args.pca9685_frequency_hz,
        pca9685_left_channel=args.pca9685_left_channel,
        pca9685_right_channel=args.pca9685_right_channel,
        pca9685_left_channels=args.pca9685_left_channels,
        pca9685_right_channels=args.pca9685_right_channels,
        pca9685_min_us=args.pca9685_min_us,
        pca9685_neutral_us=args.pca9685_neutral_us,
        pca9685_max_us=args.pca9685_max_us,
        pca9685_base_throttle_us=args.pca9685_base_throttle_us,
        pca9685_left_base_throttle_us=args.pca9685_left_base_throttle_us,
        pca9685_right_base_throttle_us=args.pca9685_right_base_throttle_us,
        pca9685_turn_gain_us=args.pca9685_turn_gain_us,
        pca9685_pid_kp_us=args.pca9685_pid_kp_us,
        pca9685_pid_ki_us=args.pca9685_pid_ki_us,
        pca9685_pid_kd_us=args.pca9685_pid_kd_us,
        pca9685_pid_integral_limit=args.pca9685_pid_integral_limit,
        pca9685_pid_derivative_filter=args.pca9685_pid_derivative_filter,
        pca9685_line_hold_ms=args.pca9685_line_hold_ms,
        pca9685_gap_crossing_enabled=args.pca9685_gap_crossing_enabled,
        pca9685_gap_straight_confirm_frames=args.pca9685_gap_straight_confirm_frames,
        pca9685_gap_reacquire_confirm_frames=args.pca9685_gap_reacquire_confirm_frames,
        pca9685_gap_crossing_timeout_ms=args.pca9685_gap_crossing_timeout_ms,
        pca9685_gap_max_entry_error=args.pca9685_gap_max_entry_error,
        pca9685_gap_max_entry_bend=args.pca9685_gap_max_entry_bend,
        pca9685_max_output_us=args.pca9685_max_output_us,
        pca9685_line_error_deadband=args.pca9685_line_error_deadband,
        pca9685_simple_line_follow=args.pca9685_simple_line_follow,
        pca9685_basic_line_follow=args.pca9685_basic_line_follow,
        pca9685_line_steering_inverted=args.pca9685_line_steering_inverted,
        pca9685_sharp_corner_maneuver_enabled=args.pca9685_sharp_corner_maneuver_enabled,
        pca9685_corner_sequence_enabled=args.pca9685_corner_sequence_enabled,
        pca9685_corner_confirm_frames=args.pca9685_corner_confirm_frames,
        pca9685_corner_min_elbow_row_contrast=args.pca9685_corner_min_elbow_row_contrast,
        pca9685_corner_min_wide_row_occupancy=args.pca9685_corner_min_wide_row_occupancy,
        pca9685_corner_approach_stop_row_ratio=args.pca9685_corner_approach_stop_row_ratio,
        pca9685_corner_approach_throttle_scale=args.pca9685_corner_approach_throttle_scale,
        pca9685_corner_approach_min_ms=args.pca9685_corner_approach_min_ms,
        pca9685_corner_approach_left_min_ms=args.pca9685_corner_approach_left_min_ms,
        pca9685_corner_brake_ms=args.pca9685_corner_brake_ms,
        pca9685_corner_pivot_speed_us=args.pca9685_corner_pivot_speed_us,
        pca9685_corner_pivot_right_ms=args.pca9685_corner_pivot_right_ms,
        pca9685_corner_pivot_left_ms=args.pca9685_corner_pivot_left_ms,
        pca9685_corner_reacquire_speed_us=args.pca9685_corner_reacquire_speed_us,
        pca9685_corner_reacquire_timeout_ms=args.pca9685_corner_reacquire_timeout_ms,
        pca9685_curve_lookahead_gain=args.pca9685_curve_lookahead_gain,
        pca9685_curve_throttle_scale=args.pca9685_curve_throttle_scale,
        pca9685_ordinary_sharp_hold_ms=args.pca9685_ordinary_sharp_hold_ms,
        pca9685_sharp_curve_threshold=args.pca9685_sharp_curve_threshold,
        pca9685_sharp_curve_hold_ms=args.pca9685_sharp_curve_hold_ms,
        pca9685_sharp_curve_outer_scale=args.pca9685_sharp_curve_outer_scale,
        pca9685_sharp_curve_inner_reverse_scale=args.pca9685_sharp_curve_inner_reverse_scale,
        pca9685_sharp_curve_entry_min_row_ratio=args.pca9685_sharp_curve_entry_min_row_ratio,
        pca9685_sharp_curve_visible_commit_ms=args.pca9685_sharp_curve_visible_commit_ms,
        pca9685_sharp_curve_finish_inner_reverse_scale=args.pca9685_sharp_curve_finish_inner_reverse_scale,
        pca9685_sharp_curve_exit_settle_ms=args.pca9685_sharp_curve_exit_settle_ms,
        pca9685_sharp_curve_exit_throttle_scale=args.pca9685_sharp_curve_exit_throttle_scale,
        pca9685_sharp_curve_exit_max_correction_us=args.pca9685_sharp_curve_exit_max_correction_us,
        pca9685_sharp_curve_recovery_ms=args.pca9685_sharp_curve_recovery_ms,
        pca9685_sharp_curve_recovery_max_correction_us=args.pca9685_sharp_curve_recovery_max_correction_us,
        pca9685_green_turn_us=args.pca9685_green_turn_us,
        pca9685_green_half_turn_us=args.pca9685_green_half_turn_us,
        pca9685_green_half_turn_ms=args.pca9685_green_half_turn_ms,
        pca9685_green_half_turn_first_ms=args.pca9685_green_half_turn_first_ms,
        pca9685_green_half_turn_second_ms=args.pca9685_green_half_turn_second_ms,
        pca9685_green_half_turn_reverse_ms=args.pca9685_green_half_turn_reverse_ms,
        pca9685_green_maneuvers_enabled=args.pca9685_green_maneuvers_enabled,
        pca9685_start_disarmed=args.pca9685_start_disarmed,
        pca9685_left_inverted=args.pca9685_left_inverted,
        pca9685_right_inverted=args.pca9685_right_inverted,
        line_only=args.line_only,
        enable_leds=args.enable_leds,
        led1_gpio=args.led1_gpio,
        led2_gpio=args.led2_gpio,
        enable_master_switch=args.enable_master_switch,
        master_switch_gpio=args.master_switch_gpio,
        master_switch_debounce_ms=args.master_switch_debounce_ms,
        profile_name=(args.profile or "").strip() or None,
        recordings_root=(args.recordings_root or "").strip() or None,
    )
    remote_server: RemoteDashboardServer | None = None
    runner.start()
    try:
        if args.headless:
            remote_server = RemoteDashboardServer(
                runner.bus,
                host=args.remote_bind,
                port=args.remote_port,
                stream_fps=args.remote_stream_fps,
                jpeg_quality=args.remote_jpeg_quality,
            )
            remote_server.start()
            runner._publish_log(
                "INFO",
                (
                    f"remote dashboard relay listening on {args.remote_bind}:{args.remote_port} "
                    f"fps={args.remote_stream_fps} jpeg_q={args.remote_jpeg_quality}"
                ),
            )
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                return 0
        if __package__ in (None, ""):
            from src.ui_overengineering.dashboard import run_dashboard
        else:
            from .ui_overengineering.dashboard import run_dashboard
        return run_dashboard(
            event_bus=runner.bus,
            camera_index=args.camera_index,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
            mock=False,
            config_path=args.config,
        )
    finally:
        if remote_server is not None:
            remote_server.stop()
        runner.stop()


if __name__ == "__main__":
    raise SystemExit(main())
