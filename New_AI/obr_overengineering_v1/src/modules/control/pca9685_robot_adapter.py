from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

try:
    from ...core.event_bus import EventBus, EventTopic, LogEvent, UICommandEvent, VisionDetectionEvent
    from ...core.state_machine import RobotState
except ImportError:  # pragma: no cover
    from core.event_bus import EventBus, EventTopic, LogEvent, UICommandEvent, VisionDetectionEvent
    from core.state_machine import RobotState


class PwmDriver(Protocol):
    def set_pwm_us(self, channel: int, pulse_us: int) -> None:
        ...

    def disable_all(self) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass(slots=True)
class Pca9685RobotConfig:
    i2c_address: int = 0x40
    frequency_hz: int = 50
    # Physical wiring confirmed on the robot: right rear/front = 0/1 and
    # left rear/front = 4/5.
    left_channel: int = 4
    right_channel: int = 0
    min_us: int = 1000
    # Bench calibration: all four continuous-rotation channels stop at 1600 us.
    neutral_us: int = 1600
    max_us: int = 2000
    # Legacy symmetric throttle fallback.  When omitted, the calibrated
    # per-side values below are used.
    base_throttle_us: int | None = None
    # Physical straight-line calibration: left=1900 us and right=1400 us
    # correspond to logical forward speed 300 and 200 from the 1600 us stop.
    left_base_throttle_us: int | None = None
    right_base_throttle_us: int | None = None
    turn_gain_us: int = 220
    # PID gains for normalized line error. If kp is omitted, turn_gain_us
    # remains the backwards-compatible proportional-only setting.
    pid_kp_us: float | None = None
    pid_ki_us: float = 0.0
    pid_kd_us: float = 0.0
    pid_integral_limit: float = 0.25
    pid_derivative_filter: float = 0.45
    max_output_us: int = 360
    # Ignore tiny visual jitter around the image center. The runner enables a
    # small deadband for field operation; zero keeps the adapter legacy-neutral.
    line_error_deadband: float = 0.0
    green_turn_us: int = 260
    green_hold_ms: int = 900
    # Two valid green squares on opposite sides command the same calibrated
    # in-place pivot used by two 90-degree turns, without changing one-green
    # behavior that will be tuned separately.
    green_half_turn_us: int = 300
    # Total powered pivot time. The maneuver divides this into two calibrated
    # 90-degree segments with a neutral brake before and between them.
    green_half_turn_ms: int = 4000
    green_half_turn_first_ms: int | None = None
    green_half_turn_second_ms: int | None = None
    green_half_turn_reverse_ms: int = 550
    green_half_turn_confidence_floor: float = 0.90
    # A geometrically validated pair is already a strong signal, so the
    # half-turn must preempt line/corner control on its first live frame.
    # Single-green instructions retain the independent legacy streak below.
    green_half_turn_trigger_streak: int = 1
    # A lone marker requires two matching live observations.  The confirmation
    # pause tolerates one short camera dropout; a validated pair remains
    # immediate and always has priority.
    green_trigger_streak: int = 2
    # A single marker may command a corner only when the vision pipeline has
    # separated it clearly from the transverse black line. Ambiguous and
    # after-line markers suppress that intersection's structural L instead.
    green_single_relation_confidence_floor: float = 0.55
    green_single_near_y_ratio: float = 0.55
    green_single_clear_frames: int = 3
    # Minimum stationary confirmation time for one green marker.  The live
    # runner enables a short hold; zero preserves direct adapter compatibility.
    green_single_confirm_hold_ms: int = 0
    green_cooldown_ms: int = 6000
    # A double-green encounter is one-shot. Re-arm only after the maneuver has
    # ended and several consecutive frames contain no green marker at all.
    green_rearm_clear_frames: int = 5
    line_confidence_floor: float = 0.18
    control_timeout_ms: int = 700
    # Keep the last forward/steering command for a very short camera gap.
    # This prevents one dropped frame from stopping and restarting sideways;
    # a real loss still reaches the normal watchdog stop.
    line_hold_ms: int = 180
    line_hold_throttle_scale: float = 0.72
    # Cross a real break in an otherwise straight line without re-enabling the
    # legacy global gap FSM. The deployment opts in; defaults preserve every
    # existing caller and maneuver.
    gap_crossing_enabled: bool = False
    gap_straight_confirm_frames: int = 3
    gap_reacquire_confirm_frames: int = 2
    # Do not mistake the flickering tail of the entry segment for the line on
    # the far side. Keep driving straight long enough for the chassis to enter
    # the white gap before line reacquisition is allowed.
    gap_reacquire_min_ms: int = 2400
    gap_crossing_timeout_ms: int = 2200
    gap_max_entry_error: float = 0.22
    gap_max_entry_bend: float = 0.18
    # Keep a recently proven straight approach armed while the visible line
    # tip slides toward the edge of the USB-camera frame. A real curve clears
    # this latch immediately; lateral endpoint drift alone does not.
    gap_straight_memory_ms: int = 1200
    # Deliberate bench mode: drive only with the calibrated straight base when
    # a confident line is visible. No PID, steering correction, or line hold.
    simple_line_follow: bool = False
    # Basic proportional line follower for bench/field validation. It uses
    # only the current image offset; integral, derivative, and line hold stay
    # disabled in this mode.
    basic_line_follow: bool = False
    # Some camera/chassis mountings mirror the physical steering response:
    # a line on the right side of the image requires a physical left command.
    # Keep this separate from motor inversion, which only maps signed speed to
    # each continuous-rotation servo's pulse direction.
    line_steering_inverted: bool = False
    # Keep the detector-driven closed 90-degree maneuver independently
    # switchable. Ordinary proportional and sharp-curve steering remain
    # available when this is disabled.
    sharp_corner_maneuver_enabled: bool = True
    # The calibrated sequence is opt-in so legacy callers keep their previous
    # corner behavior. Field deployment enables it together with the detector
    # gate above. It owns a confirmed L from approach through exit and never
    # lets ordinary curve steering interrupt the fixed physical pivot.
    corner_sequence_enabled: bool = False
    corner_confirm_frames: int = 2
    corner_min_elbow_row_contrast: float = 1.25
    corner_min_wide_row_occupancy: float = 0.68
    corner_min_abs_bend: float = 0.18
    # USB-camera fallback for a real L whose binary right-angle flag flickers
    # off.  It deliberately requires a near-field, wide, high-contrast elbow
    # so the validated open curve keeps using ordinary lookahead steering.
    corner_fallback_min_row_ratio: float = 0.78
    corner_fallback_min_elbow_row_contrast: float = 1.55
    # The USB camera can see a left L slightly narrower than the symmetric
    # corridor gate.  Keep this fallback left-only and require three stable
    # frames so ordinary open curves (whose row contrast stays near 1.0) are
    # never promoted to the calibrated 90-degree sequence.
    corner_left_fallback_confirm_frames: int = 3
    corner_left_fallback_min_row_ratio: float = 0.70
    corner_left_fallback_min_elbow_row_contrast: float = 1.45
    corner_left_fallback_min_wide_row_occupancy: float = 0.55
    corner_left_fallback_min_width_ratio: float = 0.60
    corner_left_fallback_min_height_ratio: float = 0.75
    corner_approach_stop_row_ratio: float = 0.40
    corner_approach_throttle_scale: float = 0.72
    # Optional deterministic straight creep after a structural L is confirmed.
    # The field runner enables this so the chassis crosses the elbow slightly
    # before braking; zero preserves the legacy row-gated approach.
    corner_approach_min_ms: int = 0
    # The left 90-degree corner needs a slightly deeper placement over the L.
    # None keeps the shared duration for callers that do not opt in.
    corner_approach_left_min_ms: int | None = None
    corner_approach_timeout_ms: int = 1400
    corner_approach_lost_frames: int = 3
    corner_brake_ms: int = 500
    corner_pivot_speed_us: int = 300
    corner_pivot_right_ms: int = 1900
    corner_pivot_left_ms: int = 2100
    # Pulse 1300 (left pivot) rotates this chassis substantially faster than
    # pulse 1900 (the proven right pivot).  Use a deliberately slower left-only
    # command and let vision brake on the first outgoing line after the incoming
    # leg has left the camera.  Right-side timing and power remain untouched.
    corner_pivot_left_speed_us: int = 150
    corner_pivot_left_vision_min_ms: int = 600
    corner_pivot_left_lost_confirm_frames: int = 2
    corner_reacquire_speed_us: int = 130
    corner_reacquire_timeout_ms: int = 1200
    corner_reacquire_confirm_frames: int = 3
    corner_reacquire_max_offset: float = 0.38
    corner_reacquire_max_bend: float = 0.28
    corner_exit_neutral_ms: int = 150
    corner_exit_straight_ms: int = 320
    corner_cooldown_ms: int = 800
    # A wide camera sees the path curvature before the near-field line offset
    # changes. Blend a bounded amount of that bend into ordinary steering and
    # slow only while curve geometry is present. Defaults preserve legacy
    # behavior for callers that do not opt in.
    curve_lookahead_gain: float = 0.0
    curve_throttle_scale: float = 1.0
    ordinary_sharp_hold_ms: int | None = None
    # Tight turns stop the inside side instead of merely slowing it. A
    # detector-confirmed 90-degree corner counter-rotates the inside side at
    # the calibrated base speed, producing an in-place skid-steer pivot.
    sharp_curve_threshold: float = 0.55
    sharp_curve_hold_ms: int = 900
    sharp_curve_outer_scale: float = 1.0
    sharp_curve_inner_reverse_scale: float = 1.0
    # A wide camera sees the elbow before it reaches the chassis.  Delay the
    # closed pivot until the detector's widest-row band is near the bottom of
    # the line ROI. Zero preserves the legacy immediate-entry behavior.
    sharp_curve_entry_min_row_ratio: float = 0.0
    # Keep the closed pivot through the camera blind spot. The recorded 35%
    # finish produced a forward arc, moved the camera away from the outgoing
    # leg, and then hit the old 2100 ms timeout on white floor. Keep equal
    # counter-rotation until a centered outgoing straight is confirmed.
    sharp_curve_visible_commit_ms: int = 2600
    sharp_curve_finish_inner_reverse_scale: float = 1.0
    sharp_curve_exit_settle_ms: int = 320
    sharp_curve_exit_throttle_scale: float = 1.0
    sharp_curve_exit_max_correction_us: float = 0.0
    sharp_curve_recovery_ms: int = 520
    sharp_curve_recovery_max_correction_us: float = 45.0
    monitor_interval_ms: int = 50
    # Bench calibration: left channels 4/5 go forward above neutral; right
    # channels 0/1 go forward below neutral.
    left_inverted: bool = False
    right_inverted: bool = True
    enable_green_maneuvers: bool = True
    dry_run: bool = False
    # A real-PWM runner can come online safely without moving.  The dashboard
    # START command explicitly arms outputs; STOP disarms them again.
    start_disarmed: bool = False
    # A side may drive more than one continuous-rotation servo.  The legacy
    # singular fields remain supported so existing callers keep working.
    left_channels: tuple[int, ...] = ()
    right_channels: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        left = tuple(int(channel) for channel in self.left_channels) or (int(self.left_channel),)
        right = tuple(int(channel) for channel in self.right_channels) or (int(self.right_channel),)
        channels = left + right
        if not left or not right:
            raise ValueError("PCA9685 must have at least one left and one right channel")
        if any(channel < 0 or channel > 15 for channel in channels):
            raise ValueError("PCA9685 channels must be between 0 and 15")
        if len(set(channels)) != len(channels):
            raise ValueError("PCA9685 left/right channel groups must not overlap")
        self.left_channels = left
        self.right_channels = right
        self.left_channel = left[0]
        self.right_channel = right[0]

        legacy_base = None if self.base_throttle_us is None else max(0, int(self.base_throttle_us))
        left_base = self.left_base_throttle_us
        right_base = self.right_base_throttle_us
        self.left_base_throttle_us = max(
            0,
            int(300 if left_base is None and legacy_base is None else legacy_base if left_base is None else left_base),
        )
        self.right_base_throttle_us = max(
            0,
            int(200 if right_base is None and legacy_base is None else legacy_base if right_base is None else right_base),
        )
        self.pid_kp_us = max(
            0.0,
            float(self.turn_gain_us if self.pid_kp_us is None else self.pid_kp_us),
        )
        self.pid_ki_us = float(self.pid_ki_us)
        self.pid_kd_us = float(self.pid_kd_us)
        self.pid_integral_limit = max(0.0, float(self.pid_integral_limit))
        self.pid_derivative_filter = max(0.0, min(0.99, float(self.pid_derivative_filter)))
        self.line_error_deadband = max(0.0, min(0.25, float(self.line_error_deadband)))
        self.green_turn_us = max(0, min(400, int(self.green_turn_us)))
        self.green_hold_ms = max(100, min(5000, int(self.green_hold_ms)))
        self.green_half_turn_us = max(80, min(400, int(self.green_half_turn_us)))
        self.green_half_turn_ms = max(500, min(8000, int(self.green_half_turn_ms)))
        self.green_half_turn_confidence_floor = max(
            0.0,
            min(1.0, float(self.green_half_turn_confidence_floor)),
        )
        self.green_half_turn_trigger_streak = max(
            1,
            min(4, int(self.green_half_turn_trigger_streak)),
        )
        self.green_trigger_streak = max(1, min(8, int(self.green_trigger_streak)))
        self.green_single_relation_confidence_floor = max(
            0.0,
            min(1.0, float(self.green_single_relation_confidence_floor)),
        )
        self.green_single_near_y_ratio = max(
            0.0,
            min(1.0, float(self.green_single_near_y_ratio)),
        )
        self.green_single_clear_frames = max(
            1,
            min(12, int(self.green_single_clear_frames)),
        )
        self.green_single_confirm_hold_ms = max(
            0,
            min(1500, int(self.green_single_confirm_hold_ms)),
        )
        self.start_disarmed = bool(self.start_disarmed)
        self.green_cooldown_ms = max(0, min(30000, int(self.green_cooldown_ms)))
        self.green_rearm_clear_frames = max(1, min(30, int(self.green_rearm_clear_frames)))
        self.line_hold_ms = max(0, int(self.line_hold_ms))
        self.line_hold_throttle_scale = max(0.0, min(1.0, float(self.line_hold_throttle_scale)))
        self.gap_crossing_enabled = bool(self.gap_crossing_enabled)
        self.gap_straight_confirm_frames = max(
            2,
            min(12, int(self.gap_straight_confirm_frames)),
        )
        self.gap_reacquire_confirm_frames = max(
            1,
            min(8, int(self.gap_reacquire_confirm_frames)),
        )
        self.gap_reacquire_min_ms = max(
            100,
            min(2500, int(self.gap_reacquire_min_ms)),
        )
        self.gap_crossing_timeout_ms = max(
            300,
            min(5000, int(self.gap_crossing_timeout_ms)),
        )
        self.gap_max_entry_error = max(
            0.05,
            min(0.45, float(self.gap_max_entry_error)),
        )
        self.gap_max_entry_bend = max(
            0.05,
            min(0.45, float(self.gap_max_entry_bend)),
        )
        self.gap_straight_memory_ms = max(
            200,
            min(2500, int(self.gap_straight_memory_ms)),
        )
        self.corner_confirm_frames = max(2, min(8, int(self.corner_confirm_frames)))
        self.corner_min_elbow_row_contrast = max(
            1.05,
            min(4.0, float(self.corner_min_elbow_row_contrast)),
        )
        self.corner_min_wide_row_occupancy = max(
            0.35,
            min(1.0, float(self.corner_min_wide_row_occupancy)),
        )
        self.corner_min_abs_bend = max(0.05, min(0.8, float(self.corner_min_abs_bend)))
        self.corner_fallback_min_row_ratio = max(
            0.60,
            min(1.0, float(self.corner_fallback_min_row_ratio)),
        )
        self.corner_fallback_min_elbow_row_contrast = max(
            1.25,
            min(4.0, float(self.corner_fallback_min_elbow_row_contrast)),
        )
        self.corner_approach_stop_row_ratio = max(
            0.35,
            min(1.0, float(self.corner_approach_stop_row_ratio)),
        )
        self.corner_approach_throttle_scale = max(
            0.45,
            min(1.0, float(self.corner_approach_throttle_scale)),
        )
        self.corner_approach_min_ms = max(
            0,
            min(500, int(self.corner_approach_min_ms)),
        )
        left_approach_ms = (
            self.corner_approach_min_ms
            if self.corner_approach_left_min_ms is None
            else int(self.corner_approach_left_min_ms)
        )
        self.corner_approach_left_min_ms = max(0, min(1000, left_approach_ms))
        self.corner_approach_timeout_ms = max(
            250,
            min(3000, int(self.corner_approach_timeout_ms)),
        )
        self.corner_approach_lost_frames = max(
            1,
            min(10, int(self.corner_approach_lost_frames)),
        )
        self.corner_brake_ms = max(100, min(800, int(self.corner_brake_ms)))
        self.corner_pivot_speed_us = max(80, min(400, int(self.corner_pivot_speed_us)))
        self.corner_pivot_right_ms = max(250, min(4000, int(self.corner_pivot_right_ms)))
        self.corner_pivot_left_ms = max(250, min(4000, int(self.corner_pivot_left_ms)))
        if self.green_half_turn_first_ms is None and self.green_half_turn_second_ms is None:
            first_ms = min(self.corner_pivot_right_ms, self.green_half_turn_ms - 250)
            second_ms = self.green_half_turn_ms - first_ms
        else:
            first_ms = int(
                self.corner_pivot_right_ms
                if self.green_half_turn_first_ms is None
                else self.green_half_turn_first_ms
            )
            second_ms = int(
                max(250, self.green_half_turn_ms - first_ms)
                if self.green_half_turn_second_ms is None
                else self.green_half_turn_second_ms
            )
        self.green_half_turn_first_ms = max(250, min(4000, int(first_ms)))
        self.green_half_turn_second_ms = max(250, min(4000, int(second_ms)))
        self.green_half_turn_reverse_ms = max(
            0,
            min(1500, int(self.green_half_turn_reverse_ms)),
        )
        self.green_half_turn_ms = (
            int(self.green_half_turn_first_ms) + int(self.green_half_turn_second_ms)
        )
        self.corner_reacquire_speed_us = max(
            60,
            min(self.corner_pivot_speed_us, int(self.corner_reacquire_speed_us)),
        )
        self.corner_reacquire_timeout_ms = max(
            300,
            min(3000, int(self.corner_reacquire_timeout_ms)),
        )
        self.corner_reacquire_confirm_frames = max(
            2,
            min(8, int(self.corner_reacquire_confirm_frames)),
        )
        self.corner_reacquire_max_offset = max(
            0.10,
            min(0.50, float(self.corner_reacquire_max_offset)),
        )
        self.corner_reacquire_max_bend = max(
            0.08,
            min(0.50, float(self.corner_reacquire_max_bend)),
        )
        self.corner_exit_neutral_ms = max(50, min(600, int(self.corner_exit_neutral_ms)))
        self.corner_exit_straight_ms = max(0, min(1000, int(self.corner_exit_straight_ms)))
        self.corner_cooldown_ms = max(0, min(3000, int(self.corner_cooldown_ms)))
        self.sharp_curve_threshold = max(0.25, min(0.95, float(self.sharp_curve_threshold)))
        self.sharp_curve_hold_ms = max(0, min(1500, int(self.sharp_curve_hold_ms)))
        self.curve_lookahead_gain = max(0.0, min(1.0, float(self.curve_lookahead_gain)))
        self.curve_throttle_scale = max(0.45, min(1.0, float(self.curve_throttle_scale)))
        self.ordinary_sharp_hold_ms = max(
            0,
            min(
                1500,
                int(
                    self.sharp_curve_hold_ms
                    if self.ordinary_sharp_hold_ms is None
                    else self.ordinary_sharp_hold_ms
                ),
            ),
        )
        self.sharp_curve_outer_scale = max(0.5, min(1.35, float(self.sharp_curve_outer_scale)))
        self.sharp_curve_inner_reverse_scale = max(
            0.0,
            min(1.0, float(self.sharp_curve_inner_reverse_scale)),
        )
        self.sharp_curve_entry_min_row_ratio = max(
            0.0,
            min(1.0, float(self.sharp_curve_entry_min_row_ratio)),
        )
        self.sharp_curve_visible_commit_ms = max(
            self.sharp_curve_hold_ms,
            min(3500, int(self.sharp_curve_visible_commit_ms)),
        )
        self.sharp_curve_finish_inner_reverse_scale = max(
            0.0,
            min(
                self.sharp_curve_inner_reverse_scale,
                float(self.sharp_curve_finish_inner_reverse_scale),
            ),
        )
        self.sharp_curve_exit_settle_ms = max(
            0,
            min(1000, int(self.sharp_curve_exit_settle_ms)),
        )
        self.sharp_curve_exit_throttle_scale = max(
            0.45,
            min(1.0, float(self.sharp_curve_exit_throttle_scale)),
        )
        self.sharp_curve_exit_max_correction_us = max(
            0.0,
            min(180.0, float(self.sharp_curve_exit_max_correction_us)),
        )
        self.sharp_curve_recovery_ms = max(
            0,
            min(1500, int(self.sharp_curve_recovery_ms)),
        )
        self.sharp_curve_recovery_max_correction_us = max(
            0.0,
            min(120.0, float(self.sharp_curve_recovery_max_correction_us)),
        )


class _AdafruitPca9685Driver:
    def __init__(self, config: Pca9685RobotConfig) -> None:
        try:
            import board  # type: ignore
            import busio  # type: ignore
            from adafruit_pca9685 import PCA9685  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on Pi packages
            raise RuntimeError(
                "PCA9685 dependencies missing. Install adafruit-circuitpython-pca9685 on the Raspberry Pi."
            ) from exc

        self._frequency_hz = max(1, int(config.frequency_hz))
        self._i2c = busio.I2C(board.SCL, board.SDA)
        self._pca = PCA9685(self._i2c, address=int(config.i2c_address))
        self._pca.frequency = self._frequency_hz

    def _set_duty_cycle_with_retry(self, channel: int, duty_cycle: int) -> None:
        attempts = 4
        for attempt in range(attempts):
            try:
                self._pca.channels[int(channel)].duty_cycle = int(duty_cycle)
                return
            except OSError:
                if attempt >= attempts - 1:
                    raise
                # Motor switching noise can produce a transient Linux I2C
                # errno 121.  Keep the safety command bounded but retry fast.
                time.sleep(0.005)

    def set_pwm_us(self, channel: int, pulse_us: int) -> None:
        period_us = 1_000_000.0 / float(self._frequency_hz)
        duty_cycle = int(max(0, min(0xFFFF, round((float(pulse_us) / period_us) * 0xFFFF))))
        self._set_duty_cycle_with_retry(int(channel), duty_cycle)

    def disable_all(self) -> None:
        # A hard PWM-off is safer than relying on a motor driver's neutral
        # interpretation when stopping or recovering from a stale process.
        for channel in range(16):
            self._set_duty_cycle_with_retry(channel, 0)

    def close(self) -> None:
        try:
            self._pca.deinit()
        except Exception:
            pass


class Pca9685RobotAdapter:
    def __init__(
        self,
        event_bus: EventBus,
        *,
        config: Pca9685RobotConfig,
        pwm_factory: Callable[[Pca9685RobotConfig], PwmDriver] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._event_bus = event_bus
        self._config = config
        self._pwm_factory = pwm_factory or _AdafruitPca9685Driver
        self._monotonic = monotonic

        self._driver: PwmDriver | None = None
        self._running = False
        self._estop_latched = False
        self._failsafe_active = False
        self._motor_armed = bool(config.dry_run or not config.start_disarmed)
        self._green_streak = 0
        self._green_candidate_signature = ""
        self._last_green_trigger_at = 0.0
        self._green_half_turn_armed = True
        self._green_rearm_clear_streak = 0
        self._green_half_turn_phase = "IDLE"
        self._green_half_turn_phase_started_at = 0.0
        self._green_half_turn_first_ms = 0
        self._green_half_turn_second_ms = 0
        self._monitor_error_streak = 0
        self._green_single_encounter = "IDLE"
        self._green_single_clear_streak = 0
        self._green_single_pending_since = 0.0
        self._green_single_relation_confidence = 0.0
        self._green_single_relation_delta_y = 0.0
        self._green_route_decision = "NONE"
        self._last_detection_at = 0.0
        self._last_command_at = 0.0
        self._manual_until = 0.0
        self._maneuver_until = 0.0
        self._control_mode = "STOPPED"
        self._assist_kind = "none"
        self._line_error = 0.0
        self._pid_output = 0.0
        self._vision_line_offset = 0.0
        self._line_offset_source = "none"
        self._line_candidate_count = 0
        self._steering_decision = "STOP"
        self._requested_left_speed_us = 0.0
        self._requested_right_speed_us = 0.0
        self._pid_integral = 0.0
        self._pid_previous_error: float | None = None
        self._pid_derivative = 0.0
        self._pid_last_at = 0.0
        self._last_valid_line_at = 0.0
        self._last_valid_line_error = 0.0
        self._last_valid_line_correction = 0.0
        self._gap_crossing_active = False
        self._gap_started_at = 0.0
        self._gap_straight_streak = 0
        self._gap_straight_confirmed_at = 0.0
        self._gap_entry_allowed = False
        self._gap_reacquire_streak = 0
        self._sharp_curve_active = False
        self._last_sharp_curve_at = 0.0
        self._last_sharp_curve_error = 0.0
        self._last_sharp_curve_correction = 0.0
        self._last_sharp_curve_reverse_inner = False
        self._last_sharp_curve_throttle_scale = 1.0
        self._corner_reacquire_streak = 0
        self._corner_started_at = 0.0
        self._corner_exit_until = 0.0
        self._corner_recovery_until = 0.0
        self._corner_exit_direction = 0.0
        self._corner_phase = "IDLE"
        self._corner_direction = 0
        self._corner_phase_started_at = 0.0
        self._corner_sequence_started_at = 0.0
        self._corner_confirm_streak = 0
        self._corner_confirm_direction = 0
        self._corner_confirm_max_row_ratio = 0.0
        self._corner_approach_lost_streak = 0
        self._corner_pivot_line_lost_streak = 0
        self._corner_pivot_line_lost_seen = False
        self._corner_cooldown_until = 0.0
        self._corner_elbow_row_contrast = 0.0
        self._corner_geometry_fallback_active = False
        self._curve_clear_streak = 0
        self._curve_signal = 0.0
        self._obstacle_state = "CLEAR"
        self._green_instruction = "NO_GREEN"
        self._green_marker_count = 0
        self._green_maneuver_duration_ms = 0
        self._last_left_us = int(config.neutral_us)
        self._last_right_us = int(config.neutral_us)
        self._last_event_line = ""
        self._last_event_at = 0.0

        self._lock = threading.RLock()
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._detection_sub = self._event_bus.subscribe(EventTopic.VISION_DETECTIONS, self._on_detection)
        self._ui_sub = self._event_bus.subscribe(EventTopic.UI_COMMAND, self._on_ui_command)

    @property
    def connected(self) -> bool:
        return self._config.dry_run or self._driver is not None

    def start(self) -> None:
        with self._lock:
            self._running = True
            if self._config.dry_run:
                self._publish_log("INFO", "pca9685 robot adapter running in dry-run mode")
            else:
                self._driver = self._pwm_factory(self._config)
                self._publish_log(
                    "INFO",
                    f"pca9685 connected address=0x{self._config.i2c_address:02x} freq={self._config.frequency_hz}",
                )
            self._disable_outputs_locked()
            self._neutralize_locked(
                mode="STOPPED",
                assist_kind=("none" if self._motor_armed else "motor_disarmed"),
                failsafe=False,
            )

        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, name="pca9685-robot-monitor", daemon=True)
        self._monitor_thread.start()

    def stop(self) -> None:
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=1.0)
        self._monitor_thread = None

        try:
            self.request_safe_stop(reason="adapter_shutdown", state="SHUTDOWN", emergency=False)
        except Exception:
            pass

        with self._lock:
            self._running = False
            driver = self._driver
            self._driver = None

        try:
            self._detection_sub.unsubscribe()
        except Exception:
            pass
        try:
            self._ui_sub.unsubscribe()
        except Exception:
            pass
        if driver is not None:
            driver.close()

    def request_safe_stop(self, *, reason: str, state: str, emergency: bool, force: bool = True) -> bool:
        del force
        with self._lock:
            if emergency:
                self._estop_latched = True
                self._failsafe_active = True
                self._motor_armed = False
            self._last_event_line = f"{'ESTOP' if emergency else 'STOP'} {reason}"
            self._last_event_at = self._monotonic()
            self._maneuver_until = 0.0
            self._reset_green_half_turn_sequence_locked()
            self._reset_green_single_encounter_locked()
            self._reset_calibrated_corner_sequence_locked()
            self._neutralize_locked(
                mode="ESTOP" if emergency else "STOPPED",
                assist_kind="estop" if emergency else "stop",
                failsafe=emergency,
                disable_outputs=True,
            )
        self._publish_log("WARNING" if emergency else "INFO", f"pca9685 {'estop' if emergency else 'stop'} ({reason})", state=state)
        return True

    def send_stop(self, *, reason: str, state: str, force: bool = True) -> bool:
        return self.request_safe_stop(reason=reason, state=state, emergency=False, force=force)

    def send_estop(self, *, reason: str, state: str, force: bool = True) -> bool:
        return self.request_safe_stop(reason=reason, state=state, emergency=True, force=force)

    def clear_estop(self, *, reason: str, state: str) -> bool:
        with self._lock:
            self._estop_latched = False
            self._failsafe_active = False
            if not self._config.start_disarmed:
                self._motor_armed = True
            self._neutralize_locked(mode="STOPPED", assist_kind="clear_estop", failsafe=False)
        self._publish_log("INFO", f"pca9685 estop cleared ({reason})", state=state)
        return True

    def update_line_control(
        self,
        *,
        pid_kp_us: float | None = None,
        pid_ki_us: float | None = None,
        pid_kd_us: float | None = None,
        pid_integral_limit: float | None = None,
        pid_derivative_filter: float | None = None,
        max_output_us: int | None = None,
        line_hold_ms: int | None = None,
        left_base_throttle_us: int | None = None,
        right_base_throttle_us: int | None = None,
        line_error_deadband: float | None = None,
    ) -> dict[str, float | int]:
        """Apply bounded line-control values without restarting the runner.

        Updating the gains also clears the accumulated integral and derivative
        state so a value change cannot inherit a stale steering impulse.
        """
        with self._lock:
            if pid_kp_us is not None:
                self._config.pid_kp_us = max(0.0, min(2000.0, float(pid_kp_us)))
            if pid_ki_us is not None:
                self._config.pid_ki_us = max(0.0, min(200.0, float(pid_ki_us)))
            if pid_kd_us is not None:
                self._config.pid_kd_us = max(0.0, min(500.0, float(pid_kd_us)))
            if pid_integral_limit is not None:
                self._config.pid_integral_limit = max(0.0, min(1.0, float(pid_integral_limit)))
            if pid_derivative_filter is not None:
                self._config.pid_derivative_filter = max(0.0, min(0.99, float(pid_derivative_filter)))
            if max_output_us is not None:
                self._config.max_output_us = max(0, min(1000, int(max_output_us)))
            if line_hold_ms is not None:
                self._config.line_hold_ms = max(0, min(2000, int(line_hold_ms)))
            if left_base_throttle_us is not None:
                self._config.left_base_throttle_us = max(0, min(500, int(left_base_throttle_us)))
            if right_base_throttle_us is not None:
                self._config.right_base_throttle_us = max(0, min(500, int(right_base_throttle_us)))
            if line_error_deadband is not None:
                self._config.line_error_deadband = max(0.0, min(0.25, float(line_error_deadband)))

            self._reset_line_pid_locked()
            self._last_event_line = "line control tuning updated"
            self._last_event_at = self._monotonic()
            return {
                "pid_kp_us": round(float(self._config.pid_kp_us), 2),
                "pid_ki_us": round(float(self._config.pid_ki_us), 2),
                "pid_kd_us": round(float(self._config.pid_kd_us), 2),
                "pid_integral_limit": round(float(self._config.pid_integral_limit), 3),
                "pid_derivative_filter": round(float(self._config.pid_derivative_filter), 3),
                "max_output_us": int(self._config.max_output_us),
                "line_hold_ms": int(self._config.line_hold_ms),
                "left_base_throttle_us": int(self._config.left_base_throttle_us),
                "right_base_throttle_us": int(self._config.right_base_throttle_us),
                "line_error_deadband": round(float(self._config.line_error_deadband), 3),
            }

    def update_corner_timing(
        self,
        *,
        approach_min_ms: int | None = None,
        pivot_ms: int | None = None,
    ) -> dict[str, int]:
        """Apply bounded 90-degree timing for the next corner atomically."""
        with self._lock:
            if self._corner_phase not in {"IDLE", "TIMEOUT"}:
                raise RuntimeError(
                    f"corner timing cannot change during {self._corner_phase.lower()}"
                )
            if approach_min_ms is not None:
                self._config.corner_approach_min_ms = max(
                    0,
                    min(500, int(approach_min_ms)),
                )
            if pivot_ms is not None:
                bounded_pivot = max(250, min(4000, int(pivot_ms)))
                self._config.corner_pivot_right_ms = bounded_pivot
                self._config.corner_pivot_left_ms = bounded_pivot
            self._last_event_line = "corner timing updated"
            self._last_event_at = self._monotonic()
            return {
                "approach_min_ms": int(self._config.corner_approach_min_ms),
                "pivot_right_ms": int(self._config.corner_pivot_right_ms),
                "pivot_left_ms": int(self._config.corner_pivot_left_ms),
            }

    def update_left_corner_timing(
        self,
        *,
        approach_min_ms: int | None = None,
        pivot_ms: int | None = None,
    ) -> dict[str, int]:
        """Apply only the left 90-degree placement and pivot limits."""
        with self._lock:
            if self._corner_phase not in {"IDLE", "TIMEOUT"}:
                raise RuntimeError(
                    f"left corner timing cannot change during {self._corner_phase.lower()}"
                )
            if approach_min_ms is not None:
                self._config.corner_approach_left_min_ms = max(
                    0,
                    min(1000, int(approach_min_ms)),
                )
            if pivot_ms is not None:
                self._config.corner_pivot_left_ms = max(
                    250,
                    min(4000, int(pivot_ms)),
                )
            self._last_event_line = "left corner timing updated"
            self._last_event_at = self._monotonic()
            return {
                "approach_left_min_ms": int(
                    self._config.corner_approach_left_min_ms
                ),
                "pivot_left_ms": int(self._config.corner_pivot_left_ms),
            }

    def update_green_half_turn_timing(
        self,
        *,
        duration_ms: int | None = None,
        first_ms: int | None = None,
        second_ms: int | None = None,
        reverse_ms: int | None = None,
    ) -> dict[str, int]:
        """Apply independent two-green pivot and reverse timings at runtime."""
        with self._lock:
            if (
                self._control_mode == "GREEN"
                and self._maneuver_until > self._monotonic()
            ):
                raise RuntimeError(
                    "green half-turn timing cannot change during an active green maneuver"
                )
            if first_ms is None and second_ms is None and duration_ms is not None:
                total_ms = max(500, min(8000, int(duration_ms)))
                first_ms = min(int(self._config.corner_pivot_right_ms), total_ms - 250)
                second_ms = total_ms - int(first_ms)
            if first_ms is not None:
                self._config.green_half_turn_first_ms = max(
                    250,
                    min(4000, int(first_ms)),
                )
            if second_ms is not None:
                self._config.green_half_turn_second_ms = max(
                    250,
                    min(4000, int(second_ms)),
                )
            if reverse_ms is not None:
                self._config.green_half_turn_reverse_ms = max(
                    0,
                    min(1500, int(reverse_ms)),
                )
            self._config.green_half_turn_ms = (
                int(self._config.green_half_turn_first_ms)
                + int(self._config.green_half_turn_second_ms)
            )
            self._last_event_line = "green half-turn timing updated"
            self._last_event_at = self._monotonic()
            return {
                "green_half_turn_ms": int(self._config.green_half_turn_ms),
                "green_half_turn_first_ms": int(self._config.green_half_turn_first_ms),
                "green_half_turn_second_ms": int(self._config.green_half_turn_second_ms),
                "green_half_turn_reverse_ms": int(self._config.green_half_turn_reverse_ms),
            }

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            state = "dry-run" if self._config.dry_run else ("connected" if self._driver is not None else "waiting")
            green_half_turn_left_us, green_half_turn_right_us = (
                self._green_half_turn_pivot_speeds_locked(
                    float(self._config.green_half_turn_us)
                )
            )
            green_half_turn_brake_ms = int(self._config.corner_brake_ms)
            green_half_turn_mid_brake_ms = self._green_half_turn_mid_brake_ms_locked()
            configured_first_ms, configured_second_ms = (
                self._green_half_turn_segment_durations_locked()
            )
            telemetry = {
                "mode": self._control_mode,
                "line_error": round(float(self._line_error), 4),
                "pid": round(float(self._pid_output), 2),
                "vision_line_offset": round(float(self._vision_line_offset), 4),
                "line_offset_source": self._line_offset_source,
                "line_candidate_count": int(self._line_candidate_count),
                "steering_decision": self._steering_decision,
                "requested_left_speed_us": round(float(self._requested_left_speed_us), 2),
                "requested_right_speed_us": round(float(self._requested_right_speed_us), 2),
                "green": self._green_instruction,
                "green_marker_count": int(self._green_marker_count),
                "green_maneuver_duration_ms": int(self._green_maneuver_duration_ms),
                "green_maneuver_remaining_ms": int(
                    max(0.0, self._maneuver_until - self._monotonic()) * 1000.0
                ),
                "green_half_turn_ms": int(self._config.green_half_turn_ms),
                "green_half_turn_us": int(self._config.green_half_turn_us),
                "green_half_turn_left_us": round(abs(green_half_turn_left_us), 2),
                "green_half_turn_right_us": round(abs(green_half_turn_right_us), 2),
                "green_half_turn_phase": self._green_half_turn_phase,
                "green_half_turn_first_ms": configured_first_ms,
                "green_half_turn_second_ms": configured_second_ms,
                "green_half_turn_reverse_ms": int(self._config.green_half_turn_reverse_ms),
                "green_half_turn_reverse_left_us": int(self._config.left_base_throttle_us),
                "green_half_turn_reverse_right_us": int(self._config.right_base_throttle_us),
                "green_half_turn_brake_ms": green_half_turn_brake_ms,
                "green_half_turn_mid_brake_ms": green_half_turn_mid_brake_ms,
                "green_half_turn_armed": bool(self._green_half_turn_armed),
                "green_rearm_clear_streak": int(self._green_rearm_clear_streak),
                "green_rearm_clear_frames": int(self._config.green_rearm_clear_frames),
                "green_single_encounter": self._green_single_encounter,
                "green_single_clear_streak": int(self._green_single_clear_streak),
                "green_single_confirm_elapsed_ms": int(
                    max(0.0, self._monotonic() - self._green_single_pending_since)
                    * 1000.0
                )
                if self._green_single_pending_since > 0.0
                else 0,
                "green_single_relation_confidence": round(
                    float(self._green_single_relation_confidence), 3
                ),
                "green_single_relation_delta_y": round(
                    float(self._green_single_relation_delta_y), 2
                ),
                "obstacle": self._obstacle_state,
                "failsafe": bool(self._failsafe_active or self._estop_latched),
                "motor_armed": bool(self._motor_armed),
                "green_route_decision": self._green_route_decision,
                "left_pwm": int(self._last_left_us),
                "right_pwm": int(self._last_right_us),
                "left_channels": list(self._config.left_channels),
                "right_channels": list(self._config.right_channels),
                "left_base_throttle_us": int(self._config.left_base_throttle_us),
                "right_base_throttle_us": int(self._config.right_base_throttle_us),
                "pid_kp_us": round(float(self._config.pid_kp_us), 2),
                "pid_ki_us": round(float(self._config.pid_ki_us), 2),
                "pid_kd_us": round(float(self._config.pid_kd_us), 2),
                "pid_integral_limit": round(float(self._config.pid_integral_limit), 3),
                "pid_derivative_filter": round(float(self._config.pid_derivative_filter), 3),
                "max_output_us": int(self._config.max_output_us),
                "line_hold_ms": int(self._config.line_hold_ms),
                "gap_crossing_enabled": bool(self._config.gap_crossing_enabled),
                "gap_crossing_active": bool(self._gap_crossing_active),
                "gap_entry_allowed": bool(self._gap_entry_allowed),
                "gap_straight_streak": int(self._gap_straight_streak),
                "gap_straight_memory_ms": int(self._config.gap_straight_memory_ms),
                "gap_reacquire_min_ms": int(self._config.gap_reacquire_min_ms),
                "gap_straight_age_ms": int(
                    max(0.0, self._monotonic() - self._gap_straight_confirmed_at)
                    * 1000.0
                )
                if self._gap_straight_confirmed_at > 0.0
                else 0,
                "gap_reacquire_streak": int(self._gap_reacquire_streak),
                "gap_elapsed_ms": int(
                    max(0.0, self._monotonic() - self._gap_started_at) * 1000.0
                )
                if self._gap_started_at > 0.0
                else 0,
                "simple_line_follow": bool(self._config.simple_line_follow),
                "basic_line_follow": bool(self._config.basic_line_follow),
                "line_steering_inverted": bool(self._config.line_steering_inverted),
                "sharp_corner_maneuver_enabled": bool(
                    self._config.sharp_corner_maneuver_enabled
                ),
                "corner_sequence_enabled": bool(self._config.corner_sequence_enabled),
                "corner_phase": self._corner_phase,
                "corner_direction": (
                    "RIGHT"
                    if self._corner_direction > 0
                    else "LEFT"
                    if self._corner_direction < 0
                    else "NONE"
                ),
                "corner_phase_elapsed_ms": int(
                    max(0.0, self._monotonic() - self._corner_phase_started_at) * 1000.0
                )
                if self._corner_phase_started_at > 0.0
                else 0,
                "corner_confirm_streak": int(self._corner_confirm_streak),
                "corner_elbow_row_contrast": round(
                    float(self._corner_elbow_row_contrast), 3
                ),
                "corner_geometry_fallback_active": bool(
                    self._corner_geometry_fallback_active
                ),
                "corner_approach_min_ms": int(self._config.corner_approach_min_ms),
                "corner_approach_left_min_ms": int(
                    self._config.corner_approach_left_min_ms
                ),
                "corner_pivot_right_ms": int(self._config.corner_pivot_right_ms),
                "corner_pivot_left_ms": int(self._config.corner_pivot_left_ms),
                "corner_pivot_speed_us": int(self._config.corner_pivot_speed_us),
                "corner_pivot_left_speed_us": int(
                    self._config.corner_pivot_left_speed_us
                ),
                "corner_pivot_line_lost_seen": bool(
                    self._corner_pivot_line_lost_seen
                ),
                "curve_lookahead_gain": round(float(self._config.curve_lookahead_gain), 3),
                "curve_throttle_scale": round(float(self._config.curve_throttle_scale), 3),
                "ordinary_sharp_hold_ms": int(self._config.ordinary_sharp_hold_ms),
                "sharp_curve_active": bool(self._sharp_curve_active),
                "sharp_curve_threshold": round(float(self._config.sharp_curve_threshold), 3),
                "sharp_curve_hold_ms": int(self._config.sharp_curve_hold_ms),
                "sharp_curve_inner_reverse_scale": round(
                    float(self._config.sharp_curve_inner_reverse_scale), 3
                ),
                "sharp_curve_entry_min_row_ratio": round(
                    float(self._config.sharp_curve_entry_min_row_ratio), 3
                ),
                "sharp_curve_reverse_inner": bool(self._last_sharp_curve_reverse_inner),
                "corner_reacquire_streak": int(self._corner_reacquire_streak),
                "corner_exit_active": bool(self._corner_exit_until > self._monotonic()),
                "corner_recovery_active": bool(
                    self._corner_exit_until <= self._monotonic()
                    < self._corner_recovery_until
                ),
                "curve_signal": round(float(self._curve_signal), 4),
                "line_error_deadband": round(float(self._config.line_error_deadband), 3),
            }
            return {
                "state": state,
                "connected": bool(self.connected),
                "port": f"i2c:0x{self._config.i2c_address:02x}",
                "backend": "pca9685",
                "heartbeat_ok": not bool(self._failsafe_active),
                "heartbeat_age_ms": None,
                "telemetry_age_ms": None
                if self._last_command_at <= 0
                else int(max(0.0, (self._monotonic() - self._last_command_at) * 1000.0)),
                "ack": "PWM",
                "ack_age_ms": None,
                "event": self._last_event_line,
                "event_age_ms": None
                if self._last_event_at <= 0
                else int(max(0.0, (self._monotonic() - self._last_event_at) * 1000.0)),
                "assist_kind": self._assist_kind,
                "control_mode": self._control_mode,
                "line_error": self._line_error,
                "pid_output": self._pid_output,
                "vision_line_offset": self._vision_line_offset,
                "line_offset_source": self._line_offset_source,
                "line_candidate_count": int(self._line_candidate_count),
                "steering_decision": self._steering_decision,
                "requested_left_speed_us": self._requested_left_speed_us,
                "requested_right_speed_us": self._requested_right_speed_us,
                "obstacle_state": self._obstacle_state,
                "green_instruction": self._green_instruction,
                "green_marker_count": int(self._green_marker_count),
                "green_maneuver_duration_ms": int(self._green_maneuver_duration_ms),
                "green_maneuver_remaining_ms": int(
                    max(0.0, self._maneuver_until - self._monotonic()) * 1000.0
                ),
                "green_half_turn_ms": int(self._config.green_half_turn_ms),
                "green_half_turn_us": int(self._config.green_half_turn_us),
                "green_half_turn_left_us": round(abs(green_half_turn_left_us), 2),
                "green_half_turn_right_us": round(abs(green_half_turn_right_us), 2),
                "green_half_turn_phase": self._green_half_turn_phase,
                "green_half_turn_first_ms": configured_first_ms,
                "green_half_turn_second_ms": configured_second_ms,
                "green_half_turn_reverse_ms": int(self._config.green_half_turn_reverse_ms),
                "green_half_turn_reverse_left_us": int(self._config.left_base_throttle_us),
                "green_half_turn_reverse_right_us": int(self._config.right_base_throttle_us),
                "green_half_turn_brake_ms": green_half_turn_brake_ms,
                "green_half_turn_mid_brake_ms": green_half_turn_mid_brake_ms,
                "green_half_turn_armed": bool(self._green_half_turn_armed),
                "green_rearm_clear_streak": int(self._green_rearm_clear_streak),
                "green_rearm_clear_frames": int(self._config.green_rearm_clear_frames),
                "green_single_encounter": self._green_single_encounter,
                "green_single_clear_streak": int(self._green_single_clear_streak),
                "green_single_confirm_elapsed_ms": int(
                    max(0.0, self._monotonic() - self._green_single_pending_since)
                    * 1000.0
                )
                if self._green_single_pending_since > 0.0
                else 0,
                "green_single_relation_confidence": float(
                    self._green_single_relation_confidence
                ),
                "green_single_relation_delta_y": float(
                    self._green_single_relation_delta_y
                ),
                "failsafe": bool(self._failsafe_active or self._estop_latched),
                "motor_armed": bool(self._motor_armed),
                "green_route_decision": self._green_route_decision,
                "left_channels": list(self._config.left_channels),
                "right_channels": list(self._config.right_channels),
                "left_base_throttle_us": int(self._config.left_base_throttle_us),
                "right_base_throttle_us": int(self._config.right_base_throttle_us),
                "pid_kp_us": float(self._config.pid_kp_us),
                "pid_ki_us": float(self._config.pid_ki_us),
                "pid_kd_us": float(self._config.pid_kd_us),
                "pid_integral_limit": float(self._config.pid_integral_limit),
                "pid_derivative_filter": float(self._config.pid_derivative_filter),
                "max_output_us": int(self._config.max_output_us),
                "line_hold_ms": int(self._config.line_hold_ms),
                "gap_crossing_enabled": bool(self._config.gap_crossing_enabled),
                "gap_crossing_active": bool(self._gap_crossing_active),
                "gap_entry_allowed": bool(self._gap_entry_allowed),
                "gap_straight_streak": int(self._gap_straight_streak),
                "gap_straight_memory_ms": int(self._config.gap_straight_memory_ms),
                "gap_reacquire_min_ms": int(self._config.gap_reacquire_min_ms),
                "gap_straight_age_ms": int(
                    max(0.0, self._monotonic() - self._gap_straight_confirmed_at)
                    * 1000.0
                )
                if self._gap_straight_confirmed_at > 0.0
                else 0,
                "gap_reacquire_streak": int(self._gap_reacquire_streak),
                "gap_elapsed_ms": int(
                    max(0.0, self._monotonic() - self._gap_started_at) * 1000.0
                )
                if self._gap_started_at > 0.0
                else 0,
                "simple_line_follow": bool(self._config.simple_line_follow),
                "basic_line_follow": bool(self._config.basic_line_follow),
                "line_steering_inverted": bool(self._config.line_steering_inverted),
                "sharp_corner_maneuver_enabled": bool(
                    self._config.sharp_corner_maneuver_enabled
                ),
                "corner_sequence_enabled": bool(self._config.corner_sequence_enabled),
                "corner_phase": self._corner_phase,
                "corner_direction": (
                    "RIGHT"
                    if self._corner_direction > 0
                    else "LEFT"
                    if self._corner_direction < 0
                    else "NONE"
                ),
                "corner_phase_elapsed_ms": int(
                    max(0.0, self._monotonic() - self._corner_phase_started_at) * 1000.0
                )
                if self._corner_phase_started_at > 0.0
                else 0,
                "corner_confirm_streak": int(self._corner_confirm_streak),
                "corner_elbow_row_contrast": float(self._corner_elbow_row_contrast),
                "corner_geometry_fallback_active": bool(
                    self._corner_geometry_fallback_active
                ),
                "corner_approach_min_ms": int(self._config.corner_approach_min_ms),
                "corner_approach_left_min_ms": int(
                    self._config.corner_approach_left_min_ms
                ),
                "corner_pivot_right_ms": int(self._config.corner_pivot_right_ms),
                "corner_pivot_left_ms": int(self._config.corner_pivot_left_ms),
                "corner_pivot_speed_us": int(self._config.corner_pivot_speed_us),
                "corner_pivot_left_speed_us": int(
                    self._config.corner_pivot_left_speed_us
                ),
                "corner_pivot_line_lost_seen": bool(
                    self._corner_pivot_line_lost_seen
                ),
                "curve_lookahead_gain": float(self._config.curve_lookahead_gain),
                "curve_throttle_scale": float(self._config.curve_throttle_scale),
                "ordinary_sharp_hold_ms": int(self._config.ordinary_sharp_hold_ms),
                "sharp_curve_active": bool(self._sharp_curve_active),
                "sharp_curve_threshold": float(self._config.sharp_curve_threshold),
                "sharp_curve_hold_ms": int(self._config.sharp_curve_hold_ms),
                "sharp_curve_inner_reverse_scale": float(
                    self._config.sharp_curve_inner_reverse_scale
                ),
                "sharp_curve_entry_min_row_ratio": float(
                    self._config.sharp_curve_entry_min_row_ratio
                ),
                "sharp_curve_reverse_inner": bool(self._last_sharp_curve_reverse_inner),
                "sharp_curve_visible_commit_ms": int(
                    self._config.sharp_curve_visible_commit_ms
                ),
                "sharp_curve_finish_inner_reverse_scale": float(
                    self._config.sharp_curve_finish_inner_reverse_scale
                ),
                "sharp_curve_exit_settle_ms": int(
                    self._config.sharp_curve_exit_settle_ms
                ),
                "corner_reacquire_streak": int(self._corner_reacquire_streak),
                "corner_exit_active": bool(self._corner_exit_until > self._monotonic()),
                "corner_recovery_active": bool(
                    self._corner_exit_until <= self._monotonic()
                    < self._corner_recovery_until
                ),
                "curve_signal": float(self._curve_signal),
                "line_error_deadband": float(self._config.line_error_deadband),
                "telemetry": telemetry,
            }

    def latest_telemetry(self) -> dict[str, Any]:
        return dict(self.status_payload().get("telemetry", {}))

    def _on_detection(self, event: VisionDetectionEvent) -> None:
        if not self._running or not isinstance(event, VisionDetectionEvent):
            return

        metadata = event.metadata if isinstance(event.metadata, Mapping) else {}
        with self._lock:
            self._vision_line_offset = max(
                -1.0,
                min(1.0, float(metadata.get("line_offset_norm", 0.0) or 0.0)),
            )
            self._line_offset_source = str(metadata.get("line_offset_source", "none") or "none")
            geometry = metadata.get("line_geometry")
            if isinstance(geometry, Mapping):
                self._line_candidate_count = max(0, int(float(geometry.get("candidate_count", 0) or 0)))
            else:
                self._line_candidate_count = 0
            if self._estop_latched:
                self._neutralize_locked(
                    mode="ESTOP", assist_kind="estop", failsafe=True, disable_outputs=True
                )
                return
            if not self._motor_armed:
                self._neutralize_locked(
                    mode="STOPPED",
                    assist_kind="motor_disarmed",
                    failsafe=False,
                    disable_outputs=True,
                )
                return

            now = self._monotonic()
            self._last_detection_at = now
            if self._maybe_handle_obstacle_locked(event.state, metadata):
                self._reset_gap_crossing_locked()
                return
            if self._maybe_handle_green_locked(event, metadata):
                self._reset_gap_crossing_locked()
                return
            if self._maneuver_until > self._monotonic():
                self._reset_gap_crossing_locked()
                return
            if self._handle_active_gap_crossing_locked(event, metadata, now):
                return
            single_green_suppresses_corner = (
                self._single_green_suppresses_corner_locked(event, metadata)
            )
            if (
                self._config.basic_line_follow
                and self._config.sharp_corner_maneuver_enabled
                and self._config.corner_sequence_enabled
                and not single_green_suppresses_corner
                and self._handle_calibrated_corner_sequence_locked(event, metadata, now)
            ):
                self._reset_gap_crossing_locked()
                return
            if event.state != RobotState.FOLLOWING_LINE.value or not bool(event.line):
                self._green_streak = 0
                if self._config.basic_line_follow and self._hold_sharp_curve_command_locked(now):
                    self._reset_gap_crossing_locked()
                    return
                if self._config.basic_line_follow and self._start_gap_crossing_locked(now):
                    return
                if not self._config.simple_line_follow and not self._config.basic_line_follow and self._hold_last_line_command_locked(now):
                    return
                self._reset_line_pid_locked()
                self._neutralize_locked(
                    mode="STOPPED", assist_kind="line_lost", failsafe=False, disable_outputs=True
                )
                return

            confidence = float(metadata.get("line_confidence", 0.0) or 0.0)
            if confidence < float(self._config.line_confidence_floor):
                # Never leave a previous drive command active while the
                # detector is uncertain for too long. A brief camera gap is
                # bridged with the last safe forward/steering command.
                if self._config.basic_line_follow and self._hold_sharp_curve_command_locked(now):
                    self._reset_gap_crossing_locked()
                    return
                if not self._config.simple_line_follow and not self._config.basic_line_follow and self._hold_last_line_command_locked(now):
                    return
                self._reset_line_pid_locked()
                self._neutralize_locked(
                    mode="STOPPED",
                    assist_kind="line_confidence_low",
                    failsafe=False,
                    disable_outputs=True,
                )
                return

            offset = max(-1.0, min(1.0, float(metadata.get("line_offset_norm", 0.0) or 0.0)))
            if self._config.simple_line_follow:
                self._reset_line_pid_locked()
                self._last_valid_line_at = now
                self._last_valid_line_error = offset
                self._last_valid_line_correction = 0.0
                self._apply_drive_locked(
                    left_speed_us=float(self._config.left_base_throttle_us),
                    right_speed_us=float(self._config.right_base_throttle_us),
                    mode="FOLLOW_LINE",
                    assist_kind="line_simple",
                    line_error=offset,
                    pid_output=0.0,
                )
                return

            if self._config.basic_line_follow:
                # The USB camera currently supplies about 10 control frames/s.
                # A three-frame median retained an old steering sign for one or
                # two frames (100-200 ms), which is enough to create the small
                # observed zig-zag. The detector already provides the selected
                # ground-path offset, so act on the current observation.
                filtered_offset = offset
                right_angle = False
                turn_corridor = False
                bend_signal = 0.0
                dominant_row_y_ratio = 1.0
                track_width_ratio = 0.0
                track_height_ratio = 1.0
                track_center_range_ratio = 0.0
                bottom_row_occupancy = 0.40
                if isinstance(geometry, Mapping):
                    right_angle = bool(float(geometry.get("right_angle_corridor", 0.0) or 0.0) > 0.0)
                    turn_corridor = bool(float(geometry.get("turn_corridor", 0.0) or 0.0) > 0.0)
                    bend_signal = max(
                        -1.0,
                        min(1.0, float(geometry.get("path_bend_delta_norm", 0.0) or 0.0)),
                    )
                    dominant_row_y_ratio = max(
                        0.0,
                        min(1.0, float(geometry.get("dominant_row_y_ratio", 1.0) or 0.0)),
                    )
                    track_width_ratio = max(
                        0.0,
                        min(1.0, float(geometry.get("width_ratio", 0.0) or 0.0)),
                    )
                    track_height_ratio = max(
                        0.0,
                        min(1.0, float(geometry.get("height_ratio", 1.0) or 0.0)),
                    )
                    track_center_range_ratio = max(
                        0.0,
                        min(1.0, float(geometry.get("center_range_ratio", 0.0) or 0.0)),
                    )
                    bottom_row_occupancy = max(
                        0.0,
                        min(1.0, float(geometry.get("bottom_row_occupancy", 0.40) or 0.0)),
                    )
                if self._config.corner_sequence_enabled:
                    # The calibrated phase machine above owns every confirmed
                    # 90-degree corner. A frame that has not passed its elbow
                    # and temporal gates remains an ordinary open-curve frame;
                    # it must never fall through to the legacy immediate pivot.
                    right_angle = False
                line_angle_deg = float(metadata.get("line_angle_deg", 90.0) or 90.0)

                # A 90-degree bend can still have a nearly centered centroid.
                # The installed camera/chassis mapping was validated physically:
                # the bend sign must pass through the same steering inversion as
                # ordinary line error. Once selected, the direction is latched;
                # the contour's bend sign naturally flips while the same L shape
                # rotates under the camera and must not reverse the maneuver.
                steering_error = filtered_offset
                reverse_inner = False
                inner_reverse_scale_override: float | None = None
                curve_throttle_scale = 1.0
                curve_lookahead_active = False
                sharp_triggered = False
                corner_exit = False
                corner_recovery = False
                assist_kind = "line_basic"
                corner_elapsed = (
                    max(0.0, now - self._corner_started_at)
                    if self._corner_started_at > 0.0
                    else 0.0
                )
                visible_commit_s = float(self._config.sharp_curve_visible_commit_ms) / 1000.0
                committed_reverse_corner = bool(
                    self._sharp_curve_active
                    and self._last_sharp_curve_reverse_inner
                    and self._corner_started_at > 0.0
                )

                # Once the outgoing line has been accepted, keep the exact
                # calibrated straight command for a short bounded interval.
                # Do not let a lingering right-angle classification restart
                # the same pivot while the chassis is settling.
                if self._corner_exit_until > now:
                    corner_exit = True
                    assist_kind = "line_corner_exit"
                elif self._corner_recovery_until > now:
                    corner_recovery = True
                    assist_kind = "line_corner_recovery"
                elif committed_reverse_corner and corner_elapsed >= visible_commit_s:
                    # A timeout is not evidence that the outgoing leg is
                    # aligned.  Driving straight here made the USB-camera run
                    # lock onto the incoming line behind the robot. Stop at
                    # calibrated neutral and wait for a fresh visual command.
                    self._neutralize_locked(
                        mode="STOPPED",
                        assist_kind="line_corner_timeout",
                        failsafe=False,
                        disable_outputs=True,
                    )
                    return
                elif right_angle and self._config.sharp_corner_maneuver_enabled:
                    if self._sharp_curve_active and self._last_sharp_curve_reverse_inner:
                        reverse_inner = True
                        sharp_triggered = True
                        corner_signal = self._curve_signal
                        steering_error = self._last_sharp_curve_error
                        self._corner_reacquire_streak = 0
                        if corner_elapsed > (
                            float(self._config.sharp_curve_hold_ms) / 1000.0
                        ):
                            inner_reverse_scale_override = float(
                                self._config.sharp_curve_finish_inner_reverse_scale
                            )
                            assist_kind = "line_sharp_corner_finish"
                    elif (
                        abs(bend_signal) >= 0.08
                        and dominant_row_y_ratio
                        >= float(self._config.sharp_curve_entry_min_row_ratio)
                    ):
                        reverse_inner = True
                        sharp_triggered = True
                        corner_signal = bend_signal
                        steering_error = math.copysign(
                            max(abs(filtered_offset), float(self._config.sharp_curve_threshold)),
                            corner_signal,
                        )
                    elif abs(bend_signal) >= 0.08:
                        # The wide USB camera already sees the elbow, but the
                        # incoming leg is still under the chassis. Continue
                        # ground-path steering until the elbow reaches the
                        # configured near-field row instead of pivoting early.
                        corner_signal = bend_signal
                        assist_kind = "line_corner_approach"
                    else:
                        # Wait for a reliable bend sign instead of committing
                        # from one ambiguous right-angle frame.
                        corner_signal = 0.0
                    self._curve_signal = corner_signal
                    if sharp_triggered and assist_kind == "line_basic":
                        assist_kind = "line_sharp_corner"
                else:
                    latched_curve_signal = self._curve_signal
                    committed_corner = bool(
                        committed_reverse_corner
                        and corner_elapsed < visible_commit_s
                    )
                    if committed_corner:
                        # A flat contour far to one side is not enough to end
                        # the pivot: the physical recording showed that doing
                        # so released the robot while the outgoing leg was
                        # still near the edge, after which calibrated straight
                        # motion drove it off the line. Require a real,
                        # single-candidate ground path to enter the central
                        # corridor before releasing the 90-degree commitment.
                        # Recorded field runs exposed two false exits only
                        # 160-280 ms after the corner was first recognized.
                        # Those flat frames were the incoming leg flickering
                        # under the rotating camera, not the outgoing straight.
                        # Require the strong-pivot interval to finish before a
                        # flat contour is allowed to end a confirmed 90-degree
                        # commitment.
                        minimum_commit_s = (
                            float(self._config.sharp_curve_hold_ms) / 1000.0
                        )
                        reacquire_max_offset = min(
                            0.40,
                            max(
                                0.30,
                                float(self._config.sharp_curve_threshold) * 0.70,
                            ),
                        )
                        straight_reacquired = bool(
                            corner_elapsed >= minimum_commit_s
                            and abs(filtered_offset) <= reacquire_max_offset
                            and self._line_offset_source == "ground_path"
                            and self._line_candidate_count == 1
                            # The wide lens can classify an ordinary vertical
                            # line as a turn corridor and gives it a perspective
                            # bend around 0.15-0.20. Use direct straight-track
                            # geometry instead of those two old close-camera
                            # assumptions.
                            and 65.0 <= line_angle_deg <= 115.0
                            and track_height_ratio >= 0.85
                            and track_width_ratio <= 0.66
                            and track_center_range_ratio <= 0.22
                            and 0.18 <= bottom_row_occupancy <= 0.68
                        )
                        if straight_reacquired:
                            self._corner_reacquire_streak += 1
                        else:
                            self._corner_reacquire_streak = 0
                        if self._corner_reacquire_streak < 3:
                            reverse_inner = True
                            sharp_triggered = True
                            steering_error = self._last_sharp_curve_error
                            self._curve_signal = latched_curve_signal
                            finish_phase = bool(
                                corner_elapsed
                                > (float(self._config.sharp_curve_hold_ms) / 1000.0)
                            )
                            if finish_phase:
                                inner_reverse_scale_override = float(
                                    self._config.sharp_curve_finish_inner_reverse_scale
                                )
                                assist_kind = "line_sharp_corner_finish"
                            else:
                                assist_kind = "line_sharp_corner_bridge"
                        else:
                            exit_direction = self._last_sharp_curve_correction
                            corner_exit = self._start_corner_exit_locked(
                                now,
                                exit_direction,
                            )
                            assist_kind = "line_corner_exit"
                    if not sharp_triggered and not corner_exit:
                        self._corner_exit_until = 0.0
                        self._corner_exit_direction = 0.0
                        curve_evidence = bool(
                            abs(bend_signal) >= 0.18
                            and (
                                turn_corridor
                                or abs(filtered_offset) >= 0.45
                            )
                        )
                        curve_lookahead_active = bool(
                            curve_evidence
                            and (
                                float(self._config.curve_lookahead_gain) > 0.0
                                or float(self._config.curve_throttle_scale) < 1.0
                            )
                        )
                        if curve_lookahead_active:
                            steering_error = max(
                                -1.0,
                                min(
                                    1.0,
                                    filtered_offset
                                    + (
                                        float(self._config.curve_lookahead_gain)
                                        * bend_signal
                                    ),
                                ),
                            )
                            curve_throttle_scale = float(
                                self._config.curve_throttle_scale
                            )
                            assist_kind = "line_curve_lookahead"
                        self._curve_signal = steering_error
                        sharp_triggered = bool(
                            abs(steering_error) >= float(self._config.sharp_curve_threshold)
                        )
                        if sharp_triggered:
                            assist_kind = (
                                "line_sharp_curve"
                                if curve_lookahead_active
                                else "line_sharp"
                            )

                correction = self._compute_basic_line_correction_locked(steering_error)
                if corner_exit:
                    # Straight calibration was physically verified at exactly
                    # left=300/right=200. Scaling or retaining even 70 us of
                    # steering here breaks that calibration and caused the
                    # post-corner turn seen in the field test.
                    exit_limit = float(self._config.sharp_curve_exit_max_correction_us)
                    correction = max(-exit_limit, min(exit_limit, correction))
                    if (
                        self._corner_exit_direction != 0.0
                        and (correction * self._corner_exit_direction) < 0.0
                    ):
                        # Do not bounce immediately to the opposite side while
                        # chassis inertia is still completing the corner.
                        correction = 0.0
                elif corner_recovery:
                    recovery_limit = float(
                        self._config.sharp_curve_recovery_max_correction_us
                    )
                    correction = max(-recovery_limit, min(recovery_limit, correction))
                sharp_curve = sharp_triggered
                # Bridge very short right-angle classification flicker while
                # the line is still visible. Do not refresh the timer here:
                # this cannot turn an uncertain frame into an unbounded pivot.
                visible_bridge_s = min(
                    0.16,
                    float(self._config.sharp_curve_hold_ms) / 1000.0,
                )
                if (
                    not sharp_triggered
                    and self._sharp_curve_active
                    and self._last_sharp_curve_at > 0.0
                    and (now - self._last_sharp_curve_at) <= visible_bridge_s
                ):
                    sharp_curve = True
                    steering_error = self._last_sharp_curve_error
                    correction = self._last_sharp_curve_correction
                    reverse_inner = self._last_sharp_curve_reverse_inner
                    curve_throttle_scale = self._last_sharp_curve_throttle_scale
                    assist_kind = "line_sharp_bridge"

                self._reset_line_pid_locked()
                self._last_valid_line_at = now
                self._last_valid_line_error = steering_error
                self._last_valid_line_correction = correction
                if sharp_triggered and abs(correction) > 0.5:
                    self._sharp_curve_active = True
                    if reverse_inner and (
                        self._corner_started_at <= 0.0
                        or not self._last_sharp_curve_reverse_inner
                    ):
                        self._corner_started_at = now
                        self._corner_exit_until = 0.0
                        self._corner_exit_direction = 0.0
                    elif not reverse_inner:
                        self._corner_started_at = 0.0
                    if right_angle or not reverse_inner:
                        self._last_sharp_curve_at = now
                    self._last_sharp_curve_error = steering_error
                    self._last_sharp_curve_correction = correction
                    self._last_sharp_curve_reverse_inner = reverse_inner
                    self._last_sharp_curve_throttle_scale = curve_throttle_scale
                    self._curve_clear_streak = 0
                elif not corner_exit and not corner_recovery:
                    self._curve_clear_streak = 0

                # A confirmed 90-degree corner counter-rotates its inside side
                # conservatively; a large ordinary offset still only stops the
                # inside side. Straight and gentle mixing remain unchanged.
                left_speed, right_speed = self._line_drive_speeds_locked(
                    correction,
                    throttle_scale=(
                        float(self._config.sharp_curve_exit_throttle_scale)
                        if corner_exit
                        else curve_throttle_scale
                    ),
                    sharp_turn=sharp_curve,
                    reverse_inner=reverse_inner,
                    inner_reverse_scale_override=inner_reverse_scale_override,
                )
                # One current, non-curve line frame is enough to authorize a
                # straight gap. The physical endpoint drifts laterally in the
                # USB image, so centering may refine telemetry but must never
                # delay the mandatory 2400 ms forward crossing.
                gap_entry_safe = bool(
                    self._config.gap_crossing_enabled
                    and not sharp_curve
                    and not curve_lookahead_active
                    and not corner_exit
                    and not corner_recovery
                    and self._corner_phase == "IDLE"
                    and self._green_half_turn_phase == "IDLE"
                    and abs(float(bend_signal))
                    <= float(self._config.gap_max_entry_bend)
                )
                self._gap_entry_allowed = gap_entry_safe
                gap_entry_straight = bool(
                    gap_entry_safe
                    and abs(float(steering_error))
                    <= float(self._config.gap_max_entry_error)
                )
                if gap_entry_straight:
                    self._gap_straight_streak = min(
                        int(self._config.gap_straight_confirm_frames),
                        self._gap_straight_streak + 1,
                    )
                    if self._gap_straight_streak >= int(
                        self._config.gap_straight_confirm_frames
                    ):
                        self._gap_straight_confirmed_at = float(now)
                else:
                    self._gap_straight_streak = 0
                    explicit_curve_evidence = bool(
                        sharp_curve
                        or curve_lookahead_active
                        or corner_exit
                        or corner_recovery
                        or self._corner_phase != "IDLE"
                        or self._green_half_turn_phase != "IDLE"
                        or abs(float(bend_signal))
                        > float(self._config.gap_max_entry_bend)
                    )
                    if explicit_curve_evidence:
                        self._gap_straight_confirmed_at = 0.0
                self._apply_drive_locked(
                    left_speed_us=left_speed,
                    right_speed_us=right_speed,
                    mode="FOLLOW_LINE",
                    assist_kind=assist_kind,
                    line_error=steering_error,
                    pid_output=correction,
                )
                return

            correction = self._compute_line_pid_locked(offset, now)
            self._last_valid_line_at = now
            self._last_valid_line_error = offset
            self._last_valid_line_correction = correction
            # line_offset_norm is positive when the detected line is to the
            # right of the image center. The robot must then turn right:
            # speed up the left side and slow the right side.
            left_speed, right_speed = self._line_drive_speeds_locked(correction)
            self._apply_drive_locked(
                left_speed_us=left_speed,
                right_speed_us=right_speed,
                mode="FOLLOW_LINE",
                assist_kind="line",
                line_error=offset,
                pid_output=correction,
            )

    def _on_ui_command(self, event: UICommandEvent) -> None:
        if not self._running or not isinstance(event, UICommandEvent):
            return
        command = str(event.command or "").strip().lower()
        params = event.params if isinstance(event.params, Mapping) else {}

        with self._lock:
            if command in {"robot.start", "robot.arm"}:
                self._estop_latched = False
                self._failsafe_active = False
                self._motor_armed = True
                self._manual_until = 0.0
                self._maneuver_until = 0.0
                self._reset_green_half_turn_sequence_locked()
                self._reset_green_single_encounter_locked()
                self._reset_calibrated_corner_sequence_locked()
                self._reset_line_pid_locked()
                self._green_instruction = "NO_GREEN"
                self._green_marker_count = 0
                self._last_event_line = "motors armed from dashboard"
                self._last_event_at = self._monotonic()
                self._neutralize_locked(
                    mode="STOPPED",
                    assist_kind="motor_armed_waiting_frame",
                    failsafe=False,
                )
                return
            if (
                command == "robot.forward_test"
                and not self._estop_latched
                and self._motor_armed
            ):
                duration_ms = max(1, int(params.get("duration_ms", 1200)))
                now = self._monotonic()
                self._manual_until = now + (duration_ms / 1000.0)
                self._apply_drive_locked(
                    left_speed_us=float(self._config.left_base_throttle_us),
                    right_speed_us=float(self._config.right_base_throttle_us),
                    mode="MANUAL",
                    assist_kind="forward_test",
                    line_error=0.0,
                    pid_output=0.0,
                )
                return
            if (
                command == "robot.reverse_test"
                and not self._estop_latched
                and self._motor_armed
            ):
                duration_ms = max(1, int(params.get("duration_ms", 1200)))
                now = self._monotonic()
                self._manual_until = now + (duration_ms / 1000.0)
                self._apply_drive_locked(
                    left_speed_us=-float(self._config.left_base_throttle_us),
                    right_speed_us=-float(self._config.right_base_throttle_us),
                    mode="MANUAL",
                    assist_kind="reverse_test",
                    line_error=0.0,
                    pid_output=0.0,
                )
                return
            if command == "robot.stop":
                self._motor_armed = False
                self._manual_until = 0.0
                self._maneuver_until = 0.0
                self._reset_green_half_turn_sequence_locked()
                self._reset_green_single_encounter_locked()
                self._reset_calibrated_corner_sequence_locked()
                self._green_instruction = "NO_GREEN"
                self._green_marker_count = 0
                self._neutralize_locked(
                    mode="STOPPED", assist_kind="stop", failsafe=False, disable_outputs=True
                )
                return
            if command in {"robot.force_stop", "robot.estop"}:
                self._estop_latched = True
                self._failsafe_active = True
                self._motor_armed = False
                self._manual_until = 0.0
                self._maneuver_until = 0.0
                self._reset_green_half_turn_sequence_locked()
                self._reset_green_single_encounter_locked()
                self._reset_calibrated_corner_sequence_locked()
                self._neutralize_locked(
                    mode="ESTOP", assist_kind="estop", failsafe=True, disable_outputs=True
                )
                return
            if command in {"robot.clear_estop", "robot.reset_estop"}:
                self._estop_latched = False
                self._failsafe_active = False
                if not self._config.start_disarmed:
                    self._motor_armed = True
                self._neutralize_locked(mode="STOPPED", assist_kind="clear_estop", failsafe=False)
                return
            if command in {"robot.obstacle_test", "robot.obstacle_ahead"}:
                self._obstacle_state = "TEST" if command == "robot.obstacle_test" else "AHEAD"
                self._neutralize_locked(
                    mode="OBSTACLE", assist_kind="obstacle", failsafe=False, disable_outputs=True
                )
                return
            if command in {"robot.obstacle_clear", "robot.clear_obstacle"}:
                self._obstacle_state = "CLEAR"
                self._neutralize_locked(mode="STOPPED", assist_kind="obstacle_clear", failsafe=False)

    def _maybe_handle_obstacle_locked(self, state: str, metadata: Mapping[str, Any]) -> bool:
        del state
        raw = metadata.get("obstacle")
        if isinstance(raw, Mapping):
            obstacle_state = str(raw.get("state", "CLEAR")).strip().upper()
        else:
            obstacle_state = str(metadata.get("obstacle_state", "")).strip().upper()
        if not obstacle_state:
            return False
        self._obstacle_state = obstacle_state
        if obstacle_state != "CLEAR":
            self._neutralize_locked(
                mode="OBSTACLE", assist_kind="obstacle", failsafe=False, disable_outputs=True
            )
            return True
        return False

    def _maybe_handle_green_locked(self, event: VisionDetectionEvent, metadata: Mapping[str, Any]) -> bool:
        now = self._monotonic()
        if not self._config.enable_green_maneuvers:
            self._green_streak = 0
            self._green_candidate_signature = ""
            return False

        # Once a half-turn starts, every subsequent view of that same marker
        # belongs to the same encounter. A time-only cooldown can expire while
        # the T is still in view and command a second turn, so use visual rearm
        # instead. Clear frames observed during the pivot do not count because
        # the camera is expected to sweep away from the marker while rotating.
        if not self._green_half_turn_armed:
            self._green_streak = 0
            self._green_candidate_signature = ""
            if self._maneuver_until > now:
                self._green_rearm_clear_streak = 0
                return False
            if bool(event.green):
                self._green_rearm_clear_streak = 0
                return False
            self._green_rearm_clear_streak += 1
            if self._green_rearm_clear_streak >= int(self._config.green_rearm_clear_frames):
                self._green_half_turn_armed = True
                self._green_rearm_clear_streak = 0
                self._green_instruction = "NO_GREEN"
                self._green_marker_count = 0
            return False

        if not bool(event.green):
            if self._green_single_encounter == "PENDING":
                # One camera dropout must not release the robot halfway through
                # the requested stop-and-confirm sequence.  Keep neutral and
                # preserve the live-vote count for a short bounded window; a
                # real disappearance still resumes normal line following.
                self._green_single_clear_streak += 1
                if self._green_single_clear_streak < int(
                    self._config.green_single_clear_frames
                ):
                    self._neutralize_locked(
                        mode="GREEN",
                        assist_kind="green_single_confirm_gap",
                        failsafe=False,
                        disable_outputs=True,
                    )
                    return True
                self._reset_green_single_encounter_locked()
            self._green_streak = 0
            self._green_candidate_signature = ""
            return False
        instruction = str(metadata.get("green_instruction", "NO_GREEN")).strip().upper().replace(" ", "_")
        side = str(metadata.get("green_side", "NONE")).strip().upper()
        if instruction in {"", "NO_GREEN", "NONE"} or side == "NONE":
            self._green_streak = 0
            self._green_candidate_signature = ""
            return False

        marker_count = max(
            0,
            int(metadata.get("green_marker_count", 2 if side == "BOTH" else 1) or 0),
        )
        confidence = max(
            0.0,
            min(1.0, float(metadata.get("green_marker_confidence", 0.0) or 0.0)),
        )
        half_turn_signal = instruction == "VERDE_MEIA_VOLTA" or side == "BOTH"
        is_half_turn = bool(
            instruction == "VERDE_MEIA_VOLTA"
            and side == "BOTH"
            and marker_count >= 2
            and bool(event.line)
            and confidence >= float(self._config.green_half_turn_confidence_floor)
        )
        # Never degrade an inconsistent two-green signal into a one-green turn.
        if half_turn_signal and not is_half_turn:
            self._green_streak = 0
            self._green_candidate_signature = ""
            return False

        if is_half_turn and self._config.corner_sequence_enabled and (
            self._corner_phase != "IDLE" or now < self._corner_cooldown_until
        ):
            # Preserve the proven two-green ownership rule exactly: it never
            # interrupts a calibrated corner already in progress.
            self._green_streak = 0
            self._green_candidate_signature = ""
            return False

        # A running calibrated corner remains authoritative once braking or
        # pivoting begins.  During APPROACH only, a single marker may still
        # cancel/replace the structural choice because it is the route command
        # for that intersection.  This closes the race seen in the field logs,
        # where the L detector entered APPROACH one frame before the marker.
        corner_owns_control = bool(
            self._config.corner_sequence_enabled
            and (
                self._green_single_encounter == "EXECUTE"
                or self._corner_phase not in {"IDLE", "APPROACH"}
                or now < self._corner_cooldown_until
            )
        )
        if corner_owns_control:
            self._green_streak = 0
            self._green_candidate_signature = ""
            return False

        if is_half_turn:
            signature = f"{instruction}:{side}:{marker_count}"
            if signature != self._green_candidate_signature:
                self._green_candidate_signature = signature
                self._green_streak = 1
            else:
                self._green_streak += 1
            if self._green_streak < int(self._config.green_half_turn_trigger_streak):
                return False

            duration_ms = int(self._config.green_half_turn_ms)
            self._green_instruction = instruction
            self._green_marker_count = marker_count
            self._green_maneuver_duration_ms = duration_ms
            self._last_green_trigger_at = now
            self._green_streak = 0
            self._green_candidate_signature = ""
            self._green_half_turn_armed = False
            self._green_rearm_clear_streak = 0
            self._reset_green_single_encounter_locked()
            self._green_route_decision = "HALF_TURN"
            self._reset_line_pid_locked()
            self._reset_calibrated_corner_sequence_locked()
            self._start_green_half_turn_sequence_locked(now, duration_ms)
            return True

        # Never degrade another malformed signal into a one-marker turn.
        if marker_count != 1 or side not in {"LEFT", "RIGHT"}:
            self._green_streak = 0
            self._green_candidate_signature = ""
            return False

        # A confirmed AFTER marker owns this intersection until it and the L
        # geometry have cleared.  Continuing to reconfirm the same visible
        # marker produced a stop/go pulse every other frame in the field run.
        if self._green_single_encounter == "IGNORE":
            self._green_streak = 0
            self._green_candidate_signature = ""
            return False

        relation_confidence = max(
            0.0,
            min(
                1.0,
                float(metadata.get("green_relation_confidence", 0.0) or 0.0),
            ),
        )
        relation_delta_y = float(metadata.get("green_relation_delta_y", 0.0) or 0.0)
        self._green_single_relation_confidence = relation_confidence
        self._green_single_relation_delta_y = relation_delta_y
        if self._green_single_encounter == "IDLE":
            self._green_single_encounter = "PENDING"
            self._green_single_pending_since = float(now)
        self._green_single_clear_streak = 0

        relation_is_clear = bool(
            relation_confidence
            >= float(self._config.green_single_relation_confidence_floor)
        )
        semantic_instruction = instruction in {"VERDE_ANTES", "VERDE_DEPOIS"}
        candidate_ready = bool(semantic_instruction and relation_is_clear)
        signature = f"{instruction}:{side}:{marker_count}"
        if candidate_ready:
            if signature != self._green_candidate_signature:
                self._green_candidate_signature = signature
                self._green_streak = 1
                self._green_single_pending_since = float(now)
            else:
                self._green_streak += 1
        else:
            self._green_streak = 0
            self._green_candidate_signature = ""
            self._green_single_pending_since = 0.0

        required_streak = int(self._config.green_trigger_streak)
        confirm_elapsed_ms = (
            max(0.0, float(now) - self._green_single_pending_since) * 1000.0
            if self._green_single_pending_since > 0.0
            else 0.0
        )
        confirmed = bool(
            self._green_streak >= required_streak
            and confirm_elapsed_ms
            >= float(self._config.green_single_confirm_hold_ms)
        )
        if instruction == "VERDE_DEPOIS" and confirmed:
            # Lock the whole encounter as straight-through.  The structural L
            # is suppressed until both marker and elbow have cleared, so it
            # cannot command a delayed turn after the green leaves the frame.
            if self._corner_phase == "APPROACH":
                self._reset_calibrated_corner_sequence_locked()
            self._green_single_encounter = "IGNORE"
            self._green_instruction = instruction
            self._green_route_decision = "AFTER_STRAIGHT"
            self._green_marker_count = 1
            self._green_streak = 0
            self._green_candidate_signature = ""
            return False

        if instruction == "VERDE_ANTES" and confirmed:
            visual_direction = -1 if side == "LEFT" else 1
            physical_direction = (
                -visual_direction
                if self._config.line_steering_inverted
                else visual_direction
            )
            self._green_instruction = instruction
            self._green_marker_count = 1
            self._green_maneuver_duration_ms = int(
                self._config.corner_pivot_right_ms
                if physical_direction > 0
                else self._config.corner_pivot_left_ms
            )
            self._last_green_trigger_at = now
            self._green_streak = 0
            self._green_candidate_signature = ""
            self._reset_green_single_encounter_locked(preserve_observation=True)
            self._start_single_green_corner_locked(now, physical_direction)
            self._green_single_encounter = "EXECUTE"
            self._green_route_decision = (
                "BEFORE_RIGHT" if physical_direction > 0 else "BEFORE_LEFT"
            )
            return True

        # Stop only while a clear semantic reading is accumulating.  This gives
        # the requested confirmation pause without freezing forever on an
        # ambiguous perspective overlap.  Ambiguous frames keep line following
        # but still suppress the structural corner through the helper below.
        if candidate_ready:
            self._neutralize_locked(
                mode="GREEN",
                assist_kind="green_single_confirm",
                failsafe=False,
                disable_outputs=True,
            )
            return True
        return False

    def _single_green_suppresses_corner_locked(
        self,
        event: VisionDetectionEvent,
        metadata: Mapping[str, Any],
    ) -> bool:
        """Keep the structural L from overriding one marker encounter."""
        if self._green_single_encounter not in {"PENDING", "IGNORE"}:
            return False

        if bool(event.green):
            self._green_single_clear_streak = 0
            return True

        geometry = metadata.get("line_geometry")
        geometry = geometry if isinstance(geometry, Mapping) else {}
        right_angle = bool(
            float(geometry.get("right_angle_corridor", 0.0) or 0.0) > 0.0
        )
        bend = abs(float(geometry.get("path_bend_delta_norm", 0.0) or 0.0))
        max_row = float(geometry.get("max_row_occupancy", 0.0) or 0.0)
        elbow_still_visible = bool(
            right_angle
            or (
                bend >= float(self._config.corner_min_abs_bend)
                and max_row >= min(
                    0.45,
                    float(self._config.corner_min_wide_row_occupancy),
                )
            )
        )
        if elbow_still_visible:
            self._green_single_clear_streak = 0
            return True

        self._green_single_clear_streak += 1
        if self._green_single_clear_streak < int(
            self._config.green_single_clear_frames
        ):
            return True
        self._reset_green_single_encounter_locked()
        self._green_instruction = "NO_GREEN"
        self._green_marker_count = 0
        return False

    def _reset_green_single_encounter_locked(
        self,
        *,
        preserve_observation: bool = False,
    ) -> None:
        self._green_single_encounter = "IDLE"
        self._green_single_clear_streak = 0
        self._green_single_pending_since = 0.0
        self._green_route_decision = "NONE"
        if not preserve_observation:
            self._green_single_relation_confidence = 0.0
            self._green_single_relation_delta_y = 0.0

    def _start_single_green_corner_locked(
        self,
        now: float,
        physical_direction: int,
    ) -> None:
        """Route one BEFORE marker through the proven calibrated 90-degree FSM."""
        self._reset_line_pid_locked()
        self._reset_calibrated_corner_sequence_locked()
        self._corner_direction = 1 if physical_direction > 0 else -1
        self._corner_sequence_started_at = float(now)
        self._corner_approach_lost_streak = 0
        self._corner_reacquire_streak = 0
        self._sharp_curve_active = False
        self._last_sharp_curve_reverse_inner = False
        self._set_corner_phase_locked("APPROACH", now)
        self._last_event_line = (
            "single green calibrated corner "
            + ("RIGHT" if self._corner_direction > 0 else "LEFT")
        )
        self._last_event_at = float(now)
        if self._corner_approach_duration_ms_locked() > 0:
            self._apply_corner_approach_locked(0.0, 0.0)
        else:
            self._set_corner_phase_locked("BRAKE", now)
            self._apply_corner_neutral_locked("line_corner_brake")

    def _start_green_half_turn_sequence_locked(self, now: float, pivot_ms: int) -> None:
        """Brake, execute two proven 90-degree pivots, and brake again."""
        self._green_half_turn_first_ms, self._green_half_turn_second_ms = (
            self._green_half_turn_segment_durations_locked(pivot_ms)
        )
        brake_ms = int(self._config.corner_brake_ms)
        mid_brake_ms = self._green_half_turn_mid_brake_ms_locked()
        exit_brake_ms = int(self._config.corner_exit_neutral_ms)
        reverse_ms = int(self._config.green_half_turn_reverse_ms)
        total_sequence_ms = (
            brake_ms
            + self._green_half_turn_first_ms
            + mid_brake_ms
            + reverse_ms
            + exit_brake_ms
            + self._green_half_turn_second_ms
            + exit_brake_ms
        )
        self._green_maneuver_duration_ms = total_sequence_ms
        self._green_half_turn_phase = "BRAKE_1"
        self._green_half_turn_phase_started_at = float(now)
        self._maneuver_until = float(now) + (total_sequence_ms / 1000.0)
        self._neutralize_locked(
            mode="GREEN",
            assist_kind="green_half_turn_brake_1",
            failsafe=False,
            disable_outputs=True,
        )

    def _tick_green_half_turn_sequence_locked(self, now: float) -> bool:
        phase = self._green_half_turn_phase
        if phase == "IDLE":
            return False

        elapsed_ms = max(0.0, float(now) - self._green_half_turn_phase_started_at) * 1000.0
        if phase == "BRAKE_1":
            if elapsed_ms >= float(self._config.corner_brake_ms):
                self._apply_green_half_turn_pivot_locked(stage=1)
                self._green_half_turn_phase = "PIVOT_1"
                self._green_half_turn_phase_started_at = float(now)
            return True

        if phase == "PIVOT_1":
            if elapsed_ms >= float(self._green_half_turn_first_ms):
                self._neutralize_locked(
                    mode="GREEN",
                    assist_kind="green_half_turn_brake_mid",
                    failsafe=False,
                    disable_outputs=True,
                )
                self._green_half_turn_phase = "BRAKE_MID"
                self._green_half_turn_phase_started_at = float(now)
            return True

        if phase == "BRAKE_MID":
            if elapsed_ms >= float(self._green_half_turn_mid_brake_ms_locked()):
                if int(self._config.green_half_turn_reverse_ms) <= 0:
                    self._apply_green_half_turn_pivot_locked(stage=2)
                    self._green_half_turn_phase = "PIVOT_2"
                    self._green_half_turn_phase_started_at = float(now)
                    return True
                self._apply_drive_locked(
                    left_speed_us=-float(self._config.left_base_throttle_us),
                    right_speed_us=-float(self._config.right_base_throttle_us),
                    mode="GREEN",
                    assist_kind="green_half_turn_reverse",
                    line_error=0.0,
                    pid_output=0.0,
                )
                self._steering_decision = "REVERSE_BETWEEN_90"
                self._green_half_turn_phase = "REVERSE"
                self._green_half_turn_phase_started_at = float(now)
            return True

        if phase == "REVERSE":
            if elapsed_ms >= float(self._config.green_half_turn_reverse_ms):
                self._neutralize_locked(
                    mode="GREEN",
                    assist_kind="green_half_turn_brake_reverse",
                    failsafe=False,
                    disable_outputs=True,
                )
                self._green_half_turn_phase = "BRAKE_REVERSE"
                self._green_half_turn_phase_started_at = float(now)
            return True

        if phase == "BRAKE_REVERSE":
            if elapsed_ms >= float(self._config.corner_exit_neutral_ms):
                self._apply_green_half_turn_pivot_locked(stage=2)
                self._green_half_turn_phase = "PIVOT_2"
                self._green_half_turn_phase_started_at = float(now)
            return True

        if phase == "PIVOT_2":
            if elapsed_ms >= float(self._green_half_turn_second_ms):
                self._neutralize_locked(
                    mode="GREEN",
                    assist_kind="green_half_turn_brake_exit",
                    failsafe=False,
                    disable_outputs=True,
                )
                self._green_half_turn_phase = "BRAKE_EXIT"
                self._green_half_turn_phase_started_at = float(now)
            return True

        if phase == "BRAKE_EXIT":
            if elapsed_ms >= float(self._config.corner_exit_neutral_ms):
                self._neutralize_locked(
                    mode="STOPPED",
                    assist_kind="green_half_turn_complete",
                    failsafe=False,
                    disable_outputs=True,
                )
                self._maneuver_until = 0.0
                self._reset_green_half_turn_sequence_locked()
            return True

        self._maneuver_until = 0.0
        self._reset_green_half_turn_sequence_locked()
        self._neutralize_locked(
            mode="STOPPED",
            assist_kind="green_half_turn_invalid_phase",
            failsafe=False,
            disable_outputs=True,
        )
        return True

    def _apply_green_half_turn_pivot_locked(self, *, stage: int) -> None:
        left_speed, right_speed = self._green_half_turn_pivot_speeds_locked(
            float(self._config.green_half_turn_us)
        )
        self._apply_drive_locked(
            left_speed_us=left_speed,
            right_speed_us=right_speed,
            mode="GREEN",
            assist_kind=f"green_half_turn_pivot_{int(stage)}",
            line_error=0.0,
            pid_output=0.0,
        )
        self._steering_decision = f"RIGHT_180_STAGE_{int(stage)}"

    def _green_half_turn_mid_brake_ms_locked(self) -> int:
        return max(150, min(400, int(self._config.corner_brake_ms) // 2))

    def _green_half_turn_segment_durations_locked(
        self,
        pivot_ms: int | None = None,
    ) -> tuple[int, int]:
        del pivot_ms
        return (
            int(self._config.green_half_turn_first_ms),
            int(self._config.green_half_turn_second_ms),
        )

    def _reset_green_half_turn_sequence_locked(self) -> None:
        self._green_half_turn_phase = "IDLE"
        self._green_half_turn_phase_started_at = 0.0
        self._green_half_turn_first_ms = 0
        self._green_half_turn_second_ms = 0

    def _tick_calibrated_corner_timing_locked(self, now: float) -> bool:
        """Advance time-only corner phases with monitor-thread precision."""
        if not self._config.corner_sequence_enabled:
            return False
        elapsed_ms = max(0.0, now - self._corner_phase_started_at) * 1000.0
        if self._corner_phase == "BRAKE" and elapsed_ms >= float(
            self._config.corner_brake_ms
        ):
            self._set_corner_phase_locked("PIVOT", now)
            self._apply_corner_pivot_locked(
                float(self._config.corner_pivot_speed_us),
                "line_corner_pivot",
            )
            return True
        if self._corner_phase == "PIVOT" and elapsed_ms >= float(
            self._corner_pivot_duration_ms_locked()
        ):
            self._set_corner_phase_locked("REACQUIRE", now)
            self._corner_reacquire_streak = 0
            self._apply_corner_pivot_locked(
                float(self._config.corner_reacquire_speed_us),
                "line_corner_reacquire",
            )
            return True
        if self._corner_phase == "REACQUIRE" and elapsed_ms >= float(
            self._config.corner_reacquire_timeout_ms
        ):
            self._enter_corner_timeout_locked(now, "line_corner_reacquire_timeout")
            return True
        if self._corner_phase == "EXIT":
            neutral_ms = float(self._config.corner_exit_neutral_ms)
            total_ms = neutral_ms + float(self._config.corner_exit_straight_ms)
            if elapsed_ms >= total_ms:
                self._apply_corner_neutral_locked("line_corner_exit_complete")
                self._corner_cooldown_until = now + (
                    float(self._config.corner_cooldown_ms) / 1000.0
                )
                self._reset_calibrated_corner_sequence_locked(preserve_cooldown=True)
                return True
            if elapsed_ms >= neutral_ms:
                self._apply_drive_locked(
                    left_speed_us=float(self._config.left_base_throttle_us),
                    right_speed_us=float(self._config.right_base_throttle_us),
                    mode="FOLLOW_LINE",
                    assist_kind="line_corner_exit_straight",
                    line_error=0.0,
                    pid_output=0.0,
                )
                return True
        return False

    def _monitor_loop(self) -> None:
        interval = max(0.02, float(self._config.monitor_interval_ms) / 1000.0)
        while not self._monitor_stop.wait(interval):
            with self._lock:
                if not self._running or self._estop_latched or not self._motor_armed:
                    continue
                now = self._monotonic()
                try:
                    if self._manual_until > 0 and now >= self._manual_until:
                        self._manual_until = 0.0
                        self._neutralize_locked(
                            mode="STOPPED", assist_kind="manual_timeout", failsafe=False, disable_outputs=True
                        )
                        self._monitor_error_streak = 0
                        continue
                    if self._tick_green_half_turn_sequence_locked(now):
                        self._monitor_error_streak = 0
                        continue
                    if self._maneuver_until > 0 and now >= self._maneuver_until:
                        self._maneuver_until = 0.0
                        self._neutralize_locked(
                            mode="STOPPED", assist_kind="green_timeout", failsafe=False, disable_outputs=True
                        )
                        self._monitor_error_streak = 0
                        continue
                    self._tick_calibrated_corner_timing_locked(now)
                    timeout_s = max(0.1, float(self._config.control_timeout_ms) / 1000.0)
                    if self._control_mode == "FOLLOW_LINE" and self._last_detection_at > 0 and (now - self._last_detection_at) > timeout_s:
                        self._neutralize_locked(
                            mode="STOPPED", assist_kind="detection_timeout", failsafe=False, disable_outputs=True
                        )
                    self._monitor_error_streak = 0
                except OSError as exc:
                    # Motor switching can briefly disturb the shared I2C bus.
                    # Preserve the current phase so the next monitor tick
                    # retries the command instead of abandoning the second
                    # half of the calibrated 180-degree maneuver.
                    self._monitor_error_streak += 1
                    self._last_event_line = (
                        f"monitor I2C retry {self._monitor_error_streak}: {exc}"
                    )
                    self._last_event_at = float(now)
                    self._publish_log(
                        "WARNING",
                        f"PCA monitor transient I2C failure; retrying phase "
                        f"{self._green_half_turn_phase}: {exc}",
                    )
                    if self._monitor_error_streak < 3:
                        continue
                    self._estop_latched = True
                    self._failsafe_active = True
                    self._motor_armed = False
                    self._maneuver_until = 0.0
                    self._reset_green_half_turn_sequence_locked()
                    self._reset_calibrated_corner_sequence_locked()
                    self._disable_outputs_locked()
                    self._publish_log(
                        "ERROR",
                        "PCA monitor failed three consecutive writes; outputs disabled",
                    )

    def _apply_drive_locked(
        self,
        *,
        left_speed_us: float,
        right_speed_us: float,
        mode: str,
        assist_kind: str,
        line_error: float,
        pid_output: float,
    ) -> None:
        if not self._motor_armed:
            self._neutralize_locked(
                mode="STOPPED",
                assist_kind="motor_disarmed",
                failsafe=False,
                disable_outputs=True,
            )
            return
        allow_basic_correction = bool(self._config.basic_line_follow)
        left_pulse = self._pulse_for_speed(
            left_speed_us,
            inverted=bool(self._config.left_inverted),
            allow_base_plus_correction=allow_basic_correction,
        )
        right_pulse = self._pulse_for_speed(
            right_speed_us,
            inverted=bool(self._config.right_inverted),
            allow_base_plus_correction=allow_basic_correction,
        )
        self._set_pair_locked(left_pulse, right_pulse)
        self._control_mode = mode
        self._assist_kind = assist_kind
        self._line_error = float(line_error)
        self._pid_output = float(pid_output)
        self._requested_left_speed_us = float(left_speed_us)
        self._requested_right_speed_us = float(right_speed_us)
        if mode == "FOLLOW_LINE":
            physical_correction = (
                -float(pid_output)
                if self._config.line_steering_inverted
                else float(pid_output)
            )
            if physical_correction > 0.5:
                self._steering_decision = "RIGHT"
            elif physical_correction < -0.5:
                self._steering_decision = "LEFT"
            else:
                self._steering_decision = "STRAIGHT"
            if "sharp" in str(assist_kind).lower() and self._steering_decision != "STRAIGHT":
                self._steering_decision = f"{self._steering_decision}_SHARP"
        else:
            self._steering_decision = str(mode or "STOPPED").upper()
        self._failsafe_active = False
        self._last_command_at = self._monotonic()

    def _handle_calibrated_corner_sequence_locked(
        self,
        event: VisionDetectionEvent,
        metadata: Mapping[str, Any],
        now: float,
    ) -> bool:
        """Own a structurally confirmed 90-degree L through its full exit."""
        geometry = metadata.get("line_geometry")
        if not isinstance(geometry, Mapping):
            geometry = {}

        confidence = float(metadata.get("line_confidence", 0.0) or 0.0)
        offset = max(
            -1.0,
            min(1.0, float(metadata.get("line_offset_norm", 0.0) or 0.0)),
        )
        bend = max(
            -1.5,
            min(1.5, float(geometry.get("path_bend_delta_norm", 0.0) or 0.0)),
        )
        right_angle = bool(
            float(geometry.get("right_angle_corridor", 0.0) or 0.0) > 0.0
        )
        dominant_row = max(
            0.0,
            min(1.0, float(geometry.get("dominant_row_y_ratio", 0.0) or 0.0)),
        )
        max_row_occupancy = max(
            0.0,
            min(1.0, float(geometry.get("max_row_occupancy", 0.0) or 0.0)),
        )
        median_row_occupancy = max(
            0.0,
            min(1.0, float(geometry.get("median_row_occupancy", 0.0) or 0.0)),
        )
        track_width_ratio = max(
            0.0,
            min(1.0, float(geometry.get("width_ratio", 0.0) or 0.0)),
        )
        track_height_ratio = max(
            0.0,
            min(1.0, float(geometry.get("height_ratio", 0.0) or 0.0)),
        )
        elbow_contrast = (
            max_row_occupancy / median_row_occupancy
            if median_row_occupancy >= 0.05
            else 0.0
        )
        self._corner_elbow_row_contrast = elbow_contrast
        line_valid = bool(
            event.state == RobotState.FOLLOWING_LINE.value
            and bool(event.line)
            and confidence >= float(self._config.line_confidence_floor)
        )
        geometry_fallback = bool(
            line_valid
            and not right_angle
            and dominant_row >= float(self._config.corner_fallback_min_row_ratio)
            and abs(bend) >= max(float(self._config.corner_min_abs_bend), 0.25)
            and max_row_occupancy
            >= float(self._config.corner_min_wide_row_occupancy)
            and elbow_contrast
            >= float(self._config.corner_fallback_min_elbow_row_contrast)
        )
        visual_direction = 1 if bend > 0.0 else -1 if bend < 0.0 else 0
        physical_direction = (
            -visual_direction
            if self._config.line_steering_inverted
            else visual_direction
        )
        left_geometry_fallback = bool(
            line_valid
            and not right_angle
            and physical_direction < 0
            and bend >= 0.28
            and dominant_row
            >= float(self._config.corner_left_fallback_min_row_ratio)
            and max_row_occupancy
            >= float(self._config.corner_left_fallback_min_wide_row_occupancy)
            and elbow_contrast
            >= float(self._config.corner_left_fallback_min_elbow_row_contrast)
            and track_width_ratio
            >= float(self._config.corner_left_fallback_min_width_ratio)
            and track_height_ratio
            >= float(self._config.corner_left_fallback_min_height_ratio)
        )
        geometry_fallback = bool(geometry_fallback or left_geometry_fallback)
        self._corner_geometry_fallback_active = geometry_fallback
        structural_elbow = bool(
            line_valid
            and (right_angle or geometry_fallback)
            and abs(bend) >= float(self._config.corner_min_abs_bend)
            and (
                left_geometry_fallback
                or (
                    max_row_occupancy
                    >= float(self._config.corner_min_wide_row_occupancy)
                    and elbow_contrast
                    >= float(self._config.corner_min_elbow_row_contrast)
                )
            )
        )

        if self._corner_phase == "IDLE":
            if now < self._corner_cooldown_until:
                self._corner_confirm_streak = 0
                self._corner_confirm_direction = 0
                self._corner_confirm_max_row_ratio = 0.0
                return False
            if not structural_elbow or physical_direction == 0:
                self._corner_confirm_streak = 0
                self._corner_confirm_direction = 0
                self._corner_confirm_max_row_ratio = 0.0
                return False
            if physical_direction != self._corner_confirm_direction:
                self._corner_confirm_direction = physical_direction
                self._corner_confirm_streak = 1
                self._corner_confirm_max_row_ratio = dominant_row
            else:
                self._corner_confirm_streak += 1
                self._corner_confirm_max_row_ratio = max(
                    self._corner_confirm_max_row_ratio,
                    dominant_row,
                )
            required_confirm_frames = int(self._config.corner_confirm_frames)
            if left_geometry_fallback:
                required_confirm_frames = max(
                    required_confirm_frames,
                    int(self._config.corner_left_fallback_confirm_frames),
                )
            if self._corner_confirm_streak < required_confirm_frames:
                return False

            self._corner_direction = self._corner_confirm_direction
            self._corner_sequence_started_at = now
            self._corner_approach_lost_streak = 0
            self._corner_reacquire_streak = 0
            self._sharp_curve_active = False
            self._last_sharp_curve_reverse_inner = False
            approach_min_ms = self._corner_approach_duration_ms_locked()
            if approach_min_ms > 0:
                self._set_corner_phase_locked("APPROACH", now)
                # Keep this short placement movement straight.  Steering from
                # the L geometry would start the turn before the calibrated
                # in-place pivot and recreate the open-corner failure.
                self._apply_corner_approach_locked(0.0, 0.0)
            elif (
                self._corner_confirm_max_row_ratio
                >= float(self._config.corner_approach_stop_row_ratio)
            ):
                self._set_corner_phase_locked("BRAKE", now)
                self._apply_corner_neutral_locked("line_corner_brake")
            else:
                self._set_corner_phase_locked("APPROACH", now)
                self._apply_corner_approach_locked(offset, bend)
            return True

        if self._corner_phase == "APPROACH":
            phase_elapsed_ms = (now - self._corner_phase_started_at) * 1000.0
            if phase_elapsed_ms >= float(self._config.corner_approach_timeout_ms):
                self._enter_corner_timeout_locked(now, "line_corner_approach_timeout")
                return True
            approach_min_ms = self._corner_approach_duration_ms_locked()
            if approach_min_ms > 0:
                if phase_elapsed_ms < float(approach_min_ms):
                    self._apply_corner_approach_locked(0.0, 0.0)
                else:
                    self._set_corner_phase_locked("BRAKE", now)
                    self._apply_corner_neutral_locked("line_corner_brake")
                return True
            same_direction = structural_elbow and physical_direction == self._corner_direction
            if same_direction:
                self._corner_approach_lost_streak = 0
                if dominant_row >= float(self._config.corner_approach_stop_row_ratio):
                    self._set_corner_phase_locked("BRAKE", now)
                    self._apply_corner_neutral_locked("line_corner_brake")
                    return True
                self._apply_corner_approach_locked(offset, bend)
                return True

            self._corner_approach_lost_streak += 1
            if self._corner_approach_lost_streak > int(
                self._config.corner_approach_lost_frames
            ):
                # A smooth curve can briefly resemble an L. Abort before any
                # brake or reverse command and return control to the already
                # validated ordinary curve follower on this same frame.
                self._reset_calibrated_corner_sequence_locked()
            return False

        if self._corner_phase == "BRAKE":
            if (now - self._corner_phase_started_at) * 1000.0 >= float(
                self._config.corner_brake_ms
            ):
                self._set_corner_phase_locked("PIVOT", now)
                self._apply_corner_pivot_locked(
                    float(self._config.corner_pivot_speed_us),
                    "line_corner_pivot",
                )
            else:
                self._apply_corner_neutral_locked("line_corner_brake")
            return True

        if self._corner_phase == "PIVOT":
            pivot_ms = self._corner_pivot_duration_ms_locked()
            pivot_elapsed_ms = (now - self._corner_phase_started_at) * 1000.0
            if self._corner_direction < 0 and pivot_elapsed_ms >= float(
                self._config.corner_pivot_left_vision_min_ms
            ):
                if line_valid:
                    self._corner_pivot_line_lost_streak = 0
                else:
                    self._corner_pivot_line_lost_streak += 1
                    if self._corner_pivot_line_lost_streak >= int(
                        self._config.corner_pivot_left_lost_confirm_frames
                    ):
                        self._corner_pivot_line_lost_seen = True
                if self._corner_pivot_line_lost_seen and (
                    self._corner_outgoing_line_reacquired_locked(
                        event,
                        metadata,
                        geometry,
                        offset,
                        bend,
                        right_angle,
                        confidence,
                    )
                ):
                    self._set_corner_phase_locked("REACQUIRE", now)
                    self._corner_reacquire_streak = 1
                    self._apply_corner_neutral_locked(
                        "line_corner_reacquire_verify"
                    )
                    return True
            if pivot_elapsed_ms >= float(pivot_ms):
                self._set_corner_phase_locked("REACQUIRE", now)
                self._corner_reacquire_streak = 0
                self._apply_corner_pivot_locked(
                    float(self._config.corner_reacquire_speed_us),
                    "line_corner_reacquire",
                )
            else:
                self._apply_corner_pivot_locked(
                    float(self._config.corner_pivot_speed_us),
                    "line_corner_pivot",
                )
            return True

        if self._corner_phase == "REACQUIRE":
            if self._corner_outgoing_line_reacquired_locked(
                event,
                metadata,
                geometry,
                offset,
                bend,
                right_angle,
                confidence,
            ):
                self._corner_reacquire_streak += 1
            else:
                self._corner_reacquire_streak = 0
            if self._corner_reacquire_streak >= int(
                self._config.corner_reacquire_confirm_frames
            ):
                self._set_corner_phase_locked("EXIT", now)
                self._apply_corner_neutral_locked("line_corner_exit_neutral")
            elif self._corner_reacquire_streak > 0:
                # Brake on the first plausible outgoing-line frame and verify
                # it while stationary. Continuing the slow pivot during the
                # confirmation streak caused both the left overshoot and the
                # intermittent right-side miss in the recorded field runs.
                self._apply_corner_neutral_locked("line_corner_reacquire_verify")
            elif (now - self._corner_phase_started_at) * 1000.0 >= float(
                self._config.corner_reacquire_timeout_ms
            ):
                self._enter_corner_timeout_locked(now, "line_corner_reacquire_timeout")
            else:
                self._apply_corner_pivot_locked(
                    float(self._config.corner_reacquire_speed_us),
                    "line_corner_reacquire",
                )
            return True

        if self._corner_phase == "EXIT":
            exit_elapsed_ms = (now - self._corner_phase_started_at) * 1000.0
            if exit_elapsed_ms < float(self._config.corner_exit_neutral_ms):
                self._apply_corner_neutral_locked("line_corner_exit_neutral")
                return True
            if exit_elapsed_ms < float(
                self._config.corner_exit_neutral_ms
                + self._config.corner_exit_straight_ms
            ):
                self._apply_drive_locked(
                    left_speed_us=float(self._config.left_base_throttle_us),
                    right_speed_us=float(self._config.right_base_throttle_us),
                    mode="FOLLOW_LINE",
                    assist_kind="line_corner_exit_straight",
                    line_error=0.0,
                    pid_output=0.0,
                )
                return True
            self._corner_cooldown_until = now + (
                float(self._config.corner_cooldown_ms) / 1000.0
            )
            self._reset_calibrated_corner_sequence_locked(preserve_cooldown=True)
            return False

        if self._corner_phase == "TIMEOUT":
            # TIMEOUT is a bounded neutral recovery state, not a latched ESTOP.
            # The field logs showed failsafe=false while valid line frames were
            # ignored forever; pressing "Clear ESTOP" only appeared to help
            # because it incidentally reset this corner phase.  Once a fresh
            # ground-path line is visible, hold neutral briefly, reset with a
            # cooldown, and let the normal proven line follower handle the same
            # frame.  A truly missing line remains safely stopped.
            line_angle_deg = float(metadata.get("line_angle_deg", 90.0) or 90.0)
            candidate_count = int(float(geometry.get("candidate_count", 0) or 0))
            recovery_line_visible = bool(
                line_valid
                and self._line_offset_source == "ground_path"
                and candidate_count >= 1
                and 45.0 <= line_angle_deg <= 135.0
            )
            timeout_elapsed_ms = (now - self._corner_phase_started_at) * 1000.0
            if recovery_line_visible and timeout_elapsed_ms >= float(
                self._config.corner_exit_neutral_ms
            ):
                self._corner_cooldown_until = now + (
                    float(self._config.corner_cooldown_ms) / 1000.0
                )
                self._reset_calibrated_corner_sequence_locked(preserve_cooldown=True)
                return False
            self._apply_corner_neutral_locked("line_corner_timeout", mode="STOPPED")
            return True

        self._reset_calibrated_corner_sequence_locked()
        return False

    def _corner_outgoing_line_reacquired_locked(
        self,
        event: VisionDetectionEvent,
        metadata: Mapping[str, Any],
        geometry: Mapping[str, Any],
        offset: float,
        bend: float,
        right_angle: bool,
        confidence: float,
    ) -> bool:
        del bend, right_angle
        if (
            event.state != RobotState.FOLLOWING_LINE.value
            or not bool(event.line)
            or confidence < float(self._config.line_confidence_floor)
            or abs(offset) > float(self._config.corner_reacquire_max_offset)
        ):
            return False
        line_angle_deg = float(metadata.get("line_angle_deg", 90.0) or 90.0)
        candidate_count = int(float(geometry.get("candidate_count", 0) or 0))
        return bool(
            self._line_offset_source == "ground_path"
            and candidate_count >= 1
            and 55.0 <= line_angle_deg <= 125.0
        )

    def _apply_corner_approach_locked(self, offset: float, bend: float) -> None:
        steering_error = max(
            -1.0,
            min(
                1.0,
                float(offset) + (float(self._config.curve_lookahead_gain) * float(bend)),
            ),
        )
        correction = self._compute_basic_line_correction_locked(steering_error)
        left_speed, right_speed = self._line_drive_speeds_locked(
            correction,
            throttle_scale=float(self._config.corner_approach_throttle_scale),
            sharp_turn=False,
        )
        self._apply_drive_locked(
            left_speed_us=left_speed,
            right_speed_us=right_speed,
            mode="FOLLOW_LINE",
            assist_kind="line_corner_approach",
            line_error=steering_error,
            pid_output=correction,
        )

    def _apply_corner_neutral_locked(
        self,
        assist_kind: str,
        *,
        mode: str = "FOLLOW_LINE",
    ) -> None:
        self._apply_drive_locked(
            left_speed_us=0.0,
            right_speed_us=0.0,
            mode=mode,
            assist_kind=assist_kind,
            line_error=0.0,
            pid_output=0.0,
        )
        self._steering_decision = "STOP"

    def _apply_corner_pivot_locked(self, speed_us: float, assist_kind: str) -> None:
        magnitude = max(0.0, float(speed_us))
        if self._corner_direction > 0:
            left_speed, right_speed = magnitude, -magnitude
            decision = "RIGHT_90"
        else:
            if assist_kind == "line_corner_pivot":
                magnitude = max(
                    0.0,
                    float(self._config.corner_pivot_left_speed_us),
                )
            left_speed, right_speed = -magnitude, magnitude
            decision = "LEFT_90"
        self._apply_drive_locked(
            left_speed_us=left_speed,
            right_speed_us=right_speed,
            mode="FOLLOW_LINE",
            assist_kind=assist_kind,
            line_error=0.0,
            pid_output=0.0,
        )
        self._steering_decision = decision

    def _corner_pivot_duration_ms_locked(self) -> int:
        return int(
            self._config.corner_pivot_right_ms
            if self._corner_direction > 0
            else self._config.corner_pivot_left_ms
        )

    def _corner_approach_duration_ms_locked(self) -> int:
        return int(
            self._config.corner_approach_left_min_ms
            if self._corner_direction < 0
            else self._config.corner_approach_min_ms
        )

    def _set_corner_phase_locked(self, phase: str, now: float) -> None:
        self._corner_phase = str(phase).upper()
        self._corner_phase_started_at = float(now)
        if self._corner_phase == "PIVOT":
            self._corner_pivot_line_lost_streak = 0
            self._corner_pivot_line_lost_seen = False

    def _enter_corner_timeout_locked(self, now: float, assist_kind: str) -> None:
        self._set_corner_phase_locked("TIMEOUT", now)
        self._apply_corner_neutral_locked(assist_kind, mode="STOPPED")

    def _reset_calibrated_corner_sequence_locked(
        self,
        *,
        preserve_cooldown: bool = False,
    ) -> None:
        cooldown_until = self._corner_cooldown_until if preserve_cooldown else 0.0
        self._corner_phase = "IDLE"
        self._corner_direction = 0
        self._corner_phase_started_at = 0.0
        self._corner_sequence_started_at = 0.0
        self._corner_confirm_streak = 0
        self._corner_confirm_direction = 0
        self._corner_confirm_max_row_ratio = 0.0
        self._corner_approach_lost_streak = 0
        self._corner_pivot_line_lost_streak = 0
        self._corner_pivot_line_lost_seen = False
        self._corner_reacquire_streak = 0
        self._corner_elbow_row_contrast = 0.0
        self._corner_geometry_fallback_active = False
        self._corner_cooldown_until = cooldown_until
        if self._green_single_encounter == "EXECUTE":
            # The completed marker-triggered corner becomes one ignored
            # encounter until the marker/elbow clears, preventing retriggers.
            self._green_single_encounter = "IGNORE"
            self._green_single_clear_streak = 0
            self._green_single_pending_since = 0.0

    def _line_drive_speeds_locked(
        self,
        correction: float,
        *,
        throttle_scale: float = 1.0,
        sharp_turn: bool = False,
        reverse_inner: bool = False,
        inner_reverse_scale_override: float | None = None,
    ) -> tuple[float, float]:
        """Mix a bounded steering correction into calibrated forward speeds."""
        scale = max(0.0, min(1.0, float(throttle_scale)))
        physical_correction = float(correction)
        if self._config.line_steering_inverted:
            physical_correction = -physical_correction
        if sharp_turn and abs(physical_correction) > 0.5:
            outer_scale = max(0.5, min(1.35, float(self._config.sharp_curve_outer_scale)))
            inner_reverse_scale = (
                max(
                    0.0,
                    min(
                        1.0,
                        float(
                            self._config.sharp_curve_inner_reverse_scale
                            if inner_reverse_scale_override is None
                            else inner_reverse_scale_override
                        ),
                    ),
                )
                if reverse_inner
                else 0.0
            )
            if physical_correction > 0.0:
                # Physical right turn: left is outside, right is inside.
                return (
                    float(self._config.left_base_throttle_us) * scale * outer_scale,
                    -float(self._config.right_base_throttle_us) * scale * inner_reverse_scale,
                )
            # Physical left turn: right is outside, left is inside.
            return (
                -float(self._config.left_base_throttle_us) * scale * inner_reverse_scale,
                float(self._config.right_base_throttle_us) * scale * outer_scale,
            )
        return (
            max(0.0, (float(self._config.left_base_throttle_us) * scale) + physical_correction),
            max(0.0, (float(self._config.right_base_throttle_us) * scale) - physical_correction),
        )

    def _compute_basic_line_correction_locked(self, error: float) -> float:
        """Use only the current image offset for a smooth basic correction."""
        deadband = max(0.0, min(0.25, float(self._config.line_error_deadband)))
        magnitude = abs(float(error))
        if magnitude <= deadband:
            return 0.0

        # Start at zero outside the center deadband and grow linearly. This
        # is proportional-only steering, deliberately without I, D, or hold.
        normalized = min(1.0, max(0.0, (magnitude - deadband) / max(1e-6, 1.0 - deadband)))
        limit = abs(float(self._config.max_output_us))
        if limit <= 0.0:
            limit = abs(float(self._config.turn_gain_us))
        correction = min(limit, abs(float(self._config.turn_gain_us)) * normalized)
        return correction if float(error) > 0.0 else -correction

    def _hold_sharp_curve_command_locked(self, now: float) -> bool:
        """Keep a bounded pivot through the brief blind spot of a 90-degree turn."""
        if not self._sharp_curve_active or self._last_sharp_curve_at <= 0.0:
            return False
        corner_hold_s = float(self._config.sharp_curve_hold_ms) / 1000.0
        corner_elapsed = (
            max(0.0, float(now) - self._corner_started_at)
            if self._corner_started_at > 0.0
            else 0.0
        )
        total_bound_s = float(self._config.sharp_curve_visible_commit_ms) / 1000.0
        if (
            self._last_sharp_curve_reverse_inner
            and self._corner_started_at > 0.0
            and corner_elapsed >= total_bound_s
        ):
            # With no visible line there is no evidence that driving straight
            # is safe. End the corner commitment at its total bound and let the
            # caller apply calibrated neutral.
            self._sharp_curve_active = False
            return False
        committed_reverse_corner = bool(
            self._last_sharp_curve_reverse_inner
            and self._corner_started_at > 0.0
        )
        hold_s = (
            corner_hold_s
            if committed_reverse_corner
            else float(self._config.ordinary_sharp_hold_ms) / 1000.0
        )
        if hold_s <= 0.0:
            return False
        elapsed = max(0.0, float(now) - self._last_sharp_curve_at)
        if not committed_reverse_corner and elapsed > hold_s:
            self._sharp_curve_active = False
            return False

        # A detector-confirmed 90-degree corner is expected to leave the
        # camera completely while the chassis is still rotating. The field
        # recording showed that measuring this hold from the last visible
        # frame stopped several pivots at 0.7-1.2 s, before the new leg could
        # enter the image. Keep the latched direction through that blind spot,
        # but only until the absolute corner bound checked above. Ordinary
        # sharp corrections still use the short last-visible-frame hold.

        finish_phase = bool(
            self._last_sharp_curve_reverse_inner
            and self._corner_started_at > 0.0
            and corner_elapsed > corner_hold_s
        )
        inner_reverse_scale_override = (
            float(self._config.sharp_curve_finish_inner_reverse_scale)
            if finish_phase
            else None
        )

        left_speed, right_speed = self._line_drive_speeds_locked(
            self._last_sharp_curve_correction,
            throttle_scale=(
                1.0
                if committed_reverse_corner
                else self._last_sharp_curve_throttle_scale
            ),
            sharp_turn=True,
            reverse_inner=self._last_sharp_curve_reverse_inner,
            inner_reverse_scale_override=inner_reverse_scale_override,
        )
        self._apply_drive_locked(
            left_speed_us=left_speed,
            right_speed_us=right_speed,
            mode="FOLLOW_LINE",
            assist_kind=("line_sharp_finish_hold" if finish_phase else "line_sharp_hold"),
            line_error=self._last_sharp_curve_error,
            pid_output=self._last_sharp_curve_correction,
        )
        return True

    def _green_half_turn_pivot_speeds_locked(
        self,
        turn_us: float,
    ) -> tuple[float, float]:
        """Reuse the exact symmetric pivot proven by the calibrated 90-degree turn."""
        magnitude = max(0.0, float(turn_us))
        return magnitude, -magnitude

    def _start_corner_exit_locked(self, now: float, correction: float) -> bool:
        """Enter a bounded, one-direction stabilization after a sharp turn."""
        self._sharp_curve_active = False
        self._last_sharp_curve_reverse_inner = False
        self._corner_reacquire_streak = 0
        self._corner_started_at = 0.0
        self._curve_clear_streak = 0
        settle_s = float(self._config.sharp_curve_exit_settle_ms) / 1000.0
        self._corner_exit_until = float(now) + settle_s
        recovery_s = float(self._config.sharp_curve_recovery_ms) / 1000.0
        self._corner_recovery_until = self._corner_exit_until + recovery_s
        self._corner_exit_direction = (
            math.copysign(1.0, float(correction))
            if abs(float(correction)) > 0.5
            else 0.0
        )
        return settle_s > 0.0

    def _hold_last_line_command_locked(self, now: float) -> bool:
        """Bridge a brief detector gap without stopping and restarting sideways."""
        if self._control_mode != "FOLLOW_LINE" or self._last_valid_line_at <= 0.0:
            return False
        hold_s = float(self._config.line_hold_ms) / 1000.0
        if hold_s <= 0.0:
            return False
        elapsed = max(0.0, float(now) - self._last_valid_line_at)
        if elapsed > hold_s:
            return False

        # Keep the same steering sign, decay it, and slow both sides while
        # uncertain. The line mixer never allows a reverse command here.
        progress = min(1.0, elapsed / hold_s)
        correction = self._last_valid_line_correction * (1.0 - (0.55 * progress))
        left_speed, right_speed = self._line_drive_speeds_locked(
            correction,
            throttle_scale=self._config.line_hold_throttle_scale,
        )
        self._apply_drive_locked(
            left_speed_us=max(0.0, left_speed),
            right_speed_us=max(0.0, right_speed),
            mode="FOLLOW_LINE",
            assist_kind="line_hold",
            line_error=self._last_valid_line_error,
            pid_output=correction,
        )
        return True

    def _start_gap_crossing_locked(self, now: float) -> bool:
        """Continue at the proven straight calibration across a real line gap."""
        if not self._config.gap_crossing_enabled or self._gap_crossing_active:
            return False
        if not self._gap_entry_allowed:
            return False
        if self._control_mode != "FOLLOW_LINE" or self._last_valid_line_at <= 0.0:
            return False
        # Only continue the line that disappeared immediately ahead. This is
        # never a generic search command over an arbitrary white floor.
        if max(0.0, float(now) - self._last_valid_line_at) > 0.35:
            return False
        if (
            self._corner_phase != "IDLE"
            or self._green_half_turn_phase != "IDLE"
            or self._sharp_curve_active
        ):
            return False

        self._gap_crossing_active = True
        self._gap_entry_allowed = False
        self._gap_started_at = float(now)
        self._gap_reacquire_streak = 0
        self._last_event_line = "straight gap crossing started"
        self._last_event_at = float(now)
        self._drive_gap_straight_locked(assist_kind="line_gap_crossing")
        return True

    def _handle_active_gap_crossing_locked(
        self,
        event: VisionDetectionEvent,
        metadata: Mapping[str, Any],
        now: float,
    ) -> bool:
        if not self._gap_crossing_active:
            return False

        elapsed_ms = max(0.0, float(now) - self._gap_started_at) * 1000.0
        if elapsed_ms > float(self._config.gap_crossing_timeout_ms):
            self._last_event_line = "straight gap crossing timeout"
            self._last_event_at = float(now)
            self._neutralize_locked(
                mode="STOPPED",
                assist_kind="line_gap_timeout",
                failsafe=False,
                disable_outputs=True,
            )
            return True

        confidence = float(metadata.get("line_confidence", 0.0) or 0.0)
        line_reacquired = bool(
            event.state == RobotState.FOLLOWING_LINE.value
            and bool(event.line)
            and confidence >= float(self._config.line_confidence_floor)
            and elapsed_ms >= float(self._config.gap_reacquire_min_ms)
        )
        if line_reacquired:
            self._gap_reacquire_streak += 1
            if self._gap_reacquire_streak >= int(
                self._config.gap_reacquire_confirm_frames
            ):
                self._gap_crossing_active = False
                self._gap_started_at = 0.0
                self._gap_straight_streak = 0
                self._gap_straight_confirmed_at = 0.0
                self._gap_entry_allowed = False
                self._gap_reacquire_streak = 0
                self._last_event_line = "straight gap line reacquired"
                self._last_event_at = float(now)
                return False
            assist_kind = "line_gap_reacquire"
        else:
            self._gap_reacquire_streak = 0
            assist_kind = "line_gap_crossing"

        self._drive_gap_straight_locked(assist_kind=assist_kind)
        return True

    def _drive_gap_straight_locked(self, *, assist_kind: str) -> None:
        # Exact straight-line calibration already proven on the physical robot.
        # No stale correction or curve signal is allowed across the white gap.
        self._reset_line_pid_locked()
        self._apply_drive_locked(
            left_speed_us=float(self._config.left_base_throttle_us),
            right_speed_us=float(self._config.right_base_throttle_us),
            mode="FOLLOW_LINE",
            assist_kind=str(assist_kind),
            line_error=0.0,
            pid_output=0.0,
        )

    def _reset_gap_crossing_locked(self) -> None:
        self._gap_crossing_active = False
        self._gap_started_at = 0.0
        self._gap_straight_streak = 0
        self._gap_straight_confirmed_at = 0.0
        self._gap_entry_allowed = False
        self._gap_reacquire_streak = 0

    def _reset_line_pid_locked(self) -> None:
        self._pid_integral = 0.0
        self._pid_previous_error = None
        self._pid_derivative = 0.0
        self._pid_last_at = 0.0

    def _compute_line_pid_locked(self, error: float, now: float) -> float:
        """Compute a bounded, filtered PID correction for normalized line error."""
        error = max(-1.0, min(1.0, float(error)))
        deadband = float(self._config.line_error_deadband)
        if deadband > 0.0:
            magnitude = abs(error)
            if magnitude <= deadband:
                error = 0.0
            else:
                # Preserve the full-scale response outside the quiet center,
                # while making small detector jitter command straight ahead.
                error = math.copysign((magnitude - deadband) / (1.0 - deadband), error)
        previous = self._pid_previous_error
        dt = 0.0 if self._pid_last_at <= 0.0 else float(now - self._pid_last_at)
        if dt <= 0.0 or dt > 0.30 or previous is None:
            derivative = 0.0
            integral = 0.0
        else:
            derivative_raw = (error - previous) / max(0.01, dt)
            filter_weight = float(self._config.pid_derivative_filter)
            derivative = (filter_weight * self._pid_derivative) + ((1.0 - filter_weight) * derivative_raw)
            integral = max(
                -float(self._config.pid_integral_limit),
                min(float(self._config.pid_integral_limit), self._pid_integral + (error * dt)),
            )

        output = (
            (float(self._config.pid_kp_us) * error)
            + (float(self._config.pid_ki_us) * integral)
            + (float(self._config.pid_kd_us) * derivative)
        )
        max_output = abs(float(self._config.max_output_us))
        bounded = max(-max_output, min(max_output, output))

        # Anti-windup: do not keep integrating while the command is saturated
        # in the same direction as the current error.
        saturated = abs(output) > max_output
        integral_pushes_outward = saturated and (
            (output > 0.0 and error > 0.0) or (output < 0.0 and error < 0.0)
        )
        if not saturated or not integral_pushes_outward:
            self._pid_integral = integral
        self._pid_previous_error = error
        self._pid_derivative = derivative
        self._pid_last_at = float(now)
        return float(bounded)

    def _neutralize_locked(
        self,
        *,
        mode: str,
        assist_kind: str,
        failsafe: bool,
        disable_outputs: bool = False,
    ) -> None:
        self._set_pair_locked(int(self._config.neutral_us), int(self._config.neutral_us))
        if disable_outputs:
            # Keep an active calibrated neutral pulse. Clearing the PCA outputs
            # here removes the neutral signal from continuous-rotation servos.
            pass
        self._control_mode = mode
        self._assist_kind = assist_kind
        self._line_error = 0.0
        self._pid_output = 0.0
        self._steering_decision = "STOP"
        self._requested_left_speed_us = 0.0
        self._requested_right_speed_us = 0.0
        self._reset_line_pid_locked()
        self._last_valid_line_at = 0.0
        self._last_valid_line_error = 0.0
        self._last_valid_line_correction = 0.0
        self._reset_gap_crossing_locked()
        self._sharp_curve_active = False
        self._last_sharp_curve_at = 0.0
        self._last_sharp_curve_error = 0.0
        self._last_sharp_curve_correction = 0.0
        self._last_sharp_curve_reverse_inner = False
        self._last_sharp_curve_throttle_scale = 1.0
        self._corner_reacquire_streak = 0
        self._corner_started_at = 0.0
        self._corner_exit_until = 0.0
        self._corner_recovery_until = 0.0
        self._corner_exit_direction = 0.0
        self._reset_calibrated_corner_sequence_locked()
        self._curve_clear_streak = 0
        self._curve_signal = 0.0
        self._failsafe_active = bool(failsafe)
        self._last_command_at = self._monotonic()

    def _disable_outputs_locked(self) -> None:
        if self._driver is None or self._config.dry_run:
            return
        disable_all = getattr(self._driver, "disable_all", None)
        if callable(disable_all):
            try:
                disable_all()
            except Exception as exc:
                self._publish_log("ERROR", f"PCA hard PWM stop failed: {exc}")

    def _set_pair_locked(self, left_us: int, right_us: int) -> None:
        left_us = self._clamp_pulse(left_us)
        right_us = self._clamp_pulse(right_us)
        self._last_left_us = left_us
        self._last_right_us = right_us
        if self._config.dry_run:
            return
        if self._driver is None:
            raise RuntimeError("PCA9685 driver unavailable")
        for channel in self._config.left_channels:
            self._driver.set_pwm_us(int(channel), left_us)
        for channel in self._config.right_channels:
            self._driver.set_pwm_us(int(channel), right_us)

    def _pulse_for_speed(
        self,
        speed_us: float,
        *,
        inverted: bool,
        allow_base_plus_correction: bool = False,
    ) -> int:
        # max_output_us must not clip the separately calibrated straight-drive
        # base (left=300, right=200 on this robot). Basic line mode additionally
        # allows its current-error correction to add to that base.
        calibrated_base = max(
            abs(int(self._config.left_base_throttle_us)),
            abs(int(self._config.right_base_throttle_us)),
        )
        max_output = min(
            (
                calibrated_base + abs(int(self._config.max_output_us))
                if allow_base_plus_correction
                else max(abs(int(self._config.max_output_us)), calibrated_base)
            ),
            abs(int(self._config.max_us) - int(self._config.neutral_us)),
            abs(int(self._config.neutral_us) - int(self._config.min_us)),
        )
        signed = max(-max_output, min(max_output, float(speed_us)))
        if inverted:
            signed = -signed
        return self._clamp_pulse(round(float(self._config.neutral_us) + signed))

    def _clamp_pulse(self, value: int) -> int:
        return int(max(int(self._config.min_us), min(int(self._config.max_us), int(value))))

    def _publish_log(self, level: str, message: str, *, state: str = "") -> None:
        try:
            self._event_bus.publish(
                EventTopic.SYSTEM_LOG,
                LogEvent(
                    timestamp=time.time(),
                    level=str(level).upper(),
                    message=str(message),
                    source="pca9685_robot_adapter",
                    state=str(state),
                ),
            )
        except Exception:
            pass
