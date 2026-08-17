from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.core.event_bus import EventBus, EventTopic, UICommandEvent, VisionDetectionEvent
from src.modules.control.pca9685_robot_adapter import (
    Pca9685RobotAdapter,
    Pca9685RobotConfig,
    _AdafruitPca9685Driver,
)


@dataclass
class FakePwmDriver:
    pulses: list[tuple[int, int]] = field(default_factory=list)
    closed: bool = False
    disabled: int = 0

    def set_pwm_us(self, channel: int, pulse_us: int) -> None:
        self.pulses.append((int(channel), int(pulse_us)))

    def close(self) -> None:
        self.closed = True

    def disable_all(self) -> None:
        self.disabled += 1


@dataclass
class FailOncePwmDriver(FakePwmDriver):
    fail_on_pulse: int = 0
    failed: bool = False

    def set_pwm_us(self, channel: int, pulse_us: int) -> None:
        if int(pulse_us) == int(self.fail_on_pulse) and not self.failed:
            self.failed = True
            raise OSError(121, "Remote I/O error")
        super().set_pwm_us(channel, pulse_us)


class FlakyDutyChannel:
    def __init__(self, failures: int) -> None:
        self.failures = int(failures)
        self.attempts = 0
        self.value: int | None = None

    @property
    def duty_cycle(self) -> int | None:
        return self.value

    @duty_cycle.setter
    def duty_cycle(self, value: int) -> None:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise OSError(121, "Remote I/O error")
        self.value = int(value)


def test_adafruit_driver_retries_transient_i2c_write(monkeypatch) -> None:
    channel = FlakyDutyChannel(failures=2)
    driver = _AdafruitPca9685Driver.__new__(_AdafruitPca9685Driver)
    driver._frequency_hz = 50
    driver._pca = type("FakePca", (), {"channels": [channel]})()
    sleeps: list[float] = []
    monkeypatch.setattr(
        "src.modules.control.pca9685_robot_adapter.time.sleep",
        lambda seconds: sleeps.append(float(seconds)),
    )

    driver.set_pwm_us(0, 1600)

    assert channel.attempts == 3
    assert channel.value is not None
    assert sleeps == [0.005, 0.005]

class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


def _drain(bus: EventBus) -> None:
    bus._queue.join()


def _latest_pair(driver: FakePwmDriver) -> tuple[int, int]:
    latest: dict[int, int] = {}
    for channel, pulse in driver.pulses:
        latest[channel] = pulse
    return latest[0], latest[1]


def test_pca9685_adapter_neutralizes_outputs_on_start_and_stop() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(left_channel=0, right_channel=1, neutral_us=1500),
        pwm_factory=lambda config: driver,
    )

    try:
        adapter.start()
        assert _latest_pair(driver) == (1500, 1500)

        adapter.stop()
        assert _latest_pair(driver) == (1500, 1500)
        assert driver.disabled >= 1
        assert driver.closed is True
    finally:
        bus.stop()


def test_pca9685_adapter_follows_line_with_differential_pwm() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channel=0,
            right_channel=1,
            neutral_us=1500,
            base_throttle_us=120,
            turn_gain_us=200,
            left_inverted=True,
            right_inverted=False,
            line_confidence_floor=0.2,
        ),
        pwm_factory=lambda config: driver,
    )

    try:
        adapter.start()
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={"line_offset_norm": 0.25, "line_confidence": 0.8},
            ),
        )
        _drain(bus)

        left, right = _latest_pair(driver)
        assert left == 1330
        assert right == 1570
        assert adapter.status_payload()["control_mode"] == "FOLLOW_LINE"
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_adapter_pid_derivative_reacts_to_a_fast_line_shift() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channel=0,
            right_channel=1,
            neutral_us=1500,
            base_throttle_us=120,
            pid_kp_us=200,
            pid_ki_us=10,
            pid_kd_us=30,
            pid_integral_limit=0.25,
            pid_derivative_filter=0.0,
            left_inverted=True,
            right_inverted=False,
            line_confidence_floor=0.2,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    try:
        adapter.start()
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={"line_offset_norm": 0.05, "line_confidence": 0.8},
            ),
        )
        _drain(bus)
        clock.advance(0.05)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={"line_offset_norm": 0.35, "line_confidence": 0.8},
            ),
        )
        _drain(bus)

        assert adapter.status_payload()["pid_output"] > 80.0
        left, right = _latest_pair(driver)
        assert left < 1300
        assert right == 1500
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_adapter_updates_line_control_and_clears_pid_state() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            pid_kp_us=500,
            pid_ki_us=3,
            pid_kd_us=18,
            line_hold_ms=180,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    try:
        adapter.start()
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={"line_offset_norm": 0.25, "line_confidence": 0.9},
            ),
        )
        _drain(bus)
        assert adapter.status_payload()["pid_output"] != 0.0

        applied = adapter.update_line_control(
            pid_kp_us=320,
            pid_ki_us=0,
            pid_kd_us=12,
            pid_integral_limit=0.15,
            pid_derivative_filter=0.60,
            max_output_us=240,
            line_hold_ms=120,
        )
        assert applied == {
            "pid_kp_us": 320.0,
            "pid_ki_us": 0.0,
            "pid_kd_us": 12.0,
            "pid_integral_limit": 0.15,
            "pid_derivative_filter": 0.6,
            "max_output_us": 240,
            "line_hold_ms": 120,
            "left_base_throttle_us": 300,
            "right_base_throttle_us": 200,
            "line_error_deadband": 0.0,
        }
        assert adapter.status_payload()["pid_kp_us"] == 320.0
        assert adapter.status_payload()["pid_derivative_filter"] == 0.6
        assert adapter.status_payload()["max_output_us"] == 240
        assert adapter.status_payload()["line_hold_ms"] == 120
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_adapter_deadband_keeps_small_center_error_straight() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            pid_kp_us=500,
            pid_kd_us=0,
            line_error_deadband=0.05,
        ),
        pwm_factory=lambda config: driver,
    )

    try:
        adapter.start()
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={"line_offset_norm": 0.03, "line_confidence": 0.9},
            ),
        )
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert adapter.status_payload()["pid_output"] == 0.0
        assert latest[4] == latest[5] == 1900
        assert latest[0] == latest[1] == 1400
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_adapter_basic_line_follow_corrects_from_current_offset_only() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            turn_gain_us=300,
            max_output_us=180,
            line_error_deadband=0.04,
            basic_line_follow=True,
        ),
        pwm_factory=lambda config: driver,
    )

    try:
        adapter.start()
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={"line_offset_norm": 0.10, "line_confidence": 0.9},
            ),
        )
        _drain(bus)

        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert adapter.status_payload()["assist_kind"] == "line_basic"
        assert adapter.status_payload()["pid_output"] > 0.0
        assert adapter.status_payload()["steering_decision"] == "RIGHT"
        assert latest[4] > 1900  # line right: speed up the left side
        assert latest[0] > 1400  # and slow the right side toward neutral

        # A sign change must affect this same control frame. At the field
        # camera's 10 FPS, retaining the previous sign for one frame is a
        # visible 100 ms wrong-way correction.
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={"line_offset_norm": -0.10, "line_confidence": 0.9},
            ),
        )
        _drain(bus)
        assert adapter.status_payload()["pid_output"] < 0.0
        assert adapter.status_payload()["steering_decision"] == "LEFT"
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_basic_line_follow_has_consistent_left_center_right_contract() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            turn_gain_us=220,
            max_output_us=360,
            line_error_deadband=0.04,
            basic_line_follow=True,
        ),
        pwm_factory=lambda config: driver,
    )

    try:
        adapter.start()
        expected = (
            (-0.50, "LEFT"),
            (0.00, "STRAIGHT"),
            (0.50, "RIGHT"),
        )
        observed: dict[str, tuple[int, int]] = {}
        for offset, decision in expected:
            bus.publish(
                EventTopic.VISION_DETECTIONS,
                VisionDetectionEvent(
                    state="FOLLOWING_LINE",
                    line=True,
                    metadata={
                        "line_offset_norm": offset,
                        "line_offset_source": "ground_path",
                        "line_confidence": 0.9,
                        "line_geometry": {"candidate_count": 1.0},
                    },
                ),
            )
            _drain(bus)
            status = adapter.status_payload()
            assert status["steering_decision"] == decision
            assert status["line_offset_source"] == "ground_path"
            assert status["line_candidate_count"] == 1
            latest = {channel: pulse for channel, pulse in driver.pulses}
            observed[decision] = (latest[4], latest[0])

        assert observed["STRAIGHT"] == (1900, 1400)
        assert observed["LEFT"][0] < 1900  # slow left
        assert observed["LEFT"][1] < 1400  # speed up right
        assert observed["RIGHT"][0] > 1900  # speed up left
        assert observed["RIGHT"][1] > 1400  # slow right
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_basic_line_follow_can_invert_physical_steering_for_camera_mount() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            turn_gain_us=220,
            max_output_us=360,
            line_error_deadband=0.04,
            basic_line_follow=True,
            line_steering_inverted=True,
        ),
        pwm_factory=lambda config: driver,
    )

    try:
        adapter.start()
        for _ in range(3):
            bus.publish(
                EventTopic.VISION_DETECTIONS,
                VisionDetectionEvent(
                    state="FOLLOWING_LINE",
                    line=True,
                    metadata={"line_offset_norm": 0.50, "line_confidence": 0.9},
                ),
            )
            _drain(bus)

        status = adapter.status_payload()
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert status["line_steering_inverted"] is True
        assert status["steering_decision"] == "LEFT"
        assert status["pid_output"] > 0.0  # vision error keeps its original sign
        assert latest[4] < 1900  # physical left command: slow left
        assert latest[0] < 1400  # and speed up right
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_can_disable_closed_corner_without_disabling_sharp_steering() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            basic_line_follow=True,
            line_steering_inverted=True,
            line_error_deadband=0.04,
            sharp_corner_maneuver_enabled=False,
            sharp_curve_threshold=0.55,
        ),
        pwm_factory=lambda config: driver,
    )

    def publish(offset: float, bend: float) -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    "line_offset_norm": offset,
                    "line_offset_source": "ground_path",
                    "line_confidence": 0.9,
                    "line_geometry": {
                        "candidate_count": 1.0,
                        "right_angle_corridor": 1.0,
                        "turn_corridor": 1.0,
                        "path_bend_delta_norm": bend,
                        "dominant_row_y_ratio": 1.0,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        publish(0.70, 0.30)
        status = adapter.status_payload()
        assert status["sharp_corner_maneuver_enabled"] is False
        assert status["assist_kind"] == "line_sharp"
        assert status["sharp_curve_active"] is True
        assert status["sharp_curve_reverse_inner"] is False

        # Even if the detector keeps calling the contour a right angle, an
        # ordinary curve remains responsive to the current visual side.
        publish(-0.70, -0.30)
        status = adapter.status_payload()
        assert status["assist_kind"] == "line_sharp"
        assert status["sharp_curve_reverse_inner"] is False
        assert status["pid_output"] < 0.0
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_open_curve_lookahead_anticipates_reversal_and_slows_only_curve() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            turn_gain_us=260,
            max_output_us=360,
            line_error_deadband=0.04,
            basic_line_follow=True,
            line_steering_inverted=True,
            sharp_corner_maneuver_enabled=False,
            curve_lookahead_gain=0.55,
            curve_throttle_scale=0.72,
            ordinary_sharp_hold_ms=180,
        ),
        pwm_factory=lambda config: driver,
    )

    def publish(offset: float, bend: float, *, turn: bool) -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    "line_offset_norm": offset,
                    "line_offset_source": "ground_path",
                    "line_confidence": 1.0,
                    "line_geometry": {
                        "candidate_count": 1.0,
                        "turn_corridor": float(turn),
                        "path_bend_delta_norm": bend,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        publish(0.0, 0.0, turn=False)
        status = adapter.status_payload()
        assert status["assist_kind"].startswith("line_")
        assert status["requested_left_speed_us"] == 300.0
        assert status["requested_right_speed_us"] == 200.0

        # The near-field line is still slightly on the old side, but the wide
        # camera already sees the S-curve bend reverse. Lookahead must reverse
        # steering now instead of waiting another 300-600 ms for the centroid.
        publish(0.16, -0.40, turn=True)
        status = adapter.status_payload()
        assert status["assist_kind"] == "line_curve_lookahead"
        assert status["line_error"] < 0.0
        assert status["steering_decision"] == "RIGHT"
        assert status["requested_left_speed_us"] < 300.0
        assert status["requested_right_speed_us"] < 200.0

        # At the severe edge captured in the field session, preserve the turn
        # direction but reduce forward travel from 300 to 216 us-equivalent.
        publish(-0.47, -0.80, turn=True)
        status = adapter.status_payload()
        assert status["assist_kind"] == "line_sharp_curve"
        assert status["steering_decision"] == "RIGHT_SHARP"
        assert status["requested_left_speed_us"] == 216.0
        assert status["requested_right_speed_us"] == 0.0
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_open_curve_uses_short_hold_then_stops() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            basic_line_follow=True,
            line_steering_inverted=True,
            curve_lookahead_gain=0.55,
            curve_throttle_scale=0.72,
            ordinary_sharp_hold_ms=180,
            sharp_curve_hold_ms=800,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    try:
        adapter.start()
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    "line_offset_norm": -0.80,
                    "line_confidence": 1.0,
                    "line_geometry": {
                        "turn_corridor": 1.0,
                        "path_bend_delta_norm": -0.50,
                    },
                },
            ),
        )
        _drain(bus)
        assert adapter.status_payload()["assist_kind"] == "line_sharp_curve"

        clock.advance(0.15)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(state="FOLLOWING_LINE", line=False, metadata={}),
        )
        _drain(bus)
        status = adapter.status_payload()
        assert status["assist_kind"] == "line_sharp_hold"
        assert status["requested_left_speed_us"] == 216.0

        clock.advance(0.04)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(state="FOLLOWING_LINE", line=False, metadata={}),
        )
        _drain(bus)
        status = adapter.status_payload()
        assert status["assist_kind"] == "line_lost"
        assert status["control_mode"] == "STOPPED"
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_basic_line_follow_pivots_and_holds_a_right_angle_corner() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            turn_gain_us=220,
            max_output_us=360,
            line_error_deadband=0.04,
            basic_line_follow=True,
            line_steering_inverted=True,
            sharp_curve_threshold=0.55,
            sharp_curve_hold_ms=650,
            sharp_curve_inner_reverse_scale=0.55,
            sharp_curve_visible_commit_ms=1450,
            sharp_curve_finish_inner_reverse_scale=0.22,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    try:
        adapter.start()
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    # This reproduces the physical recording and installed
                    # camera mapping: a positive bend commits LEFT_SHARP.
                    "line_offset_norm": 0.10,
                    "line_confidence": 0.9,
                    "line_geometry": {
                        "candidate_count": 1.0,
                        "right_angle_corridor": 1.0,
                        "path_bend_delta_norm": 0.30,
                    },
                },
            ),
        )
        _drain(bus)

        latest = {channel: pulse for channel, pulse in driver.pulses}
        status = adapter.status_payload()
        assert status["assist_kind"] == "line_sharp_corner"
        assert status["steering_decision"] == "LEFT_SHARP"
        assert status["sharp_curve_active"] is True
        assert status["sharp_curve_reverse_inner"] is True
        assert status["curve_signal"] == 0.3
        assert latest[4] == latest[5] == 1435  # left/inside side reversing
        assert latest[0] == latest[1] == 1400  # right/outside side forward

        # Preserve the committed direction when the L geometry flickers while
        # rotating, then bridge a short visible non-corner frame.
        clock.advance(0.04)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    "line_offset_norm": -0.45,
                    "line_confidence": 0.9,
                    "line_geometry": {
                        "right_angle_corridor": 1.0,
                        # The sign flips as the same L rotates under the camera.
                        "path_bend_delta_norm": -0.40,
                    },
                },
            ),
        )
        _drain(bus)
        assert adapter.status_payload()["steering_decision"] == "LEFT_SHARP"

        clock.advance(0.04)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    "line_offset_norm": -0.90,
                    "line_confidence": 0.9,
                    "line_geometry": {
                        "turn_corridor": 1.0,
                        "path_bend_delta_norm": -0.30,
                    },
                },
            ),
        )
        _drain(bus)
        assert adapter.status_payload()["assist_kind"] == "line_sharp_corner_bridge"
        assert adapter.status_payload()["steering_decision"] == "LEFT_SHARP"

        # The line can briefly leave the camera while the chassis rotates.
        clock.advance(0.30)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(state="FOLLOWING_LINE", line=False, metadata={}),
        )
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert adapter.status_payload()["assist_kind"] == "line_sharp_hold"
        assert latest[4] == latest[5] == 1435
        assert latest[0] == latest[1] == 1400

        # After the strong phase, keep the same direction at finish strength.
        clock.advance(0.35)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(state="FOLLOWING_LINE", line=False, metadata={}),
        )
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert adapter.status_payload()["assist_kind"] == "line_sharp_finish_hold"
        assert latest[4] == latest[5] == 1534
        assert latest[0] == latest[1] == 1400

        # The absolute corner bound still stops safely if nothing is reacquired.
        clock.advance(0.73)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(state="FOLLOWING_LINE", line=False, metadata={}),
        )
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert adapter.status_payload()["assist_kind"] == "line_lost"
        assert latest[4] == latest[5] == 1600
        assert latest[0] == latest[1] == 1600

        # A large ordinary tracking error may stop the inside side, but only a
        # detector-confirmed 90-degree corner is allowed to reverse it.
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={"line_offset_norm": 0.90, "line_confidence": 0.9},
            ),
        )
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        status = adapter.status_payload()
        assert status["steering_decision"] == "LEFT_SHARP"
        assert status["sharp_curve_reverse_inner"] is False
        assert latest[4] == latest[5] == 1600
        assert latest[0] == latest[1] == 1400
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_right_angle_commit_requires_centered_ground_path_before_exit() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            basic_line_follow=True,
            line_steering_inverted=True,
            line_error_deadband=0.04,
            sharp_curve_hold_ms=650,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    try:
        adapter.start()
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    "line_offset_norm": 0.05,
                    "line_confidence": 0.9,
                    "line_geometry": {
                        "right_angle_corridor": 1.0,
                        "turn_corridor": 1.0,
                        "path_bend_delta_norm": 0.30,
                    },
                },
            ),
        )
        _drain(bus)
        assert adapter.status_payload()["steering_decision"] == "LEFT_SHARP"

        # Flat frames immediately after detection are the incoming leg
        # flickering under the rotating camera, not a completed 90-degree turn.
        for _ in range(3):
            clock.advance(0.04)
            bus.publish(
                EventTopic.VISION_DETECTIONS,
                VisionDetectionEvent(
                    state="FOLLOWING_LINE",
                    line=True,
                    metadata={
                        # A flat contour at the edge is not aligned yet.
                        "line_offset_norm": 0.70,
                        "line_confidence": 0.9,
                        "line_geometry": {
                            "turn_corridor": 0.0,
                            "path_bend_delta_norm": 0.01,
                        },
                    },
                ),
            )
            _drain(bus)
            assert adapter.status_payload()["sharp_curve_active"] is True

        # Even after the minimum interval, a flat line at the image edge is
        # not an aligned outgoing leg and must not release the pivot.
        clock.advance(0.54)
        for _ in range(3):
            clock.advance(0.04)
            bus.publish(
                EventTopic.VISION_DETECTIONS,
                VisionDetectionEvent(
                    state="FOLLOWING_LINE",
                    line=True,
                    metadata={
                        "line_offset_norm": 0.70,
                        "line_offset_source": "ground_path",
                        "line_confidence": 0.9,
                        "line_geometry": {
                            "candidate_count": 1.0,
                            "turn_corridor": 0.0,
                            "path_bend_delta_norm": 0.01,
                        },
                    },
                ),
            )
            _drain(bus)
            assert adapter.status_payload()["sharp_curve_active"] is True

        # Three stable, centered, single-candidate ground-path frames confirm
        # that the outgoing straight is under the chassis.
        # Three centered frames release the turn; there is no extra filter lag.
        for expected_still_committed in (True, True, False):
            clock.advance(0.04)
            bus.publish(
                EventTopic.VISION_DETECTIONS,
                VisionDetectionEvent(
                    state="FOLLOWING_LINE",
                    line=True,
                    metadata={
                        "line_offset_norm": 0.25,
                        "line_offset_source": "ground_path",
                        "line_confidence": 0.9,
                        "line_geometry": {
                            "candidate_count": 1.0,
                            "turn_corridor": 0.0,
                            "path_bend_delta_norm": 0.01,
                        },
                    },
                ),
            )
            _drain(bus)
            assert adapter.status_payload()["sharp_curve_active"] is expected_still_committed

        status = adapter.status_payload()
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert status["assist_kind"] == "line_corner_exit"
        assert status["steering_decision"] == "STRAIGHT"
        assert status["sharp_curve_reverse_inner"] is False
        assert status["corner_exit_active"] is True
        assert latest[4] == latest[5] == 1900
        assert latest[0] == latest[1] == 1400
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_right_angle_blind_hold_keeps_finish_strength_and_total_bound() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            basic_line_follow=True,
            line_steering_inverted=True,
            line_error_deadband=0.04,
            sharp_curve_hold_ms=900,
            sharp_curve_visible_commit_ms=2100,
            sharp_curve_inner_reverse_scale=0.55,
            sharp_curve_finish_inner_reverse_scale=0.35,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish_corner() -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    "line_offset_norm": 0.10,
                    "line_confidence": 0.9,
                    "line_geometry": {
                        "right_angle_corridor": 1.0,
                        "turn_corridor": 1.0,
                        "path_bend_delta_norm": 0.30,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        publish_corner()

        # The physical recording loses the line early. The confirmed corner
        # must keep its full 900 ms minimum before switching to finish strength.
        clock.advance(0.45)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(state="FOLLOWING_LINE", line=False, metadata={}),
        )
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        status = adapter.status_payload()
        assert status["assist_kind"] == "line_sharp_hold"
        assert status["steering_decision"] == "LEFT_SHARP"
        assert latest[4] == latest[5] == 1435
        assert latest[0] == latest[1] == 1400

        # Even though the last visible corner is now older than the former
        # 650 ms hold, the bounded blind commitment must continue at 35%.
        clock.advance(0.55)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(state="FOLLOWING_LINE", line=False, metadata={}),
        )
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        status = adapter.status_payload()
        assert status["assist_kind"] == "line_sharp_finish_hold"
        assert status["steering_decision"] == "LEFT_SHARP"
        assert latest[4] == latest[5] == 1495
        assert latest[0] == latest[1] == 1400

        # At 2100 ms total, no visible line means neutral—not another pivot and
        # not an unverified forward command.
        clock.advance(1.11)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(state="FOLLOWING_LINE", line=False, metadata={}),
        )
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert adapter.status_payload()["assist_kind"] == "line_lost"
        assert latest[4] == latest[5] == 1600
        assert latest[0] == latest[1] == 1600
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_default_right_angle_is_a_closed_symmetric_pivot() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            basic_line_follow=True,
            line_steering_inverted=True,
            line_error_deadband=0.04,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish_corner(bend: float) -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    "line_offset_norm": 0.05,
                    "line_offset_source": "ground_path",
                    "line_confidence": 1.0,
                    "line_geometry": {
                        "candidate_count": 1.0,
                        "right_angle_corridor": 1.0,
                        "turn_corridor": 1.0,
                        "path_bend_delta_norm": bend,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        publish_corner(0.30)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert adapter.status_payload()["steering_decision"] == "LEFT_SHARP"
        assert latest[4] == latest[5] == 1300  # left reverses at calibrated base
        assert latest[0] == latest[1] == 1400  # right advances at calibrated base

        # Crossing the old 900 ms phase boundary must not weaken the inside
        # side and turn an in-place pivot into a forward arc.
        clock.advance(1.00)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(state="FOLLOWING_LINE", line=False, metadata={}),
        )
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert adapter.status_payload()["assist_kind"] == "line_sharp_finish_hold"
        assert latest[4] == latest[5] == 1300
        assert latest[0] == latest[1] == 1400

        # The enlarged bound remains finite and neutralizes a blind turn.
        clock.advance(1.61)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(state="FOLLOWING_LINE", line=False, metadata={}),
        )
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert adapter.status_payload()["assist_kind"] == "line_lost"
        assert latest[4] == latest[5] == 1600
        assert latest[0] == latest[1] == 1600

        # The mirrored corner uses the same two calibrated base magnitudes.
        clock.advance(0.10)
        publish_corner(-0.30)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert adapter.status_payload()["steering_decision"] == "RIGHT_SHARP"
        assert latest[4] == latest[5] == 1900  # left advances at calibrated base
        assert latest[0] == latest[1] == 1800  # right reverses at calibrated base
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_right_angle_finishes_gently_then_suppresses_exit_bounce() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            basic_line_follow=True,
            line_steering_inverted=True,
            line_error_deadband=0.04,
            sharp_curve_hold_ms=650,
            sharp_curve_visible_commit_ms=1450,
            sharp_curve_inner_reverse_scale=0.55,
            sharp_curve_finish_inner_reverse_scale=0.22,
            sharp_curve_exit_settle_ms=320,
            sharp_curve_exit_throttle_scale=1.0,
            sharp_curve_exit_max_correction_us=0.0,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish_line(offset: float, *, right_angle: bool, turn_corridor: bool, bend: float) -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    "line_offset_norm": offset,
                    "line_offset_source": "ground_path",
                    "line_confidence": 0.9,
                    "line_geometry": {
                        "candidate_count": 1.0,
                        "right_angle_corridor": float(right_angle),
                        "turn_corridor": float(turn_corridor),
                        "path_bend_delta_norm": bend,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        publish_line(0.05, right_angle=True, turn_corridor=True, bend=0.30)

        # Once the blind-spot hold has elapsed, a still-visible corner keeps
        # the committed direction but reduces inside reverse from 55% to 22%.
        clock.advance(0.70)
        publish_line(0.75, right_angle=False, turn_corridor=True, bend=0.30)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        status = adapter.status_payload()
        assert status["assist_kind"] == "line_sharp_corner_finish"
        assert status["steering_decision"] == "LEFT_SHARP"
        assert status["sharp_curve_reverse_inner"] is True
        assert latest[4] == latest[5] == 1534
        assert latest[0] == latest[1] == 1400

        # Losing the line after the strong interval keeps the bounded finish
        # pivot instead of stopping in the middle of the physical turn.
        clock.advance(0.01)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(state="FOLLOWING_LINE", line=False, metadata={}),
        )
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert adapter.status_payload()["assist_kind"] == "line_sharp_finish_hold"
        assert latest[4] == latest[5] == 1534
        assert latest[0] == latest[1] == 1400

        # The total commitment bound still stops a blind pivot safely.
        clock.advance(0.75)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(state="FOLLOWING_LINE", line=False, metadata={}),
        )
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert adapter.status_payload()["assist_kind"] == "line_lost"
        assert latest[4] == latest[5] == 1600
        assert latest[0] == latest[1] == 1600

        # Start another corner and confirm a flat outgoing straight only once
        # it is inside the central corridor. During the exit window every
        # steering correction is suppressed and the physically calibrated
        # 300/200 straight command is restored exactly.
        clock.advance(0.10)
        publish_line(0.05, right_angle=True, turn_corridor=True, bend=0.30)
        clock.advance(0.66)
        for _ in range(3):
            clock.advance(0.04)
            publish_line(0.25, right_angle=False, turn_corridor=False, bend=0.01)
        for _ in range(2):
            clock.advance(0.04)
            publish_line(-0.80, right_angle=False, turn_corridor=False, bend=-0.02)

        latest = {channel: pulse for channel, pulse in driver.pulses}
        status = adapter.status_payload()
        assert status["assist_kind"] == "line_corner_exit"
        assert status["corner_exit_active"] is True
        assert status["steering_decision"] == "STRAIGHT"
        assert status["pid_output"] == 0.0
        assert latest[4] == latest[5] == 1900
        assert latest[0] == latest[1] == 1400

        # After exact-straight stabilization, large residual offset is limited
        # to a gentle all-forward correction instead of restarting a pivot.
        clock.advance(0.33)
        publish_line(-0.80, right_angle=False, turn_corridor=False, bend=-0.02)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        status = adapter.status_payload()
        assert status["corner_exit_active"] is False
        assert status["corner_recovery_active"] is True
        assert status["assist_kind"] == "line_corner_recovery"
        assert status["pid_output"] == -45.0
        assert status["steering_decision"] == "RIGHT"
        assert latest[4] == latest[5] == 1945
        assert latest[0] == latest[1] == 1445

        # Once recovery expires, the ordinary follower regains full authority.
        clock.advance(0.53)
        publish_line(-0.80, right_angle=False, turn_corridor=False, bend=-0.02)
        status = adapter.status_payload()
        assert status["corner_recovery_active"] is False
        assert status["assist_kind"] == "line_sharp"
        assert status["pid_output"] < 0.0
        assert status["steering_decision"] == "RIGHT_SHARP"
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_wide_camera_delays_corner_pivot_until_elbow_is_near() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            basic_line_follow=True,
            line_steering_inverted=True,
            line_error_deadband=0.04,
            sharp_curve_entry_min_row_ratio=0.94,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish_corner(row_ratio: float) -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    "line_offset_norm": 0.05,
                    "line_confidence": 0.9,
                    "line_angle_deg": 90,
                    "line_geometry": {
                        "candidate_count": 1.0,
                        "right_angle_corridor": 1.0,
                        "turn_corridor": 1.0,
                        "path_bend_delta_norm": 0.30,
                        "dominant_row_y_ratio": row_ratio,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        publish_corner(0.82)
        status = adapter.status_payload()
        assert status["assist_kind"] == "line_corner_approach"
        assert status["sharp_curve_reverse_inner"] is False
        assert status["steering_decision"] != "LEFT_SHARP"

        clock.advance(0.04)
        publish_corner(0.97)
        status = adapter.status_payload()
        assert status["assist_kind"] == "line_sharp_corner"
        assert status["sharp_curve_reverse_inner"] is True
        assert status["steering_decision"] == "LEFT_SHARP"
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_usb_perspective_reacquires_straight_even_if_turn_corridor_lingers() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            basic_line_follow=True,
            line_steering_inverted=True,
            line_error_deadband=0.04,
            sharp_curve_hold_ms=650,
            sharp_curve_visible_commit_ms=2000,
            sharp_curve_entry_min_row_ratio=0.94,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish(offset: float, *, right_angle: bool, row_ratio: float, bend: float) -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    "line_offset_norm": offset,
                    "line_offset_source": "ground_path",
                    "line_confidence": 0.9,
                    "line_angle_deg": 90,
                    "line_geometry": {
                        "candidate_count": 1.0,
                        "right_angle_corridor": float(right_angle),
                        # This remains true on the USB lens even for a normal
                        # perspective-stretched straight line.
                        "turn_corridor": 1.0,
                        "path_bend_delta_norm": bend,
                        "dominant_row_y_ratio": row_ratio,
                        "height_ratio": 1.0,
                        "width_ratio": 0.54,
                        "center_range_ratio": 0.14,
                        "bottom_row_occupancy": 0.40,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        publish(0.10, right_angle=True, row_ratio=0.97, bend=0.30)
        assert adapter.status_payload()["steering_decision"] == "LEFT_SHARP"

        clock.advance(0.66)
        for expected_active in (True, True, False):
            clock.advance(0.04)
            publish(0.20, right_angle=False, row_ratio=0.50, bend=0.18)
            assert adapter.status_payload()["sharp_curve_active"] is expected_active

        status = adapter.status_payload()
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert status["assist_kind"] == "line_corner_exit"
        assert status["steering_decision"] == "STRAIGHT"
        assert latest[4] == latest[5] == 1900
        assert latest[0] == latest[1] == 1400
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_persistent_unaligned_right_angle_stops_at_hard_bound() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            basic_line_follow=True,
            line_steering_inverted=True,
            line_error_deadband=0.04,
            sharp_curve_hold_ms=650,
            sharp_curve_visible_commit_ms=1450,
            sharp_curve_inner_reverse_scale=0.55,
            sharp_curve_finish_inner_reverse_scale=0.22,
            sharp_curve_exit_settle_ms=320,
            sharp_curve_exit_throttle_scale=1.0,
            sharp_curve_exit_max_correction_us=0.0,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish_corner() -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    "line_offset_norm": 0.70,
                    "line_confidence": 0.9,
                    "line_geometry": {
                        "right_angle_corridor": 1.0,
                        "turn_corridor": 1.0,
                        "path_bend_delta_norm": 0.30,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        publish_corner()
        assert adapter.status_payload()["assist_kind"] == "line_sharp_corner"

        # A sticky detector first reduces reverse strength after 650 ms.
        clock.advance(0.70)
        publish_corner()
        latest = {channel: pulse for channel, pulse in driver.pulses}
        status = adapter.status_payload()
        assert status["assist_kind"] == "line_sharp_corner_finish"
        assert latest[4] == latest[5] == 1534
        assert latest[0] == latest[1] == 1400

        # A timeout does not prove alignment. It must neutralize rather than
        # drive forward onto the incoming line behind the robot.
        clock.advance(0.76)
        publish_corner()
        latest = {channel: pulse for channel, pulse in driver.pulses}
        status = adapter.status_payload()
        assert status["assist_kind"] == "line_corner_timeout"
        assert status["steering_decision"] == "STOP"
        assert status["pid_output"] == 0.0
        assert latest[4] == latest[5] == 1600
        assert latest[0] == latest[1] == 1600

        # A fresh visual event may start a new bounded maneuver; the timeout
        # itself never leaves a stale forward command active.
        clock.advance(0.10)
        publish_corner()
        assert adapter.status_payload()["assist_kind"] == "line_sharp_corner"
        assert adapter.status_payload()["steering_decision"] == "LEFT_SHARP"
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_ordinary_curve_remains_responsive_without_corner_exit_state() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            basic_line_follow=True,
            line_steering_inverted=True,
            line_error_deadband=0.04,
            sharp_curve_exit_settle_ms=320,
            sharp_curve_exit_throttle_scale=1.0,
            sharp_curve_exit_max_correction_us=0.0,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish(offset: float, *, turn_corridor: bool = False, bend: float = 0.0) -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    "line_offset_norm": offset,
                    "line_confidence": 0.9,
                    "line_geometry": {
                        "turn_corridor": float(turn_corridor),
                        "path_bend_delta_norm": bend,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        publish(0.90)
        assert adapter.status_payload()["steering_decision"] == "LEFT_SHARP"
        assert adapter.status_payload()["sharp_curve_reverse_inner"] is False

        # An ordinary-curve sign flip is applied on the current frame and must
        # not enter the special 90-degree-corner exit state.
        clock.advance(0.04)
        publish(-0.90)
        status = adapter.status_payload()
        assert status["assist_kind"] == "line_sharp"
        assert status["corner_exit_active"] is False
        assert status["steering_decision"] == "RIGHT_SHARP"
        assert status["pid_output"] < 0.0

        clock.advance(0.33)
        publish(-0.90)
        status = adapter.status_payload()
        assert status["corner_exit_active"] is False
        assert status["assist_kind"] == "line_sharp"
        assert status["steering_decision"] == "RIGHT_SHARP"

        # A real S-curve remains immediately steerable as well.
        clock.advance(0.10)
        publish(0.90, turn_corridor=True, bend=0.25)
        status = adapter.status_payload()
        assert status["assist_kind"] == "line_sharp"
        assert status["steering_decision"] == "LEFT_SHARP"
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_adapter_holds_forward_correction_across_a_brief_line_gap() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            line_hold_ms=180,
            line_hold_throttle_scale=0.72,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    try:
        adapter.start()
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={"line_offset_norm": 0.12, "line_confidence": 0.9},
            ),
        )
        _drain(bus)
        clock.advance(0.05)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(state="FOLLOWING_LINE", line=False, metadata={}),
        )
        _drain(bus)

        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert adapter.status_payload()["assist_kind"] == "line_hold"
        assert latest[4] > 1600
        assert latest[0] < 1600

        clock.advance(0.20)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(state="FOLLOWING_LINE", line=False, metadata={}),
        )
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert latest[4] == latest[5] == 1600
        assert latest[0] == latest[1] == 1600
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_gap_crossing_keeps_calibrated_straight_until_line_reacquired() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            neutral_us=1600,
            left_base_throttle_us=300,
            right_base_throttle_us=200,
            right_inverted=True,
            basic_line_follow=True,
            gap_crossing_enabled=True,
            gap_straight_confirm_frames=3,
            gap_reacquire_confirm_frames=2,
            gap_reacquire_min_ms=2400,
            gap_crossing_timeout_ms=2800,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish(*, line: bool, offset: float = 0.0, bend: float = 0.0) -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=line,
                metadata={
                    "line_offset_norm": offset,
                    "line_confidence": 0.95 if line else 0.0,
                    "line_geometry": {
                        "candidate_count": 1 if line else 0,
                        "path_bend_delta_norm": bend,
                        "right_angle_corridor": 0.0,
                        "turn_corridor": 0.0,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        # No multi-frame straight confirmation: one safe basic-line frame is
        # enough, then the first loss must immediately start the 2400 ms drive.
        publish(line=True)
        clock.advance(0.10)

        publish(line=False)
        status = adapter.status_payload()
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert status["assist_kind"] == "line_gap_crossing"
        assert status["gap_crossing_active"] is True
        assert latest[4] == latest[5] == 1900
        assert latest[0] == latest[1] == 1400

        # Residual views of the entry tip must not end the crossing before the
        # motors have had enough time to move the chassis into the white gap.
        clock.advance(0.10)
        publish(line=True)
        assert adapter.status_payload()["assist_kind"] == "line_gap_crossing"
        assert adapter.status_payload()["gap_crossing_active"] is True

        clock.advance(0.10)
        publish(line=True)
        assert adapter.status_payload()["assist_kind"] == "line_gap_crossing"
        assert adapter.status_payload()["gap_crossing_active"] is True

        clock.advance(2.21)
        publish(line=True)
        assert adapter.status_payload()["assist_kind"] == "line_gap_reacquire"
        assert adapter.status_payload()["gap_crossing_active"] is True

        clock.advance(0.10)
        publish(line=True)
        status = adapter.status_payload()
        assert status["assist_kind"] == "line_basic"
        assert status["gap_crossing_active"] is False
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_gap_crossing_does_not_start_after_a_curve() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            basic_line_follow=True,
            gap_crossing_enabled=True,
            gap_straight_confirm_frames=3,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish(*, line: bool, offset: float = 0.0, bend: float = 0.0) -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=line,
                metadata={
                    "line_offset_norm": offset,
                    "line_confidence": 0.95 if line else 0.0,
                    "line_geometry": {
                        "candidate_count": 1 if line else 0,
                        "path_bend_delta_norm": bend,
                        "right_angle_corridor": 0.0,
                        "turn_corridor": 0.0,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        for _ in range(3):
            publish(line=True)
            clock.advance(0.10)
        publish(line=True, offset=0.35, bend=0.30)
        clock.advance(0.10)
        publish(line=False)

        status = adapter.status_payload()
        assert status["gap_crossing_active"] is False
        assert status["assist_kind"] == "line_lost"
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_gap_crossing_keeps_recent_straight_arm_through_endpoint_drift() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            neutral_us=1600,
            left_base_throttle_us=300,
            right_base_throttle_us=200,
            right_inverted=True,
            basic_line_follow=True,
            gap_crossing_enabled=True,
            gap_straight_confirm_frames=3,
            gap_straight_memory_ms=1200,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish(*, line: bool, offset: float = 0.0, bend: float = 0.0) -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=line,
                metadata={
                    "line_offset_norm": offset,
                    "line_confidence": 0.95 if line else 0.0,
                    "line_geometry": {
                        "candidate_count": 1 if line else 0,
                        "path_bend_delta_norm": bend,
                        "right_angle_corridor": 0.0,
                        "turn_corridor": 0.0,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        for _ in range(3):
            publish(line=True, offset=0.0, bend=0.04)
            clock.advance(0.10)

        # The recorded USB-camera endpoint drifts beyond the centering gate
        # for about 0.8 s even though it remains a straight, low-bend segment.
        for _ in range(7):
            publish(line=True, offset=-0.30, bend=0.04)
            clock.advance(0.10)

        publish(line=False, bend=0.04)
        status = adapter.status_payload()
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert status["gap_crossing_active"] is True
        assert status["assist_kind"] == "line_gap_crossing"
        assert latest[4] == latest[5] == 1900
        assert latest[0] == latest[1] == 1400
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_gap_crossing_stops_at_bounded_timeout() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            basic_line_follow=True,
            gap_crossing_enabled=True,
            gap_straight_confirm_frames=2,
            gap_crossing_timeout_ms=500,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish(line: bool) -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=line,
                metadata={
                    "line_offset_norm": 0.0,
                    "line_confidence": 0.95 if line else 0.0,
                    "line_geometry": {
                        "candidate_count": 1 if line else 0,
                        "path_bend_delta_norm": 0.0,
                        "right_angle_corridor": 0.0,
                        "turn_corridor": 0.0,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        publish(True)
        clock.advance(0.10)
        publish(True)
        clock.advance(0.10)
        publish(False)
        assert adapter.status_payload()["gap_crossing_active"] is True

        clock.advance(0.51)
        publish(False)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        status = adapter.status_payload()
        assert status["gap_crossing_active"] is False
        assert status["assist_kind"] == "line_gap_timeout"
        assert latest[4] == latest[5] == 1600
        assert latest[0] == latest[1] == 1600
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_calibrated_right_corner_runs_locked_2100_ms_sequence() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            neutral_us=1600,
            basic_line_follow=True,
            line_steering_inverted=True,
            sharp_corner_maneuver_enabled=True,
            corner_sequence_enabled=True,
            corner_confirm_frames=3,
            corner_brake_ms=250,
            corner_pivot_speed_us=300,
            corner_pivot_right_ms=2100,
            corner_reacquire_speed_us=130,
            corner_reacquire_confirm_frames=3,
            corner_exit_neutral_ms=150,
            corner_exit_straight_ms=320,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish(
        *,
        line: bool = True,
        right_angle: bool = True,
        bend: float = -0.50,
        row_ratio: float = 0.86,
        offset: float = -0.05,
        outgoing: bool = False,
    ) -> None:
        geometry = {
            "candidate_count": 1.0,
            "right_angle_corridor": float(right_angle),
            "turn_corridor": float(right_angle),
            "path_bend_delta_norm": bend,
            "dominant_row_y_ratio": row_ratio,
            "max_row_occupancy": 0.90,
            "median_row_occupancy": 0.40,
            "height_ratio": 1.0,
            "width_ratio": 0.54 if outgoing else 0.92,
            "center_range_ratio": 0.14 if outgoing else 0.48,
            "bottom_row_occupancy": 0.40,
        }
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=line,
                metadata={
                    "line_offset_norm": offset,
                    "line_offset_source": "ground_path" if line else "none",
                    "line_confidence": 0.9 if line else 0.0,
                    "line_angle_deg": 90,
                    "line_geometry": geometry,
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        for _ in range(3):
            publish()
            clock.advance(0.04)
        assert adapter.status_payload()["corner_phase"] == "BRAKE"
        assert {channel: pulse for channel, pulse in driver.pulses}[4] == 1600

        clock.advance(0.25)
        publish()
        status = adapter.status_payload()
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert status["corner_phase"] == "PIVOT"
        assert status["corner_direction"] == "RIGHT"
        assert status["requested_left_speed_us"] == 300.0
        assert status["requested_right_speed_us"] == -300.0
        assert latest[4] == latest[5] == 1900
        assert latest[0] == latest[1] == 1900

        # The contour sign flips while the chassis rotates. The physical side
        # remains locked and the calibrated command is unchanged.
        clock.advance(1.0)
        publish(line=False, bend=0.80, row_ratio=0.10)
        status = adapter.status_payload()
        assert status["corner_phase"] == "PIVOT"
        assert status["corner_direction"] == "RIGHT"
        assert status["requested_left_speed_us"] == 300.0
        assert status["requested_right_speed_us"] == -300.0

        clock.advance(1.11)
        publish(line=False, bend=0.80, row_ratio=0.10)
        status = adapter.status_payload()
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert status["corner_phase"] == "REACQUIRE"
        assert status["requested_left_speed_us"] == 130.0
        assert status["requested_right_speed_us"] == -130.0
        assert latest[4] == latest[5] == 1730
        assert latest[0] == latest[1] == 1730

        for _ in range(3):
            clock.advance(0.04)
            publish(
                line=True,
                right_angle=False,
                bend=0.01,
                row_ratio=0.50,
                offset=0.08,
                outgoing=True,
            )
        assert adapter.status_payload()["corner_phase"] == "EXIT"
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert latest[4] == latest[5] == latest[0] == latest[1] == 1600

        clock.advance(0.15)
        publish(
            line=True,
            right_angle=False,
            bend=0.01,
            row_ratio=0.50,
            offset=0.08,
            outgoing=True,
        )
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert latest[4] == latest[5] == 1900
        assert latest[0] == latest[1] == 1400
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_corner_reacquire_brakes_before_confirming_recorded_line() -> None:
    """A plausible outgoing line must stop the slow pivot before verification."""
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channel=0,
            right_channel=1,
            neutral_us=1600,
            basic_line_follow=True,
            corner_sequence_enabled=True,
            corner_reacquire_confirm_frames=3,
            corner_reacquire_max_offset=0.38,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish(offset: float, angle: float) -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    "line_offset_norm": offset,
                    "line_offset_source": "ground_path",
                    "line_confidence": 1.0,
                    "line_angle_deg": angle,
                    # Deliberately violate the former contour-shape gates. The
                    # recorded outgoing line was real, but those fields flicker
                    # while the chassis is rotating.
                    "line_geometry": {
                        "candidate_count": 2.0,
                        "right_angle_corridor": 0.0,
                        "path_bend_delta_norm": 0.62,
                        "height_ratio": 0.55,
                        "width_ratio": 0.88,
                        "center_range_ratio": 0.44,
                        "bottom_row_occupancy": 0.74,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        with adapter._lock:
            adapter._corner_direction = -1
            adapter._set_corner_phase_locked("REACQUIRE", clock())

        publish(0.215, 101.0)
        status = adapter.status_payload()
        assert status["corner_phase"] == "REACQUIRE"
        assert status["corner_reacquire_streak"] == 1
        assert status["assist_kind"] == "line_corner_reacquire_verify"
        assert status["requested_left_speed_us"] == 0.0
        assert status["requested_right_speed_us"] == 0.0
        assert _latest_pair(driver) == (1600, 1600)

        clock.advance(0.04)
        publish(0.29, 105.0)
        assert adapter.status_payload()["corner_reacquire_streak"] == 2
        assert _latest_pair(driver) == (1600, 1600)

        clock.advance(0.04)
        publish(0.31, 109.0)
        status = adapter.status_payload()
        assert status["corner_phase"] == "EXIT"
        assert status["assist_kind"] == "line_corner_exit_neutral"
        assert _latest_pair(driver) == (1600, 1600)
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_corner_timeout_recovers_without_clear_estop() -> None:
    """A visible line releases TIMEOUT after a short neutral hold, never ESTOP."""
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channel=0,
            right_channel=1,
            neutral_us=1600,
            basic_line_follow=True,
            corner_sequence_enabled=True,
            corner_exit_neutral_ms=150,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish(*, green: bool = False, source: str = "ground_path") -> None:
        metadata = {
            "line_offset_norm": 0.65,
            "line_offset_source": source,
            "line_confidence": 1.0,
            "line_angle_deg": 68.0,
            "line_geometry": {
                "candidate_count": 1.0,
                "right_angle_corridor": 0.0,
                "path_bend_delta_norm": 0.0,
            },
        }
        if green:
            metadata.update(
                {
                    "green_instruction": "VERDE_DEPOIS",
                    "green_side": "LEFT",
                    "green_marker_count": 1,
                    "green_marker_confidence": 0.99,
                }
            )
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                green=green,
                metadata=metadata,
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        with adapter._lock:
            adapter._corner_direction = -1
            adapter._set_corner_phase_locked("TIMEOUT", clock())

        # This reproduces the extra single-green command from the bad left
        # recording. It must be ignored while the corner owns control.
        publish(green=True, source="none")
        status = adapter.status_payload()
        assert status["corner_phase"] == "TIMEOUT"
        assert status["control_mode"] == "STOPPED"
        assert status["green_instruction"] == "NO_GREEN"
        assert adapter._maneuver_until == 0.0

        publish()
        assert adapter.status_payload()["corner_phase"] == "TIMEOUT"
        assert _latest_pair(driver) == (1600, 1600)

        clock.advance(0.151)
        publish()
        status = adapter.status_payload()
        assert status["corner_phase"] == "IDLE"
        assert status["control_mode"] == "FOLLOW_LINE"
        assert status["assist_kind"].startswith("line_")
        assert status["failsafe"] is False
        assert _latest_pair(driver) != (1600, 1600)
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_recorded_filled_l_brakes_before_pivot() -> None:
    """The close USB-camera L must not fall through to ordinary curve following."""
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            neutral_us=1600,
            basic_line_follow=True,
            line_steering_inverted=True,
            sharp_corner_maneuver_enabled=True,
            corner_sequence_enabled=True,
            corner_confirm_frames=2,
            corner_min_elbow_row_contrast=1.25,
            corner_approach_stop_row_ratio=0.40,
            corner_approach_min_ms=350,
            corner_approach_left_min_ms=550,
            corner_brake_ms=500,
            corner_pivot_speed_us=300,
            corner_pivot_left_ms=1900,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish(*, max_row: float, median_row: float, bend: float) -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    "line_offset_norm": 0.18,
                    "line_offset_source": "ground_path",
                    "line_confidence": 1.0,
                    "line_geometry": {
                        "candidate_count": 1.0,
                        "right_angle_corridor": 1.0,
                        "turn_corridor": 1.0,
                        "path_bend_delta_norm": bend,
                        "dominant_row_y_ratio": 0.68,
                        "max_row_occupancy": max_row,
                        "median_row_occupancy": median_row,
                        "width_ratio": 0.83,
                        "height_ratio": 1.0,
                        "center_range_ratio": 0.11,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        # Values are from the consecutive 06.129 s and 06.167 s field frames
        # in session_20260618_033042_147.
        publish(max_row=0.822, median_row=0.628, bend=0.217)
        clock.advance(0.04)
        publish(max_row=0.822, median_row=0.639, bend=0.211)

        status = adapter.status_payload()
        assert status["corner_phase"] == "APPROACH"
        assert status["assist_kind"] == "line_corner_approach"
        assert status["corner_approach_min_ms"] == 350
        assert status["corner_approach_left_min_ms"] == 550
        # The placement creep is straight and uses the already calibrated
        # asymmetric base magnitudes; L geometry must not steer during it.
        assert status["requested_left_speed_us"] == 216.0
        assert status["requested_right_speed_us"] == 144.0

        clock.advance(0.549)
        publish(max_row=0.822, median_row=0.639, bend=0.211)
        assert adapter.status_payload()["corner_phase"] == "APPROACH"

        clock.advance(0.002)
        publish(max_row=0.822, median_row=0.639, bend=0.211)
        status = adapter.status_payload()
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert status["corner_phase"] == "BRAKE"
        assert status["assist_kind"] == "line_corner_brake"
        assert status["requested_left_speed_us"] == 0.0
        assert status["requested_right_speed_us"] == 0.0
        assert latest[4] == latest[5] == latest[0] == latest[1] == 1600

        clock.advance(0.499)
        publish(max_row=0.822, median_row=0.639, bend=0.211)
        assert adapter.status_payload()["corner_phase"] == "BRAKE"

        clock.advance(0.002)
        publish(max_row=0.822, median_row=0.639, bend=0.211)
        status = adapter.status_payload()
        assert status["corner_phase"] == "PIVOT"
        assert status["corner_direction"] == "LEFT"
        assert status["requested_left_speed_us"] == -150.0
        assert status["requested_right_speed_us"] == 150.0

        clock.advance(1.899)
        publish(max_row=0.822, median_row=0.639, bend=0.211)
        assert adapter.status_payload()["corner_phase"] == "PIVOT"

        clock.advance(0.002)
        publish(max_row=0.822, median_row=0.639, bend=0.211)
        status = adapter.status_payload()
        assert status["corner_phase"] == "REACQUIRE"
        assert status["assist_kind"] == "line_corner_reacquire"
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_recorded_l_fallback_survives_right_angle_flag_flicker() -> None:
    """The missed field L must trigger without widening the open-curve gate."""
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            neutral_us=1600,
            basic_line_follow=True,
            line_steering_inverted=True,
            sharp_corner_maneuver_enabled=True,
            corner_sequence_enabled=True,
            corner_confirm_frames=2,
            corner_approach_min_ms=350,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish(*, bend: float, row: float, max_row: float, median_row: float) -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    "line_offset_norm": -0.25,
                    "line_offset_source": "ground_path",
                    "line_confidence": 1.0,
                    "line_geometry": {
                        "candidate_count": 1.0,
                        "right_angle_corridor": 0.0,
                        "turn_corridor": 1.0,
                        "path_bend_delta_norm": bend,
                        "dominant_row_y_ratio": row,
                        "max_row_occupancy": max_row,
                        "median_row_occupancy": median_row,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        # Consecutive missed-L frames at 15.260 s and 15.356 s in
        # session_20260618_004519_609.
        publish(bend=-0.311, row=0.910, max_row=0.691, median_row=0.416)
        status = adapter.status_payload()
        assert status["corner_phase"] == "IDLE"
        assert status["corner_confirm_streak"] == 1
        assert status["corner_geometry_fallback_active"] is True

        clock.advance(0.096)
        publish(bend=-0.372, row=0.814, max_row=0.681, median_row=0.419)
        status = adapter.status_payload()
        assert status["corner_phase"] == "APPROACH"
        assert status["corner_direction"] == "RIGHT"
        assert status["assist_kind"] == "line_corner_approach"
        assert status["requested_left_speed_us"] == 216.0
        assert status["requested_right_speed_us"] == 144.0
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_narrow_recorded_left_l_uses_three_frame_fallback() -> None:
    """The narrow left L in session ...002622_528 must not become an open curve."""
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            neutral_us=1600,
            basic_line_follow=True,
            line_steering_inverted=True,
            sharp_corner_maneuver_enabled=True,
            corner_sequence_enabled=True,
            corner_confirm_frames=2,
            corner_approach_min_ms=350,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    recorded_frames = (
        # bend, dominant row, maximum row, median row, width, height
        (0.288, 0.93, 0.58, 0.38, 0.67, 1.00),
        (0.352, 0.79, 0.59, 0.38, 0.66, 1.00),
        (0.320, 0.78, 0.59, 0.38, 0.66, 0.92),
    )

    try:
        adapter.start()
        for index, (bend, row, max_row, median_row, width, height) in enumerate(
            recorded_frames,
            start=1,
        ):
            bus.publish(
                EventTopic.VISION_DETECTIONS,
                VisionDetectionEvent(
                    state="FOLLOWING_LINE",
                    line=True,
                    metadata={
                        "line_offset_norm": 0.40,
                        "line_offset_source": "ground_path",
                        "line_confidence": 1.0,
                        "line_angle_deg": 92,
                        "line_geometry": {
                            "candidate_count": 1.0,
                            "right_angle_corridor": 0.0,
                            "turn_corridor": 0.0,
                            "path_bend_delta_norm": bend,
                            "dominant_row_y_ratio": row,
                            "max_row_occupancy": max_row,
                            "median_row_occupancy": median_row,
                            "width_ratio": width,
                            "height_ratio": height,
                            "bottom_row_occupancy": max_row,
                        },
                    },
                ),
            )
            _drain(bus)
            status = adapter.status_payload()
            if index < 3:
                assert status["corner_phase"] == "IDLE"
                assert status["corner_confirm_streak"] == index
            clock.advance(0.10)

        status = adapter.status_payload()
        assert status["corner_phase"] == "APPROACH"
        assert status["corner_direction"] == "LEFT"
        assert status["corner_geometry_fallback_active"] is True
        assert status["assist_kind"] == "line_corner_approach"
        assert status["requested_left_speed_us"] == 216.0
        assert status["requested_right_speed_us"] == 144.0
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_left_precision_pivot_brakes_on_first_outgoing_line() -> None:
    """After the incoming leg leaves view, the left pivot must not reach 180 degrees."""
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            neutral_us=1600,
            basic_line_follow=True,
            line_steering_inverted=True,
            sharp_corner_maneuver_enabled=True,
            corner_sequence_enabled=True,
            corner_confirm_frames=2,
            corner_approach_min_ms=0,
            corner_approach_stop_row_ratio=0.40,
            corner_brake_ms=250,
            corner_pivot_speed_us=300,
            corner_pivot_left_speed_us=150,
            corner_pivot_left_ms=1900,
            corner_pivot_left_vision_min_ms=600,
            corner_pivot_left_lost_confirm_frames=2,
            corner_reacquire_confirm_frames=3,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish(
        *,
        line: bool = True,
        right_angle: bool = True,
        bend: float = 0.50,
        offset: float = 0.05,
        angle: float = 90.0,
    ) -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=line,
                metadata={
                    "line_offset_norm": offset if line else 0.0,
                    "line_offset_source": "ground_path" if line else "none",
                    "line_confidence": 0.9 if line else 0.0,
                    "line_angle_deg": angle,
                    "line_geometry": {
                        "candidate_count": 1.0 if line else 0.0,
                        "right_angle_corridor": float(right_angle),
                        "turn_corridor": float(right_angle),
                        "path_bend_delta_norm": bend,
                        "dominant_row_y_ratio": 0.86,
                        "max_row_occupancy": 0.90,
                        "median_row_occupancy": 0.40,
                        "width_ratio": 0.90 if right_angle else 0.45,
                        "height_ratio": 1.0,
                        "bottom_row_occupancy": 0.40,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        publish()
        clock.advance(0.04)
        publish()
        assert adapter.status_payload()["corner_phase"] == "BRAKE"

        clock.advance(0.251)
        publish()
        status = adapter.status_payload()
        assert status["corner_phase"] == "PIVOT"
        assert status["requested_left_speed_us"] == -150.0
        assert status["requested_right_speed_us"] == 150.0

        clock.advance(0.61)
        publish(line=False, right_angle=False)
        clock.advance(0.10)
        publish(line=False, right_angle=False)
        status = adapter.status_payload()
        assert status["corner_phase"] == "PIVOT"
        assert status["corner_pivot_line_lost_seen"] is True

        clock.advance(0.10)
        publish(
            line=True,
            right_angle=False,
            bend=0.04,
            offset=0.12,
            angle=91.0,
        )
        status = adapter.status_payload()
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert status["corner_phase"] == "REACQUIRE"
        assert status["corner_reacquire_streak"] == 1
        assert status["assist_kind"] == "line_corner_reacquire_verify"
        assert status["requested_left_speed_us"] == 0.0
        assert status["requested_right_speed_us"] == 0.0
        assert latest[4] == latest[5] == latest[0] == latest[1] == 1600

        for _ in range(2):
            clock.advance(0.10)
            publish(
                line=True,
                right_angle=False,
                bend=0.03,
                offset=0.10,
                angle=90.0,
            )
        assert adapter.status_payload()["corner_phase"] == "EXIT"
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_corner_timing_can_be_applied_for_the_next_corner() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            corner_approach_min_ms=250,
            corner_pivot_right_ms=1900,
            corner_pivot_left_ms=1900,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
    )
    try:
        adapter.start()
        applied = adapter.update_corner_timing(approach_min_ms=350, pivot_ms=1850)
        assert applied == {
            "approach_min_ms": 350,
            "pivot_right_ms": 1850,
            "pivot_left_ms": 1850,
        }
        status = adapter.status_payload()
        assert status["corner_approach_min_ms"] == 350
        assert status["corner_pivot_right_ms"] == 1850
        assert status["corner_pivot_left_ms"] == 1850
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_left_corner_timing_changes_only_the_left_corner() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            corner_approach_min_ms=350,
            corner_approach_left_min_ms=550,
            corner_pivot_right_ms=1900,
            corner_pivot_left_ms=1900,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
    )
    try:
        adapter.start()
        applied = adapter.update_left_corner_timing(
            approach_min_ms=625,
            pivot_ms=2050,
        )
        assert applied == {
            "approach_left_min_ms": 625,
            "pivot_left_ms": 2050,
        }
        status = adapter.status_payload()
        assert status["corner_approach_left_min_ms"] == 625
        assert status["corner_pivot_left_ms"] == 2050
        assert status["corner_approach_min_ms"] == 350
        assert status["corner_pivot_right_ms"] == 1900
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_green_half_turn_timing_can_be_applied_live() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            green_half_turn_ms=4000,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
    )
    try:
        adapter.start()
        applied = adapter.update_green_half_turn_timing(duration_ms=4100)
        assert applied == {
            "green_half_turn_ms": 4100,
            "green_half_turn_first_ms": 1900,
            "green_half_turn_second_ms": 2200,
            "green_half_turn_reverse_ms": 550,
        }
        status = adapter.status_payload()
        assert status["green_half_turn_ms"] == 4100
        assert status["telemetry"]["green_half_turn_ms"] == 4100
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_calibrated_left_corner_uses_slow_precision_pivot() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            neutral_us=1600,
            basic_line_follow=True,
            line_steering_inverted=True,
            sharp_corner_maneuver_enabled=True,
            corner_sequence_enabled=True,
            corner_confirm_frames=3,
            corner_brake_ms=250,
            corner_pivot_speed_us=300,
            corner_pivot_left_ms=2100,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish() -> None:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={
                    "line_offset_norm": 0.05,
                    "line_offset_source": "ground_path",
                    "line_confidence": 0.9,
                    "line_angle_deg": 90,
                    "line_geometry": {
                        "candidate_count": 1.0,
                        "right_angle_corridor": 1.0,
                        "turn_corridor": 1.0,
                        "path_bend_delta_norm": 0.50,
                        "dominant_row_y_ratio": 0.86,
                        "max_row_occupancy": 0.90,
                        "median_row_occupancy": 0.40,
                    },
                },
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        for _ in range(3):
            publish()
            clock.advance(0.04)
        clock.advance(0.25)
        publish()

        status = adapter.status_payload()
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert status["corner_phase"] == "PIVOT"
        assert status["corner_direction"] == "LEFT"
        assert status["requested_left_speed_us"] == -150.0
        assert status["requested_right_speed_us"] == 150.0
        assert status["corner_pivot_left_speed_us"] == 150
        assert latest[4] == latest[5] == 1450
        assert latest[0] == latest[1] == 1450
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_open_curve_never_enters_calibrated_corner_sequence() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            neutral_us=1600,
            basic_line_follow=True,
            line_steering_inverted=True,
            sharp_corner_maneuver_enabled=True,
            corner_sequence_enabled=True,
            curve_lookahead_gain=0.55,
            curve_throttle_scale=0.72,
            ordinary_sharp_hold_ms=180,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    try:
        adapter.start()
        for _ in range(8):
            bus.publish(
                EventTopic.VISION_DETECTIONS,
                VisionDetectionEvent(
                    state="FOLLOWING_LINE",
                    line=True,
                    metadata={
                        "line_offset_norm": -0.10,
                        "line_offset_source": "ground_path",
                        "line_confidence": 0.9,
                        "line_geometry": {
                            "candidate_count": 1.0,
                            "right_angle_corridor": 1.0,
                            "turn_corridor": 1.0,
                            "path_bend_delta_norm": -0.75,
                            "dominant_row_y_ratio": 0.95,
                            # Recorded open curves have no abrupt horizontal
                            # elbow: their widest and median rows are similar.
                            "max_row_occupancy": 0.90,
                            "median_row_occupancy": 0.80,
                        },
                    },
                ),
            )
            _drain(bus)
            clock.advance(0.04)

        status = adapter.status_payload()
        assert status["corner_phase"] == "IDLE"
        assert status["corner_confirm_streak"] == 0
        assert status["assist_kind"] == "line_curve_lookahead"
        assert status["requested_left_speed_us"] >= 0.0
        assert status["requested_right_speed_us"] >= 0.0
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_adapter_line_mixer_never_commands_reverse() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            pid_kp_us=1000,
            max_output_us=360,
            left_inverted=False,
            right_inverted=True,
        ),
        pwm_factory=lambda config: driver,
    )

    try:
        adapter.start()
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={"line_offset_norm": 0.8, "line_confidence": 0.9},
            ),
        )
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert latest[4] <= 1960
        assert latest[0] >= 1600
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_adapter_preserves_calibrated_base_above_pid_limit() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            max_output_us=240,
        ),
        pwm_factory=lambda config: driver,
    )

    try:
        adapter.start()
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={"line_offset_norm": 0.0, "line_confidence": 0.9},
            ),
        )
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert latest[4] == latest[5] == 1900
        assert latest[0] == latest[1] == 1400
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_adapter_stop_and_estop_hold_neutral_until_clear() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(left_channel=0, right_channel=1, neutral_us=1500, base_throttle_us=120),
        pwm_factory=lambda config: driver,
    )

    try:
        adapter.start()
        bus.publish(EventTopic.UI_COMMAND, UICommandEvent(command="robot.estop", params={}))
        _drain(bus)
        assert _latest_pair(driver) == (1500, 1500)
        assert adapter.status_payload()["failsafe"] is True

        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={"line_offset_norm": 0.0, "line_confidence": 0.9},
            ),
        )
        _drain(bus)
        assert _latest_pair(driver) == (1500, 1500)

        bus.publish(EventTopic.UI_COMMAND, UICommandEvent(command="robot.clear_estop", params={}))
        _drain(bus)
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={"line_offset_norm": 0.0, "line_confidence": 0.9},
            ),
        )
        _drain(bus)
        assert _latest_pair(driver) != (1500, 1500)
        assert adapter.status_payload()["failsafe"] is False
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_adapter_green_meia_volta_commands_spin_pwm() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    clock = FakeClock()
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channel=0,
            right_channel=1,
            neutral_us=1500,
            green_trigger_streak=2,
            green_turn_us=260,
            green_half_turn_us=300,
            green_half_turn_ms=4000,
            left_inverted=True,
            right_inverted=False,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    try:
        adapter.start()
        event = VisionDetectionEvent(
            state="FOLLOWING_LINE",
            line=True,
            green=True,
            metadata={
                "green_instruction": "VERDE_MEIA_VOLTA",
                "green_side": "BOTH",
                "green_marker_count": 2,
                "green_marker_confidence": 0.92,
            },
        )
        bus.publish(EventTopic.VISION_DETECTIONS, event)
        _drain(bus)
        assert _latest_pair(driver) == (1500, 1500)
        status = adapter.status_payload()
        assert status["control_mode"] == "GREEN"
        assert status["green_instruction"] == "VERDE_MEIA_VOLTA"
        assert status["green_marker_count"] == 2
        assert status["green_maneuver_duration_ms"] == 5600
        assert status["green_half_turn_armed"] is False
        assert status["green_half_turn_phase"] == "BRAKE_1"
        assert status["green_half_turn_first_ms"] == 1900
        assert status["green_half_turn_second_ms"] == 2100
        assert status["green_half_turn_reverse_ms"] == 550
        assert status["requested_left_speed_us"] == 0.0
        assert status["requested_right_speed_us"] == 0.0
        assert status["green_half_turn_left_us"] == 300.0
        assert status["green_half_turn_right_us"] == 300.0
        assert abs(adapter._maneuver_until - 105.60) < 1e-9

        clock.advance(0.501)
        assert adapter._tick_green_half_turn_sequence_locked(clock()) is True
        assert _latest_pair(driver) == (1200, 1200)
        assert adapter.status_payload()["green_half_turn_phase"] == "PIVOT_1"

        clock.advance(1.901)
        assert adapter._tick_green_half_turn_sequence_locked(clock()) is True
        assert _latest_pair(driver) == (1500, 1500)
        assert adapter.status_payload()["green_half_turn_phase"] == "BRAKE_MID"

        clock.advance(0.251)
        assert adapter._tick_green_half_turn_sequence_locked(clock()) is True
        assert _latest_pair(driver) == (1800, 1300)
        assert adapter.status_payload()["green_half_turn_phase"] == "REVERSE"

        clock.advance(0.551)
        assert adapter._tick_green_half_turn_sequence_locked(clock()) is True
        assert _latest_pair(driver) == (1500, 1500)
        assert adapter.status_payload()["green_half_turn_phase"] == "BRAKE_REVERSE"

        clock.advance(0.151)
        assert adapter._tick_green_half_turn_sequence_locked(clock()) is True
        assert _latest_pair(driver) == (1200, 1200)
        assert adapter.status_payload()["green_half_turn_phase"] == "PIVOT_2"

        clock.advance(2.101)
        assert adapter._tick_green_half_turn_sequence_locked(clock()) is True
        assert _latest_pair(driver) == (1500, 1500)
        assert adapter.status_payload()["green_half_turn_phase"] == "BRAKE_EXIT"

        clock.advance(0.151)
        assert adapter._tick_green_half_turn_sequence_locked(clock()) is True
        assert adapter.status_payload()["green_half_turn_phase"] == "IDLE"
        assert adapter.status_payload()["control_mode"] == "STOPPED"
        assert adapter._maneuver_until == 0.0
    finally:
        adapter.stop()
        bus.stop()


def test_green_half_turn_retries_reverse_without_skipping_phase_after_i2c_error() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    clock = FakeClock()
    driver = FailOncePwmDriver(fail_on_pulse=1800)
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channel=0,
            right_channel=1,
            neutral_us=1500,
            left_base_throttle_us=300,
            right_base_throttle_us=200,
            green_half_turn_first_ms=1900,
            green_half_turn_second_ms=2100,
            green_half_turn_reverse_ms=550,
            left_inverted=True,
            right_inverted=False,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )
    try:
        adapter.start()
        adapter._green_half_turn_phase = "BRAKE_MID"
        adapter._green_half_turn_phase_started_at = clock()
        clock.advance(0.251)

        with pytest.raises(OSError, match="Remote I/O error"):
            adapter._tick_green_half_turn_sequence_locked(clock())
        assert adapter.status_payload()["green_half_turn_phase"] == "BRAKE_MID"

        assert adapter._tick_green_half_turn_sequence_locked(clock()) is True
        assert adapter.status_payload()["green_half_turn_phase"] == "REVERSE"
        assert adapter.status_payload()["steering_decision"] == "REVERSE_BETWEEN_90"
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_adapter_half_turn_is_one_shot_until_green_clears() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    clock = FakeClock()
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channel=0,
            right_channel=1,
            neutral_us=1500,
            green_half_turn_us=300,
            green_half_turn_ms=1000,
            green_cooldown_ms=0,
            green_rearm_clear_frames=3,
            left_inverted=True,
            right_inverted=False,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )
    pair = VisionDetectionEvent(
        state="FOLLOWING_LINE",
        line=True,
        green=True,
        metadata={
            "green_instruction": "VERDE_MEIA_VOLTA",
            "green_side": "BOTH",
            "green_marker_count": 2,
            "green_marker_confidence": 0.96,
            "line_offset_norm": 0.0,
            "line_confidence": 0.9,
        },
    )
    clear = VisionDetectionEvent(
        state="FOLLOWING_LINE",
        line=True,
        green=False,
        metadata={"line_offset_norm": 0.0, "line_confidence": 0.9},
    )

    try:
        adapter.start()
        bus.publish(EventTopic.VISION_DETECTIONS, pair)
        _drain(bus)
        first_until = adapter._maneuver_until
        assert adapter.status_payload()["green_half_turn_armed"] is False

        clock.advance(3.0)
        bus.publish(EventTopic.VISION_DETECTIONS, pair)
        _drain(bus)
        assert adapter._maneuver_until == first_until
        assert adapter.status_payload()["green_half_turn_armed"] is False

        for _ in range(2):
            bus.publish(EventTopic.VISION_DETECTIONS, clear)
            _drain(bus)
        assert adapter.status_payload()["green_half_turn_armed"] is False

        bus.publish(EventTopic.VISION_DETECTIONS, clear)
        _drain(bus)
        assert adapter.status_payload()["green_half_turn_armed"] is True

        bus.publish(EventTopic.VISION_DETECTIONS, pair)
        _drain(bus)
        assert adapter.status_payload()["control_mode"] == "GREEN"
        assert adapter.status_payload()["green_half_turn_armed"] is False
        assert adapter._maneuver_until > first_until
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_adapter_double_green_preempts_pending_single_green() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channel=0,
            right_channel=1,
            neutral_us=1500,
            green_trigger_streak=4,
            green_half_turn_trigger_streak=1,
            green_cooldown_ms=6000,
            left_inverted=True,
            right_inverted=False,
        ),
        pwm_factory=lambda config: driver,
    )
    single = VisionDetectionEvent(
        state="FOLLOWING_LINE",
        line=True,
        green=True,
        metadata={
            "green_instruction": "VERDE_ANTES",
            "green_side": "LEFT",
            "green_marker_count": 1,
            "green_marker_confidence": 0.98,
            "line_offset_norm": 0.0,
            "line_confidence": 0.9,
        },
    )
    pair = VisionDetectionEvent(
        state="FOLLOWING_LINE",
        line=True,
        green=True,
        metadata={
            "green_instruction": "VERDE_MEIA_VOLTA",
            "green_side": "BOTH",
            "green_marker_count": 2,
            "green_marker_confidence": 0.96,
            "line_offset_norm": 0.0,
            "line_confidence": 0.9,
        },
    )

    try:
        adapter.start()
        for _ in range(2):
            bus.publish(EventTopic.VISION_DETECTIONS, single)
            _drain(bus)
        assert adapter.status_payload()["control_mode"] != "GREEN"

        bus.publish(EventTopic.VISION_DETECTIONS, pair)
        _drain(bus)
        assert adapter.status_payload()["control_mode"] == "GREEN"
        assert adapter.status_payload()["green_instruction"] == "VERDE_MEIA_VOLTA"
        assert adapter.status_payload()["green_maneuver_duration_ms"] == 5600
    finally:
        adapter.stop()
        bus.stop()


@pytest.mark.parametrize(
    ("image_side", "physical_side", "approach_ms", "pivot_ms"),
    [
        ("RIGHT", "LEFT", 550, 2100),
        ("LEFT", "RIGHT", 350, 1900),
    ],
)
def test_pca9685_single_green_before_uses_calibrated_mirrored_corner(
    image_side: str,
    physical_side: str,
    approach_ms: int,
    pivot_ms: int,
) -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            neutral_us=1600,
            basic_line_follow=True,
            line_steering_inverted=True,
            corner_sequence_enabled=True,
            corner_approach_min_ms=350,
            corner_approach_left_min_ms=550,
            corner_pivot_right_ms=1900,
            corner_pivot_left_ms=2100,
            green_trigger_streak=2,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )
    event = VisionDetectionEvent(
        state="FOLLOWING_LINE",
        line=True,
        green=True,
        metadata={
            "green_instruction": "VERDE_ANTES",
            # The USB image side is mirrored by the proven installation.
            "green_side": image_side,
            "green_marker_count": 1,
            "green_marker_confidence": 0.98,
            "green_relation_confidence": 0.95,
            "green_relation_delta_y": 34.0,
            "green_center_y": 154.0,
            "preprocessor": {"output_shape": {"width": 320, "height": 200}},
            "line_offset_norm": 0.0,
            "line_confidence": 1.0,
            "line_geometry": {},
        },
    )

    try:
        adapter.start()
        bus.publish(EventTopic.VISION_DETECTIONS, event)
        _drain(bus)
        assert adapter.status_payload()["assist_kind"] == "green_single_confirm"
        assert _latest_pair(driver) == (1600, 1600)

        bus.publish(EventTopic.VISION_DETECTIONS, event)
        _drain(bus)
        status = adapter.status_payload()
        assert status["corner_phase"] == "APPROACH"
        assert status["corner_direction"] == physical_side
        assert (
            status["corner_approach_left_min_ms"]
            if physical_side == "LEFT"
            else status["corner_approach_min_ms"]
        ) == approach_ms
        assert (
            status["corner_pivot_left_ms"]
            if physical_side == "LEFT"
            else status["corner_pivot_right_ms"]
        ) == pivot_ms
        assert status["assist_kind"] == "line_corner_approach"
        assert adapter._maneuver_until == 0.0

        # The marker remains visible while the chassis creeps over the L.
        # It must not restart confirmation and reset the calibrated corner on
        # every camera frame.
        phase_started_at = adapter._corner_phase_started_at
        clock.advance(0.1)
        bus.publish(EventTopic.VISION_DETECTIONS, event)
        _drain(bus)
        status = adapter.status_payload()
        assert status["green_single_encounter"] == "EXECUTE"
        assert status["corner_phase"] == "APPROACH"
        assert status["corner_direction"] == physical_side
        assert adapter._corner_phase_started_at == phase_started_at
        assert status["assist_kind"] == "line_corner_approach"
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_single_green_confirmation_stays_stopped_across_one_dropout() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            neutral_us=1600,
            basic_line_follow=True,
            line_steering_inverted=True,
            corner_sequence_enabled=True,
            corner_approach_min_ms=350,
            green_trigger_streak=2,
            green_single_clear_frames=3,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )
    green = VisionDetectionEvent(
        state="FOLLOWING_LINE",
        line=True,
        green=True,
        metadata={
            "green_instruction": "VERDE_ANTES",
            "green_side": "LEFT",
            "green_marker_count": 1,
            "green_marker_confidence": 0.98,
            "green_relation_confidence": 0.95,
            "green_relation_delta_y": -42.0,
            "line_offset_norm": 0.0,
            "line_confidence": 1.0,
            "line_geometry": {},
        },
    )
    dropout = VisionDetectionEvent(
        state="FOLLOWING_LINE",
        line=True,
        green=False,
        metadata={
            "line_offset_norm": 0.0,
            "line_confidence": 1.0,
            "line_geometry": {},
        },
    )

    try:
        adapter.start()
        bus.publish(EventTopic.VISION_DETECTIONS, green)
        _drain(bus)
        assert adapter.status_payload()["assist_kind"] == "green_single_confirm"
        assert _latest_pair(driver) == (1600, 1600)

        bus.publish(EventTopic.VISION_DETECTIONS, dropout)
        _drain(bus)
        status = adapter.status_payload()
        assert status["green_single_encounter"] == "PENDING"
        assert status["assist_kind"] == "green_single_confirm_gap"
        assert _latest_pair(driver) == (1600, 1600)

        bus.publish(EventTopic.VISION_DETECTIONS, green)
        _drain(bus)
        status = adapter.status_payload()
        assert status["corner_phase"] == "APPROACH"
        assert status["corner_direction"] == "RIGHT"
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_single_green_waits_for_stationary_confirmation_hold() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            neutral_us=1600,
            basic_line_follow=True,
            line_steering_inverted=True,
            corner_sequence_enabled=True,
            green_trigger_streak=2,
            green_single_confirm_hold_ms=180,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )
    event = VisionDetectionEvent(
        state="FOLLOWING_LINE",
        line=True,
        green=True,
        metadata={
            "green_instruction": "VERDE_ANTES",
            "green_side": "RIGHT",
            "green_marker_count": 1,
            "green_marker_confidence": 0.98,
            "green_relation_confidence": 0.95,
            "green_relation_delta_y": 34.0,
            "line_offset_norm": 0.0,
            "line_confidence": 1.0,
            "line_geometry": {},
        },
    )

    try:
        adapter.start()
        bus.publish(EventTopic.VISION_DETECTIONS, event)
        _drain(bus)
        clock.advance(0.10)
        bus.publish(EventTopic.VISION_DETECTIONS, event)
        _drain(bus)
        assert adapter.status_payload()["green_single_encounter"] == "PENDING"
        assert adapter.status_payload()["assist_kind"] == "green_single_confirm"
        assert _latest_pair(driver) == (1600, 1600)

        clock.advance(0.09)
        bus.publish(EventTopic.VISION_DETECTIONS, event)
        _drain(bus)
        status = adapter.status_payload()
        assert status["green_single_encounter"] == "EXECUTE"
        assert status["green_route_decision"] == "BEFORE_LEFT"
        assert status["corner_direction"] == "LEFT"
        assert status["corner_phase"] in {"APPROACH", "BRAKE"}
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_single_green_after_ignores_entire_structural_intersection() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    clock = FakeClock()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            neutral_us=1600,
            basic_line_follow=True,
            line_steering_inverted=True,
            corner_sequence_enabled=True,
            green_trigger_streak=2,
            green_single_clear_frames=3,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    def publish(*, green: bool, elbow: bool) -> None:
        metadata = {
            "line_offset_norm": 0.0,
            "line_confidence": 1.0,
            "line_geometry": {
                "right_angle_corridor": 1.0 if elbow else 0.0,
                "path_bend_delta_norm": 0.45 if elbow else 0.0,
                "max_row_occupancy": 0.82 if elbow else 0.35,
                "median_row_occupancy": 0.30,
            },
        }
        if green:
            metadata.update(
                {
                    "green_instruction": "VERDE_DEPOIS",
                    "green_side": "LEFT",
                    "green_marker_count": 1,
                    "green_marker_confidence": 0.98,
                    "green_relation_confidence": 0.96,
                    "green_relation_delta_y": -31.0,
                    "green_center_y": 58.0,
                    "preprocessor": {
                        "output_shape": {"width": 320, "height": 200}
                    },
                }
            )
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                green=green,
                metadata=metadata,
            ),
        )
        _drain(bus)

    try:
        adapter.start()
        publish(green=True, elbow=True)
        assert adapter.status_payload()["assist_kind"] == "green_single_confirm"

        publish(green=True, elbow=True)
        status = adapter.status_payload()
        assert status["green_single_encounter"] == "IGNORE"
        assert status["green_instruction"] == "VERDE_DEPOIS"
        assert status["corner_phase"] == "IDLE"
        assert status["assist_kind"] == "line_basic"

        # A lingering AFTER marker belongs to the already confirmed IGNORE
        # encounter. It must not stop again and start another confirmation.
        publish(green=True, elbow=True)
        status = adapter.status_payload()
        assert status["green_single_encounter"] == "IGNORE"
        assert status["corner_phase"] == "IDLE"
        assert status["assist_kind"] == "line_basic"
        assert _latest_pair(driver) != (1600, 1600)

        for _ in range(4):
            publish(green=False, elbow=True)
            assert adapter.status_payload()["corner_phase"] == "IDLE"
            assert adapter.status_payload()["green_single_encounter"] == "IGNORE"

        for _ in range(2):
            publish(green=False, elbow=False)
            assert adapter.status_payload()["green_single_encounter"] == "IGNORE"
        publish(green=False, elbow=False)
        assert adapter.status_payload()["green_single_encounter"] == "IDLE"
        assert adapter.status_payload()["corner_phase"] == "IDLE"
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_dashboard_start_arms_and_stop_disarms_real_outputs() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channel=0,
            right_channel=1,
            neutral_us=1600,
            left_base_throttle_us=300,
            right_base_throttle_us=200,
            basic_line_follow=True,
            start_disarmed=True,
            monitor_interval_ms=100_000,
        ),
        pwm_factory=lambda config: driver,
    )
    line = VisionDetectionEvent(
        state="FOLLOWING_LINE",
        line=True,
        metadata={
            "line_offset_norm": 0.0,
            "line_confidence": 1.0,
            "line_geometry": {},
        },
    )

    try:
        adapter.start()
        assert adapter.status_payload()["motor_armed"] is False
        bus.publish(EventTopic.VISION_DETECTIONS, line)
        _drain(bus)
        assert _latest_pair(driver) == (1600, 1600)

        bus.publish(
            EventTopic.UI_COMMAND,
            UICommandEvent(command="robot.start", params={}),
        )
        _drain(bus)
        assert adapter.status_payload()["motor_armed"] is True
        bus.publish(EventTopic.VISION_DETECTIONS, line)
        _drain(bus)
        assert _latest_pair(driver) != (1600, 1600)

        bus.publish(
            EventTopic.UI_COMMAND,
            UICommandEvent(command="robot.stop", params={}),
        )
        _drain(bus)
        assert adapter.status_payload()["motor_armed"] is False
        assert _latest_pair(driver) == (1600, 1600)
        bus.publish(EventTopic.VISION_DETECTIONS, line)
        _drain(bus)
        assert _latest_pair(driver) == (1600, 1600)
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_adapter_rejects_half_turn_when_both_markers_are_not_confirmed() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    clock = FakeClock()
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channel=0,
            right_channel=1,
            neutral_us=1500,
            green_trigger_streak=2,
            green_half_turn_us=300,
            green_half_turn_ms=3800,
            left_inverted=True,
            right_inverted=False,
        ),
        pwm_factory=lambda config: driver,
        monotonic=clock,
    )

    try:
        adapter.start()
        inconsistent = VisionDetectionEvent(
            state="FOLLOWING_LINE",
            line=True,
            green=True,
            metadata={
                "green_instruction": "VERDE_MEIA_VOLTA",
                "green_side": "LEFT",
                "green_marker_count": 1,
                "green_marker_confidence": 0.95,
            },
        )
        bus.publish(EventTopic.VISION_DETECTIONS, inconsistent)
        bus.publish(EventTopic.VISION_DETECTIONS, inconsistent)
        _drain(bus)

        assert _latest_pair(driver) == (1500, 1500)
        assert adapter.status_payload()["control_mode"] == "STOPPED"
        assert adapter.status_payload()["green_instruction"] == "NO_GREEN"
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_adapter_dry_run_does_not_create_driver() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(dry_run=True),
        pwm_factory=lambda config: (_ for _ in ()).throw(AssertionError("driver should not be created")),
    )

    try:
        adapter.start()
        bus.publish(EventTopic.UI_COMMAND, UICommandEvent(command="robot.forward_test", params={"duration_ms": 500}))
        _drain(bus)
        assert adapter.status_payload()["state"] == "dry-run"
        assert adapter.status_payload()["control_mode"] == "MANUAL"
    finally:
        adapter.stop()
        bus.stop()


def test_pca9685_adapter_drives_all_channels_in_each_side_group() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(
            left_channels=(4, 5),
            right_channels=(0, 1),
            neutral_us=1500,
            base_throttle_us=120,
            left_inverted=True,
            right_inverted=False,
        ),
        pwm_factory=lambda config: driver,
    )

    try:
        adapter.start()
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={"line_offset_norm": 0.0, "line_confidence": 0.9},
            ),
        )
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert latest[4] == latest[5] == 1380
        assert latest[0] == latest[1] == 1620
        assert adapter.status_payload()["left_channels"] == [4, 5]
        assert adapter.status_payload()["right_channels"] == [0, 1]
    finally:
        adapter.stop()
        bus.stop()


def test_calibrated_defaults_drive_forward_and_reverse_in_the_same_logical_direction() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(left_channels=(4, 5), right_channels=(0, 1)),
        pwm_factory=lambda config: driver,
    )

    try:
        adapter.start()
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert latest[4] == latest[5] == latest[0] == latest[1] == 1600

        bus.publish(EventTopic.UI_COMMAND, UICommandEvent(command="robot.forward_test", params={"duration_ms": 5000}))
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert latest[4] == latest[5] == 1900
        assert latest[0] == latest[1] == 1400

        bus.publish(EventTopic.UI_COMMAND, UICommandEvent(command="robot.reverse_test", params={"duration_ms": 5000}))
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert latest[4] == latest[5] == 1300
        assert latest[0] == latest[1] == 1800
    finally:
        adapter.stop()
        bus.stop()


def test_calibrated_line_following_starts_with_the_verified_straight_pulses() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    driver = FakePwmDriver()
    adapter = Pca9685RobotAdapter(
        bus,
        config=Pca9685RobotConfig(left_channels=(4, 5), right_channels=(0, 1)),
        pwm_factory=lambda config: driver,
    )

    try:
        adapter.start()
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                metadata={"line_offset_norm": 0.0, "line_confidence": 0.9},
            ),
        )
        _drain(bus)
        latest = {channel: pulse for channel, pulse in driver.pulses}
        assert latest[4] == latest[5] == 1900
        assert latest[0] == latest[1] == 1400
        assert adapter.status_payload()["left_base_throttle_us"] == 300
        assert adapter.status_payload()["right_base_throttle_us"] == 200
    finally:
        adapter.stop()
        bus.stop()
