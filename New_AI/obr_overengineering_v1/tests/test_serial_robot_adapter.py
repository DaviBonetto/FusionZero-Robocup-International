from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from conftest import wait_until
from src.core.event_bus import EventBus, EventTopic, LogEvent, UICommandEvent, VisionDetectionEvent
from src.modules.control.serial_robot_adapter import RobotSerialConfig, SerialRobotAdapter


@dataclass
class FakeSerial:
    port: str
    scripted_reads: list[list[str]] = field(default_factory=list)
    writes: list[bytes] = field(default_factory=list)
    flushed: int = 0
    closed: bool = False

    def __post_init__(self) -> None:
        self._read_queue: deque[bytes] = deque()

    def write(self, payload: bytes) -> int:
        self.writes.append(payload)
        if self.scripted_reads:
            for line in self.scripted_reads.pop(0):
                self._read_queue.append(f"{line}\n".encode("ascii"))
        return len(payload)

    def flush(self) -> None:
        self.flushed += 1

    def readline(self) -> bytes:
        if self._read_queue:
            return self._read_queue.popleft()
        return b""

    def close(self) -> None:
        self.closed = True


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


def _drain(bus: EventBus) -> None:
    bus._queue.join()


def test_robot_adapter_triggers_green_assist_on_green_streak() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    clock = FakeClock()
    transport = FakeSerial(
        port="/dev/ttyACM0",
        scripted_reads=[
            ["READY FUSIONZERO", "PONG"],
            ["ACK ASST GREEN"],
            ["ACK STOP"],
        ],
    )
    adapter = SerialRobotAdapter(
        bus,
        config=RobotSerialConfig(
            port="/dev/ttyACM0",
            green_trigger_streak=2,
            green_cooldown_ms=6000,
            green_hold_ms=900,
            heartbeat_interval_ms=60000,
            reconnect_interval_ms=0,
        ),
        serial_factory=lambda port, baud: transport,
        monotonic=clock,
    )
    adapter.start()

    received_logs: list[LogEvent] = []
    sub = bus.subscribe(EventTopic.SYSTEM_LOG, lambda event: received_logs.append(event))  # type: ignore[arg-type]

    try:
        for _ in range(2):
            bus.publish(
                EventTopic.VISION_DETECTIONS,
                VisionDetectionEvent(
                    state="FOLLOWING_LINE",
                    green=True,
                    metadata={"green_instruction": "VERDE DEPOIS", "green_side": "LEFT"},
                ),
            )
            _drain(bus)

        assert transport.writes[:2] == [
            b"PING\n",
            b"ASST GREEN found=1 instruction=VERDE_DEPOIS side=LEFT conf=1.000 hold_ms=900 source=vision\n",
        ]
        assert any("ASST GREEN" in event.message for event in received_logs)
    finally:
        sub.unsubscribe()
        adapter.stop()
        bus.stop()


def test_robot_adapter_sends_line_assist_from_follow_line_metadata() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    clock = FakeClock()
    transport = FakeSerial(
        port="/dev/ttyACM0",
        scripted_reads=[
            ["PONG"],
            ["ACK ASST LINE"],
            ["ACK STOP"],
        ],
    )
    adapter = SerialRobotAdapter(
        bus,
        config=RobotSerialConfig(
            port="/dev/ttyACM0",
            green_trigger_streak=1,
            heartbeat_interval_ms=60000,
            reconnect_interval_ms=0,
        ),
        serial_factory=lambda port, baud: transport,
        monotonic=clock,
    )
    adapter.start()
    try:
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(
                state="FOLLOWING_LINE",
                line=True,
                green=False,
                metadata={
                    "line_offset_norm": 0.25,
                    "line_angle_deg": 108.0,
                    "line_confidence": 0.84,
                    "line_gap_frames": 0,
                },
            ),
        )
        _drain(bus)
        assert [payload for payload in transport.writes if payload.startswith(b"ASST LINE")] == [
            b"ASST LINE found=1 offset=0.250 angle=108.000 conf=0.840 gap=0 source=vision\n",
        ]
    finally:
        adapter.stop()
        bus.stop()


def test_robot_adapter_respects_green_cooldown() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    clock = FakeClock()
    transport = FakeSerial(
        port="/dev/ttyACM0",
        scripted_reads=[
            ["PONG"],
            ["ACK ASST GREEN"],
            ["ACK ASST GREEN"],
            ["ACK STOP"],
        ],
    )
    adapter = SerialRobotAdapter(
        bus,
        config=RobotSerialConfig(
            port="/dev/ttyACM0",
            green_trigger_streak=1,
            green_cooldown_ms=6000,
            green_hold_ms=900,
            heartbeat_interval_ms=60000,
            reconnect_interval_ms=0,
        ),
        serial_factory=lambda port, baud: transport,
        monotonic=clock,
    )
    adapter.start()
    try:
        event = VisionDetectionEvent(
            state="FOLLOWING_LINE",
            green=True,
            metadata={"green_instruction": "VERDE ANTES", "green_side": "RIGHT"},
        )
        bus.publish(EventTopic.VISION_DETECTIONS, event)
        _drain(bus)
        bus.publish(EventTopic.VISION_DETECTIONS, event)
        _drain(bus)
        assert [payload for payload in transport.writes if payload.startswith(b"ASST GREEN")] == [
            b"ASST GREEN found=1 instruction=VERDE_ANTES side=RIGHT conf=1.000 hold_ms=900 source=vision\n"
        ]

        clock.advance(6.1)
        bus.publish(EventTopic.VISION_DETECTIONS, event)
        _drain(bus)
        assert [payload for payload in transport.writes if payload.startswith(b"ASST GREEN")] == [
            b"ASST GREEN found=1 instruction=VERDE_ANTES side=RIGHT conf=1.000 hold_ms=900 source=vision\n",
            b"ASST GREEN found=1 instruction=VERDE_ANTES side=RIGHT conf=1.000 hold_ms=900 source=vision\n",
        ]
    finally:
        adapter.stop()
        bus.stop()


def test_robot_adapter_accepts_manual_ui_commands_force_stop_and_obstacle_triggers() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    clock = FakeClock()
    transport = FakeSerial(
        port="/dev/ttyACM0",
        scripted_reads=[
            ["PONG"],
            ["ACK FORWARD 1200"],
            ["ACK STOP"],
            ["ACK ESTOP"],
            ["ACK ASST OBSTACLE"],
            ["ACK ASST OBSTACLE"],
            ["ACK STOP"],
        ],
    )
    adapter = SerialRobotAdapter(
        bus,
        config=RobotSerialConfig(
            port="/dev/ttyACM0",
            green_trigger_streak=1,
            heartbeat_interval_ms=60000,
            reconnect_interval_ms=0,
        ),
        serial_factory=lambda port, baud: transport,
        monotonic=clock,
    )
    adapter.start()
    try:
        bus.publish(
            EventTopic.UI_COMMAND,
            UICommandEvent(command="robot.forward_test", params={"duration_ms": 1200}),
        )
        _drain(bus)
        bus.publish(EventTopic.UI_COMMAND, UICommandEvent(command="robot.stop", params={}))
        _drain(bus)
        bus.publish(EventTopic.UI_COMMAND, UICommandEvent(command="robot.force_stop", params={}))
        _drain(bus)
        bus.publish(EventTopic.UI_COMMAND, UICommandEvent(command="robot.obstacle_test", params={}))
        _drain(bus)
        bus.publish(EventTopic.UI_COMMAND, UICommandEvent(command="robot.obstacle_clear", params={}))
        _drain(bus)

        assert transport.writes[:6] == [
            b"PING\n",
            b"CMD FORWARD 1200\n",
            b"CMD STOP 0\n",
            b"CMD ESTOP 0\n",
            b"ASST OBSTACLE state=TEST conf=1.000 hold_ms=1200 source=dashboard\n",
            b"ASST OBSTACLE state=CLEAR conf=1.000 hold_ms=1200 source=dashboard\n",
        ]
    finally:
        adapter.stop()
        bus.stop()


def test_robot_adapter_retries_after_ack_timeout() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    transport = FakeSerial(
        port="/dev/ttyACM0",
        scripted_reads=[
            ["PONG"],
            [],
            ["PONG"],
            ["ACK ASST LINE"],
            ["ACK STOP"],
        ],
    )
    adapter = SerialRobotAdapter(
        bus,
        config=RobotSerialConfig(
            port="/dev/ttyACM0",
            ack_timeout_ms=20,
            max_retries=1,
            heartbeat_interval_ms=60000,
            reconnect_interval_ms=0,
        ),
        serial_factory=lambda port, baud: transport,
    )
    adapter.start()

    logs: list[LogEvent] = []
    sub = bus.subscribe(EventTopic.SYSTEM_LOG, lambda event: logs.append(event))  # type: ignore[arg-type]
    try:
        assert adapter.send_line_assist(
            offset_norm=0.18,
            angle_deg=96.0,
            confidence=0.62,
            gap_frames=0,
            reason="retry_test",
            state="MANUAL",
        ) is True
        _drain(bus)
        assert transport.writes[:4] == [
            b"PING\n",
            b"ASST LINE found=1 offset=0.180 angle=96.000 conf=0.620 gap=0 source=vision\n",
            b"PING\n",
            b"ASST LINE found=1 offset=0.180 angle=96.000 conf=0.620 gap=0 source=vision\n",
        ]
        assert any("retry 2/2 line=ASST LINE" in event.message for event in logs)
    finally:
        sub.unsubscribe()
        adapter.stop()
        bus.stop()


def test_robot_adapter_heartbeat_marks_link_unhealthy() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    transport = FakeSerial(
        port="/dev/ttyACM0",
        scripted_reads=[
            ["PONG"],
            [],
        ],
    )
    adapter = SerialRobotAdapter(
        bus,
        config=RobotSerialConfig(
            port="/dev/ttyACM0",
            ack_timeout_ms=20,
            max_retries=0,
            heartbeat_interval_ms=50,
            reconnect_interval_ms=10000,
        ),
        serial_factory=lambda port, baud: transport,
    )
    logs: list[LogEvent] = []
    sub = bus.subscribe(EventTopic.SYSTEM_LOG, lambda event: logs.append(event))  # type: ignore[arg-type]
    adapter.start()
    try:
        assert wait_until(lambda: any("robot heartbeat lost" in event.message for event in logs), timeout=0.6)
        assert transport.closed is True
    finally:
        sub.unsubscribe()
        adapter.stop()
        bus.stop()


def test_robot_adapter_preserves_dry_run_without_touching_serial() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    adapter = SerialRobotAdapter(
        bus,
        config=RobotSerialConfig(
            auto_detect=False,
            dry_run=True,
        ),
        serial_factory=lambda port, baud: (_ for _ in ()).throw(AssertionError("serial factory should not run")),
    )
    adapter.start()
    try:
        bus.publish(
            EventTopic.UI_COMMAND,
            UICommandEvent(command="robot.forward_test", params={"duration_ms": 750}),
        )
        _drain(bus)
        bus.publish(EventTopic.UI_COMMAND, UICommandEvent(command="robot.force_stop", params={}))
        _drain(bus)
    finally:
        adapter.stop()
        bus.stop()


def test_robot_adapter_tracks_telemetry_from_serial_lines() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    transport = FakeSerial(
        port="/dev/ttyACM0",
        scripted_reads=[
            ["TLM mode=FOLLOW_LINE line_error=-0.100 pid=14.500 front=330 failsafe=0", "PONG"],
            ["TLM mode=FOLLOW_LINE line_error=-0.070 pid=11.250 front=315 obstacle=AHEAD green=VERDE_DEPOIS", "ACK ASST LINE"],
            ["ACK STOP"],
        ],
    )
    adapter = SerialRobotAdapter(
        bus,
        config=RobotSerialConfig(
            port="/dev/ttyACM0",
            heartbeat_interval_ms=60000,
            reconnect_interval_ms=0,
        ),
        serial_factory=lambda port, baud: transport,
    )
    adapter.start()
    try:
        assert adapter.send_line_assist(
            offset_norm=-0.07,
            angle_deg=88.0,
            confidence=0.71,
            gap_frames=0,
            reason="telemetry_test",
            state="FOLLOWING_LINE",
        )
        _drain(bus)
        status = adapter.status_payload()
        telemetry = status["telemetry"]
        assert telemetry["mode"] == "FOLLOW_LINE"
        assert telemetry["line_error"] == -0.07
        assert telemetry["pid"] == 11.25
        assert telemetry["front"] == 315
        assert telemetry["obstacle"] == "AHEAD"
        assert telemetry["green"] == "VERDE_DEPOIS"
        assert status["state"] == "connected"
    finally:
        adapter.stop()
        bus.stop()
