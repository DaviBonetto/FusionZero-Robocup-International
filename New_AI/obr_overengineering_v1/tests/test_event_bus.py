from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pytest

from core.event_bus import (
    BaseEvent,
    EventBus,
    EventBusError,
    EventBusFullError,
    EventTopic,
    LogEvent,
)


@dataclass(slots=True)
class SequenceEvent(BaseEvent):
    seq: int = 0
    publisher: int = 0


def wait_until(condition, *, timeout: float = 1.5, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return bool(condition())


def test_event_bus_preserves_order_for_single_publisher() -> None:
    bus = EventBus(max_queue_size=128)
    bus.register_topic("test.sequence", (SequenceEvent,))
    received: list[int] = []
    try:
        bus.subscribe("test.sequence", lambda event: received.append(event.seq))
        for idx in range(80):
            bus.publish("test.sequence", SequenceEvent(seq=idx))
        assert wait_until(lambda: len(received) == 80, timeout=1.5)
        assert received == list(range(80))
    finally:
        bus.stop()


def test_event_bus_isolates_subscriber_exceptions() -> None:
    bus = EventBus(max_queue_size=32)
    received: list[str] = []
    try:
        def broken_handler(_event: LogEvent) -> None:
            raise RuntimeError("simulated handler failure")

        def healthy_handler(event: LogEvent) -> None:
            received.append(event.message)

        bus.subscribe(EventTopic.SYSTEM_LOG, broken_handler)
        bus.subscribe(EventTopic.SYSTEM_LOG, healthy_handler)
        bus.publish(EventTopic.SYSTEM_LOG, LogEvent(message="ok", source="test"))

        assert wait_until(lambda: len(received) == 1, timeout=1.5)
        assert received == ["ok"]
    finally:
        bus.stop()


def test_event_bus_supports_concurrent_publishers_without_message_loss() -> None:
    bus = EventBus(max_queue_size=2048)
    bus.register_topic("test.concurrent", (SequenceEvent,))
    total_publishers = 4
    per_publisher = 120
    expected = total_publishers * per_publisher

    lock = threading.Lock()
    received: set[tuple[int, int]] = set()

    def handler(event: SequenceEvent) -> None:
        with lock:
            received.add((event.publisher, event.seq))

    def publisher(pid: int) -> None:
        for seq in range(per_publisher):
            bus.publish(
                "test.concurrent",
                SequenceEvent(seq=seq, publisher=pid),
                block=True,
                timeout=0.5,
            )

    try:
        bus.subscribe("test.concurrent", handler)
        threads = [threading.Thread(target=publisher, args=(pid,), daemon=True) for pid in range(total_publishers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3.0)
            assert not thread.is_alive()

        assert wait_until(lambda: len(received) == expected, timeout=2.0)
        assert len(received) == expected
    finally:
        bus.stop()


def test_event_bus_reports_queue_full_when_not_dropping_oldest() -> None:
    bus = EventBus(max_queue_size=4, drop_oldest=False)
    full_errors = 0
    try:
        def slow_subscriber(_event: LogEvent) -> None:
            time.sleep(0.02)

        bus.subscribe(EventTopic.SYSTEM_LOG, slow_subscriber)
        for idx in range(300):
            try:
                bus.publish(
                    EventTopic.SYSTEM_LOG,
                    LogEvent(message=f"log-{idx}", source="test"),
                    block=False,
                )
            except EventBusFullError:
                full_errors += 1

        assert full_errors > 0
    finally:
        bus.stop()


def test_event_bus_drop_oldest_policy_avoids_full_queue_errors() -> None:
    bus = EventBus(max_queue_size=4, drop_oldest=True)
    full_errors = 0
    try:
        def slow_subscriber(_event: LogEvent) -> None:
            time.sleep(0.02)

        bus.subscribe(EventTopic.SYSTEM_LOG, slow_subscriber)
        for idx in range(300):
            try:
                bus.publish(
                    EventTopic.SYSTEM_LOG,
                    LogEvent(message=f"log-{idx}", source="test"),
                    block=False,
                )
            except EventBusFullError:
                full_errors += 1

        assert full_errors == 0
    finally:
        bus.stop()


def test_event_bus_publish_after_stop_raises_error() -> None:
    bus = EventBus(max_queue_size=8)
    bus.stop()
    with pytest.raises(EventBusError, match="event bus is stopped"):
        bus.publish(EventTopic.SYSTEM_LOG, LogEvent(message="late", source="test"))
