from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.core.event_bus import EventBus, EventTopic, UICommandEvent
from src.modules.control.gpio_master_switch import GpioMasterSwitchController


@dataclass
class FakeSwitchInput:
    is_pressed: bool = False
    when_pressed: Callable[[], None] | None = None
    when_released: Callable[[], None] | None = None
    closed: bool = False

    def press(self) -> None:
        self.is_pressed = True
        if self.when_pressed is not None:
            self.when_pressed()

    def release(self) -> None:
        self.is_pressed = False
        if self.when_released is not None:
            self.when_released()

    def close(self) -> None:
        self.closed = True


def test_master_switch_closed_starts_and_open_stops() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    switch_input = FakeSwitchInput(is_pressed=False)
    commands: list[UICommandEvent] = []
    subscription = bus.subscribe(EventTopic.UI_COMMAND, lambda event: commands.append(event))  # type: ignore[arg-type]
    controller = GpioMasterSwitchController(
        bus,
        gpio=17,
        debounce_ms=80,
        input_factory=lambda gpio, debounce_ms: switch_input,
    )
    try:
        controller.start()
        bus._queue.join()
        assert [event.command for event in commands] == ["system.stop"]
        assert controller.status_payload()["state"] == "STOP"
        assert controller.start_permitted is False

        switch_input.press()
        bus._queue.join()
        assert commands[-1].command == "system.start"
        assert commands[-1].params["source"] == "gpio_master_switch"
        assert controller.status_payload()["state"] == "RUN"
        assert controller.start_permitted is True

        switch_input.release()
        bus._queue.join()
        assert commands[-1].command == "system.stop"
        assert controller.status_payload()["state"] == "STOP"
        assert controller.start_permitted is False

        switch_input.press()
        bus._queue.join()
        assert commands[-1].command == "system.start"
        assert controller.status_payload()["state"] == "RUN"

        switch_input.release()
        bus._queue.join()
        assert commands[-1].command == "system.stop"
    finally:
        controller.stop()
        subscription.unsubscribe()
        bus.stop()
    assert switch_input.closed is True


def test_master_switch_closed_at_boot_requires_open_then_close() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    switch_input = FakeSwitchInput(is_pressed=True)
    commands: list[UICommandEvent] = []
    subscription = bus.subscribe(EventTopic.UI_COMMAND, lambda event: commands.append(event))  # type: ignore[arg-type]
    controller = GpioMasterSwitchController(
        bus,
        gpio=17,
        input_factory=lambda gpio, debounce_ms: switch_input,
    )
    try:
        controller.start()
        bus._queue.join()
        assert [event.command for event in commands] == ["system.stop"]
        assert controller.status_payload()["state"] == "LOCKED"
        assert controller.start_permitted is False

        switch_input.press()
        bus._queue.join()
        assert [event.command for event in commands] == ["system.stop"]

        switch_input.release()
        bus._queue.join()
        assert [event.command for event in commands] == ["system.stop", "system.stop"]
        assert controller.status_payload()["state"] == "STOP"

        switch_input.press()
        bus._queue.join()
        assert commands[-1].command == "system.start"
        assert controller.status_payload()["state"] == "RUN"
    finally:
        controller.stop()
        subscription.unsubscribe()
        bus.stop()
