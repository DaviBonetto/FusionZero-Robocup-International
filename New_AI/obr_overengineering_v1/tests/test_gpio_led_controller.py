from __future__ import annotations

from dataclasses import dataclass

from src.core.event_bus import EventBus, EventTopic, UICommandEvent
from src.modules.control.gpio_led_controller import GpioLedController


@dataclass
class FakeOutput:
    on_calls: int = 0
    off_calls: int = 0
    closed: bool = False

    def on(self) -> None:
        self.on_calls += 1

    def off(self) -> None:
        self.off_calls += 1

    def close(self) -> None:
        self.closed = True


def test_led_controller_starts_low_and_handles_dashboard_commands() -> None:
    bus = EventBus(max_queue_size=64, drop_oldest=False)
    outputs: list[FakeOutput] = []

    def factory(pin: int) -> FakeOutput:
        del pin
        output = FakeOutput()
        outputs.append(output)
        return output

    controller = GpioLedController(bus, led1_gpio=18, led2_gpio=23, output_factory=factory)
    try:
        controller.start()
        assert len(outputs) == 2
        assert controller.status_payload()["led1"] is False
        assert controller.status_payload()["led2"] is False

        bus.publish(EventTopic.UI_COMMAND, UICommandEvent(command="leds.on", params={}))
        bus._queue.join()
        assert controller.status_payload()["led1"] is True
        assert controller.status_payload()["led2"] is True

        bus.publish(EventTopic.UI_COMMAND, UICommandEvent(command="led1.toggle", params={}))
        bus._queue.join()
        assert controller.status_payload()["led1"] is False
        assert controller.status_payload()["led2"] is True

        bus.publish(EventTopic.UI_COMMAND, UICommandEvent(command="leds.off", params={}))
        bus._queue.join()
        assert controller.status_payload()["led1"] is False
        assert controller.status_payload()["led2"] is False
    finally:
        controller.stop()
        assert all(output.closed for output in outputs)
        bus.stop()
