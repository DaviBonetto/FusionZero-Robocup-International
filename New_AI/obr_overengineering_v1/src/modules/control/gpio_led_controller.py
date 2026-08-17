from __future__ import annotations

import threading
import time
from typing import Any, Callable, Mapping, Protocol

try:
    from ...core.event_bus import EventBus, EventTopic, LogEvent, UICommandEvent
except ImportError:  # pragma: no cover
    from core.event_bus import EventBus, EventTopic, LogEvent, UICommandEvent


class LedOutput(Protocol):
    is_active: bool

    def on(self) -> None:
        ...

    def off(self) -> None:
        ...

    def close(self) -> None:
        ...


class GpioLedController:
    """Own two optional status LEDs and expose only explicit UI commands.

    The outputs start LOW and are never enabled automatically.  Each physical
    LED must have a suitable series resistor before using ``leds.on``.
    """

    def __init__(
        self,
        event_bus: EventBus,
        *,
        led1_gpio: int = 18,
        led2_gpio: int = 23,
        output_factory: Callable[[int], LedOutput] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._pins = (int(led1_gpio), int(led2_gpio))
        self._output_factory = output_factory or self._default_output_factory
        self._outputs: list[LedOutput] = []
        self._states = [False, False]
        self._running = False
        self._lock = threading.RLock()
        self._ui_sub: Any = None

    @staticmethod
    def _default_output_factory(pin: int) -> LedOutput:
        try:
            from gpiozero import OutputDevice  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on Pi packages
            raise RuntimeError("gpiozero is required for the LED outputs") from exc
        return OutputDevice(pin, active_high=True, initial_value=False)

    @property
    def pins(self) -> tuple[int, int]:
        return self._pins

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            created: list[LedOutput] = []
            try:
                for pin in self._pins:
                    output = self._output_factory(pin)
                    output.off()
                    created.append(output)
            except Exception:
                for output in created:
                    try:
                        output.off()
                        output.close()
                    except Exception:
                        pass
                raise
            self._outputs = created
            self._states = [False, False]
            self._running = True
            self._ui_sub = self._event_bus.subscribe(EventTopic.UI_COMMAND, self._on_ui_command)
        self._publish_log("INFO", f"LED outputs ready gpio={self._pins[0]},{self._pins[1]}")

    def stop(self) -> None:
        with self._lock:
            if not self._running and not self._outputs:
                return
            self._running = False
            subscription = self._ui_sub
            self._ui_sub = None
            outputs = list(self._outputs)
            self._outputs = []
            self._states = [False, False]
        if subscription is not None:
            try:
                subscription.unsubscribe()
            except Exception:
                pass
        for output in outputs:
            try:
                output.off()
                output.close()
            except Exception:
                pass

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": bool(self._running),
                "pins": list(self._pins),
                "led1": bool(self._states[0]),
                "led2": bool(self._states[1]),
            }

    def _on_ui_command(self, event: UICommandEvent) -> None:
        if not self._running or not isinstance(event, UICommandEvent):
            return
        command = str(event.command or "").strip().lower()
        params = event.params if isinstance(event.params, Mapping) else {}
        if command in {"leds.on", "leds.off", "leds.toggle"}:
            action = command.rsplit(".", 1)[-1]
            self._set_all(action)
            return
        if command in {"led1.on", "led1.off", "led1.toggle", "led2.on", "led2.off", "led2.toggle"}:
            led_index = 0 if command.startswith("led1.") else 1
            action = command.rsplit(".", 1)[-1]
            self._set_one(led_index, action)
            return
        if command == "leds.set":
            self._set_one(0, "on" if bool(params.get("led1", False)) else "off")
            self._set_one(1, "on" if bool(params.get("led2", False)) else "off")

    def _set_all(self, action: str) -> None:
        for index in range(2):
            self._set_one(index, action)

    def _set_one(self, index: int, action: str) -> None:
        with self._lock:
            if not self._running or index not in (0, 1) or len(self._outputs) != 2:
                return
            if action == "toggle":
                enabled = not self._states[index]
            else:
                enabled = action == "on"
            output = self._outputs[index]
            if enabled:
                output.on()
            else:
                output.off()
            self._states[index] = enabled
        self._publish_log("INFO", f"LED{index + 1} {'ON' if enabled else 'OFF'}")

    def _publish_log(self, level: str, message: str) -> None:
        try:
            self._event_bus.publish(
                EventTopic.SYSTEM_LOG,
                LogEvent(
                    timestamp=time.time(),
                    level=str(level).upper(),
                    message=str(message),
                    source="gpio_led_controller",
                    state="",
                ),
            )
        except Exception:
            pass
