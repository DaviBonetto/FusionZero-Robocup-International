from __future__ import annotations

import threading
import time
from typing import Any, Callable, Protocol

try:
    from ...core.event_bus import EventBus, EventTopic, LogEvent, UICommandEvent
except ImportError:  # pragma: no cover
    from core.event_bus import EventBus, EventTopic, LogEvent, UICommandEvent


class SwitchInput(Protocol):
    is_pressed: bool
    when_pressed: Callable[[], None] | None
    when_released: Callable[[], None] | None

    def close(self) -> None:
        ...


class GpioMasterSwitchController:
    """Debounced master switch between a BCM GPIO and ground.

    The physical contract is direct and fail-safe: contact closed requests
    START; contact open requests STOP.  A contact held closed during process
    startup is interlocked until opened once, so rebooting the Raspberry Pi
    cannot make the robot move unexpectedly.
    """

    def __init__(
        self,
        event_bus: EventBus,
        *,
        gpio: int = 17,
        debounce_ms: int = 20,
        input_factory: Callable[[int, int], SwitchInput] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._gpio = int(gpio)
        self._debounce_ms = max(10, int(debounce_ms))
        self._input_factory = input_factory or self._default_input_factory
        self._input: SwitchInput | None = None
        self._running = False
        self._closed = False
        self._run_requested = False
        self._release_required = False
        self._lock = threading.RLock()

    @staticmethod
    def _default_input_factory(gpio: int, debounce_ms: int) -> SwitchInput:
        try:
            from gpiozero import Button  # type: ignore
        except ImportError as exc:  # pragma: no cover - Raspberry Pi package
            raise RuntimeError("gpiozero is required for the GPIO master switch") from exc
        return Button(
            int(gpio),
            pull_up=True,
            bounce_time=max(0.01, int(debounce_ms) / 1000.0),
        )

    @property
    def gpio(self) -> int:
        return self._gpio

    @property
    def start_permitted(self) -> bool:
        with self._lock:
            return bool(
                self._running
                and self._closed
                and self._run_requested
                and not self._release_required
            )

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            switch_input = self._input_factory(self._gpio, self._debounce_ms)
            closed = bool(switch_input.is_pressed)
            self._input = switch_input
            self._closed = closed
            self._run_requested = False
            self._release_required = closed
            self._running = True
            switch_input.when_pressed = self._on_closed
            switch_input.when_released = self._on_opened

        # Startup is always stopped.  If the contact booted closed, only a
        # deliberate open -> close edge can request START.
        self._publish_master_command("system.stop", reason="gpio_switch_startup")
        if closed:
            self._publish_log(
                "WARNING",
                f"master switch GPIO{self._gpio} closed at startup; open then close to enable",
            )
        else:
            self._publish_log("INFO", f"master switch ready GPIO{self._gpio} OPEN/STOP")

    def stop(self) -> None:
        with self._lock:
            switch_input = self._input
            self._input = None
            self._running = False
            self._closed = False
            self._run_requested = False
            self._release_required = False
        if switch_input is not None:
            try:
                switch_input.when_pressed = None
                switch_input.when_released = None
                switch_input.close()
            except Exception:
                pass

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            if not self._running:
                state = "UNAVAILABLE"
            elif self._release_required:
                state = "LOCKED"
            elif self._run_requested:
                state = "RUN"
            else:
                state = "STOP"
            return {
                "enabled": bool(self._running),
                "gpio": self._gpio,
                "closed": bool(self._closed),
                "run_requested": bool(self._run_requested),
                "release_required": bool(self._release_required),
                "start_permitted": bool(
                    self._running
                    and self._closed
                    and self._run_requested
                    and not self._release_required
                ),
                "debounce_ms": self._debounce_ms,
                "state": state,
            }

    def _on_closed(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._closed = True
            locked = self._release_required
            if not locked:
                self._run_requested = True
        if locked:
            self._publish_log(
                "WARNING",
                f"master START ignored on GPIO{self._gpio}; open the switch first",
            )
            return
        self._publish_master_command("system.start", reason="gpio_switch_closed")
        self._publish_log("INFO", f"master switch GPIO{self._gpio} CLOSED/START")

    def _on_opened(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._closed = False
            was_locked = self._release_required
            self._release_required = False
            self._run_requested = False
        self._publish_master_command("system.stop", reason="gpio_switch_open")
        if was_locked:
            self._publish_log("INFO", f"master button GPIO{self._gpio} released/ready")
        else:
            self._publish_log("INFO", f"master switch GPIO{self._gpio} OPEN/STOP")

    def _publish_master_command(self, command: str, *, reason: str) -> None:
        try:
            self._event_bus.publish(
                EventTopic.UI_COMMAND,
                UICommandEvent(
                    command=str(command),
                    params={
                        "source": "gpio_master_switch",
                        "gpio": self._gpio,
                        "reason": str(reason),
                    },
                ),
            )
        except Exception as exc:
            self._publish_log("ERROR", f"failed to publish {command}: {exc}")

    def _publish_log(self, level: str, message: str) -> None:
        try:
            self._event_bus.publish(
                EventTopic.SYSTEM_LOG,
                LogEvent(
                    timestamp=time.time(),
                    level=str(level).upper(),
                    message=str(message),
                    source="gpio_master_switch",
                    state="",
                ),
            )
        except Exception:
            pass
