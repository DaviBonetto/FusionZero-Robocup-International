from __future__ import annotations

import glob
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

try:
    import serial as pyserial
except ImportError:  # pragma: no cover
    pyserial = None

try:
    from ...core.event_bus import EventBus, EventTopic, LogEvent, UICommandEvent, VisionDetectionEvent
    from ...core.state_machine import RobotState
    from .robot_link_protocol import (
        GreenAssist,
        LineAssist,
        ObstacleAssist,
        decode_telemetry_line,
        encode_green_assist,
        encode_line_assist,
        encode_obstacle_assist,
    )
except ImportError:  # pragma: no cover
    from core.event_bus import EventBus, EventTopic, LogEvent, UICommandEvent, VisionDetectionEvent
    from core.state_machine import RobotState
    from modules.control.robot_link_protocol import (
        GreenAssist,
        LineAssist,
        ObstacleAssist,
        decode_telemetry_line,
        encode_green_assist,
        encode_line_assist,
        encode_obstacle_assist,
    )


class SerialTransport(Protocol):
    @property
    def port(self) -> str:
        ...

    def write(self, payload: bytes) -> int:
        ...

    def flush(self) -> None:
        ...

    def readline(self) -> bytes:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True, slots=True)
class ProtocolExpectation:
    exact: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()

    def matches(self, response: str) -> bool:
        return response in self.exact or any(response.startswith(prefix) for prefix in self.prefixes)

    def describe(self) -> str:
        parts = [*self.exact, *self.prefixes]
        return " or ".join(parts) if parts else "<any-response>"


@dataclass(slots=True)
class RobotSerialConfig:
    port: str | None = None
    baud_rate: int = 115200
    green_forward_ms: int = 5000
    green_trigger_streak: int = 2
    green_cooldown_ms: int = 6000
    green_hold_ms: int = 900
    obstacle_hold_ms: int = 1200
    min_send_interval_ms: int = 120
    assist_refresh_ms: int = 280
    assist_delta_offset: float = 0.06
    assist_delta_angle_deg: float = 6.0
    line_confidence_floor: float = 0.18
    ack_timeout_ms: int = 250
    max_retries: int = 2
    heartbeat_interval_ms: int = 400
    reconnect_interval_ms: int = 800
    connect_probe_timeout_ms: int = 2500
    status_log_interval_ms: int = 2500
    telemetry_stale_ms: int = 1200
    auto_detect: bool = True
    dry_run: bool = False


class SerialRobotAdapter:
    AUTO_PORT_PATTERNS = (
        "/dev/serial/by-id/*",
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
        "COM*",
    )

    _PONG = ProtocolExpectation(exact=("PONG",))
    _ACK_STOP = ProtocolExpectation(exact=("ACK STOP",), prefixes=("ACK STOP ",))
    _ACK_ESTOP = ProtocolExpectation(exact=("ACK ESTOP",), prefixes=("ACK ESTOP ",))
    _ACK_RESET_ESTOP = ProtocolExpectation(exact=("ACK RESET_ESTOP",), prefixes=("ACK RESET_ESTOP ",))
    _ACK_LINE = ProtocolExpectation(exact=("ACK ASST LINE",), prefixes=("ACK ASST LINE ",))
    _ACK_GREEN = ProtocolExpectation(exact=("ACK ASST GREEN",), prefixes=("ACK ASST GREEN ",))
    _ACK_OBSTACLE = ProtocolExpectation(exact=("ACK ASST OBSTACLE",), prefixes=("ACK ASST OBSTACLE ",))

    def __init__(
        self,
        event_bus: EventBus,
        *,
        config: RobotSerialConfig,
        serial_factory: Callable[[str, int], SerialTransport] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._event_bus = event_bus
        self._config = config
        self._serial_factory = serial_factory or self._default_serial_factory
        self._monotonic = monotonic

        self._transport: SerialTransport | None = None
        self._resolved_port: str | None = None
        self._last_send_at = 0.0
        self._last_green_trigger_at = 0.0
        self._last_protocol_ok_at = 0.0
        self._last_connect_attempt_at = 0.0
        self._green_streak = 0
        self._running = False
        self._heartbeat_fault_active = False
        self._io_lock = threading.RLock()
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._throttled_log_at: dict[str, float] = {}

        self._last_line_assist_sent_at = 0.0
        self._last_line_assist: LineAssist | None = None
        self._last_green_assist: GreenAssist | None = None
        self._last_obstacle_assist: ObstacleAssist | None = None
        self._last_assist_kind = "none"
        self._last_telemetry: dict[str, Any] = {}
        self._last_telemetry_at = 0.0
        self._last_ack_line = ""
        self._last_ack_at = 0.0
        self._last_event_line = ""
        self._last_event_at = 0.0
        self._failsafe_active = False
        self._last_obstacle_state = "CLEAR"
        self._last_green_instruction = "NO_GREEN"

        self._detection_sub = self._event_bus.subscribe(EventTopic.VISION_DETECTIONS, self._on_detection)
        self._ui_sub = self._event_bus.subscribe(EventTopic.UI_COMMAND, self._on_ui_command)

    @property
    def enabled(self) -> bool:
        return bool(self._config.dry_run or self._config.port or self._config.auto_detect)

    @property
    def connected(self) -> bool:
        return self._transport is not None

    @property
    def resolved_port(self) -> str | None:
        return self._resolved_port

    def start(self) -> None:
        self._running = True
        if not self.enabled:
            self._publish_log("INFO", "robot serial adapter disabled")
            return

        if self._config.dry_run:
            self._publish_log("INFO", "robot serial adapter running in dry-run mode")
            return

        self._ensure_transport()
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="robot-serial-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=1.5)
        self._monitor_thread = None

        try:
            self.request_safe_stop(reason="adapter_shutdown", state="SHUTDOWN", emergency=False, force=True)
        except Exception:
            pass

        self._running = False
        try:
            self._detection_sub.unsubscribe()
        except Exception:
            pass
        try:
            self._ui_sub.unsubscribe()
        except Exception:
            pass
        self._close_transport()

    def send_forward(self, *, duration_ms: int, reason: str, state: str) -> bool:
        duration_ms = max(1, int(duration_ms))
        return self._dispatch_request(
            line=f"CMD FORWARD {duration_ms}",
            expectation=ProtocolExpectation(exact=(f"ACK FORWARD {duration_ms}",)),
            reason=reason,
            state=state,
            record_send=True,
            bypass_rate_limit=False,
            retries=None,
            label="command",
            failure_hint="motion command was not acknowledged",
        )

    def send_line_assist(
        self,
        *,
        offset_norm: float,
        angle_deg: float,
        confidence: float,
        gap_frames: int,
        reason: str,
        state: str,
        found: bool = True,
    ) -> bool:
        payload = LineAssist(
            found=bool(found),
            offset_norm=float(offset_norm),
            angle_deg=float(angle_deg),
            confidence=float(confidence),
            gap_frames=int(gap_frames),
            source="vision",
        )
        success = self._dispatch_request(
            line=encode_line_assist(payload),
            expectation=self._ACK_LINE,
            reason=reason,
            state=state,
            record_send=True,
            bypass_rate_limit=False,
            retries=None,
            label="assist",
            failure_hint="arduino will continue local line following without this assist",
        )
        if success:
            self._last_line_assist = payload
            self._last_line_assist_sent_at = self._monotonic()
            self._last_assist_kind = "line"
        return success

    def send_green_assist(
        self,
        *,
        instruction: str,
        side: str,
        confidence: float,
        hold_ms: int,
        reason: str,
        state: str,
        bypass_rate_limit: bool = False,
    ) -> bool:
        payload = GreenAssist(
            found=True,
            instruction=instruction,
            side=side,
            confidence=float(confidence),
            hold_ms=int(hold_ms),
            source="vision",
        )
        success = self._dispatch_request(
            line=encode_green_assist(payload),
            expectation=self._ACK_GREEN,
            reason=reason,
            state=state,
            record_send=True,
            bypass_rate_limit=bypass_rate_limit,
            retries=None,
            label="assist",
            failure_hint="arduino will ignore the maneuver hint and remain on local control",
        )
        if success:
            self._last_green_assist = payload
            self._last_green_instruction = str(payload.instruction).replace(" ", "_").upper()
            self._last_assist_kind = "green"
        return success

    def send_obstacle_assist(
        self,
        *,
        obstacle_state: str,
        confidence: float,
        hold_ms: int,
        source: str,
        reason: str,
        state: str,
        bypass_rate_limit: bool = False,
    ) -> bool:
        payload = ObstacleAssist(
            state=obstacle_state,
            confidence=float(confidence),
            hold_ms=int(hold_ms),
            source=source,
        )
        success = self._dispatch_request(
            line=encode_obstacle_assist(payload),
            expectation=self._ACK_OBSTACLE,
            reason=reason,
            state=state,
            record_send=True,
            bypass_rate_limit=bypass_rate_limit,
            retries=None,
            label="assist",
            failure_hint="arduino will stay on local obstacle handling and failsafe rules",
        )
        if success:
            self._last_obstacle_assist = payload
            self._last_obstacle_state = str(payload.state).replace(" ", "_").upper()
            self._last_assist_kind = "obstacle"
        return success

    def send_stop(self, *, reason: str, state: str, force: bool = True) -> bool:
        return self._dispatch_request(
            line="CMD STOP 0",
            expectation=self._ACK_STOP,
            reason=reason,
            state=state,
            record_send=False,
            bypass_rate_limit=force,
            retries=None,
            label="safe-stop",
            failure_hint="arduino watchdog should stop motors if the robot was moving",
        )

    def send_estop(self, *, reason: str, state: str, force: bool = True) -> bool:
        return self._dispatch_request(
            line="CMD ESTOP 0",
            expectation=self._ACK_ESTOP,
            reason=reason,
            state=state,
            record_send=False,
            bypass_rate_limit=force,
            retries=None,
            label="emergency-stop",
            failure_hint="arduino watchdog should stop motors if the robot was moving",
        )

    def clear_estop(self, *, reason: str, state: str) -> bool:
        return self._dispatch_request(
            line="CMD RESET_ESTOP 0",
            expectation=self._ACK_RESET_ESTOP,
            reason=reason,
            state=state,
            record_send=False,
            bypass_rate_limit=True,
            retries=None,
            label="command",
            failure_hint="ESTOP latch remains active until this command is acknowledged",
        )

    def request_safe_stop(self, *, reason: str, state: str, emergency: bool, force: bool = True) -> bool:
        if emergency:
            return self.send_estop(reason=reason, state=state, force=force)
        return self.send_stop(reason=reason, state=state, force=force)

    def latest_telemetry(self) -> dict[str, Any]:
        return dict(self._last_telemetry)

    def status_payload(self) -> dict[str, Any]:
        now = self._monotonic()
        heartbeat_age_ms = (
            None if self._last_protocol_ok_at <= 0 else int(max(0.0, (now - self._last_protocol_ok_at) * 1000.0))
        )
        telemetry_age_ms = (
            None if self._last_telemetry_at <= 0 else int(max(0.0, (now - self._last_telemetry_at) * 1000.0))
        )
        telemetry_stale_ms = max(0, int(self._config.telemetry_stale_ms))
        telemetry_fresh = telemetry_age_ms is not None and telemetry_age_ms <= telemetry_stale_ms

        if self._config.dry_run:
            state = "dry-run"
        elif self._heartbeat_fault_active:
            state = "error"
        elif self.connected:
            state = "connected"
        elif self.enabled:
            state = "waiting"
        else:
            state = "disabled"

        telemetry = dict(self._last_telemetry) if telemetry_fresh else {}
        control_mode = str(telemetry.get("mode", telemetry.get("controller_state", ""))).strip().upper()
        pid_output = telemetry.get("pid", telemetry.get("pid_output"))
        line_error = telemetry.get("line_error", telemetry.get("error"))
        failsafe = bool(telemetry.get("failsafe", False) or self._failsafe_active)

        return {
            "state": state,
            "connected": bool(self.connected),
            "port": str(self._resolved_port or ""),
            "heartbeat_ok": not self._heartbeat_fault_active,
            "heartbeat_age_ms": heartbeat_age_ms,
            "telemetry_age_ms": telemetry_age_ms,
            "ack": self._last_ack_line,
            "ack_age_ms": None if self._last_ack_at <= 0 else int(max(0.0, (now - self._last_ack_at) * 1000.0)),
            "event": self._last_event_line,
            "event_age_ms": None if self._last_event_at <= 0 else int(max(0.0, (now - self._last_event_at) * 1000.0)),
            "assist_kind": self._last_assist_kind,
            "control_mode": control_mode,
            "line_error": line_error,
            "pid_output": pid_output,
            "obstacle_state": str(telemetry.get("obstacle", self._last_obstacle_state)).strip().upper(),
            "green_instruction": str(telemetry.get("green", self._last_green_instruction)).strip().upper(),
            "failsafe": failsafe,
            "telemetry": telemetry,
        }

    def _on_detection(self, event: VisionDetectionEvent) -> None:
        if not self._running or not isinstance(event, VisionDetectionEvent):
            return

        metadata = event.metadata if isinstance(event.metadata, Mapping) else {}
        self._maybe_send_obstacle_assist_from_detection(event.state, metadata)

        if event.state != RobotState.FOLLOWING_LINE.value:
            self._green_streak = 0
            return

        self._maybe_send_line_assist(event, metadata)
        self._maybe_send_green_assist(event, metadata)

    def _maybe_send_line_assist(self, event: VisionDetectionEvent, metadata: Mapping[str, Any]) -> None:
        gap_frames = max(0, int(metadata.get("line_gap_frames", 0)))

        if not bool(event.line):
            if self._last_line_assist is not None and self._last_line_assist.found:
                self.send_line_assist(
                    offset_norm=0.0,
                    angle_deg=float(metadata.get("line_angle_deg", 90.0)),
                    confidence=0.0,
                    gap_frames=gap_frames,
                    reason="vision_line_lost",
                    state=event.state,
                    found=False,
                )
            return

        confidence = float(metadata.get("line_confidence", 0.0))
        if confidence < float(self._config.line_confidence_floor):
            return

        payload = LineAssist(
            found=True,
            offset_norm=float(metadata.get("line_offset_norm", 0.0)),
            angle_deg=float(metadata.get("line_angle_deg", 90.0)),
            confidence=confidence,
            gap_frames=gap_frames,
            source="vision",
        )
        if not self._should_refresh_line_assist(payload):
            return

        self.send_line_assist(
            offset_norm=payload.offset_norm,
            angle_deg=payload.angle_deg,
            confidence=payload.confidence,
            gap_frames=payload.gap_frames,
            reason="vision_line_follow",
            state=event.state,
            found=payload.found,
        )

    def _maybe_send_green_assist(self, event: VisionDetectionEvent, metadata: Mapping[str, Any]) -> None:
        if not bool(event.green):
            self._green_streak = 0
            return

        instruction = str(metadata.get("green_instruction", "NO GREEN")).strip().upper()
        side = str(metadata.get("green_side", "NONE")).strip().upper()
        if instruction == "NO GREEN" or side == "NONE":
            self._green_streak = 0
            return

        self._green_streak += 1
        if self._green_streak < max(1, int(self._config.green_trigger_streak)):
            return

        now = self._monotonic()
        cooldown_s = max(0.0, float(self._config.green_cooldown_ms) / 1000.0)
        if (now - self._last_green_trigger_at) < cooldown_s:
            return

        confidence = float(metadata.get("green_marker_confidence", 1.0) or 1.0)
        success = self.send_green_assist(
            instruction=instruction,
            side=side,
            confidence=confidence,
            hold_ms=int(self._config.green_hold_ms),
            reason=f"green_detected instruction={instruction} side={side}",
            state=event.state,
            bypass_rate_limit=True,
        )
        if success:
            self._last_green_trigger_at = now
            self._green_streak = 0

    def _maybe_send_obstacle_assist_from_detection(self, state: str, metadata: Mapping[str, Any]) -> None:
        raw = metadata.get("obstacle")
        if isinstance(raw, Mapping):
            obstacle_state = str(raw.get("state", "CLEAR")).strip().upper()
            confidence = float(raw.get("confidence", 0.0) or 0.0)
            source = str(raw.get("source", "vision")).strip().lower() or "vision"
        else:
            obstacle_state = str(metadata.get("obstacle_state", "")).strip().upper()
            confidence = float(metadata.get("obstacle_confidence", 0.0) or 0.0)
            source = "vision"

        if not obstacle_state:
            return

        now = self._monotonic()
        refresh_s = max(0.05, float(self._config.assist_refresh_ms) / 1000.0)
        if obstacle_state == self._last_obstacle_state and self._last_ack_at > 0 and (now - self._last_ack_at) < refresh_s:
            return

        self.send_obstacle_assist(
            obstacle_state=obstacle_state,
            confidence=confidence,
            hold_ms=int(self._config.obstacle_hold_ms),
            source=source,
            reason=f"vision_obstacle state={obstacle_state}",
            state=state,
            bypass_rate_limit=True,
        )

    def _should_refresh_line_assist(self, payload: LineAssist) -> bool:
        now = self._monotonic()
        refresh_s = max(0.05, float(self._config.assist_refresh_ms) / 1000.0)
        previous = self._last_line_assist
        if previous is None:
            return True
        if (now - self._last_line_assist_sent_at) >= refresh_s:
            return True
        if previous.found != payload.found:
            return True
        if abs(float(previous.offset_norm) - float(payload.offset_norm)) >= float(self._config.assist_delta_offset):
            return True
        if abs(float(previous.angle_deg) - float(payload.angle_deg)) >= float(self._config.assist_delta_angle_deg):
            return True
        if abs(float(previous.confidence) - float(payload.confidence)) >= 0.12:
            return True
        if int(previous.gap_frames) != int(payload.gap_frames):
            return True
        return False

    def _on_ui_command(self, event: UICommandEvent) -> None:
        if not self._running or not isinstance(event, UICommandEvent):
            return

        command = str(event.command or "").strip().lower()
        params = event.params if isinstance(event.params, Mapping) else {}

        if command == "robot.forward_test":
            duration_ms = int(params.get("duration_ms", self._config.green_forward_ms))
            self.send_forward(duration_ms=duration_ms, reason="ui_manual_forward", state="MANUAL")
            return

        if command == "robot.stop":
            self.send_stop(reason="ui_manual_stop", state="MANUAL")
            return

        if command in {"robot.force_stop", "robot.estop"}:
            self.send_estop(reason="ui_force_stop", state="MANUAL")
            return

        if command in {"robot.clear_estop", "robot.reset_estop"}:
            self.clear_estop(reason="ui_clear_estop", state="MANUAL")
            return

        if command in {"robot.obstacle_test", "robot.obstacle_ahead"}:
            obstacle_state = "TEST" if command == "robot.obstacle_test" else "AHEAD"
            self.send_obstacle_assist(
                obstacle_state=obstacle_state,
                confidence=float(params.get("confidence", 1.0) or 1.0),
                hold_ms=int(params.get("hold_ms", self._config.obstacle_hold_ms)),
                source="dashboard",
                reason=f"ui_obstacle {obstacle_state.lower()}",
                state="MANUAL",
                bypass_rate_limit=True,
            )
            return

        if command in {"robot.obstacle_clear", "robot.clear_obstacle"}:
            self.send_obstacle_assist(
                obstacle_state="CLEAR",
                confidence=float(params.get("confidence", 1.0) or 1.0),
                hold_ms=int(params.get("hold_ms", self._config.obstacle_hold_ms)),
                source="dashboard",
                reason="ui_obstacle clear",
                state="MANUAL",
                bypass_rate_limit=True,
            )

    def _dispatch_request(
        self,
        *,
        line: str,
        expectation: ProtocolExpectation,
        reason: str,
        state: str,
        record_send: bool,
        bypass_rate_limit: bool,
        retries: int | None,
        label: str,
        failure_hint: str,
        log_success: bool = True,
    ) -> bool:
        if not self.enabled:
            return False

        line = str(line).strip()
        if not line:
            return False

        now = self._monotonic()
        if record_send and not bypass_rate_limit and not self._rate_limit_ok(now):
            return False

        if self._config.dry_run:
            if record_send:
                self._last_send_at = now
            self._last_protocol_ok_at = now
            self._last_ack_line = "DRY_RUN"
            self._last_ack_at = now
            if log_success and label != "heartbeat":
                self._publish_log("INFO", f"robot dry-run -> {line} ({reason})", state=state)
            return True

        attempts = 1 + max(0, int(self._config.max_retries if retries is None else retries))
        last_error = "transport_unavailable"
        port_name = self._resolved_port or ""

        for attempt in range(1, attempts + 1):
            if not self._ensure_transport(force=True):
                last_error = "transport_unavailable"
                continue

            port_name = self._resolved_port or port_name
            try:
                response = self._exchange_once(line=line, expectation=expectation)
                now = self._monotonic()
                self._last_protocol_ok_at = now
                self._last_ack_line = response
                self._last_ack_at = now
                if record_send:
                    self._last_send_at = now
                if label == "heartbeat" and self._heartbeat_fault_active:
                    self._heartbeat_fault_active = False
                    self._publish_log("INFO", f"robot heartbeat restored port={port_name}")
                elif log_success and label != "heartbeat":
                    self._publish_log(
                        "INFO",
                        f"robot {label} -> {line} ack={response} port={port_name} attempts={attempt} ({reason})",
                        state=state,
                    )
                return True
            except Exception as exc:
                last_error = str(exc)
                self._close_transport()
                if label != "heartbeat" and attempt < attempts:
                    self._publish_log(
                        "WARNING",
                        f"robot {label} retry {attempt + 1}/{attempts} line={line} because {last_error}",
                        state=state,
                    )

        if label == "heartbeat":
            if not self._heartbeat_fault_active:
                self._heartbeat_fault_active = True
                self._publish_log(
                    "ERROR",
                    (
                        f"robot heartbeat lost error={last_error}; awaiting reconnect. "
                        "Arduino watchdog should stop motors if the robot was moving"
                    ),
                )
            return False

        message = f"robot {label} failed after {attempts} attempt(s): line={line} error={last_error}"
        if failure_hint:
            message = f"{message}; {failure_hint}"
        level = "ERROR" if "stop" in label else "WARNING"
        self._publish_log(level, message, state=state)
        return False

    def _exchange_once(self, *, line: str, expectation: ProtocolExpectation) -> str:
        with self._io_lock:
            transport = self._transport
            if transport is None:
                raise RuntimeError("serial transport unavailable")

            payload = f"{line}\n".encode("ascii")
            transport.write(payload)
            transport.flush()
            return self._wait_for_expected_response(expectation)

    def _wait_for_expected_response(self, expectation: ProtocolExpectation) -> str:
        deadline = self._monotonic() + max(0.05, float(self._config.ack_timeout_ms) / 1000.0)
        last_response = ""

        while self._monotonic() < deadline:
            transport = self._transport
            if transport is None:
                raise RuntimeError("serial transport unavailable")

            raw = transport.readline()
            if not raw:
                continue

            response = raw.decode("ascii", errors="ignore").strip()
            if not response:
                continue

            last_response = response
            if response == "READY FUSIONZERO":
                self._publish_log("INFO", f"robot serial ready port={self._resolved_port or transport.port}")
                continue

            if response.startswith("TLM ") or response.startswith("STAT "):
                self._handle_telemetry_response(response)
                continue

            if response.startswith("EVENT "):
                self._handle_event_response(response)
                continue

            if response.startswith("ERR "):
                raise RuntimeError(response)

            if expectation.matches(response):
                return response

        if last_response:
            raise TimeoutError(f"timeout waiting for {expectation.describe()} (last={last_response})")
        raise TimeoutError(f"timeout waiting for {expectation.describe()}")

    def _handle_telemetry_response(self, response: str) -> None:
        telemetry = decode_telemetry_line(response)
        if not telemetry:
            return
        self._last_telemetry = telemetry
        self._last_telemetry_at = self._monotonic()
        self._failsafe_active = bool(telemetry.get("failsafe", False))
        if "obstacle" in telemetry:
            self._last_obstacle_state = str(telemetry["obstacle"]).strip().upper()
        if "green" in telemetry:
            self._last_green_instruction = str(telemetry["green"]).strip().upper()

    def _handle_event_response(self, response: str) -> None:
        self._last_event_line = response
        self._last_event_at = self._monotonic()
        if "WATCHDOG_STOP" in response.upper():
            self._failsafe_active = True
        self._publish_log("WARNING", f"robot serial event -> {response}")

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.wait(0.05):
            if not self._running or not self.enabled or self._config.dry_run:
                continue

            if self._transport is None:
                self._ensure_transport()
                continue

            now = self._monotonic()
            heartbeat_interval_s = max(0.1, float(self._config.heartbeat_interval_ms) / 1000.0)
            if (now - self._last_protocol_ok_at) < heartbeat_interval_s:
                continue

            self._dispatch_request(
                line="PING",
                expectation=self._PONG,
                reason="heartbeat",
                state="",
                record_send=False,
                bypass_rate_limit=True,
                retries=0,
                label="heartbeat",
                failure_hint="arduino watchdog should stop motors if the robot was moving",
                log_success=False,
            )

    def _rate_limit_ok(self, now: float | None = None) -> bool:
        now = self._monotonic() if now is None else now
        min_interval_s = max(0.0, float(self._config.min_send_interval_ms) / 1000.0)
        return (now - self._last_send_at) >= min_interval_s

    def _ensure_transport(self, *, force: bool = False) -> bool:
        if self._config.dry_run:
            return True

        with self._io_lock:
            if self._transport is not None:
                return True

            now = self._monotonic()
            reconnect_interval_s = max(0.0, float(self._config.reconnect_interval_ms) / 1000.0)
            if not force and (now - self._last_connect_attempt_at) < reconnect_interval_s:
                return False
            self._last_connect_attempt_at = now

            resolved = self._resolve_port()
            if not resolved:
                self._publish_log_throttled("serial_port_missing", "WARNING", "robot serial port not found")
                return False

            transport: SerialTransport | None = None
            try:
                transport = self._serial_factory(resolved, int(self._config.baud_rate))
                self._transport = transport
                self._resolved_port = resolved
                self._probe_transport()
                self._last_protocol_ok_at = self._monotonic()
                self._heartbeat_fault_active = False
                self._failsafe_active = False
                self._publish_log("INFO", f"robot serial connected port={resolved} baud={self._config.baud_rate}")
                return True
            except Exception as exc:
                self._resolved_port = resolved
                self._transport = None
                if transport is not None:
                    try:
                        transport.close()
                    except Exception:
                        pass
                self._publish_log_throttled(
                    "serial_open_failed",
                    "ERROR",
                    f"robot serial open failed on {resolved}: {exc}",
                )
                return False

    def _probe_transport(self) -> None:
        deadline = self._monotonic() + max(0.2, float(self._config.connect_probe_timeout_ms) / 1000.0)
        last_error = "timeout waiting for PONG"

        while self._monotonic() < deadline:
            try:
                self._exchange_once(line="PING", expectation=self._PONG)
                return
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.1)

        raise TimeoutError(f"probe handshake failed: {last_error}")

    def _resolve_port(self) -> str | None:
        explicit = (self._config.port or "").strip()
        if explicit:
            return explicit
        if not self._config.auto_detect:
            return None

        for pattern in self.AUTO_PORT_PATTERNS:
            matches = sorted(glob.glob(pattern))
            for candidate in matches:
                if Path(candidate).name.startswith("ttyS"):
                    continue
                return candidate
        return None

    def _close_transport(self) -> None:
        with self._io_lock:
            transport = self._transport
            self._transport = None
        if transport is None:
            return
        try:
            transport.close()
        except Exception:
            pass

    def _publish_log_throttled(self, key: str, level: str, message: str, *, state: str = "") -> None:
        now = self._monotonic()
        interval_s = max(0.2, float(self._config.status_log_interval_ms) / 1000.0)
        if (now - self._throttled_log_at.get(key, 0.0)) < interval_s:
            return
        self._throttled_log_at[key] = now
        self._publish_log(level, message, state=state)

    def _publish_log(self, level: str, message: str, *, state: str = "") -> None:
        try:
            self._event_bus.publish(
                EventTopic.SYSTEM_LOG,
                LogEvent(
                    level=str(level).upper(),
                    message=message,
                    source="robot_serial_adapter",
                    state=state,
                ),
            )
        except Exception:
            pass

    @staticmethod
    def _default_serial_factory(port: str, baud_rate: int) -> SerialTransport:
        if pyserial is None:
            raise RuntimeError("pyserial not installed")
        return pyserial.Serial(
            port=port,
            baudrate=int(baud_rate),
            timeout=0.05,
            write_timeout=0.2,
        )
