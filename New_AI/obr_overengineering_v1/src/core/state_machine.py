from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

try:
    from .event_bus import EventBus, EventBusError, EventTopic, LogEvent, StateSnapshotEvent, StateTransitionEvent
except ImportError:  # pragma: no cover
    from event_bus import EventBus, EventBusError, EventTopic, LogEvent, StateSnapshotEvent, StateTransitionEvent


class RobotState(str, Enum):
    SEARCHING_LINE = "SEARCHING_LINE"
    FOLLOWING_LINE = "FOLLOWING_LINE"
    VALIDATING_GAP = "VALIDATING_GAP"
    CROSSING_GAP = "CROSSING_GAP"
    VICTIM_FOUND = "VICTIM_FOUND"
    RESCUE_ZONE_DETECTED = "RESCUE_ZONE_DETECTED"


class RobotEvent(str, Enum):
    ON_GAP = "ON_GAP"
    ON_LINE_FOUND = "ON_LINE_FOUND"
    ON_LINE_LOST = "ON_LINE_LOST"
    ON_VICTIM_DETECTED = "ON_VICTIM_DETECTED"
    ON_RESCUE_RED_DETECTED = "ON_RESCUE_RED_DETECTED"
    ON_INTERSECTION = "ON_INTERSECTION"
    ON_TIMEOUT = "ON_TIMEOUT"
    ON_RESET = "ON_RESET"

    @classmethod
    def coerce(cls, value: RobotEvent | str) -> RobotEvent:
        if isinstance(value, cls):
            return value
        normalized = value.strip().upper()
        normalized = ROBOT_EVENT_ALIASES.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError as exc:
            raise InvalidTransitionError(f"unknown event '{value}'") from exc


@dataclass(frozen=True, slots=True)
class TransitionRule:
    next_state: RobotState
    reason: str


class InvalidTransitionError(ValueError):
    pass


ROBOT_EVENT_ALIASES: dict[str, str] = {
    "ON_VICTIM": "ON_VICTIM_DETECTED",
    "ON_INTERSECT": "ON_INTERSECTION",
    "ON_RESCUE_RED": "ON_RESCUE_RED_DETECTED",
}


TRANSITION_TABLE: dict[RobotState, dict[RobotEvent, TransitionRule]] = {
    RobotState.SEARCHING_LINE: {
        RobotEvent.ON_GAP: TransitionRule(RobotState.SEARCHING_LINE, "gap ignored while line not locked"),
        RobotEvent.ON_LINE_FOUND: TransitionRule(RobotState.FOLLOWING_LINE, "line detected"),
        RobotEvent.ON_LINE_LOST: TransitionRule(RobotState.SEARCHING_LINE, "still searching for line"),
        RobotEvent.ON_VICTIM_DETECTED: TransitionRule(RobotState.VICTIM_FOUND, "victim detected while scanning"),
        RobotEvent.ON_RESCUE_RED_DETECTED: TransitionRule(
            RobotState.RESCUE_ZONE_DETECTED,
            "rescue red marker detected",
        ),
        RobotEvent.ON_INTERSECTION: TransitionRule(
            RobotState.SEARCHING_LINE,
            "intersection ignored until line lock",
        ),
        RobotEvent.ON_TIMEOUT: TransitionRule(RobotState.SEARCHING_LINE, "search timeout fallback"),
        RobotEvent.ON_RESET: TransitionRule(RobotState.SEARCHING_LINE, "manual/system reset"),
    },
    RobotState.FOLLOWING_LINE: {
        RobotEvent.ON_GAP: TransitionRule(RobotState.VALIDATING_GAP, "possible gap detected"),
        RobotEvent.ON_LINE_FOUND: TransitionRule(RobotState.FOLLOWING_LINE, "line tracking stable"),
        RobotEvent.ON_LINE_LOST: TransitionRule(RobotState.SEARCHING_LINE, "line lost"),
        RobotEvent.ON_VICTIM_DETECTED: TransitionRule(RobotState.VICTIM_FOUND, "victim detected on route"),
        RobotEvent.ON_RESCUE_RED_DETECTED: TransitionRule(
            RobotState.RESCUE_ZONE_DETECTED,
            "rescue zone red marker detected",
        ),
        RobotEvent.ON_INTERSECTION: TransitionRule(
            RobotState.FOLLOWING_LINE,
            "intersection handled by navigation policy",
        ),
        RobotEvent.ON_TIMEOUT: TransitionRule(RobotState.SEARCHING_LINE, "following timeout fallback"),
        RobotEvent.ON_RESET: TransitionRule(RobotState.SEARCHING_LINE, "manual/system reset"),
    },
    RobotState.VALIDATING_GAP: {
        RobotEvent.ON_GAP: TransitionRule(RobotState.CROSSING_GAP, "gap confirmed"),
        RobotEvent.ON_LINE_FOUND: TransitionRule(RobotState.FOLLOWING_LINE, "false positive gap"),
        RobotEvent.ON_LINE_LOST: TransitionRule(RobotState.SEARCHING_LINE, "line not recovered during validation"),
        RobotEvent.ON_VICTIM_DETECTED: TransitionRule(RobotState.VICTIM_FOUND, "victim detected during validation"),
        RobotEvent.ON_RESCUE_RED_DETECTED: TransitionRule(
            RobotState.RESCUE_ZONE_DETECTED,
            "rescue marker detected during validation",
        ),
        RobotEvent.ON_INTERSECTION: TransitionRule(RobotState.FOLLOWING_LINE, "intersection overrides gap validation"),
        RobotEvent.ON_TIMEOUT: TransitionRule(RobotState.SEARCHING_LINE, "validation timeout fallback"),
        RobotEvent.ON_RESET: TransitionRule(RobotState.SEARCHING_LINE, "manual/system reset"),
    },
    RobotState.CROSSING_GAP: {
        RobotEvent.ON_GAP: TransitionRule(RobotState.CROSSING_GAP, "still crossing gap"),
        RobotEvent.ON_LINE_FOUND: TransitionRule(RobotState.FOLLOWING_LINE, "line reacquired after crossing"),
        RobotEvent.ON_LINE_LOST: TransitionRule(RobotState.CROSSING_GAP, "expected line loss while crossing"),
        RobotEvent.ON_VICTIM_DETECTED: TransitionRule(RobotState.VICTIM_FOUND, "victim detected during crossing"),
        RobotEvent.ON_RESCUE_RED_DETECTED: TransitionRule(
            RobotState.RESCUE_ZONE_DETECTED,
            "rescue marker detected during crossing",
        ),
        RobotEvent.ON_INTERSECTION: TransitionRule(RobotState.CROSSING_GAP, "intersection ignored during crossing"),
        RobotEvent.ON_TIMEOUT: TransitionRule(RobotState.SEARCHING_LINE, "crossing timeout fallback"),
        RobotEvent.ON_RESET: TransitionRule(RobotState.SEARCHING_LINE, "manual/system reset"),
    },
    RobotState.VICTIM_FOUND: {
        RobotEvent.ON_GAP: TransitionRule(RobotState.VICTIM_FOUND, "gap event ignored during victim routine"),
        RobotEvent.ON_LINE_FOUND: TransitionRule(RobotState.FOLLOWING_LINE, "victim routine complete; line found"),
        RobotEvent.ON_LINE_LOST: TransitionRule(RobotState.SEARCHING_LINE, "victim routine complete; line lost"),
        RobotEvent.ON_VICTIM_DETECTED: TransitionRule(RobotState.VICTIM_FOUND, "victim confirmation"),
        RobotEvent.ON_RESCUE_RED_DETECTED: TransitionRule(
            RobotState.RESCUE_ZONE_DETECTED,
            "rescue marker detected after victim",
        ),
        RobotEvent.ON_INTERSECTION: TransitionRule(RobotState.VICTIM_FOUND, "intersection ignored during victim routine"),
        RobotEvent.ON_TIMEOUT: TransitionRule(RobotState.FOLLOWING_LINE, "victim routine timeout returns to line"),
        RobotEvent.ON_RESET: TransitionRule(RobotState.SEARCHING_LINE, "manual/system reset"),
    },
    RobotState.RESCUE_ZONE_DETECTED: {
        RobotEvent.ON_GAP: TransitionRule(RobotState.RESCUE_ZONE_DETECTED, "gap ignored inside rescue zone"),
        RobotEvent.ON_LINE_FOUND: TransitionRule(RobotState.FOLLOWING_LINE, "line exit from rescue zone"),
        RobotEvent.ON_LINE_LOST: TransitionRule(RobotState.RESCUE_ZONE_DETECTED, "continue rescue-zone behavior"),
        RobotEvent.ON_VICTIM_DETECTED: TransitionRule(RobotState.VICTIM_FOUND, "victim detected in rescue zone"),
        RobotEvent.ON_RESCUE_RED_DETECTED: TransitionRule(
            RobotState.RESCUE_ZONE_DETECTED,
            "rescue marker confirmation",
        ),
        RobotEvent.ON_INTERSECTION: TransitionRule(RobotState.RESCUE_ZONE_DETECTED, "intersection ignored in rescue zone"),
        RobotEvent.ON_TIMEOUT: TransitionRule(RobotState.FOLLOWING_LINE, "rescue-zone timeout returns to line"),
        RobotEvent.ON_RESET: TransitionRule(RobotState.SEARCHING_LINE, "manual/system reset"),
    },
}


def _validate_transition_table() -> None:
    expected_events = set(RobotEvent)
    for state in RobotState:
        if state not in TRANSITION_TABLE:
            raise RuntimeError(f"missing transition map for state '{state.value}'")
        state_map = TRANSITION_TABLE[state]
        missing = expected_events - set(state_map)
        extra = set(state_map) - expected_events
        if missing:
            names = ", ".join(sorted(event.value for event in missing))
            raise RuntimeError(f"state '{state.value}' missing events: {names}")
        if extra:
            names = ", ".join(sorted(event.value for event in extra))
            raise RuntimeError(f"state '{state.value}' has unsupported events: {names}")


_validate_transition_table()


class StateMachine:
    def __init__(
        self,
        *,
        initial_state: RobotState = RobotState.SEARCHING_LINE,
        event_bus: EventBus | None = None,
        clock: Callable[[], float] = time.time,
        logger: logging.Logger | None = None,
    ) -> None:
        self._state = initial_state
        self._event_bus = event_bus
        self._clock = clock
        self._lock = threading.RLock()
        self._logger = logger or self._build_default_logger()
        self._log_state_line(self._state, "fsm initialized")
        self._publish_state_snapshot(self._clock())

    @property
    def state(self) -> RobotState:
        with self._lock:
            return self._state

    def handle(self, event: RobotEvent | str, payload: dict[str, Any] | None = None) -> RobotState:
        resolved_event = RobotEvent.coerce(event)
        with self._lock:
            old_state = self._state
            rule = TRANSITION_TABLE[old_state][resolved_event]
            new_state = rule.next_state
            timestamp = self._clock()
            reason = self._merge_reason(rule.reason, payload)
            self._state = new_state

        self._emit_transition(timestamp, old_state, new_state, resolved_event, reason)
        return new_state

    def reset(self, payload: dict[str, Any] | None = None) -> RobotState:
        return self.handle(RobotEvent.ON_RESET, payload)

    def allowed_events(self, state: RobotState | None = None) -> tuple[RobotEvent, ...]:
        selected_state = state or self.state
        return tuple(TRANSITION_TABLE[selected_state].keys())

    def transition_table(self) -> dict[RobotState, dict[RobotEvent, RobotState]]:
        return {
            state: {event: rule.next_state for event, rule in events.items()}
            for state, events in TRANSITION_TABLE.items()
        }

    def _emit_transition(
        self,
        timestamp: float,
        old_state: RobotState,
        new_state: RobotState,
        trigger: RobotEvent,
        reason: str,
    ) -> None:
        log_message = f"{old_state.value} --{trigger.value}--> {new_state.value} | {reason}"
        self._log_state_line(new_state, log_message, timestamp=timestamp)

        transition_event = StateTransitionEvent(
            timestamp=timestamp,
            old_state=old_state.value,
            new_state=new_state.value,
            trigger=trigger.value,
            reason=reason,
        )
        state_event = StateSnapshotEvent(timestamp=timestamp, state=new_state.value)
        system_log_event = LogEvent(
            timestamp=timestamp,
            level="INFO",
            message=log_message,
            source="state_machine",
            state=new_state.value,
        )

        self._safe_publish(EventTopic.FSM_TRANSITION, transition_event)
        self._safe_publish(EventTopic.FSM_STATE, state_event)
        self._safe_publish(EventTopic.SYSTEM_LOG, system_log_event)

    def _publish_state_snapshot(self, timestamp: float) -> None:
        self._safe_publish(
            EventTopic.FSM_STATE,
            StateSnapshotEvent(timestamp=timestamp, state=self.state.value),
        )

    def _safe_publish(self, topic: EventTopic, message: StateSnapshotEvent | StateTransitionEvent | LogEvent) -> None:
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish(topic, message)
        except EventBusError as exc:
            self._log_state_line(self.state, f"event bus publish failed: {exc}")

    def _log_state_line(self, state: RobotState, message: str, *, timestamp: float | None = None) -> None:
        timestamp = self._clock() if timestamp is None else timestamp
        iso_timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="milliseconds")
        self._logger.info("%s [%s] %s", iso_timestamp, state.value, message)

    @staticmethod
    def _merge_reason(default_reason: str, payload: dict[str, Any] | None) -> str:
        if not payload:
            return default_reason
        reason = payload.get("reason")
        if reason:
            return f"{default_reason}; {reason}"
        return default_reason

    @staticmethod
    def _build_default_logger() -> logging.Logger:
        logger = logging.getLogger("core.state_machine")
        if logger.handlers:
            return logger
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
