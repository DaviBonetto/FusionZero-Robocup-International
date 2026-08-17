from __future__ import annotations

import io
import logging
import re
import time

import pytest

from core.event_bus import EventTopic, LogEvent, StateSnapshotEvent, StateTransitionEvent
from core.state_machine import InvalidTransitionError, RobotEvent, RobotState, StateMachine


EXPECTED_TRANSITIONS: dict[RobotState, dict[RobotEvent, RobotState]] = {
    RobotState.SEARCHING_LINE: {
        RobotEvent.ON_GAP: RobotState.SEARCHING_LINE,
        RobotEvent.ON_LINE_FOUND: RobotState.FOLLOWING_LINE,
        RobotEvent.ON_LINE_LOST: RobotState.SEARCHING_LINE,
        RobotEvent.ON_VICTIM_DETECTED: RobotState.VICTIM_FOUND,
        RobotEvent.ON_RESCUE_RED_DETECTED: RobotState.RESCUE_ZONE_DETECTED,
        RobotEvent.ON_INTERSECTION: RobotState.SEARCHING_LINE,
        RobotEvent.ON_TIMEOUT: RobotState.SEARCHING_LINE,
        RobotEvent.ON_RESET: RobotState.SEARCHING_LINE,
    },
    RobotState.FOLLOWING_LINE: {
        RobotEvent.ON_GAP: RobotState.VALIDATING_GAP,
        RobotEvent.ON_LINE_FOUND: RobotState.FOLLOWING_LINE,
        RobotEvent.ON_LINE_LOST: RobotState.SEARCHING_LINE,
        RobotEvent.ON_VICTIM_DETECTED: RobotState.VICTIM_FOUND,
        RobotEvent.ON_RESCUE_RED_DETECTED: RobotState.RESCUE_ZONE_DETECTED,
        RobotEvent.ON_INTERSECTION: RobotState.FOLLOWING_LINE,
        RobotEvent.ON_TIMEOUT: RobotState.SEARCHING_LINE,
        RobotEvent.ON_RESET: RobotState.SEARCHING_LINE,
    },
    RobotState.VALIDATING_GAP: {
        RobotEvent.ON_GAP: RobotState.CROSSING_GAP,
        RobotEvent.ON_LINE_FOUND: RobotState.FOLLOWING_LINE,
        RobotEvent.ON_LINE_LOST: RobotState.SEARCHING_LINE,
        RobotEvent.ON_VICTIM_DETECTED: RobotState.VICTIM_FOUND,
        RobotEvent.ON_RESCUE_RED_DETECTED: RobotState.RESCUE_ZONE_DETECTED,
        RobotEvent.ON_INTERSECTION: RobotState.FOLLOWING_LINE,
        RobotEvent.ON_TIMEOUT: RobotState.SEARCHING_LINE,
        RobotEvent.ON_RESET: RobotState.SEARCHING_LINE,
    },
    RobotState.CROSSING_GAP: {
        RobotEvent.ON_GAP: RobotState.CROSSING_GAP,
        RobotEvent.ON_LINE_FOUND: RobotState.FOLLOWING_LINE,
        RobotEvent.ON_LINE_LOST: RobotState.CROSSING_GAP,
        RobotEvent.ON_VICTIM_DETECTED: RobotState.VICTIM_FOUND,
        RobotEvent.ON_RESCUE_RED_DETECTED: RobotState.RESCUE_ZONE_DETECTED,
        RobotEvent.ON_INTERSECTION: RobotState.CROSSING_GAP,
        RobotEvent.ON_TIMEOUT: RobotState.SEARCHING_LINE,
        RobotEvent.ON_RESET: RobotState.SEARCHING_LINE,
    },
    RobotState.VICTIM_FOUND: {
        RobotEvent.ON_GAP: RobotState.VICTIM_FOUND,
        RobotEvent.ON_LINE_FOUND: RobotState.FOLLOWING_LINE,
        RobotEvent.ON_LINE_LOST: RobotState.SEARCHING_LINE,
        RobotEvent.ON_VICTIM_DETECTED: RobotState.VICTIM_FOUND,
        RobotEvent.ON_RESCUE_RED_DETECTED: RobotState.RESCUE_ZONE_DETECTED,
        RobotEvent.ON_INTERSECTION: RobotState.VICTIM_FOUND,
        RobotEvent.ON_TIMEOUT: RobotState.FOLLOWING_LINE,
        RobotEvent.ON_RESET: RobotState.SEARCHING_LINE,
    },
    RobotState.RESCUE_ZONE_DETECTED: {
        RobotEvent.ON_GAP: RobotState.RESCUE_ZONE_DETECTED,
        RobotEvent.ON_LINE_FOUND: RobotState.FOLLOWING_LINE,
        RobotEvent.ON_LINE_LOST: RobotState.RESCUE_ZONE_DETECTED,
        RobotEvent.ON_VICTIM_DETECTED: RobotState.VICTIM_FOUND,
        RobotEvent.ON_RESCUE_RED_DETECTED: RobotState.RESCUE_ZONE_DETECTED,
        RobotEvent.ON_INTERSECTION: RobotState.RESCUE_ZONE_DETECTED,
        RobotEvent.ON_TIMEOUT: RobotState.FOLLOWING_LINE,
        RobotEvent.ON_RESET: RobotState.SEARCHING_LINE,
    },
}

TRANSITION_CASES = [
    pytest.param(start_state, trigger, expected_state, id=f"{start_state.value}-{trigger.value}")
    for start_state, transitions in EXPECTED_TRANSITIONS.items()
    for trigger, expected_state in transitions.items()
]


def wait_until(condition, *, timeout: float = 1.5, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return bool(condition())


@pytest.mark.parametrize(("start_state", "trigger", "expected_state"), TRANSITION_CASES)
def test_fsm_transitions_match_frozen_contract(
    start_state: RobotState,
    trigger: RobotEvent,
    expected_state: RobotState,
) -> None:
    machine = StateMachine(initial_state=start_state, clock=lambda: 1_700_000_000.0)
    next_state = machine.handle(trigger)
    assert next_state is expected_state


def test_fsm_alias_events_are_supported() -> None:
    machine = StateMachine(clock=lambda: 1_700_000_000.0)
    assert machine.handle(" on_victim ") is RobotState.VICTIM_FOUND
    assert machine.handle("on_reset") is RobotState.SEARCHING_LINE
    assert machine.handle("on_intersect") is RobotState.SEARCHING_LINE
    assert machine.handle("ON_RESCUE_RED") is RobotState.RESCUE_ZONE_DETECTED


def test_fsm_invalid_event_raises_immediately() -> None:
    machine = StateMachine(clock=lambda: 1_700_000_000.0)
    with pytest.raises(InvalidTransitionError, match="unknown event"):
        machine.handle("ON_DOES_NOT_EXIST")


def test_fsm_logs_follow_required_format() -> None:
    stream = io.StringIO()
    logger = logging.getLogger("tests.fsm.log-format")
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    ticks = iter((1_709_999_999.111, 1_709_999_999.150, 1_709_999_999.222))
    machine = StateMachine(clock=lambda: next(ticks), logger=logger)
    machine.handle(RobotEvent.ON_LINE_FOUND, payload={"reason": "line reacquired"})

    lines = [line.strip() for line in stream.getvalue().splitlines() if line.strip()]
    assert len(lines) >= 2
    iso_line = re.compile(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+00:00 \[[A-Z_]+\] .+$"
    )
    assert iso_line.match(lines[0])
    assert iso_line.match(lines[-1])
    assert "[FOLLOWING_LINE]" in lines[-1]
    assert "SEARCHING_LINE --ON_LINE_FOUND--> FOLLOWING_LINE" in lines[-1]
    assert "line detected; line reacquired" in lines[-1]


def test_fsm_publishes_state_transition_and_log_events(event_bus) -> None:
    transitions: list[StateTransitionEvent] = []
    states: list[StateSnapshotEvent] = []
    logs: list[LogEvent] = []

    event_bus.subscribe(EventTopic.FSM_TRANSITION, lambda event: transitions.append(event))
    event_bus.subscribe(EventTopic.FSM_STATE, lambda event: states.append(event))
    event_bus.subscribe(EventTopic.SYSTEM_LOG, lambda event: logs.append(event))

    machine = StateMachine(event_bus=event_bus, clock=lambda: 1_700_000_010.0)
    machine.handle(RobotEvent.ON_LINE_FOUND)

    assert wait_until(lambda: len(transitions) >= 1 and len(states) >= 2 and len(logs) >= 1, timeout=1.5)
    assert transitions[-1].old_state == RobotState.SEARCHING_LINE.value
    assert transitions[-1].new_state == RobotState.FOLLOWING_LINE.value
    assert transitions[-1].trigger == RobotEvent.ON_LINE_FOUND.value
    assert states[-1].state == RobotState.FOLLOWING_LINE.value
    assert "SEARCHING_LINE --ON_LINE_FOUND--> FOLLOWING_LINE" in logs[-1].message
