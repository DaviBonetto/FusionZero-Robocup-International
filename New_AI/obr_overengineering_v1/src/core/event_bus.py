from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, cast


@dataclass(slots=True)
class BaseEvent:
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class FrameEvent(BaseEvent):
    frame_id: int = 0
    width: int = 0
    height: int = 0
    encoding: str = "bgr8"
    data: bytes = b""


@dataclass(slots=True)
class StateSnapshotEvent(BaseEvent):
    state: str = ""


@dataclass(slots=True)
class StateTransitionEvent(BaseEvent):
    old_state: str = ""
    new_state: str = ""
    trigger: str = ""
    reason: str = ""


@dataclass(slots=True)
class VisionDetectionEvent(BaseEvent):
    state: str = ""
    line: bool = False
    balls: int = 0
    green: bool = False
    red: bool = False
    victims: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PoseEvent(BaseEvent):
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


@dataclass(slots=True)
class PathEvent(BaseEvent):
    poses: list[PoseEvent] = field(default_factory=list)


@dataclass(slots=True)
class UICommandEvent(BaseEvent):
    command: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HealthEvent(BaseEvent):
    cpu_percent: float = 0.0
    fps_capture: float = 0.0
    fps_process: float = 0.0
    fps_ui: float = 0.0
    queue_depth: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LogEvent(BaseEvent):
    level: str = "INFO"
    message: str = ""
    source: str = ""
    state: str = ""


class EventTopic(str, Enum):
    VISION_RAW_FRAME = "vision.raw_frame"
    VISION_PROCESSED_FRAME = "vision.processed_frame"
    VISION_DETECTIONS = "vision.detections"
    FSM_STATE = "fsm.state"
    FSM_TRANSITION = "fsm.transition"
    NAV_POSE = "nav.pose"
    NAV_PATH = "nav.path"
    UI_COMMAND = "ui.command"
    SYSTEM_HEALTH = "system.health"
    SYSTEM_LOG = "system.log"


class EventBusError(RuntimeError):
    pass


class EventBusFullError(EventBusError):
    pass


class TopicTypeError(EventBusError):
    pass


EventHandler = Callable[[BaseEvent], None]


@dataclass(slots=True)
class Subscription:
    _bus: "EventBus"
    topic: str
    token: int
    _active: bool = True

    def unsubscribe(self) -> None:
        if not self._active:
            return
        self._bus.unsubscribe(self)
        self._active = False


@dataclass(frozen=True, slots=True)
class _QueuedMessage:
    topic: str
    message: BaseEvent


class EventBus:
    DEFAULT_TOPIC_TYPES: dict[str, tuple[type[BaseEvent], ...]] = {
        EventTopic.VISION_RAW_FRAME.value: (FrameEvent,),
        EventTopic.VISION_PROCESSED_FRAME.value: (FrameEvent,),
        EventTopic.VISION_DETECTIONS.value: (VisionDetectionEvent,),
        EventTopic.FSM_STATE.value: (StateSnapshotEvent,),
        EventTopic.FSM_TRANSITION.value: (StateTransitionEvent,),
        EventTopic.NAV_POSE.value: (PoseEvent,),
        EventTopic.NAV_PATH.value: (PathEvent,),
        EventTopic.UI_COMMAND.value: (UICommandEvent,),
        EventTopic.SYSTEM_HEALTH.value: (HealthEvent,),
        EventTopic.SYSTEM_LOG.value: (LogEvent,),
    }

    def __init__(self, max_queue_size: int = 512, drop_oldest: bool = False) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")

        self._queue: queue.Queue[object] = queue.Queue(maxsize=max_queue_size)
        self._drop_oldest = drop_oldest
        self._subscribers: dict[str, dict[int, EventHandler]] = {}
        self._topic_types: dict[str, tuple[type[BaseEvent], ...]] = dict(self.DEFAULT_TOPIC_TYPES)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._dispatcher: threading.Thread | None = None
        self._subscription_ids = itertools.count(1)
        self._stop_sentinel = object()
        self._logger = logging.getLogger("core.event_bus")
        self._running = False
        self.start()

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def max_queue_size(self) -> int:
        return self._queue.maxsize

    def register_topic(self, topic: str | EventTopic, payload_types: tuple[type[BaseEvent], ...]) -> None:
        topic_name = self._normalize_topic(topic)
        if not payload_types:
            raise ValueError("payload_types cannot be empty")
        with self._lock:
            self._topic_types[topic_name] = payload_types

    def subscribe(self, topic: str | EventTopic, handler: EventHandler) -> Subscription:
        topic_name = self._normalize_topic(topic)
        with self._lock:
            token = next(self._subscription_ids)
            self._subscribers.setdefault(topic_name, {})[token] = handler
        return Subscription(_bus=self, topic=topic_name, token=token)

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            handlers = self._subscribers.get(subscription.topic)
            if not handlers:
                return
            handlers.pop(subscription.token, None)
            if not handlers:
                self._subscribers.pop(subscription.topic, None)

    def publish(
        self,
        topic: str | EventTopic,
        message: BaseEvent,
        *,
        block: bool = False,
        timeout: float | None = None,
    ) -> None:
        topic_name = self._normalize_topic(topic)
        self._validate_message_type(topic_name, message)
        queued_message = _QueuedMessage(topic=topic_name, message=message)

        if not self._running:
            raise EventBusError("event bus is stopped")

        try:
            if block:
                if timeout is None:
                    self._queue.put(queued_message, block=True)
                else:
                    self._queue.put(queued_message, block=True, timeout=timeout)
            else:
                self._queue.put_nowait(queued_message)
        except queue.Full as exc:
            if self._drop_oldest:
                self._drop_oldest_message()
                try:
                    self._queue.put_nowait(queued_message)
                    return
                except queue.Full:
                    pass
            raise EventBusFullError(
                f"queue is full ({self.queue_depth}/{self.max_queue_size}) for topic '{topic_name}'"
            ) from exc

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._stop_event.clear()
            self._dispatcher = threading.Thread(
                target=self._dispatch_loop,
                name="event-bus-dispatcher",
                daemon=True,
            )
            self._running = True
            self._dispatcher.start()

    def stop(self, timeout: float = 2.0) -> None:
        dispatcher: threading.Thread | None
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()
            dispatcher = self._dispatcher
            try:
                self._queue.put_nowait(self._stop_sentinel)
            except queue.Full:
                if self._drop_oldest:
                    self._drop_oldest_message()
                    self._queue.put_nowait(self._stop_sentinel)
        if dispatcher is not None:
            dispatcher.join(timeout=timeout)

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if item is self._stop_sentinel:
                    return

                queued = cast(_QueuedMessage, item)
                with self._lock:
                    handlers = list(self._subscribers.get(queued.topic, {}).values())

                for handler in handlers:
                    try:
                        handler(queued.message)
                    except Exception:
                        self._logger.exception("subscriber failure on topic '%s'", queued.topic)
            finally:
                self._queue.task_done()

    def _drop_oldest_message(self) -> None:
        try:
            dropped = self._queue.get_nowait()
        except queue.Empty:
            return

        if dropped is self._stop_sentinel:
            self._queue.put_nowait(self._stop_sentinel)
        else:
            self._queue.task_done()

    def _validate_message_type(self, topic_name: str, message: BaseEvent) -> None:
        with self._lock:
            expected = self._topic_types.get(topic_name)

        if expected is None:
            return
        if isinstance(message, expected):
            return

        expected_names = ", ".join(cls.__name__ for cls in expected)
        raise TopicTypeError(
            f"topic '{topic_name}' expects payload type {expected_names}, got {type(message).__name__}"
        )

    @staticmethod
    def _normalize_topic(topic: str | EventTopic) -> str:
        if isinstance(topic, EventTopic):
            topic_name = topic.value
        else:
            topic_name = topic.strip()
        if not topic_name:
            raise ValueError("topic cannot be empty")
        return topic_name
