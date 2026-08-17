from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from ...core.event_bus import (
        EventBus,
        EventBusError,
        EventBusFullError,
        EventTopic,
        FrameEvent,
        StateSnapshotEvent,
        Subscription,
        VisionDetectionEvent,
    )
    from ...core.state_machine import RobotState
except ImportError:  # pragma: no cover
    from core.event_bus import (
        EventBus,
        EventBusError,
        EventBusFullError,
        EventTopic,
        FrameEvent,
        StateSnapshotEvent,
        Subscription,
        VisionDetectionEvent,
    )
    from core.state_machine import RobotState

try:
    from .pipelines import VisionConfig, VisionPipelineManager, _coerce_config, get_pipeline_manager
except ImportError:  # pragma: no cover
    from pipelines import VisionConfig, VisionPipelineManager, _coerce_config, get_pipeline_manager


class VisionNode:
    def __init__(
        self,
        event_bus: EventBus,
        *,
        config: VisionConfig | str | Path | None = None,
        publish_raw_frame: bool = True,
        publish_processed_frame: bool = True,
        debug_artifacts: bool | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._config = _coerce_config(config)
        self._debug_artifacts = bool(debug_artifacts) if debug_artifacts is not None else bool(
            (self._config.data.get("runtime", {}) if isinstance(self._config.data.get("runtime", {}), Mapping) else {}).get(
                "debug_artifacts_enabled",
                False,
            )
        )
        self._pipeline = (
            get_pipeline_manager(self._config)
            if debug_artifacts is None
            else VisionPipelineManager(self._config, debug_artifacts_enabled=self._debug_artifacts)
        )
        self._publish_raw_frame = bool(publish_raw_frame)
        self._publish_processed_frame = bool(publish_processed_frame)
        self._logger = logger or logging.getLogger("modules.vision.vision_node")

        self._state = RobotState.SEARCHING_LINE
        self._frame_id = 0
        self._lock = threading.RLock()
        self._latency_history = deque(maxlen=120)
        self._last_process_ts = 0.0
        self._fps = 0.0
        self._last_processed_frame: np.ndarray | None = None
        self._last_debug_bundle: dict[str, Any] | None = None

        self._state_sub: Subscription = self._event_bus.subscribe(EventTopic.FSM_STATE, self._on_state_event)

    @property
    def state(self) -> RobotState:
        with self._lock:
            return self._state

    @property
    def fps(self) -> float:
        with self._lock:
            return self._fps

    def set_state(self, state: RobotState | str) -> None:
        try:
            next_state = state if isinstance(state, RobotState) else RobotState(str(state))
        except Exception:
            return
        with self._lock:
            self._state = next_state

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        frame_id: int | None = None,
        timestamp: float | None = None,
    ) -> VisionDetectionEvent:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame_bgr is empty")

        ts = time.time() if timestamp is None else float(timestamp)
        if frame_id is None:
            with self._lock:
                self._frame_id += 1
                frame_id = self._frame_id
        else:
            with self._lock:
                self._frame_id = max(self._frame_id, int(frame_id))

        if self._publish_raw_frame:
            self._publish_frame(EventTopic.VISION_RAW_FRAME, ts, int(frame_id), frame_bgr)

        output = self._pipeline.run(self.state, frame_bgr)
        detection = output.event
        with self._lock:
            self._last_processed_frame = output.processed_frame.copy()
            self._last_debug_bundle = self._make_debug_bundle(
                frame_id=int(frame_id),
                detection=detection,
                views=output.debug_views,
            )

        if self._publish_processed_frame:
            self._publish_frame(
                EventTopic.VISION_PROCESSED_FRAME,
                detection.timestamp,
                int(frame_id),
                output.processed_frame,
            )

        self._publish_with_retry(EventTopic.VISION_DETECTIONS, detection)
        self._update_perf(detection.latency_ms)
        return detection

    def benchmark_snapshot(self) -> dict[str, Any]:
        state_benchmark = self._pipeline.benchmark_snapshot()
        with self._lock:
            avg_latency = float(np.mean(np.array(self._latency_history, dtype=np.float32))) if self._latency_history else 0.0
            fps = self._fps
        return {
            "fps_process": fps,
            "avg_latency_ms": avg_latency,
            "pipelines": state_benchmark,
            "active_state": self.state.value,
        }

    def get_last_processed_frame(self, *, copy: bool = True) -> np.ndarray | None:
        with self._lock:
            if self._last_processed_frame is None:
                return None
            return self._last_processed_frame.copy() if copy else self._last_processed_frame

    def get_last_debug_bundle(self, *, copy: bool = True) -> dict[str, Any] | None:
        with self._lock:
            if self._last_debug_bundle is None:
                return None
            return self._copy_debug_bundle(self._last_debug_bundle) if copy else self._last_debug_bundle

    def close(self) -> None:
        self._state_sub.unsubscribe()

    def _publish_frame(self, topic: EventTopic, timestamp: float, frame_id: int, frame_bgr: np.ndarray) -> None:
        frame_event = FrameEvent(
            timestamp=float(timestamp),
            frame_id=int(frame_id),
            width=int(frame_bgr.shape[1]),
            height=int(frame_bgr.shape[0]),
            encoding="bgr8",
            data=frame_bgr.tobytes(),
        )
        self._publish_with_retry(topic, frame_event)

    def _publish_with_retry(self, topic: EventTopic, message: FrameEvent | VisionDetectionEvent) -> None:
        backoff = [0.005, 0.010, 0.020]
        for attempt, wait_s in enumerate(backoff, start=1):
            try:
                self._event_bus.publish(topic, message)
                return
            except EventBusFullError:
                if attempt == len(backoff):
                    self._logger.warning("dropping message for %s after retries", topic.value)
                    return
                time.sleep(wait_s)
            except EventBusError as exc:
                self._logger.warning("event bus publish failed on %s: %s", topic.value, exc)
                return

    def _on_state_event(self, event: StateSnapshotEvent) -> None:
        try:
            next_state = RobotState(event.state)
        except Exception:
            return
        with self._lock:
            self._state = next_state

    def _update_perf(self, latency_ms: float) -> None:
        now = time.perf_counter()
        with self._lock:
            self._latency_history.append(float(latency_ms))
            if self._last_process_ts > 0:
                dt = now - self._last_process_ts
                if dt > 0:
                    instant_fps = 1.0 / dt
                    self._fps = instant_fps if self._fps <= 0 else (0.9 * self._fps + 0.1 * instant_fps)
            self._last_process_ts = now

    def _make_debug_bundle(
        self,
        *,
        frame_id: int,
        detection: VisionDetectionEvent,
        views: Mapping[str, np.ndarray] | None,
    ) -> dict[str, Any] | None:
        if not self._debug_artifacts:
            return None
        if not isinstance(views, Mapping) or not views:
            return {
                "frame_id": int(frame_id),
                "timestamp": float(detection.timestamp),
                "state": str(detection.state),
                "metadata": dict(detection.metadata or {}),
                "views": {},
            }
        return {
            "frame_id": int(frame_id),
            "timestamp": float(detection.timestamp),
            "state": str(detection.state),
            "metadata": dict(detection.metadata or {}),
            "views": {str(name): view.copy() for name, view in views.items() if isinstance(view, np.ndarray)},
        }

    @staticmethod
    def _copy_debug_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
        copied: dict[str, Any] = {
            "frame_id": int(bundle.get("frame_id", 0)),
            "timestamp": float(bundle.get("timestamp", 0.0)),
            "state": str(bundle.get("state", "")),
            "metadata": dict(bundle.get("metadata", {})) if isinstance(bundle.get("metadata"), Mapping) else {},
            "views": {},
        }
        views = bundle.get("views", {})
        if isinstance(views, Mapping):
            copied["views"] = {
                str(name): value.copy() for name, value in views.items() if isinstance(value, np.ndarray)
            }
        return copied
