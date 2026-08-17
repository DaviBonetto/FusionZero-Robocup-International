from __future__ import annotations

import threading
import time
from dataclasses import dataclass
import json

import cv2
import numpy as np

from core.event_bus import EventBus, EventTopic, StateSnapshotEvent, StateTransitionEvent, VisionDetectionEvent
from core.state_machine import RobotEvent, StateMachine
from modules.vision.vision_node import VisionNode


def _vision_test_config() -> dict:
    return {
        "paths": {"model_root": "."},
        "preprocessor": {
            "default_profile": "line",
            "line": {
                "roi": {"x": 0.0, "y": 0.3, "w": 1.0, "h": 0.7},
                "resize": {"width": 320, "height": 200},
                "luma": {"enabled": True, "target_mean": 128.0, "min_gain": 0.85, "max_gain": 1.25},
                "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
                "morphology": {"enabled": True, "kernel_size": 3, "open_iterations": 1, "close_iterations": 1},
            },
            "rescue": {
                "roi": {"x": 0.0, "y": 0.15, "w": 1.0, "h": 0.85},
                "resize": {"width": 320, "height": 240},
                "luma": {"enabled": True, "target_mean": 132.0, "min_gain": 0.8, "max_gain": 1.3},
                "clahe": {"enabled": True, "clip_limit": 1.8, "tile_grid_size": [8, 8]},
                "morphology": {"enabled": True, "kernel_size": 3, "open_iterations": 1, "close_iterations": 1},
            },
        },
        "detectors": {
            "line": {
                "min_black_area": 40,
                "black_h_max": 180,
                "black_s_max": 255,
                "black_v_max": 70,
                "erode_iter": 2,
                "dilate_iter": 2,
            },
            "color": {
                "red_s_min": 90,
                "red_v_min": 80,
                "red_min_area": 140,
                "red_min_ratio": 2.0,
                "red_min_long_side": 18,
            },
            "ball": {},
            "silver_line": {"enabled": False, "run_every_n_frames": 2, "confidence_threshold": 0.95},
            "dead_victim": {"enabled": False, "run_every_n_frames": 3, "confidence_threshold": 0.55},
        },
        "pipelines": {
            "SEARCHING_LINE": {"profile": "line", "detectors": ["line", "red"]},
            "FOLLOWING_LINE": {"profile": "line", "detectors": ["line", "green", "red", "silver_line"]},
            "VALIDATING_GAP": {"profile": "line", "detectors": ["line", "red"]},
            "CROSSING_GAP": {"profile": "line", "detectors": ["line"]},
            "VICTIM_FOUND": {"profile": "rescue", "detectors": ["balls", "victims", "red_zone"]},
            "RESCUE_ZONE_DETECTED": {"profile": "rescue", "detectors": ["balls", "victims", "red_zone"]},
            "default": {"profile": "line", "detectors": ["line", "red"]},
        },
    }


def _capture_frame(index: int) -> np.ndarray:
    frame = np.full((480, 640, 3), 235, dtype=np.uint8)
    if index % 5 != 0:
        cv2.rectangle(frame, (300, 140), (340, 479), (0, 0, 0), thickness=-1)
    if index % 11 == 0:
        cv2.rectangle(frame, (420, 220), (620, 290), (0, 0, 255), thickness=-1)
    return frame


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    return float(np.percentile(arr, 95))


@dataclass
class PipelineProbe:
    fsm: StateMachine
    detection_times: list[float]
    transition_times: list[float]
    ui_state_times: list[float]

    def __init__(self, bus: EventBus, fsm: StateMachine) -> None:
        self.fsm = fsm
        self.detection_times = []
        self.transition_times = []
        self.ui_state_times = []
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._expected = 0

        self._subscriptions = [
            bus.subscribe(EventTopic.VISION_DETECTIONS, self._on_detection_event),
            bus.subscribe(EventTopic.FSM_TRANSITION, self._on_transition_event),
            bus.subscribe(EventTopic.FSM_STATE, self._on_ui_state_event),
        ]

    def close(self) -> None:
        for sub in self._subscriptions:
            sub.unsubscribe()

    def expect(self, count: int) -> None:
        self._expected = int(count)
        self._done.clear()

    def wait(self, timeout: float = 3.0) -> bool:
        return self._done.wait(timeout=timeout)

    def _on_detection_event(self, event: VisionDetectionEvent) -> None:
        if not isinstance(event, VisionDetectionEvent):
            return
        dispatch_ts = time.perf_counter()
        with self._lock:
            self.detection_times.append(dispatch_ts)

        if event.red:
            trigger = RobotEvent.ON_RESCUE_RED_DETECTED
        elif event.victims > 0:
            trigger = RobotEvent.ON_VICTIM_DETECTED
        elif event.line:
            trigger = RobotEvent.ON_LINE_FOUND
        else:
            trigger = RobotEvent.ON_LINE_LOST
        self.fsm.handle(trigger, payload={"reason": "e2e_pipeline_probe"})

    def _on_transition_event(self, event: StateTransitionEvent) -> None:
        if not isinstance(event, StateTransitionEvent):
            return
        with self._lock:
            self.transition_times.append(time.perf_counter())
            if self._expected > 0 and len(self.transition_times) >= self._expected:
                self._done.set()

    def _on_ui_state_event(self, event: StateSnapshotEvent) -> None:
        if not isinstance(event, StateSnapshotEvent):
            return
        with self._lock:
            if self.detection_times:
                self.ui_state_times.append(time.perf_counter())


def test_e2e_capture_vision_fsm_ui_latency_and_stability(tmp_path) -> None:
    bus = EventBus(max_queue_size=2048, drop_oldest=False)
    config_path = tmp_path / "vision_config_e2e.json"
    config_path.write_text(json.dumps(_vision_test_config()), encoding="utf-8")
    node = VisionNode(bus, config=config_path, publish_raw_frame=True, publish_processed_frame=True)
    fsm = StateMachine(event_bus=bus)
    probe = PipelineProbe(bus, fsm)

    frame_count = 45
    probe.expect(frame_count)
    capture_start_times: list[float] = []
    capture_to_vision_ms: list[float] = []

    try:
        for idx in range(frame_count):
            frame = _capture_frame(idx)
            started = time.perf_counter()
            capture_start_times.append(started)
            node.process_frame(frame, frame_id=idx + 1, timestamp=time.time())
            ended = time.perf_counter()
            capture_to_vision_ms.append((ended - started) * 1000.0)

        assert probe.wait(timeout=4.0), "timeout waiting for e2e transition stream"

        samples = min(
            frame_count,
            len(capture_start_times),
            len(probe.detection_times),
            len(probe.transition_times),
            len(probe.ui_state_times),
        )
        assert samples >= int(frame_count * 0.95), "pipeline lost too many samples"

        capture_to_detection_ms = [
            (probe.detection_times[i] - capture_start_times[i]) * 1000.0 for i in range(samples)
        ]
        vision_to_fsm_ms = [
            (probe.transition_times[i] - probe.detection_times[i]) * 1000.0 for i in range(samples)
        ]
        fsm_to_ui_ms = [
            (probe.ui_state_times[i] - probe.transition_times[i]) * 1000.0 for i in range(samples)
        ]
        end_to_end_ms = [
            (probe.ui_state_times[i] - capture_start_times[i]) * 1000.0 for i in range(samples)
        ]

        assert _p95(capture_to_vision_ms[:samples]) < 80.0
        assert _p95(capture_to_detection_ms) < 90.0
        assert _p95(vision_to_fsm_ms) < 30.0
        assert _p95(fsm_to_ui_ms) < 30.0
        assert _p95(end_to_end_ms) < 120.0
    finally:
        probe.close()
        node.close()
        bus.stop()
