from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from conftest import wait_until
from src.core.event_bus import EventBus, EventTopic, FrameEvent, UICommandEvent
from src.session_recording import LATEST_SESSION_MARKER, SessionRecorder


def test_session_recorder_writes_events_and_frame_samples(tmp_path: Path) -> None:
    bus = EventBus(max_queue_size=128, drop_oldest=False)
    recorder = SessionRecorder(
        bus,
        output_root=tmp_path,
        default_options={"include_raw": False, "include_processed": True, "every_n_frames": 1},
    )
    frame = np.full((32, 48, 3), 120, dtype=np.uint8)
    try:
        session_dir = recorder.start(reason="pytest", context={"case": "session-recorder"})
        bus.publish(
            EventTopic.UI_COMMAND,
            UICommandEvent(timestamp=time.time(), command="calibration.snapshot", params={"view_mode": "line_mask"}),
        )
        bus.publish(
            EventTopic.VISION_PROCESSED_FRAME,
            FrameEvent(
                timestamp=time.time(),
                frame_id=3,
                width=48,
                height=32,
                encoding="bgr8",
                data=frame.tobytes(),
            ),
        )
        bus._queue.join()
        assert wait_until(lambda: recorder.status_payload()["event_count"] >= 2, timeout=2.0)

        recorder.stop(reason="done")
        recorder.close()

        events_path = session_dir / "events.jsonl"
        manifest_path = session_dir / "manifest.json"
        assert events_path.exists()
        assert manifest_path.exists()

        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        assert any(event["event_type"] == "SessionStart" for event in events)
        assert any(event["event_type"] == "UICommandEvent" for event in events)
        frame_sample = next(event for event in events if event["event_type"] == "FrameSample")
        assert (session_dir / frame_sample["fields"]["path"]).exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["stats"]["frame_count"] >= 1
        assert manifest["active"] is False
    finally:
        bus.stop()


def test_session_recorder_partial_config_preserves_existing_jpeg_quality(tmp_path: Path) -> None:
    bus = EventBus(max_queue_size=32, drop_oldest=False)
    recorder = SessionRecorder(
        bus,
        output_root=tmp_path,
        default_options={"include_raw": True, "include_processed": True, "every_n_frames": 4, "jpeg_quality": 72},
    )
    try:
        options = recorder.configure({"include_raw": False, "every_n_frames": 9})
        assert options.include_raw is False
        assert options.include_processed is True
        assert options.every_n_frames == 9
        assert options.jpeg_quality == 72
    finally:
        recorder.close()
        bus.stop()


def test_session_recorder_restores_authoritative_latest_session_after_restart(tmp_path: Path) -> None:
    first_bus = EventBus(max_queue_size=32, drop_oldest=False)
    first = SessionRecorder(first_bus, output_root=tmp_path)
    session_dir = first.start(reason="first")
    first.stop(reason="done")
    first.close()
    first_bus.stop()

    second_bus = EventBus(max_queue_size=32, drop_oldest=False)
    second = SessionRecorder(second_bus, output_root=tmp_path)
    try:
        status = second.status_payload()
        assert Path(status["session_dir"]) == session_dir.resolve()
        assert Path(status["latest_marker"]) == tmp_path / LATEST_SESSION_MARKER
    finally:
        second.close()
        second_bus.stop()


def test_latest_session_marker_ignores_wall_clock_regression(monkeypatch, tmp_path: Path) -> None:
    first_bus = EventBus(max_queue_size=32, drop_oldest=False)
    monkeypatch.setattr("src.session_recording.time.time", lambda: 2_000_000.500)
    first = SessionRecorder(first_bus, output_root=tmp_path)
    first_dir = first.start(reason="newer-clock")
    first.stop(reason="done")
    first.close()
    first_bus.stop()

    second_bus = EventBus(max_queue_size=32, drop_oldest=False)
    monkeypatch.setattr("src.session_recording.time.time", lambda: 1_000_000.250)
    second = SessionRecorder(second_bus, output_root=tmp_path)
    second_dir = second.start(reason="clock-went-back")
    second.stop(reason="done")
    second.close()
    second_bus.stop()

    marker = json.loads((tmp_path / LATEST_SESSION_MARKER).read_text(encoding="utf-8"))
    assert first_dir != second_dir
    assert marker["session_dir"] == second_dir.name
    assert marker["active"] is False
