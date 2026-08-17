from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

if __package__ in (None, ""):
    import sys

    SRC_ROOT = Path(__file__).resolve().parent
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from core.event_bus import (  # type: ignore
        EventBus,
        EventTopic,
        FrameEvent,
        Subscription,
    )
else:
    from .core.event_bus import EventBus, EventTopic, FrameEvent, Subscription


DEFAULT_RECORDINGS_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "session_recordings"
LATEST_SESSION_MARKER = "latest_session.json"


@dataclass(slots=True)
class SessionRecordingOptions:
    include_raw: bool = False
    include_processed: bool = True
    every_n_frames: int = 20
    jpeg_quality: int = 70

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "SessionRecordingOptions":
        raw = dict(payload or {})
        return cls(
            include_raw=bool(raw.get("include_raw", False)),
            include_processed=bool(raw.get("include_processed", True)),
            every_n_frames=max(1, int(raw.get("every_n_frames", 20))),
            jpeg_quality=max(35, min(95, int(raw.get("jpeg_quality", 70)))),
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _QueuedAction:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


class SessionRecorder:
    TOPICS: tuple[EventTopic, ...] = (
        EventTopic.FSM_STATE,
        EventTopic.FSM_TRANSITION,
        EventTopic.VISION_DETECTIONS,
        EventTopic.SYSTEM_HEALTH,
        EventTopic.SYSTEM_LOG,
        EventTopic.UI_COMMAND,
        EventTopic.NAV_PATH,
        EventTopic.VISION_RAW_FRAME,
        EventTopic.VISION_PROCESSED_FRAME,
    )

    def __init__(
        self,
        event_bus: EventBus,
        *,
        output_root: str | Path | None = None,
        default_options: SessionRecordingOptions | Mapping[str, Any] | None = None,
        queue_size: int = 256,
    ) -> None:
        self._event_bus = event_bus
        self._output_root = DEFAULT_RECORDINGS_ROOT if output_root is None else Path(output_root)
        self._options = (
            default_options
            if isinstance(default_options, SessionRecordingOptions)
            else SessionRecordingOptions.from_mapping(default_options)
        )
        self._queue: queue.Queue[_QueuedAction] = queue.Queue(maxsize=max(32, int(queue_size)))
        self._lock = threading.RLock()
        self._worker_stop = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, name="session-recorder", daemon=True)
        self._worker.start()

        self._subscriptions: list[Subscription] = [
            self._event_bus.subscribe(topic, self._build_handler(topic)) for topic in self.TOPICS
        ]

        self._active = False
        self._session_dir: Path | None = None
        self._frames_dir: Path | None = None
        self._events_path: Path | None = None
        self._event_count = 0
        self._frame_count = 0
        self._dropped_records = 0
        self._session_started_at = 0.0
        self._last_context: dict[str, Any] = {}
        self._raw_frame_seen = 0
        self._processed_frame_seen = 0
        self._restore_latest_session()

    def start(self, *, reason: str = "", context: Mapping[str, Any] | None = None) -> Path:
        with self._lock:
            if self._active and self._session_dir is not None:
                return self._session_dir

            now = time.time()
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
            session_dir = self._unique_session_dir(f"session_{stamp}_{int(now * 1000) % 1000:03d}")
            frames_dir = session_dir / "frames"
            session_dir.mkdir(parents=True, exist_ok=True)
            frames_dir.mkdir(parents=True, exist_ok=True)
            self._session_dir = session_dir
            self._frames_dir = frames_dir
            self._events_path = session_dir / "events.jsonl"
            self._event_count = 0
            self._frame_count = 0
            self._dropped_records = 0
            self._session_started_at = now
            self._last_context = dict(context or {})
            self._raw_frame_seen = 0
            self._processed_frame_seen = 0
            self._active = True

        self._write_manifest(active=True, reason=reason)
        self._write_latest_session_marker(active=True, reason=reason)
        self._enqueue(
            "event",
            {
                "topic": "session.meta",
                "event_type": "SessionStart",
                "recorded_at": time.time(),
                "fields": {
                    "reason": str(reason),
                    "options": self._options.to_payload(),
                    "context": dict(context or {}),
                },
            },
        )
        return session_dir

    def stop(self, *, reason: str = "") -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False

        self._enqueue(
            "event",
            {
                "topic": "session.meta",
                "event_type": "SessionStop",
                "recorded_at": time.time(),
                "fields": {"reason": str(reason)},
            },
        )
        self._write_latest_session_marker(active=False, reason=reason)
        self._enqueue("flush", {"active": False, "reason": str(reason)})

    def configure(self, payload: Mapping[str, Any] | None) -> SessionRecordingOptions:
        with self._lock:
            merged = self._options.to_payload()
            merged.update(dict(payload or {}))
            options = SessionRecordingOptions.from_mapping(merged)
            self._options = options
            active = self._active
        self._write_manifest(active=active, reason="configure")
        if active:
            self._enqueue(
                "event",
                {
                    "topic": "session.meta",
                    "event_type": "SessionConfig",
                    "recorded_at": time.time(),
                    "fields": {"options": options.to_payload()},
                },
            )
        return options

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            session_dir = self._session_dir
            return {
                "enabled": bool(self._active),
                "session_dir": str(session_dir) if session_dir is not None else "",
                "event_count": int(self._event_count),
                "frame_count": int(self._frame_count),
                "dropped_records": int(self._dropped_records),
                "started_at": float(self._session_started_at),
                "latest_marker": str(self._output_root / LATEST_SESSION_MARKER),
                "options": self._options.to_payload(),
            }

    def close(self) -> None:
        self.stop(reason="shutdown")
        for sub in self._subscriptions:
            try:
                sub.unsubscribe()
            except Exception:
                pass
        self._subscriptions.clear()
        self._enqueue("shutdown", {})
        self._worker.join(timeout=2.0)

    def _build_handler(self, topic: EventTopic):
        topic_name = topic.value

        def _handler(event: object) -> None:
            with self._lock:
                if not self._active:
                    return
                options = self._options
            if isinstance(event, FrameEvent):
                self._handle_frame(topic_name, event, options)
                return
            self._enqueue(
                "event",
                {
                    "topic": topic_name,
                    "event_type": type(event).__name__,
                    "recorded_at": time.time(),
                    "fields": asdict(event),
                },
            )

        return _handler

    def _handle_frame(self, topic: str, event: FrameEvent, options: SessionRecordingOptions) -> None:
        is_raw = topic == EventTopic.VISION_RAW_FRAME.value
        if is_raw and not options.include_raw:
            return
        if (not is_raw) and not options.include_processed:
            return

        if is_raw:
            self._raw_frame_seen += 1
            should_capture = (self._raw_frame_seen % max(1, options.every_n_frames)) == 0
        else:
            self._processed_frame_seen += 1
            should_capture = (self._processed_frame_seen % max(1, options.every_n_frames)) == 0
        if not should_capture:
            return

        self._enqueue(
            "frame",
            {
                "topic": topic,
                "recorded_at": time.time(),
                "frame_id": int(event.frame_id),
                "timestamp": float(event.timestamp),
                "width": int(event.width),
                "height": int(event.height),
                "encoding": str(event.encoding or "bgr8"),
                "data": bytes(event.data),
                "jpeg_quality": int(options.jpeg_quality),
            },
        )

    def _enqueue(self, kind: str, payload: Mapping[str, Any]) -> None:
        try:
            self._queue.put_nowait(_QueuedAction(kind=kind, payload=dict(payload)))
        except queue.Full:
            with self._lock:
                self._dropped_records += 1

    def _worker_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if item.kind == "shutdown":
                    self._worker_stop.set()
                    return
                if item.kind == "event":
                    self._write_event_line(item.payload)
                    continue
                if item.kind == "frame":
                    self._write_frame(item.payload)
                    continue
                if item.kind == "flush":
                    self._write_manifest(active=bool(item.payload.get("active", False)), reason=str(item.payload.get("reason", "")))
                    continue
            finally:
                self._queue.task_done()

    def _write_event_line(self, payload: Mapping[str, Any]) -> None:
        events_path = self._events_path
        if events_path is None:
            return
        with events_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(dict(payload), ensure_ascii=True, separators=(",", ":")) + "\n")
        with self._lock:
            self._event_count += 1

    def _write_frame(self, payload: Mapping[str, Any]) -> None:
        frames_dir = self._frames_dir
        if frames_dir is None:
            return

        frame_kind = "raw" if payload.get("topic") == EventTopic.VISION_RAW_FRAME.value else "processed"
        filename = f"{frame_kind}_{int(payload.get('frame_id', 0)):06d}.jpg"
        output_path = frames_dir / filename
        encoded = self._encode_frame(
            width=int(payload.get("width", 0)),
            height=int(payload.get("height", 0)),
            encoding=str(payload.get("encoding", "bgr8")),
            data=bytes(payload.get("data", b"")),
            jpeg_quality=int(payload.get("jpeg_quality", 70)),
        )
        if encoded is None:
            return
        output_path.write_bytes(encoded)
        self._write_event_line(
            {
                "topic": str(payload.get("topic", "")),
                "event_type": "FrameSample",
                "recorded_at": float(payload.get("recorded_at", time.time())),
                "fields": {
                    "frame_id": int(payload.get("frame_id", 0)),
                    "timestamp": float(payload.get("timestamp", time.time())),
                    "path": str(output_path.relative_to(self._session_dir or output_path.parent)),
                    "kind": frame_kind,
                },
            }
        )
        with self._lock:
            self._frame_count += 1

    @staticmethod
    def _encode_frame(*, width: int, height: int, encoding: str, data: bytes, jpeg_quality: int) -> bytes | None:
        normalized = str(encoding or "bgr8").strip().lower()
        if normalized in {"jpeg", "jpg"}:
            return bytes(data)
        if cv2 is None or width <= 0 or height <= 0:
            return None
        if len(data) < (width * height * 3):
            return None
        array = np.frombuffer(memoryview(data), dtype=np.uint8)
        expected = width * height * 3
        if array.size < expected:
            return None
        frame = array[:expected].reshape((height, width, 3))
        if normalized.startswith("rgb"):
            frame = frame[:, :, ::-1]
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            return None
        return encoded.tobytes()

    def _write_manifest(self, *, active: bool, reason: str) -> None:
        session_dir = self._session_dir
        if session_dir is None:
            return
        payload = {
            "active": bool(active),
            "reason": str(reason),
            "started_at": float(self._session_started_at),
            "config": self._options.to_payload(),
            "stats": self.status_payload(),
            "context": dict(self._last_context),
        }
        (session_dir / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _unique_session_dir(self, base_name: str) -> Path:
        candidate = self._output_root / str(base_name)
        suffix = 0
        while candidate.exists():
            suffix += 1
            candidate = self._output_root / f"{base_name}_{suffix:02d}"
        return candidate

    def _restore_latest_session(self) -> None:
        marker_path = self._output_root / LATEST_SESSION_MARKER
        if not marker_path.is_file():
            return
        try:
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
            relative_name = str(payload.get("session_dir", "")).strip()
            if not relative_name:
                return
            root = self._output_root.resolve()
            candidate = (self._output_root / relative_name).resolve()
            candidate.relative_to(root)
            if not candidate.is_dir():
                return
            self._session_dir = candidate
            self._frames_dir = candidate / "frames"
            self._events_path = candidate / "events.jsonl"
            self._session_started_at = float(payload.get("started_at", 0.0))
            context = payload.get("context", {})
            self._last_context = dict(context) if isinstance(context, Mapping) else {}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return

    def _write_latest_session_marker(self, *, active: bool, reason: str) -> None:
        session_dir = self._session_dir
        if session_dir is None:
            return
        self._output_root.mkdir(parents=True, exist_ok=True)
        marker_path = self._output_root / LATEST_SESSION_MARKER
        temporary_path = self._output_root / f".{LATEST_SESSION_MARKER}.tmp"
        payload = {
            "schema_version": 1,
            "session_dir": session_dir.name,
            "active": bool(active),
            "reason": str(reason),
            "started_at": float(self._session_started_at),
            "updated_at": float(time.time()),
            "context": dict(self._last_context),
        }
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(marker_path)
