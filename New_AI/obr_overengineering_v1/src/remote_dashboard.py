from __future__ import annotations

import itertools
import json
import socket
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

if __package__ in (None, ""):
    SRC_ROOT = Path(__file__).resolve().parent
    import sys

    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from core.event_bus import (  # type: ignore
        EventBus,
        EventBusError,
        EventBusFullError,
        EventTopic,
        FrameEvent,
        HealthEvent,
        LogEvent,
        PathEvent,
        PoseEvent,
        StateSnapshotEvent,
        StateTransitionEvent,
        Subscription,
        UICommandEvent,
        VisionDetectionEvent,
    )
else:
    from .core.event_bus import (
        EventBus,
        EventBusError,
        EventBusFullError,
        EventTopic,
        FrameEvent,
        HealthEvent,
        LogEvent,
        PathEvent,
        PoseEvent,
        StateSnapshotEvent,
        StateTransitionEvent,
        Subscription,
        UICommandEvent,
        VisionDetectionEvent,
    )


CONTROL_TOPIC = "__remote_control__"


@dataclass(slots=True)
class ControlMessage:
    timestamp: float = field(default_factory=time.time)
    kind: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    session_id: int = 0


EVENT_CLASS_BY_NAME: dict[str, type[Any]] = {
    "ControlMessage": ControlMessage,
    "FrameEvent": FrameEvent,
    "HealthEvent": HealthEvent,
    "LogEvent": LogEvent,
    "PathEvent": PathEvent,
    "PoseEvent": PoseEvent,
    "StateSnapshotEvent": StateSnapshotEvent,
    "StateTransitionEvent": StateTransitionEvent,
    "UICommandEvent": UICommandEvent,
    "VisionDetectionEvent": VisionDetectionEvent,
}

FORWARDED_TOPICS: tuple[EventTopic, ...] = (
    EventTopic.VISION_RAW_FRAME,
    EventTopic.VISION_PROCESSED_FRAME,
    EventTopic.VISION_DETECTIONS,
    EventTopic.FSM_STATE,
    EventTopic.FSM_TRANSITION,
    EventTopic.SYSTEM_HEALTH,
    EventTopic.SYSTEM_LOG,
    EventTopic.NAV_PATH,
)


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _encode_frame_payload(event: FrameEvent, jpeg_quality: int) -> tuple[dict[str, Any], bytes]:
    encoding = str(event.encoding or "bgr8").strip().lower()
    payload = bytes(event.data)
    fields = {
        "timestamp": float(event.timestamp),
        "frame_id": int(event.frame_id),
        "width": int(event.width),
        "height": int(event.height),
        "encoding": encoding,
    }
    if encoding in {"jpeg", "jpg"}:
        fields["encoding"] = "jpeg"
        return fields, payload

    if cv2 is None or event.width <= 0 or event.height <= 0:
        return fields, payload

    raw = np.frombuffer(event.data, dtype=np.uint8)
    expected = int(event.width) * int(event.height) * 3
    if raw.size < expected:
        return fields, payload

    frame = raw[:expected].reshape((int(event.height), int(event.width), 3))
    if encoding.startswith("rgb"):
        frame = frame[:, :, ::-1]

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return fields, payload

    fields["encoding"] = "jpeg"
    return fields, encoded.tobytes()


def pack_event_message(topic: str, event: object, *, jpeg_quality: int = 70) -> bytes:
    if isinstance(event, FrameEvent):
        fields, payload = _encode_frame_payload(event, jpeg_quality)
    else:
        fields = asdict(event)
        payload = b""

    header = {
        "topic": str(topic),
        "event_type": type(event).__name__,
        "payload_length": int(len(payload)),
        "fields": fields,
    }
    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return len(header_bytes).to_bytes(4, byteorder="big", signed=False) + header_bytes + payload


def recv_event_message(sock: socket.socket) -> tuple[str, object]:
    header_len = int.from_bytes(_read_exact(sock, 4), byteorder="big", signed=False)
    if header_len <= 0:
        raise ConnectionError("invalid header length")
    header = json.loads(_read_exact(sock, header_len).decode("utf-8"))
    topic = str(header["topic"])
    event_type = str(header["event_type"])
    payload_length = int(header.get("payload_length", 0))
    fields = header.get("fields", {})
    if not isinstance(fields, dict):
        raise ValueError("invalid message fields")

    payload = _read_exact(sock, payload_length) if payload_length > 0 else b""
    event_cls = EVENT_CLASS_BY_NAME.get(event_type)
    if event_cls is None:
        raise ValueError(f"unsupported event type: {event_type}")

    if event_cls is FrameEvent:
        fields["data"] = payload
    elif event_cls is PathEvent:
        poses = fields.get("poses", [])
        if isinstance(poses, list):
            fields["poses"] = [PoseEvent(**pose) if isinstance(pose, dict) else pose for pose in poses]
    event = event_cls(**fields)
    return topic, event


def pack_control_message(kind: str, *, session_id: int = 0, detail: Mapping[str, Any] | None = None) -> bytes:
    return pack_event_message(
        CONTROL_TOPIC,
        ControlMessage(
            kind=str(kind).strip().lower(),
            detail=dict(detail or {}),
            session_id=int(session_id),
        ),
    )


class RemoteDashboardServer:
    def __init__(
        self,
        event_bus: EventBus,
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        stream_fps: float = 8.0,
        jpeg_quality: int = 70,
        client_timeout: float = 6.0,
        log_replay_size: int = 25,
    ) -> None:
        self._event_bus = event_bus
        self._host = str(host)
        self._port = int(port)
        self._stream_fps = max(1.0, float(stream_fps))
        self._jpeg_quality = max(35, min(95, int(jpeg_quality)))
        self._client_timeout = max(2.0, float(client_timeout))
        self._subscriptions: list[Subscription] = []
        self._stop_event = threading.Event()
        self._server_socket: socket.socket | None = None
        self._client_socket: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._recv_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None
        self._outbound_thread: threading.Thread | None = None
        self._client_lock = threading.RLock()
        self._send_lock = threading.RLock()
        self._pending_outbound_lock = threading.RLock()
        self._outbound_wakeup = threading.Event()
        self._session_ids = itertools.count(1)
        self._client_session_id = 0
        self._last_client_activity_at = 0.0
        self._pending_latest: dict[str, object] = {}
        self._pending_ordered: deque[tuple[str, object]] = deque(maxlen=128)
        self._latest_by_topic: dict[str, object] = {}
        self._recent_logs: deque[LogEvent] = deque(maxlen=max(1, int(log_replay_size)))

    @property
    def port(self) -> int:
        return int(self._port)

    def start(self) -> None:
        if self._accept_thread is not None:
            return

        self._stop_event.clear()
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self._host, self._port))
        self._port = int(self._server_socket.getsockname()[1])
        self._server_socket.listen(1)
        self._server_socket.settimeout(1.0)

        for topic in FORWARDED_TOPICS:
            self._subscriptions.append(self._event_bus.subscribe(topic, self._build_forwarder(topic.value)))

        self._outbound_wakeup.clear()
        self._outbound_thread = threading.Thread(
            target=self._outbound_send_loop,
            name="remote-dashboard-outbound",
            daemon=True,
        )
        self._outbound_thread.start()
        self._accept_thread = threading.Thread(target=self._accept_loop, name="remote-dashboard-accept", daemon=True)
        self._accept_thread.start()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, name="remote-dashboard-monitor", daemon=True)
        self._monitor_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._outbound_wakeup.set()
        self._close_client(publish_log=False)
        server = self._server_socket
        self._server_socket = None
        if server is not None:
            try:
                server.close()
            except Exception:
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=2.0)
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
        if self._outbound_thread is not None:
            self._outbound_thread.join(timeout=2.0)
        self._accept_thread = None
        self._recv_thread = None
        self._monitor_thread = None
        self._outbound_thread = None
        with self._pending_outbound_lock:
            self._pending_latest.clear()
            self._pending_ordered.clear()
        for sub in self._subscriptions:
            try:
                sub.unsubscribe()
            except Exception:
                pass
        self._subscriptions.clear()

    def _accept_loop(self) -> None:
        while not self._stop_event.is_set():
            server = self._server_socket
            if server is None:
                return
            try:
                client, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32 * 1024)
            self._replace_client(client)

    def _replace_client(self, client: socket.socket) -> None:
        peer = self._peer_name(client)
        session_id = next(self._session_ids)
        self._close_client(reason="replaced_by_new_client", publish_log=False)
        with self._pending_outbound_lock:
            self._pending_latest.clear()
            self._pending_ordered.clear()
        with self._client_lock:
            self._client_socket = client
            self._client_session_id = session_id
            self._last_client_activity_at = time.monotonic()
        try:
            self._send_control(client, "hello", session_id=session_id, detail={"peer": peer})
            self._replay_snapshot(client)
        except Exception:
            self._close_client(reason="initial_snapshot_failed")
            return
        self._publish_local_log("INFO", f"remote dashboard client connected peer={peer} session={session_id}")
        self._recv_thread = threading.Thread(
            target=self._recv_loop,
            args=(client,),
            name="remote-dashboard-recv",
            daemon=True,
        )
        self._recv_thread.start()

    def _close_client(self, reason: str | None = None, *, publish_log: bool = True) -> None:
        with self._client_lock:
            client = self._client_socket
            self._client_socket = None
            session_id = self._client_session_id
            self._client_session_id = 0
        if client is not None:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                client.close()
            except Exception:
                pass
        if publish_log and session_id > 0 and not self._stop_event.is_set():
            message = f"remote dashboard client disconnected session={session_id}"
            if reason:
                message = f"{message} reason={reason}"
            self._publish_local_log("WARNING" if reason else "INFO", message)

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(0.5):
            with self._client_lock:
                has_client = self._client_socket is not None
                idle_for = time.monotonic() - self._last_client_activity_at
            if has_client and idle_for > self._client_timeout:
                self._close_client(reason="heartbeat_timeout")

    def _recv_loop(self, client: socket.socket) -> None:
        try:
            while not self._stop_event.is_set():
                topic, event = recv_event_message(client)
                with self._client_lock:
                    if self._client_socket is client:
                        self._last_client_activity_at = time.monotonic()
                if topic == CONTROL_TOPIC and isinstance(event, ControlMessage):
                    self._handle_control_message(client, event)
                    continue
                if topic != EventTopic.UI_COMMAND.value or not isinstance(event, UICommandEvent):
                    continue
                try:
                    self._event_bus.publish(EventTopic.UI_COMMAND, event)
                except EventBusError:
                    continue
        except Exception:
            pass
        finally:
            with self._client_lock:
                is_current = self._client_socket is client
            if is_current:
                self._close_client(reason="client_stream_closed")
            else:
                try:
                    client.close()
                except Exception:
                    pass

    def _build_forwarder(self, topic: str) -> Callable[[object], None]:
        def _forward(event: object) -> None:
            self._remember_event(topic, event)
            if self._stop_event.is_set():
                return
            with self._pending_outbound_lock:
                if topic in {
                    EventTopic.SYSTEM_LOG.value,
                    EventTopic.FSM_TRANSITION.value,
                }:
                    self._pending_ordered.append((topic, event))
                else:
                    # High-rate frames, detections and health are latest-only.
                    # This prevents network or UI latency from blocking the
                    # EventBus that also carries motor STOP/START commands.
                    self._pending_latest[topic] = event
            self._outbound_wakeup.set()

        return _forward

    def _outbound_send_loop(self) -> None:
        interval = 1.0 / self._stream_fps
        latest_topic_order = (
            EventTopic.FSM_STATE.value,
            EventTopic.VISION_DETECTIONS.value,
            EventTopic.SYSTEM_HEALTH.value,
            EventTopic.NAV_PATH.value,
            EventTopic.VISION_RAW_FRAME.value,
            EventTopic.VISION_PROCESSED_FRAME.value,
        )
        next_latest_at = time.monotonic()
        while not self._stop_event.is_set():
            timeout = max(0.0, min(0.25, next_latest_at - time.monotonic()))
            self._outbound_wakeup.wait(timeout)
            self._outbound_wakeup.clear()
            if self._stop_event.is_set():
                return

            with self._pending_outbound_lock:
                ordered = list(self._pending_ordered)
                self._pending_ordered.clear()
            for topic, event in ordered:
                self._send_event(topic, event)

            if time.monotonic() < next_latest_at:
                continue
            with self._pending_outbound_lock:
                latest = self._pending_latest
                self._pending_latest = {}
            for topic in latest_topic_order:
                event = latest.get(topic)
                if event is not None:
                    self._send_event(topic, event)
            next_latest_at = time.monotonic() + interval

    def _send_event(self, topic: str, event: object) -> bool:
        with self._client_lock:
            client = self._client_socket
        if client is None:
            return False

        try:
            packet = pack_event_message(topic, event, jpeg_quality=self._jpeg_quality)
            self._send_packet(client, packet)
            return True
        except Exception:
            self._close_client(reason="send_failed")
            return False

    def _send_packet(self, client: socket.socket, packet: bytes) -> None:
        with self._send_lock:
            client.sendall(packet)

    def _send_control(
        self,
        client: socket.socket,
        kind: str,
        *,
        session_id: int,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        self._send_packet(client, pack_control_message(kind, session_id=session_id, detail=detail))

    def _handle_control_message(self, client: socket.socket, event: ControlMessage) -> None:
        if str(event.kind).strip().lower() != "ping":
            return
        with self._client_lock:
            session_id = self._client_session_id
        self._send_control(client, "pong", session_id=session_id, detail=dict(event.detail))

    def _remember_event(self, topic: str, event: object) -> None:
        if isinstance(event, LogEvent):
            self._recent_logs.append(event)
            return
        self._latest_by_topic[topic] = event

    def _replay_snapshot(self, client: socket.socket) -> None:
        ordered_topics = (
            EventTopic.FSM_STATE.value,
            EventTopic.FSM_TRANSITION.value,
            EventTopic.VISION_DETECTIONS.value,
            EventTopic.SYSTEM_HEALTH.value,
            EventTopic.NAV_PATH.value,
            EventTopic.VISION_RAW_FRAME.value,
            EventTopic.VISION_PROCESSED_FRAME.value,
        )
        for topic in ordered_topics:
            event = self._latest_by_topic.get(topic)
            if event is None:
                continue
            self._send_packet(client, pack_event_message(topic, event, jpeg_quality=self._jpeg_quality))
        for event in list(self._recent_logs):
            self._send_packet(
                client,
                pack_event_message(EventTopic.SYSTEM_LOG.value, event, jpeg_quality=self._jpeg_quality),
            )

    def _publish_local_log(self, level: str, message: str) -> None:
        try:
            self._event_bus.publish(
                EventTopic.SYSTEM_LOG,
                LogEvent(timestamp=time.time(), level=level, message=message, source="remote_server", state=""),
            )
        except EventBusError:
            pass

    @staticmethod
    def _peer_name(client: socket.socket) -> str:
        try:
            host, port = client.getpeername()
            return f"{host}:{port}"
        except Exception:
            return "unknown"


class RemoteDashboardClient:
    def __init__(
        self,
        event_bus: EventBus,
        *,
        host: str,
        port: int = 8765,
        reconnect_interval: float = 2.0,
        heartbeat_interval: float = 1.0,
        server_timeout: float = 6.0,
    ) -> None:
        self._event_bus = event_bus
        self._host = str(host)
        self._port = int(port)
        self._reconnect_interval = max(0.5, float(reconnect_interval))
        self._heartbeat_interval = max(0.5, float(heartbeat_interval))
        self._server_timeout = max(self._heartbeat_interval * 2.0, float(server_timeout))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._socket_lock = threading.RLock()
        self._send_lock = threading.RLock()
        self._ui_subscription: Subscription | None = None
        self._last_server_activity_at = 0.0
        self._last_ping_sent_at = 0.0
        self._session_id = 0
        self._ping_sequence = 0
        self._pending_pings: dict[int, float] = {}
        self._last_rtt_ms = 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._ui_subscription = self._event_bus.subscribe(EventTopic.UI_COMMAND, self._on_ui_command)
        self._thread = threading.Thread(target=self._run_loop, name="remote-dashboard-client", daemon=True)
        self._thread.start()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, name="remote-dashboard-heartbeat", daemon=True)
        self._monitor_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._ui_subscription is not None:
            try:
                self._ui_subscription.unsubscribe()
            except Exception:
                pass
            self._ui_subscription = None
        self._close_socket()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
        self._thread = None
        self._monitor_thread = None

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            sock: socket.socket | None = None
            try:
                sock = socket.create_connection((self._host, self._port), timeout=5.0)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._socket_lock:
                    self._socket = sock
                    self._last_server_activity_at = time.monotonic()
                    self._last_ping_sent_at = 0.0
                    self._session_id = 0
                    self._ping_sequence = 0
                    self._pending_pings.clear()
                    self._last_rtt_ms = 0.0
                self._publish_local_log("INFO", f"connected to raspberry {self._host}:{self._port}")
                while not self._stop_event.is_set():
                    topic, event = recv_event_message(sock)
                    with self._socket_lock:
                        if self._socket is sock:
                            self._last_server_activity_at = time.monotonic()
                    if topic == CONTROL_TOPIC and isinstance(event, ControlMessage):
                        self._handle_control_message(sock, event)
                        continue
                    self._publish_remote_event(topic, event)
            except Exception:
                if not self._stop_event.is_set():
                    self._publish_local_log("WARNING", f"remote dashboard disconnected from {self._host}:{self._port}")
                    time.sleep(self._reconnect_interval)
            finally:
                self._close_socket()

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(0.2):
            with self._socket_lock:
                sock = self._socket
                last_server_activity = self._last_server_activity_at
                last_ping_sent = self._last_ping_sent_at
            if sock is None:
                continue

            now = time.monotonic()
            if (now - last_ping_sent) >= self._heartbeat_interval:
                with self._socket_lock:
                    self._ping_sequence += 1
                    seq = self._ping_sequence
                    self._pending_pings[seq] = now
                if self._send_control("ping", detail={"seq": seq}):
                    with self._socket_lock:
                        if self._socket is sock:
                            self._last_ping_sent_at = now
                else:
                    with self._socket_lock:
                        self._pending_pings.pop(seq, None)
                continue

            if (now - last_server_activity) > self._server_timeout:
                self._publish_local_log(
                    "WARNING",
                    f"remote dashboard heartbeat timeout from {self._host}:{self._port}",
                )
                self._close_socket()

    def _publish_remote_event(self, topic: str, event: object) -> None:
        if topic == EventTopic.SYSTEM_HEALTH.value and isinstance(event, HealthEvent):
            metadata = dict(event.metadata or {})
            metadata["network_latency_ms"] = float(self._last_rtt_ms)
            network = metadata.get("network", {})
            network_payload = dict(network) if isinstance(network, Mapping) else {}
            network_payload.update(
                {
                    "latency_ms": float(self._last_rtt_ms),
                    "state": "connected",
                    "peer": f"{self._host}:{self._port}",
                    "session_id": int(self._session_id),
                }
            )
            event.metadata = metadata
            event.metadata["network"] = network_payload
        try:
            self._event_bus.publish(topic, event)
        except EventBusFullError:
            return
        except EventBusError:
            return

    def _publish_local_log(self, level: str, message: str) -> None:
        try:
            self._event_bus.publish(
                EventTopic.SYSTEM_LOG,
                LogEvent(timestamp=time.time(), level=level, message=message, source="remote_client", state=""),
            )
        except EventBusError:
            pass

    def _handle_control_message(self, sock: socket.socket, event: ControlMessage) -> None:
        kind = str(event.kind).strip().lower()
        if kind == "hello":
            self._session_id = int(event.session_id)
            self._publish_local_log("INFO", f"remote dashboard session ready session={self._session_id}")
            return
        if kind == "pong":
            seq = int(event.detail.get("seq", 0)) if isinstance(event.detail, Mapping) else 0
            started_at = self._pending_pings.pop(seq, 0.0)
            if started_at > 0:
                self._last_rtt_ms = max(0.0, (time.monotonic() - started_at) * 1000.0)
            return
        if kind == "ping":
            self._send_control("pong")
            return

    def _on_ui_command(self, event: UICommandEvent) -> None:
        if not isinstance(event, UICommandEvent):
            return
        with self._socket_lock:
            sock = self._socket
        if sock is None:
            return
        try:
            packet = pack_event_message(EventTopic.UI_COMMAND.value, event)
            with self._send_lock:
                sock.sendall(packet)
        except Exception:
            self._close_socket()

    def _send_control(self, kind: str, detail: Mapping[str, Any] | None = None) -> bool:
        with self._socket_lock:
            sock = self._socket
            session_id = self._session_id
        if sock is None:
            return False
        try:
            packet = pack_control_message(kind, session_id=session_id, detail=detail)
            with self._send_lock:
                sock.sendall(packet)
            return True
        except Exception:
            self._close_socket()
            return False

    def _close_socket(self) -> None:
        with self._socket_lock:
            sock = self._socket
            self._socket = None
            self._session_id = 0
            self._pending_pings.clear()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass
