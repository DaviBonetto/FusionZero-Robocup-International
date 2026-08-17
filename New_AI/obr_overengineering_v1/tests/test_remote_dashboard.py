from __future__ import annotations

import socket
import time

import cv2
import numpy as np

from conftest import wait_until
from src.core.event_bus import EventBus, EventTopic, FrameEvent, HealthEvent, LogEvent, PathEvent, PoseEvent, StateSnapshotEvent, VisionDetectionEvent
from src.remote_dashboard import (
    CONTROL_TOPIC,
    ControlMessage,
    RemoteDashboardClient,
    RemoteDashboardServer,
    pack_event_message,
    recv_event_message,
)


def _drain(bus: EventBus) -> None:
    bus._queue.join()


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def test_remote_dashboard_roundtrip_control_message() -> None:
    left, right = socket.socketpair()
    try:
        event = ControlMessage(kind="ping", detail={"origin": "pc"}, session_id=3)
        right.sendall(pack_event_message(CONTROL_TOPIC, event))
        topic, restored = recv_event_message(left)
    finally:
        left.close()
        right.close()

    assert topic == CONTROL_TOPIC
    assert isinstance(restored, ControlMessage)
    assert restored.kind == "ping"
    assert restored.detail["origin"] == "pc"
    assert restored.session_id == 3


def test_remote_dashboard_roundtrip_detection_event() -> None:
    left, right = socket.socketpair()
    try:
        event = VisionDetectionEvent(
            timestamp=time.time(),
            state="RESCUE_ZONE_DETECTED",
            line=False,
            balls=2,
            green=False,
            red=True,
            victims=1,
            latency_ms=14.2,
            metadata={"silver_ball_count": 2, "green_corner_found": True},
        )
        right.sendall(pack_event_message("vision.detections", event))
        topic, restored = recv_event_message(left)
    finally:
        left.close()
        right.close()

    assert topic == "vision.detections"
    assert isinstance(restored, VisionDetectionEvent)
    assert restored.state == event.state
    assert restored.balls == 2
    assert restored.metadata["green_corner_found"] is True


def test_remote_dashboard_roundtrip_health_event_keeps_metadata() -> None:
    left, right = socket.socketpair()
    try:
        event = HealthEvent(
            timestamp=time.time(),
            cpu_percent=42.0,
            fps_capture=18.0,
            fps_process=16.5,
            fps_ui=16.5,
            queue_depth=3,
            metadata={"camera": {"state": "online"}, "recording": {"enabled": False}},
        )
        right.sendall(pack_event_message("system.health", event))
        topic, restored = recv_event_message(left)
    finally:
        left.close()
        right.close()

    assert topic == "system.health"
    assert isinstance(restored, HealthEvent)
    assert restored.metadata["camera"]["state"] == "online"


def test_remote_dashboard_roundtrip_frame_event_compresses_to_jpeg() -> None:
    left, right = socket.socketpair()
    frame = np.full((120, 160, 3), 180, dtype=np.uint8)
    cv2.rectangle(frame, (30, 20), (120, 90), (0, 220, 0), thickness=-1)
    try:
        event = FrameEvent(
            timestamp=time.time(),
            frame_id=7,
            width=160,
            height=120,
            encoding="bgr8",
            data=frame.tobytes(),
        )
        right.sendall(pack_event_message("vision.processed_frame", event, jpeg_quality=70))
        topic, restored = recv_event_message(left)
    finally:
        left.close()
        right.close()

    assert topic == "vision.processed_frame"
    assert isinstance(restored, FrameEvent)
    assert restored.encoding == "jpeg"
    decoded = cv2.imdecode(np.frombuffer(restored.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (120, 160)


def test_remote_dashboard_frame_forwarder_drops_stale_frames_without_blocking_event_bus() -> None:
    bus = EventBus(max_queue_size=128, drop_oldest=False)
    server = RemoteDashboardServer(bus, host="127.0.0.1", port=0, stream_fps=30.0)
    sent_frame_ids: list[int] = []

    def slow_send(_topic: str, event: object) -> bool:
        time.sleep(0.15)
        if isinstance(event, FrameEvent):
            sent_frame_ids.append(event.frame_id)
        return True

    server._send_event = slow_send  # type: ignore[method-assign]
    server.start()
    try:
        started_at = time.monotonic()
        for frame_id in range(8):
            bus.publish(
                EventTopic.VISION_RAW_FRAME,
                FrameEvent(
                    timestamp=time.time(),
                    frame_id=frame_id,
                    width=2,
                    height=2,
                    encoding="bgr8",
                    data=bytes(12),
                ),
            )
        _drain(bus)
        assert (time.monotonic() - started_at) < 0.10
        assert wait_until(lambda: 7 in sent_frame_ids, timeout=1.5)
        assert len(sent_frame_ids) < 8
    finally:
        server.stop()
        bus.stop()


def test_remote_dashboard_roundtrip_path_event_restores_pose_objects() -> None:
    left, right = socket.socketpair()
    try:
        event = PathEvent(
            timestamp=time.time(),
            poses=[
                PoseEvent(timestamp=time.time(), x=0.1, y=0.2, theta=0.3),
                PoseEvent(timestamp=time.time(), x=0.4, y=0.5, theta=0.6),
            ],
        )
        right.sendall(pack_event_message("nav.path", event))
        topic, restored = recv_event_message(left)
    finally:
        left.close()
        right.close()

    assert topic == "nav.path"
    assert isinstance(restored, PathEvent)
    assert len(restored.poses) == 2
    assert isinstance(restored.poses[0], PoseEvent)
    assert restored.poses[1].theta == 0.6


def test_remote_dashboard_server_replays_snapshot_to_new_client() -> None:
    bus = EventBus(max_queue_size=128, drop_oldest=False)
    server = RemoteDashboardServer(bus, host="127.0.0.1", port=0, stream_fps=30.0, log_replay_size=4)
    server.start()
    sock: socket.socket | None = None
    try:
        bus.publish(EventTopic.FSM_STATE, StateSnapshotEvent(timestamp=time.time(), state="FOLLOWING_LINE"))
        bus.publish(
            EventTopic.VISION_DETECTIONS,
            VisionDetectionEvent(timestamp=time.time(), state="FOLLOWING_LINE", line=True, green=True, metadata={}),
        )
        bus.publish(
            EventTopic.SYSTEM_LOG,
            LogEvent(timestamp=time.time(), level="INFO", message="before connect", source="test", state="FOLLOWING_LINE"),
        )
        _drain(bus)

        sock = socket.create_connection(("127.0.0.1", server.port), timeout=1.5)
        sock.settimeout(1.5)
        received: list[tuple[str, object]] = []
        deadline = time.time() + 2.0
        while time.time() < deadline and len(received) < 4:
            try:
                received.append(recv_event_message(sock))
            except socket.timeout:
                continue

        topics = [topic for topic, _ in received]
        assert CONTROL_TOPIC in topics
        assert EventTopic.FSM_STATE.value in topics
        assert EventTopic.VISION_DETECTIONS.value in topics
        assert EventTopic.SYSTEM_LOG.value in topics

        hello = next(event for topic, event in received if topic == CONTROL_TOPIC)
        assert isinstance(hello, ControlMessage)
        assert hello.kind == "hello"
        assert any(
            isinstance(event, LogEvent) and event.message == "before connect"
            for topic, event in received
            if topic == EventTopic.SYSTEM_LOG.value
        )
    finally:
        if sock is not None:
            sock.close()
        server.stop()
        bus.stop()


def test_remote_dashboard_client_reconnects_after_server_restart() -> None:
    port = _free_port()
    client_bus = EventBus(max_queue_size=128, drop_oldest=False)
    client_logs: list[LogEvent] = []
    sub = client_bus.subscribe(EventTopic.SYSTEM_LOG, lambda event: client_logs.append(event))  # type: ignore[arg-type]

    server_bus = EventBus(max_queue_size=128, drop_oldest=False)
    server = RemoteDashboardServer(server_bus, host="127.0.0.1", port=port, stream_fps=30.0, client_timeout=1.5)
    client = RemoteDashboardClient(
        client_bus,
        host="127.0.0.1",
        port=port,
        reconnect_interval=0.2,
        heartbeat_interval=0.2,
        server_timeout=1.0,
    )
    server.start()
    client.start()

    replacement_bus: EventBus | None = None
    replacement_server: RemoteDashboardServer | None = None
    try:
        server_bus.publish(
            EventTopic.SYSTEM_LOG,
            LogEvent(timestamp=time.time(), level="INFO", message="first payload", source="test", state=""),
        )
        _drain(server_bus)
        assert wait_until(lambda: any(event.message == "first payload" for event in client_logs), timeout=2.0)

        server.stop()
        server_bus.stop()
        assert wait_until(
            lambda: any("remote dashboard disconnected" in event.message for event in client_logs),
            timeout=2.0,
        )

        replacement_bus = EventBus(max_queue_size=128, drop_oldest=False)
        replacement_server = RemoteDashboardServer(
            replacement_bus,
            host="127.0.0.1",
            port=port,
            stream_fps=30.0,
            client_timeout=1.5,
        )
        replacement_server.start()
        replacement_bus.publish(
            EventTopic.SYSTEM_LOG,
            LogEvent(timestamp=time.time(), level="INFO", message="second payload", source="test", state=""),
        )
        _drain(replacement_bus)

        assert wait_until(lambda: any(event.message == "second payload" for event in client_logs), timeout=3.0)
        assert any("connected to raspberry" in event.message for event in client_logs)
        assert any("session ready" in event.message for event in client_logs)
    finally:
        sub.unsubscribe()
        client.stop()
        client_bus.stop()
        if replacement_server is not None:
            replacement_server.stop()
        else:
            server.stop()
        if replacement_bus is not None:
            replacement_bus.stop()
        else:
            server_bus.stop()


def test_remote_dashboard_client_enriches_health_with_network_latency() -> None:
    port = _free_port()
    client_bus = EventBus(max_queue_size=128, drop_oldest=False)
    server_bus = EventBus(max_queue_size=128, drop_oldest=False)
    health_events: list[HealthEvent] = []
    sub = client_bus.subscribe(EventTopic.SYSTEM_HEALTH, lambda event: health_events.append(event))  # type: ignore[arg-type]

    server = RemoteDashboardServer(server_bus, host="127.0.0.1", port=port, stream_fps=30.0, client_timeout=1.5)
    client = RemoteDashboardClient(
        client_bus,
        host="127.0.0.1",
        port=port,
        reconnect_interval=0.2,
        heartbeat_interval=0.2,
        server_timeout=1.0,
    )
    server.start()
    client.start()
    try:
        time.sleep(0.35)
        server_bus.publish(
            EventTopic.SYSTEM_HEALTH,
            HealthEvent(
                timestamp=time.time(),
                cpu_percent=55.0,
                fps_capture=20.0,
                fps_process=18.0,
                fps_ui=18.0,
                queue_depth=2,
                metadata={"network": {"state": "local"}},
            ),
        )
        _drain(server_bus)
        assert wait_until(lambda: len(health_events) > 0, timeout=2.0)
        latest = health_events[-1]
        assert latest.metadata["network"]["state"] == "connected"
        assert latest.metadata["network"]["peer"] == "127.0.0.1:{}".format(port)
        assert "network_latency_ms" in latest.metadata
    finally:
        sub.unsubscribe()
        client.stop()
        client_bus.stop()
        server.stop()
        server_bus.stop()
