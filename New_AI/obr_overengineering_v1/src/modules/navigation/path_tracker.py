from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from pathlib import Path

try:
    from ...core.event_bus import (
        EventBus,
        EventBusFullError,
        EventTopic,
        PathEvent,
        PoseEvent,
        Subscription,
    )
except ImportError:  # pragma: no cover
    from core.event_bus import (
        EventBus,
        EventBusFullError,
        EventTopic,
        PathEvent,
        PoseEvent,
        Subscription,
    )


PoseTuple = tuple[float, float, float, float]


def _angle_delta(a: float, b: float) -> float:
    delta = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(delta)


def _default_snapshot_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "runtime" / "path_snapshots"


class PathTracker:
    """Bounded path tracker for nav.pose -> nav.path with rotating JSON snapshots."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        *,
        max_points: int = 1000,
        publish_interval_ms: int = 100,
        min_step_distance: float = 1e-3,
        min_theta_delta: float = 5e-3,
        snapshot_dir: str | Path | None = None,
        snapshot_prefix: str = "nav_path_snapshot",
        snapshot_rotate: int = 4,
        snapshot_interval_s: float = 1.0,
        subscribe_nav_pose: bool = True,
        start_worker: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        if max_points <= 1:
            raise ValueError("max_points must be > 1")
        if publish_interval_ms <= 0:
            raise ValueError("publish_interval_ms must be > 0")
        if snapshot_rotate < 0:
            raise ValueError("snapshot_rotate cannot be negative")
        if snapshot_interval_s < 0:
            raise ValueError("snapshot_interval_s cannot be negative")

        self._event_bus = event_bus
        self._poses: deque[PoseTuple] = deque(maxlen=int(max_points))
        self._lock = threading.RLock()
        self._logger = logger or logging.getLogger("modules.navigation.path_tracker")

        self._publish_interval_s = max(0.01, float(publish_interval_ms) / 1000.0)
        self._min_step_distance_sq = float(min_step_distance) * float(min_step_distance)
        self._min_theta_delta = max(0.0, float(min_theta_delta))

        self._snapshot_prefix = snapshot_prefix.strip() or "nav_path_snapshot"
        self._snapshot_rotate = int(snapshot_rotate)
        self._snapshot_interval_s = float(snapshot_interval_s)
        self._snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else _default_snapshot_dir()
        if self._snapshot_rotate > 0:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._generation = 0
        self._last_published_generation = -1
        self._last_snapshot_generation = -1
        self._last_pose: PoseTuple | None = None
        self._snapshot_index = 0

        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

        self._pose_subscription: Subscription | None = None
        if self._event_bus is not None and subscribe_nav_pose:
            self._pose_subscription = self._event_bus.subscribe(EventTopic.NAV_POSE, self._on_pose_event)

        if start_worker:
            self.start()

    @property
    def max_points(self) -> int:
        return int(self._poses.maxlen or 0)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._poses)

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="path-tracker", daemon=True)
        self._worker.start()

    def stop(self, timeout: float = 1.5) -> None:
        self._stop_event.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=timeout)
        self._worker = None

    def close(self) -> None:
        self.stop()
        if self._pose_subscription is not None:
            self._pose_subscription.unsubscribe()
            self._pose_subscription = None

    def push(self, x: float, y: float, theta: float, timestamp: float | None = None) -> None:
        ts = time.time() if timestamp is None else float(timestamp)
        pose: PoseTuple = (float(x), float(y), float(theta), ts)
        self._append_pose(pose)

    def snapshot(self) -> list[PoseEvent]:
        with self._lock:
            data = list(self._poses)
        return [
            PoseEvent(timestamp=ts, x=x, y=y, theta=theta)
            for x, y, theta, ts in data
        ]

    def snapshot_tuples(self) -> list[PoseTuple]:
        with self._lock:
            return list(self._poses)

    def clear(self) -> None:
        with self._lock:
            self._poses.clear()
            self._last_pose = None
            self._generation += 1

    def recover_latest_snapshot(self) -> int:
        if self._snapshot_rotate <= 0 or not self._snapshot_dir.exists():
            return 0

        pattern = f"{self._snapshot_prefix}_*.json"
        files = sorted(
            self._snapshot_dir.glob(pattern),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        for file_path in files:
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            raw_poses = payload.get("poses")
            if not isinstance(raw_poses, list):
                continue

            loaded: list[PoseTuple] = []
            for item in raw_poses:
                pose = self._coerce_pose_dict(item)
                if pose is not None:
                    loaded.append(pose)

            if not loaded:
                continue

            with self._lock:
                self._poses.clear()
                for pose in loaded[-self.max_points :]:
                    self._poses.append(pose)
                self._last_pose = self._poses[-1]
                self._generation += 1
            return len(self._poses)

        return 0

    def _on_pose_event(self, event: PoseEvent) -> None:
        if not isinstance(event, PoseEvent):
            return
        self.push(event.x, event.y, event.theta, event.timestamp)

    def _append_pose(self, pose: PoseTuple) -> bool:
        with self._lock:
            if self._should_skip_pose(pose):
                return False
            self._poses.append(pose)
            self._last_pose = pose
            self._generation += 1
            return True

    def _should_skip_pose(self, pose: PoseTuple) -> bool:
        last = self._last_pose
        if last is None:
            return False

        dx = pose[0] - last[0]
        dy = pose[1] - last[1]
        if (dx * dx + dy * dy) > self._min_step_distance_sq:
            return False

        if _angle_delta(pose[2], last[2]) >= self._min_theta_delta:
            return False

        return True

    def _worker_loop(self) -> None:
        next_snapshot_due = time.monotonic() + self._snapshot_interval_s

        while not self._stop_event.wait(self._publish_interval_s):
            self._publish_path_if_changed()

            if self._snapshot_rotate <= 0 or self._snapshot_interval_s <= 0:
                continue

            now = time.monotonic()
            if now >= next_snapshot_due:
                self._write_snapshot_if_changed()
                next_snapshot_due = now + self._snapshot_interval_s

    def _publish_path_if_changed(self) -> None:
        if self._event_bus is None:
            return

        with self._lock:
            generation = self._generation

        if generation == self._last_published_generation:
            return

        poses = self.snapshot()
        try:
            self._event_bus.publish(
                EventTopic.NAV_PATH,
                PathEvent(timestamp=time.time(), poses=poses),
            )
        except EventBusFullError:
            self._logger.debug("nav.path publish skipped: event bus queue full")
            return
        except Exception as exc:
            self._logger.debug("nav.path publish failed: %s", exc)
            return

        self._last_published_generation = generation

    def _write_snapshot_if_changed(self) -> None:
        with self._lock:
            if self._generation == self._last_snapshot_generation:
                return
            data = list(self._poses)
            generation = self._generation

        payload = {
            "version": 1,
            "saved_at": time.time(),
            "count": len(data),
            "poses": [
                {
                    "x": x,
                    "y": y,
                    "theta": theta,
                    "timestamp": ts,
                }
                for x, y, theta, ts in data
            ],
        }

        file_path = self._snapshot_dir / f"{self._snapshot_prefix}_{self._snapshot_index:02d}.json"
        temp_path = file_path.with_suffix(".tmp")
        try:
            temp_path.write_text(
                json.dumps(payload, separators=(",", ":")),
                encoding="utf-8",
            )
            temp_path.replace(file_path)
        except Exception as exc:
            self._logger.debug("path snapshot write failed: %s", exc)
            return

        self._snapshot_index = (self._snapshot_index + 1) % max(1, self._snapshot_rotate)
        self._last_snapshot_generation = generation

    @staticmethod
    def _coerce_pose_dict(item: object) -> PoseTuple | None:
        if not isinstance(item, dict):
            return None
        try:
            x = float(item.get("x", 0.0))
            y = float(item.get("y", 0.0))
            theta = float(item.get("theta", 0.0))
            timestamp = float(item.get("timestamp", time.time()))
        except Exception:
            return None
        return (x, y, theta, timestamp)

    def __enter__(self) -> "PathTracker":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()
