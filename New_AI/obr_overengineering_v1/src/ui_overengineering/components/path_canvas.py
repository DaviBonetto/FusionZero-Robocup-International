from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Mapping, Sequence

from PyQt6.QtCore import QPointF, QRectF, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QLabel, QSizePolicy, QWidget

from .base import CardFrame

try:
    from ...core.event_bus import EventBus, EventTopic, PathEvent, PoseEvent, Subscription
except ImportError:  # pragma: no cover
    from core.event_bus import EventBus, EventTopic, PathEvent, PoseEvent, Subscription


PoseTuple = tuple[float, float, float, float]

_START_COLOR = QColor("#ADD8E6")
_END_COLOR = QColor("#00008B")
_GRID_COLOR = QColor("#2D3440")
_BG_COLOR = QColor("#101216")
_BORDER_COLOR = QColor("#3A3F48")


def _lerp_color(start: QColor, end: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, float(t)))
    r = int(start.red() + (end.red() - start.red()) * t)
    g = int(start.green() + (end.green() - start.green()) * t)
    b = int(start.blue() + (end.blue() - start.blue()) * t)
    return QColor(r, g, b)


class _PathSurface(QWidget):
    def __init__(self, owner: "PathCanvas") -> None:
        super().__init__(owner)
        self._owner = owner
        self.setObjectName("VideoLabel")
        self.setMinimumSize(280, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event: Any) -> None:  # noqa: ANN401
        super().paintEvent(event)
        self._owner._paint_surface(self)


class PathCanvas(CardFrame):
    """Bounded 2D path renderer with gradient trail for long-running UI sessions."""

    path_event_received = pyqtSignal(object)
    pose_event_received = pyqtSignal(object)

    def __init__(
        self,
        title: str = "Path Tracker",
        parent: QWidget | None = None,
        *,
        max_points: int = 1000,
        refresh_interval_ms: int = 100,
        padding_ratio: float = 0.12,
        min_pixel_distance: float = 1.0,
        event_bus: EventBus | None = None,
        subscribe_path: bool = True,
        subscribe_pose: bool = False,
    ) -> None:
        super().__init__(title, parent)
        if max_points <= 1:
            raise ValueError("max_points must be > 1")

        self._lock = threading.RLock()
        self._poses: deque[PoseTuple] = deque(maxlen=int(max_points))
        self._min_world_distance_sq = 1e-12
        self._min_pixel_distance = max(0.0, float(min_pixel_distance))
        self._padding_ratio = max(0.0, min(0.45, float(padding_ratio)))

        self._data_generation = 0
        self._cached_generation = -1
        self._cached_size: tuple[int, int] = (-1, -1)
        self._cached_points: list[QPointF] = []
        self._cached_last_theta = 0.0

        self._subscriptions: list[Subscription] = []

        self._surface = _PathSurface(self)
        self.content_layout.addWidget(self._surface, 1)

        self._meta_label = QLabel("No path data", self)
        self._meta_label.setObjectName("CardTitle")
        self._meta_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.content_layout.addWidget(self._meta_label)

        self._refresh_label = QLabel("Render 100 ms", self)
        self._refresh_label.setObjectName("CardTitle")
        self._refresh_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.content_layout.addWidget(self._refresh_label)

        self.path_event_received.connect(self._on_path_event)
        self.pose_event_received.connect(self._on_pose_event)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._on_refresh_tick)
        self.set_refresh_interval(refresh_interval_ms)
        self._refresh_timer.start()

        if event_bus is not None:
            self.attach_event_bus(
                event_bus,
                subscribe_path=subscribe_path,
                subscribe_pose=subscribe_pose,
            )

    @property
    def size_points(self) -> int:
        with self._lock:
            return len(self._poses)

    def set_refresh_interval(self, interval_ms: int) -> None:
        clamped = max(16, int(interval_ms))
        self._refresh_timer.setInterval(clamped)
        self._refresh_label.setText(f"Render {clamped} ms")

    def attach_event_bus(
        self,
        event_bus: EventBus,
        *,
        subscribe_path: bool = True,
        subscribe_pose: bool = False,
    ) -> None:
        self.detach_event_bus()

        if subscribe_path:
            self._subscriptions.append(
                event_bus.subscribe(EventTopic.NAV_PATH, lambda event: self.path_event_received.emit(event))
            )
        if subscribe_pose:
            self._subscriptions.append(
                event_bus.subscribe(EventTopic.NAV_POSE, lambda event: self.pose_event_received.emit(event))
            )

    def detach_event_bus(self) -> None:
        for sub in self._subscriptions:
            try:
                sub.unsubscribe()
            except Exception:
                pass
        self._subscriptions.clear()

    def clear(self) -> None:
        with self._lock:
            self._poses.clear()
            self._data_generation += 1

    def set_path(self, poses: Sequence[Any]) -> None:  # noqa: ANN401
        normalized: list[PoseTuple] = []
        for item in poses:
            pose = self._coerce_pose(item)
            if pose is not None:
                normalized.append(pose)

        with self._lock:
            self._poses.clear()
            if normalized:
                self._poses.extend(normalized[-int(self._poses.maxlen or 0) :])
            self._data_generation += 1

    def append_pose(self, x: float, y: float, theta: float, timestamp: float | None = None) -> None:
        ts = time.time() if timestamp is None else float(timestamp)
        pose: PoseTuple = (float(x), float(y), float(theta), ts)

        with self._lock:
            if self._poses:
                last = self._poses[-1]
                dx = pose[0] - last[0]
                dy = pose[1] - last[1]
                if (dx * dx + dy * dy) < self._min_world_distance_sq:
                    return
            self._poses.append(pose)
            self._data_generation += 1

    def snapshot(self) -> list[PoseEvent]:
        with self._lock:
            values = list(self._poses)
        return [
            PoseEvent(timestamp=ts, x=x, y=y, theta=theta)
            for x, y, theta, ts in values
        ]

    @pyqtSlot(object)
    def _on_path_event(self, event: object) -> None:
        if not isinstance(event, PathEvent):
            return
        poses = event.poses if isinstance(event.poses, list) else []
        self.set_path(poses)

    @pyqtSlot(object)
    def _on_pose_event(self, event: object) -> None:
        if not isinstance(event, PoseEvent):
            return
        self.append_pose(event.x, event.y, event.theta, event.timestamp)

    @pyqtSlot()
    def _on_refresh_tick(self) -> None:
        if self._rebuild_cache_if_needed():
            self._surface.update()

    def _rebuild_cache_if_needed(self) -> bool:
        with self._lock:
            generation = self._data_generation
            poses = list(self._poses)

        size_key = (self._surface.width(), self._surface.height())
        if generation == self._cached_generation and size_key == self._cached_size:
            return False

        draw_rect = self._draw_rect(self._surface)
        if not poses or draw_rect.width() <= 1.0 or draw_rect.height() <= 1.0:
            self._cached_points = []
            self._cached_last_theta = 0.0
            self._cached_generation = generation
            self._cached_size = size_key
            self._meta_label.setText("No path data")
            return True

        xs = [pose[0] for pose in poses]
        ys = [pose[1] for pose in poses]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        span_x = max_x - min_x
        span_y = max_y - min_y
        span = max(span_x, span_y, 1e-9)
        padded_span = span * (1.0 + 2.0 * self._padding_ratio)

        center_x = 0.5 * (min_x + max_x)
        center_y = 0.5 * (min_y + max_y)
        center_sx = draw_rect.center().x()
        center_sy = draw_rect.center().y()
        scale = min(draw_rect.width(), draw_rect.height()) / padded_span

        out_points: list[QPointF] = []
        pixel_threshold_sq = self._min_pixel_distance * self._min_pixel_distance

        for idx, (x, y, _theta, _ts) in enumerate(poses):
            sx = center_sx + (x - center_x) * scale
            sy = center_sy - (y - center_y) * scale
            point = QPointF(sx, sy)

            if out_points:
                dx = point.x() - out_points[-1].x()
                dy = point.y() - out_points[-1].y()
                dist_sq = dx * dx + dy * dy
                is_last = idx == (len(poses) - 1)
                if dist_sq < pixel_threshold_sq and not is_last:
                    continue

            out_points.append(point)

        if not out_points:
            out_points.append(QPointF(center_sx, center_sy))

        self._cached_points = out_points
        self._cached_last_theta = poses[-1][2]
        self._cached_generation = generation
        self._cached_size = size_key
        self._meta_label.setText(
            f"{len(poses)} poses | span={span:.4f} world-units"
        )
        return True

    def _paint_surface(self, widget: QWidget) -> None:
        painter = QPainter(widget)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        painter.fillRect(widget.rect(), _BG_COLOR)

        draw_rect = self._draw_rect(widget)
        painter.setPen(QPen(_BORDER_COLOR, 1.0))
        painter.drawRoundedRect(draw_rect, 8.0, 8.0)

        self._draw_grid(painter, draw_rect)

        points = self._cached_points
        if len(points) >= 2:
            total_segments = len(points) - 1
            for idx in range(1, len(points)):
                t = (idx - 1) / max(1, total_segments - 1)
                color = _lerp_color(_START_COLOR, _END_COLOR, t)
                pen = QPen(color, 2.0)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(points[idx - 1], points[idx])
        elif len(points) == 1:
            painter.setPen(QPen(_END_COLOR, 3.0))
            painter.drawPoint(points[0])

        if points:
            start = points[0]
            end = points[-1]

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_START_COLOR)
            painter.drawEllipse(start, 3.5, 3.5)

            painter.setBrush(_END_COLOR)
            painter.drawEllipse(end, 4.5, 4.5)

            self._draw_heading(painter, end, self._cached_last_theta)

    def _draw_grid(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(_GRID_COLOR, 1.0, Qt.PenStyle.DashLine))
        for ratio in (0.25, 0.5, 0.75):
            x = rect.left() + rect.width() * ratio
            y = rect.top() + rect.height() * ratio
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

    def _draw_heading(self, painter: QPainter, point: QPointF, theta: float) -> None:
        length = 14.0
        angle = -float(theta)

        tip = QPointF(
            point.x() + math.cos(angle) * length,
            point.y() + math.sin(angle) * length,
        )
        wing = 6.0
        left = QPointF(
            tip.x() - math.cos(angle - 0.65) * wing,
            tip.y() - math.sin(angle - 0.65) * wing,
        )
        right = QPointF(
            tip.x() - math.cos(angle + 0.65) * wing,
            tip.y() - math.sin(angle + 0.65) * wing,
        )

        painter.setPen(QPen(QColor("#D5E6FF"), 1.4))
        painter.drawLine(point, tip)
        painter.drawLine(tip, left)
        painter.drawLine(tip, right)

    @staticmethod
    def _draw_rect(widget: QWidget) -> QRectF:
        return QRectF(widget.rect().adjusted(8, 8, -8, -8))

    @staticmethod
    def _coerce_pose(item: object) -> PoseTuple | None:
        if isinstance(item, PoseEvent):
            return (float(item.x), float(item.y), float(item.theta), float(item.timestamp))

        if isinstance(item, Mapping):
            try:
                x = float(item.get("x", 0.0))
                y = float(item.get("y", 0.0))
                theta = float(item.get("theta", 0.0))
                ts = float(item.get("timestamp", time.time()))
            except Exception:
                return None
            return (x, y, theta, ts)

        if isinstance(item, Sequence) and len(item) >= 4:
            try:
                x = float(item[0])
                y = float(item[1])
                theta = float(item[2])
                ts = float(item[3])
            except Exception:
                return None
            return (x, y, theta, ts)

        return None

    def closeEvent(self, event: Any) -> None:  # noqa: ANN401
        self._refresh_timer.stop()
        self.detach_event_bus()
        super().closeEvent(event)
