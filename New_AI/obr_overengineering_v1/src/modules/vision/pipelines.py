from __future__ import annotations

import importlib
import json
import contextlib
import io
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

try:
    from ...core.event_bus import VisionDetectionEvent
    from ...core.state_machine import RobotState
except ImportError:  # pragma: no cover
    from core.event_bus import VisionDetectionEvent
    from core.state_machine import RobotState

try:
    from .preprocessor import VisionPreprocessor
except ImportError:  # pragma: no cover
    from preprocessor import VisionPreprocessor


_FILE = Path(__file__).resolve()
_OBR_ROOT = _FILE.parents[3]
_REPO_ROOT = _FILE.parents[5]
DEFAULT_CONFIG_PATH = _OBR_ROOT / "configs" / "vision_config.json"


def _state_name(state: RobotState | str) -> str:
    if isinstance(state, RobotState):
        return state.value
    raw = str(state).strip().upper()
    return raw if raw in RobotState._value2member_map_ else RobotState.SEARCHING_LINE.value


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _empty_mask(height: int, width: int) -> np.ndarray:
    return np.zeros((max(0, int(height)), max(0, int(width))), dtype=np.uint8)


def _mask_ratio(mask: np.ndarray | None) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask) / max(1, mask.shape[0] * mask.shape[1]))


def _safe_bbox(
    x: int | float,
    y: int | float,
    w: int | float,
    h: int | float,
    frame_shape: tuple[int, ...],
) -> dict[str, int] | None:
    frame_h = int(frame_shape[0]) if len(frame_shape) > 0 else 0
    frame_w = int(frame_shape[1]) if len(frame_shape) > 1 else 0
    if frame_w <= 0 or frame_h <= 0:
        return None

    x0 = max(0, min(frame_w - 1, int(round(x))))
    y0 = max(0, min(frame_h - 1, int(round(y))))
    ww = max(1, int(round(w)))
    hh = max(1, int(round(h)))
    if x0 + ww > frame_w:
        ww = frame_w - x0
    if y0 + hh > frame_h:
        hh = frame_h - y0
    if ww <= 0 or hh <= 0:
        return None
    return {"x": int(x0), "y": int(y0), "w": int(ww), "h": int(hh)}


def _bbox_from_circle(
    cx: int | float,
    cy: int | float,
    radius: int | float,
    frame_shape: tuple[int, ...],
) -> dict[str, int] | None:
    r = max(1, int(round(radius)))
    return _safe_bbox(int(round(cx)) - r, int(round(cy)) - r, r * 2, r * 2, frame_shape)


def _bbox_iou(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> float:
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return 0.0
    ax = int(a.get("x", -1))
    ay = int(a.get("y", -1))
    aw = int(a.get("w", -1))
    ah = int(a.get("h", -1))
    bx = int(b.get("x", -1))
    by = int(b.get("y", -1))
    bw = int(b.get("w", -1))
    bh = int(b.get("h", -1))
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = float(inter_w * inter_h)
    union = float(aw * ah + bw * bh) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


@dataclass(slots=True)
class VisionConfig:
    data: dict[str, Any]
    config_path: Path
    project_root: Path
    cache_key: str

    @property
    def preprocessor(self) -> Mapping[str, Any]:
        raw = self.data.get("preprocessor")
        return raw if isinstance(raw, Mapping) else {}

    def pipeline(self, state_name: str) -> Mapping[str, Any]:
        raw = self.data.get("pipelines", {})
        if not isinstance(raw, Mapping):
            return {}
        state_cfg = raw.get(state_name)
        if isinstance(state_cfg, Mapping):
            return state_cfg
        default_cfg = raw.get("default")
        return default_cfg if isinstance(default_cfg, Mapping) else {}

    def detector(self, detector_name: str) -> Mapping[str, Any]:
        raw = self.data.get("detectors", {})
        if not isinstance(raw, Mapping):
            return {}
        detector_cfg = raw.get(detector_name)
        return detector_cfg if isinstance(detector_cfg, Mapping) else {}

    def model_path(self, model_name: str) -> Path | None:
        raw = self.data.get("models", {})
        if not isinstance(raw, Mapping):
            return None
        candidate = raw.get(model_name)
        if not candidate:
            return None
        path = Path(str(candidate))
        return path if path.is_absolute() else (self.project_root / path).resolve()

    def benchmark(self) -> Mapping[str, Any]:
        raw = self.data.get("benchmark")
        return raw if isinstance(raw, Mapping) else {}


def load_vision_config(config_path: str | Path | None = None) -> VisionConfig:
    if config_path is None:
        resolved = DEFAULT_CONFIG_PATH.resolve()
    else:
        p = Path(config_path)
        resolved = p if p.is_absolute() else (_REPO_ROOT / p).resolve()

    with resolved.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    model_root = _REPO_ROOT
    paths_cfg = data.get("paths")
    if isinstance(paths_cfg, Mapping):
        model_root_raw = paths_cfg.get("model_root")
        if model_root_raw:
            model_root_candidate = Path(str(model_root_raw))
            model_root = (
                model_root_candidate
                if model_root_candidate.is_absolute()
                else (_REPO_ROOT / model_root_candidate).resolve()
            )

    cache_key = f"{resolved}:{resolved.stat().st_mtime_ns}"
    return VisionConfig(
        data=data,
        config_path=resolved,
        project_root=model_root,
        cache_key=cache_key,
    )


class LineDetector:
    def __init__(self, width: int, height: int, settings: Mapping[str, Any] | None = None) -> None:
        cfg = dict(settings or {})
        self.width = int(width)
        self.height = int(height)
        self.min_black_area = int(cfg.get("min_black_area", 50))
        self.prev_angle = 90
        self.gap_found = 0

        self.black_h_max = int(cfg.get("black_h_max", 180))
        self.black_s_max = int(cfg.get("black_s_max", 255))
        self.black_v_max = int(cfg.get("black_v_max", 70))
        self.erode_iter = int(cfg.get("erode_iter", 3))
        self.dilate_iter = int(cfg.get("dilate_iter", 4))
        self.erode_ksize = int(cfg.get("erode_ksize", 3))
        self.dilate_ksize = int(cfg.get("dilate_ksize", 3))
        self.line_border_margin = int(cfg.get("line_border_margin", 8))
        self.round_blob_circularity = float(cfg.get("round_blob_circularity", 0.72))
        self.round_blob_aspect = float(cfg.get("round_blob_aspect", 1.25))
        self.min_aspect_without_border = float(cfg.get("min_aspect_without_border", 1.35))
        self.min_area_ratio_for_internal = float(cfg.get("min_area_ratio_for_internal", 0.10))
        self.min_internal_long_side_ratio = float(cfg.get("min_internal_long_side_ratio", 0.45))
        self.compact_extent_reject = float(cfg.get("compact_extent_reject", 0.44))
        self.compact_solidity_reject = float(cfg.get("compact_solidity_reject", 0.70))
        self.compact_circularity_reject = float(cfg.get("compact_circularity_reject", 0.36))
        # A line-following candidate must look like a narrow ground path.  A
        # previous border-touch shortcut accepted any black object reaching
        # the image edge, including tables and clothing.
        self.min_ground_span_ratio = float(cfg.get("min_ground_span_ratio", 0.38))
        self.min_vertical_support_ratio = float(cfg.get("min_vertical_support_ratio", 0.55))
        self.min_line_aspect = float(cfg.get("min_line_aspect", 1.25))
        self.max_line_width_ratio = float(cfg.get("max_line_width_ratio", 0.52))
        self.max_line_row_occupancy = float(cfg.get("max_line_row_occupancy", 0.48))
        self.max_bottom_row_occupancy = float(cfg.get("max_bottom_row_occupancy", 0.38))
        self.bottom_band_ratio = float(cfg.get("bottom_band_ratio", 0.12))
        self.min_row_occupancy = float(cfg.get("min_row_occupancy", 0.01))
        # A nearby black track can fill much of the camera width while still
        # being a valid ground corridor. Keep this path separate from the
        # narrow-line rules so a full-width table/object is still rejected.
        self.wide_corridor_enabled = bool(cfg.get("wide_corridor_enabled", True))
        self.wide_corridor_min_height_ratio = float(cfg.get("wide_corridor_min_height_ratio", 0.85))
        self.wide_corridor_max_width_ratio = float(cfg.get("wide_corridor_max_width_ratio", 0.90))
        self.wide_corridor_min_side_gap_ratio = float(cfg.get("wide_corridor_min_side_gap_ratio", 0.08))
        self.wide_corridor_min_side_support_ratio = float(cfg.get("wide_corridor_min_side_support_ratio", 0.60))
        self.wide_corridor_max_center_range_ratio = float(cfg.get("wide_corridor_max_center_range_ratio", 0.45))
        self.wide_corridor_max_row_occupancy = float(cfg.get("wide_corridor_max_row_occupancy", 0.82))
        self.wide_corridor_max_bottom_row_occupancy = float(
            cfg.get("wide_corridor_max_bottom_row_occupancy", 0.82)
        )
        # During a tight turn the close camera can see the bend entering from
        # the top/side before it reaches the bottom band.  This is still a
        # valid track corridor, but it must show a sustained, non-flat path;
        # otherwise a black table/object that fills the frame would pass.
        self.turn_corridor_enabled = bool(cfg.get("turn_corridor_enabled", True))
        self.turn_corridor_min_area_ratio = float(cfg.get("turn_corridor_min_area_ratio", 0.30))
        self.turn_corridor_min_height_ratio = float(cfg.get("turn_corridor_min_height_ratio", 0.50))
        self.turn_corridor_max_width_ratio = float(cfg.get("turn_corridor_max_width_ratio", 1.0))
        self.turn_corridor_min_side_support_ratio = float(
            cfg.get("turn_corridor_min_side_support_ratio", 0.45)
        )
        self.turn_corridor_min_center_range_ratio = float(
            cfg.get("turn_corridor_min_center_range_ratio", 0.12)
        )
        self.turn_corridor_max_median_row_occupancy = float(
            cfg.get("turn_corridor_max_median_row_occupancy", 0.96)
        )
        self.turn_corridor_max_bottom_row_occupancy = float(
            cfg.get("turn_corridor_max_bottom_row_occupancy", 1.0)
        )
        self.turn_corridor_max_extent_ratio = float(cfg.get("turn_corridor_max_extent_ratio", 0.98))
        # A 90-degree bend can briefly become a broad diagonal corridor with
        # almost no side gap, or leave the bottom edge while the robot turns.
        # Keep this broader rule gated by top anchoring and path movement so a
        # flat black band is not promoted to a line.
        self.right_angle_corridor_enabled = bool(cfg.get("right_angle_corridor_enabled", True))
        self.right_angle_corridor_min_area_ratio = float(
            cfg.get("right_angle_corridor_min_area_ratio", 0.30)
        )
        self.right_angle_corridor_min_height_ratio = float(
            cfg.get("right_angle_corridor_min_height_ratio", 0.65)
        )
        self.right_angle_corridor_min_width_ratio = float(
            cfg.get("right_angle_corridor_min_width_ratio", 0.75)
        )
        self.right_angle_corridor_min_center_range_ratio = float(
            cfg.get("right_angle_corridor_min_center_range_ratio", 0.08)
        )
        self.right_angle_corridor_max_extent_ratio = float(
            cfg.get("right_angle_corridor_max_extent_ratio", 0.995)
        )
        # If the valid line was just seen, bridge a short, bounded sequence
        # while the close camera is inside a bend.  The USB camera can show
        # only the top/side remnant of a real 90-degree track for roughly half
        # a second.  Cold-start and stale candidates remain rejected so an
        # isolated black object cannot start or indefinitely sustain motion.
        self.turn_continuation_enabled = bool(cfg.get("turn_continuation_enabled", True))
        self.turn_continuation_min_area_ratio = float(
            cfg.get("turn_continuation_min_area_ratio", 0.75)
        )
        self.turn_continuation_min_height_ratio = float(
            cfg.get("turn_continuation_min_height_ratio", 0.85)
        )
        self.turn_continuation_min_width_ratio = float(
            cfg.get("turn_continuation_min_width_ratio", 0.85)
        )
        self.turn_continuation_min_extent_ratio = float(
            cfg.get("turn_continuation_min_extent_ratio", 0.90)
        )
        self.turn_continuation_max_extent_ratio = float(
            cfg.get("turn_continuation_max_extent_ratio", 0.999)
        )
        self.turn_continuation_max_gap_frames = max(
            0, int(cfg.get("turn_continuation_max_gap_frames", 1))
        )
        self.turn_continuation_max_frames = max(
            0, int(cfg.get("turn_continuation_max_frames", 18))
        )
        self.turn_continuation_min_fragment_area_ratio = float(
            cfg.get("turn_continuation_min_fragment_area_ratio", 0.005)
        )
        self.turn_continuation_min_fragment_height_ratio = float(
            cfg.get("turn_continuation_min_fragment_height_ratio", 0.18)
        )
        self.turn_continuation_max_fragment_width_ratio = float(
            cfg.get("turn_continuation_max_fragment_width_ratio", 0.50)
        )
        self.turn_continuation_min_fragment_aspect = float(
            cfg.get("turn_continuation_min_fragment_aspect", 1.25)
        )
        self.turn_continuation_min_wide_area_ratio = float(
            cfg.get("turn_continuation_min_wide_area_ratio", 0.05)
        )
        self.turn_continuation_min_wide_width_ratio = float(
            cfg.get("turn_continuation_min_wide_width_ratio", 0.50)
        )
        self.turn_continuation_min_wide_aspect = float(
            cfg.get("turn_continuation_min_wide_aspect", 1.80)
        )
        self.turn_continuation_min_bend_range_ratio = float(
            cfg.get("turn_continuation_min_bend_range_ratio", 0.08)
        )
        self.has_accepted_line = False
        self.turn_continuation_frames = 0
        self.last_rejection_reason = "not_evaluated"
        self.last_geometry: dict[str, float] = {}

    def black_mask(self, image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        self.last_rejection_reason = "no_black_contour"
        self.last_geometry = {}
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        black_mask = cv2.inRange(
            hsv_image,
            (0, 0, 0),
            (self.black_h_max, self.black_s_max, self.black_v_max),
        )

        if self.erode_iter > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (max(1, self.erode_ksize), max(1, self.erode_ksize)),
            )
            black_mask = cv2.erode(black_mask, kernel, iterations=self.erode_iter)
        if self.dilate_iter > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (max(1, self.dilate_ksize), max(1, self.dilate_ksize)),
            )
            black_mask = cv2.dilate(black_mask, kernel, iterations=self.dilate_iter)

        contours, _ = cv2.findContours(black_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_black_area]

        # _is_line_candidate stores the geometry it just measured on the
        # detector.  Keep that geometry paired with its contour.  Filtering
        # with a plain list comprehension used to leave ``last_geometry``
        # pointing at whichever contour happened to be examined last, even
        # when a different contour was selected below.  The control pipeline
        # then drew one line but steered from another object's path offset.
        candidates: list[tuple[np.ndarray, dict[str, float], str]] = []
        for contour in contours:
            if self._is_line_candidate(contour):
                candidates.append(
                    (
                        contour,
                        dict(self.last_geometry),
                        str(self.last_rejection_reason),
                    )
                )

        if not candidates:
            return None, black_mask
        if len(candidates) == 1:
            selected_contour, selected_geometry, selected_reason = candidates[0]
        else:
            diffs: list[float] = []
            for contour, _, _ in candidates:
                angle, _ = self.calculate_angle(contour, update_state=False)
                diffs.append(abs(float(angle) - float(self.prev_angle)))
            best = int(np.argmin(np.array(diffs, dtype=np.float32)))
            selected_contour, selected_geometry, selected_reason = candidates[best]

        selected_geometry["candidate_count"] = float(len(candidates))
        if selected_geometry.get("turn_continuation", 0.0) > 0.0:
            self.turn_continuation_frames += 1
            selected_geometry["turn_continuation_streak"] = float(self.turn_continuation_frames)
        else:
            self.turn_continuation_frames = 0
        self.last_geometry = selected_geometry
        self.last_rejection_reason = selected_reason
        return selected_contour, black_mask

    def _reject_candidate(self, reason: str) -> bool:
        self.last_rejection_reason = str(reason)
        return False

    def _is_line_candidate(self, contour: np.ndarray) -> bool:
        if contour is None or len(contour) < 3:
            return self._reject_candidate("too_few_contour_points")

        area = float(cv2.contourArea(contour))
        if area < float(self.min_black_area):
            return self._reject_candidate("area_below_minimum")

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return self._reject_candidate("invalid_bbox")

        short = float(min(w, h))
        long = float(max(w, h))
        aspect = long / max(1.0, short)
        extent = area / float(w * h) if (w * h) > 0 else 0.0

        margin = max(2, int(self.line_border_margin))
        touches_border = (
            x <= margin
            or y <= margin
            or (x + w) >= (self.width - margin)
            or (y + h) >= (self.height - margin)
        )

        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0
        if perimeter > 1e-6:
            circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 1e-6 else 0.0

        frame_area = float(max(1, self.width * self.height))
        geometry = {
            "area_ratio": float(area / frame_area),
            "aspect": float(aspect),
            "width_ratio": float(w / max(1, self.width)),
            "height_ratio": float(h / max(1, self.height)),
            "extent": float(extent),
            "circularity": float(circularity),
            "solidity": float(solidity),
        }

        # The robot's line normally enters from the bottom camera band.  A
        # tight bend is the intentional exception handled below after the
        # contour's row geometry has been measured.
        margin = max(2, int(self.line_border_margin))
        touches_bottom = (y + h) >= (self.height - margin)

        # Analyze the filled contour row by row.  A line stays narrow and has
        # support through much of its vertical span; a table/shorts-like blob
        # produces wide rows or large unsupported gaps.
        component_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], contourIdx=-1, color=255, thickness=-1)
        row_occupancy = np.count_nonzero(component_mask, axis=1).astype(np.float32) / max(1, self.width)
        bbox_rows = row_occupancy[y : min(self.height, y + h)]
        active_rows = int(np.count_nonzero(bbox_rows >= self.min_row_occupancy))
        vertical_support = active_rows / max(1.0, float(len(bbox_rows)))
        max_row_occupancy = float(np.max(bbox_rows)) if bbox_rows.size else 0.0
        dominant_row_y_ratio = 0.5
        if bbox_rows.size and max_row_occupancy > 0.0:
            # A 90-degree track produces a short band of very wide rows at
            # the elbow.  Its vertical position tells the controller whether
            # the bend is merely visible in the wide USB camera or is already
            # close enough to the chassis to start an in-place pivot.
            dominant_threshold = max_row_occupancy * 0.97
            dominant_rows = np.flatnonzero(row_occupancy >= dominant_threshold)
            if dominant_rows.size:
                dominant_row_y_ratio = float(np.mean(dominant_rows)) / max(
                    1.0,
                    float(self.height - 1),
                )
        bottom_band_h = max(3, int(round(self.height * self.bottom_band_ratio)))
        bottom_rows = row_occupancy[max(0, self.height - bottom_band_h) :]
        bottom_row_occupancy = float(np.max(bottom_rows)) if bottom_rows.size else 0.0
        row_center_samples: list[tuple[int, float]] = []
        side_gap_rows = 0
        for row_index in range(max(0, y), min(self.height, y + h)):
            xs = np.flatnonzero(component_mask[row_index] > 0)
            if xs.size == 0:
                continue
            left_gap_ratio = float(xs[0]) / max(1.0, float(self.width))
            right_gap_ratio = float(self.width - 1 - xs[-1]) / max(1.0, float(self.width))
            if max(left_gap_ratio, right_gap_ratio) >= self.wide_corridor_min_side_gap_ratio:
                side_gap_rows += 1
            row_center_samples.append((row_index, float((xs[0] + xs[-1]) / 2.0)))
        row_centers = [center for _, center in row_center_samples]
        side_gap_support = side_gap_rows / max(1.0, float(len(bbox_rows)))
        center_range_ratio = 0.0
        if row_centers:
            center_range_ratio = (max(row_centers) - min(row_centers)) / max(1.0, float(self.width))
        median_row_occupancy = float(np.median(bbox_rows)) if bbox_rows.size else 0.0
        path_center_x = float(np.mean(row_centers)) if row_centers else float(x + (w / 2.0))
        far_path_center_x = path_center_x
        ground_path_center_x = path_center_x
        if row_center_samples:
            path_band = max(3, int(round(h * 0.25)))
            far_end = y + path_band
            ground_start = y + int(round(h * 0.75))
            far_centers = [center for row, center in row_center_samples if row < far_end]
            ground_centers = [center for row, center in row_center_samples if row >= ground_start]
            if far_centers:
                far_path_center_x = float(np.mean(far_centers))
            if ground_centers:
                ground_path_center_x = float(np.mean(ground_centers))
                path_center_x = ground_path_center_x
        path_center_offset_norm = (path_center_x - (self.width / 2.0)) / max(1.0, self.width / 2.0)
        path_bend_delta_norm = (ground_path_center_x - far_path_center_x) / max(
            1.0,
            self.width / 2.0,
        )
        geometry.update(
            {
                "vertical_support_ratio": float(vertical_support),
                "max_row_occupancy": max_row_occupancy,
                "dominant_row_y_ratio": float(dominant_row_y_ratio),
                "bottom_row_occupancy": bottom_row_occupancy,
                "side_gap_support_ratio": float(side_gap_support),
                "center_range_ratio": float(center_range_ratio),
                "median_row_occupancy": median_row_occupancy,
                "path_center_offset_norm": float(path_center_offset_norm),
                "far_path_center_offset_norm": float(
                    (far_path_center_x - (self.width / 2.0)) / max(1.0, self.width / 2.0)
                ),
                "ground_path_center_offset_norm": float(path_center_offset_norm),
                "path_bend_delta_norm": float(path_bend_delta_norm),
            }
        )
        self.last_geometry = geometry

        width_ratio = w / max(1.0, float(self.width))
        top_anchored = y <= margin
        wide_ground_corridor = (
            self.wide_corridor_enabled
            and touches_bottom
            and (h / max(1.0, float(self.height))) >= self.wide_corridor_min_height_ratio
            and width_ratio <= self.wide_corridor_max_width_ratio
            and vertical_support >= self.min_vertical_support_ratio
            and side_gap_support >= self.wide_corridor_min_side_support_ratio
            and center_range_ratio <= self.wide_corridor_max_center_range_ratio
            and max_row_occupancy <= self.wide_corridor_max_row_occupancy
            and bottom_row_occupancy <= self.wide_corridor_max_bottom_row_occupancy
        )
        turn_corridor = (
            self.turn_corridor_enabled
            and top_anchored
            and (area / frame_area) >= self.turn_corridor_min_area_ratio
            and (h / max(1.0, float(self.height))) >= self.turn_corridor_min_height_ratio
            and width_ratio <= self.turn_corridor_max_width_ratio
            and vertical_support >= self.min_vertical_support_ratio
            and side_gap_support >= self.turn_corridor_min_side_support_ratio
            and center_range_ratio >= self.turn_corridor_min_center_range_ratio
            and median_row_occupancy <= self.turn_corridor_max_median_row_occupancy
            and bottom_row_occupancy <= self.turn_corridor_max_bottom_row_occupancy
            and extent <= self.turn_corridor_max_extent_ratio
        )
        right_angle_corridor = (
            self.right_angle_corridor_enabled
            and top_anchored
            and (area / frame_area) >= self.right_angle_corridor_min_area_ratio
            and (h / max(1.0, float(self.height))) >= self.right_angle_corridor_min_height_ratio
            and width_ratio >= self.right_angle_corridor_min_width_ratio
            and width_ratio <= self.turn_corridor_max_width_ratio
            and vertical_support >= self.min_vertical_support_ratio
            and center_range_ratio >= self.right_angle_corridor_min_center_range_ratio
            and extent <= self.right_angle_corridor_max_extent_ratio
        )
        # Preserve the most specific geometry even when a bend also satisfies
        # the broad ground-corridor rule.  Previously that early return hid a
        # real 90-degree corner from the controller until the line was already
        # leaving the camera.
        if right_angle_corridor:
            geometry["turn_corridor"] = 1.0
            geometry["right_angle_corridor"] = 1.0
            self.last_rejection_reason = "accepted"
            return True
        if turn_corridor:
            geometry["turn_corridor"] = 1.0
            self.last_rejection_reason = "accepted"
            return True
        if wide_ground_corridor:
            geometry["wide_ground_corridor"] = 1.0
            self.last_rejection_reason = "accepted"
            return True

        continuation_ready = (
            self.turn_continuation_enabled
            and self.has_accepted_line
            and self.gap_found <= self.turn_continuation_max_gap_frames
            and self.turn_continuation_frames < self.turn_continuation_max_frames
            and top_anchored
        )
        full_frame_continuation = (
            continuation_ready
            and (area / frame_area) >= self.turn_continuation_min_area_ratio
            and (h / max(1.0, float(self.height))) >= self.turn_continuation_min_height_ratio
            and width_ratio >= self.turn_continuation_min_width_ratio
            and width_ratio <= self.turn_corridor_max_width_ratio
            and extent >= self.turn_continuation_min_extent_ratio
            and extent <= self.turn_continuation_max_extent_ratio
        )
        height_ratio = h / max(1.0, float(self.height))
        area_ratio = area / frame_area
        compact_top_fragment = (
            continuation_ready
            and not touches_bottom
            and area_ratio >= self.turn_continuation_min_fragment_area_ratio
            and height_ratio >= self.turn_continuation_min_fragment_height_ratio
            and width_ratio <= self.turn_continuation_max_fragment_width_ratio
            and aspect >= self.turn_continuation_min_fragment_aspect
            and vertical_support >= self.min_vertical_support_ratio
        )
        wide_top_fragment = (
            continuation_ready
            and not touches_bottom
            and area_ratio >= self.turn_continuation_min_wide_area_ratio
            and height_ratio >= max(0.15, self.turn_continuation_min_fragment_height_ratio * 0.75)
            and width_ratio >= self.turn_continuation_min_wide_width_ratio
            and aspect >= self.turn_continuation_min_wide_aspect
            and vertical_support >= self.min_vertical_support_ratio
        )
        tall_bend_fragment = (
            continuation_ready
            and not touches_bottom
            and height_ratio >= self.min_ground_span_ratio
            and center_range_ratio >= self.turn_continuation_min_bend_range_ratio
            and vertical_support >= self.min_vertical_support_ratio
        )
        if full_frame_continuation or compact_top_fragment or wide_top_fragment or tall_bend_fragment:
            geometry["turn_corridor"] = 1.0
            geometry["turn_continuation"] = 1.0
            if compact_top_fragment:
                geometry["turn_continuation_compact_top"] = 1.0
            if wide_top_fragment:
                geometry["turn_continuation_wide_top"] = 1.0
            if tall_bend_fragment:
                geometry["turn_continuation_tall_bend"] = 1.0
            self.last_rejection_reason = "accepted"
            return True

        if height_ratio < self.min_ground_span_ratio:
            return self._reject_candidate("insufficient_vertical_span")

        if not touches_bottom:
            return self._reject_candidate("not_ground_anchored")

        if aspect < self.min_line_aspect:
            return self._reject_candidate("not_line_like_aspect")

        if width_ratio > self.max_line_width_ratio:
            return self._reject_candidate("line_too_wide")

        if vertical_support < self.min_vertical_support_ratio:
            return self._reject_candidate("insufficient_vertical_support")
        if max_row_occupancy > self.max_line_row_occupancy:
            return self._reject_candidate("rows_too_wide")
        if bottom_row_occupancy > self.max_bottom_row_occupancy:
            return self._reject_candidate("bottom_band_too_wide")

        if (not touches_border) and circularity >= self.round_blob_circularity and aspect <= self.round_blob_aspect:
            return self._reject_candidate("round_blob")

        if (not touches_border) and (
            extent >= self.compact_extent_reject
            and solidity >= self.compact_solidity_reject
            and circularity >= self.compact_circularity_reject
            and aspect <= max(self.round_blob_aspect, 1.45)
        ):
            return self._reject_candidate("compact_blob")

        if not touches_border:
            if (area / frame_area) < self.min_area_ratio_for_internal:
                return self._reject_candidate("internal_area_too_small")
            min_long_side = float(min(self.width, self.height)) * max(0.05, self.min_internal_long_side_ratio)
            if long < min_long_side:
                return self._reject_candidate("internal_span_too_short")
        self.last_rejection_reason = "accepted"
        return True

    def calculate_angle(self, contour: np.ndarray | None, *, update_state: bool = True) -> tuple[int, int]:
        if contour is None:
            if update_state:
                self.gap_found += 1
                if self.gap_found > self.turn_continuation_max_gap_frames:
                    self.turn_continuation_frames = 0
            return self.prev_angle, self.gap_found

        points = contour.reshape(-1, 2)
        top_points = points[points[:, 1] <= 5]
        left_points = points[points[:, 0] <= 10]
        right_points = points[points[:, 0] >= self.width - 10]
        ref_point: tuple[int, int] | None = None

        if len(top_points) > 0:
            ref_point = (int(np.mean(top_points[:, 0])), 0)
        elif len(left_points) > 0 and len(right_points) > 0:
            if self.prev_angle <= 90:
                ref_point = (0, int(np.mean(left_points[:, 1])))
            else:
                ref_point = (self.width - 1, int(np.mean(right_points[:, 1])))
        else:
            top_idx = int(np.argmin(points[:, 1]))
            top = points[top_idx]
            ref_point = (int(top[0]), int(top[1]))

        if ref_point is None:
            if update_state:
                self.gap_found += 1
            return self.prev_angle, self.gap_found

        bottom = (self.width // 2, self.height)
        dx = bottom[0] - ref_point[0]
        dy = bottom[1] - ref_point[1]
        angle = int(np.degrees(np.arctan2(dy, dx)))
        if update_state:
            self.prev_angle = angle
            self.gap_found = 0
            self.has_accepted_line = True
        return angle, self.gap_found


class BallDetector:
    def __init__(self, width: int, height: int, settings: Mapping[str, Any] | None = None) -> None:
        cfg = dict(settings or {})
        self.width = int(width)
        self.height = int(height)
        self.crop_size = int(cfg.get("crop_size", 200))
        self.silver_blur = int(cfg.get("silver_blur", 7))
        self.silver_hough_dp = float(cfg.get("silver_hough_dp", 1.2))
        self.silver_hough_min_distance = int(cfg.get("silver_hough_min_distance", 60))
        self.silver_hough_param1 = int(cfg.get("silver_hough_param1", 120))
        self.silver_hough_param2 = int(cfg.get("silver_hough_param2", 30))
        self.silver_hough_min_radius = int(cfg.get("silver_hough_min_radius", 8))
        self.silver_hough_max_radius = int(cfg.get("silver_hough_max_radius", 120))
        self.silver_specular_v_min = int(cfg.get("silver_specular_v_min", 175))
        self.silver_specular_s_max = int(cfg.get("silver_specular_s_max", 95))
        self.silver_specular_min_area = float(cfg.get("silver_specular_min_area", 120))
        self.silver_specular_max_area_ratio = float(cfg.get("silver_specular_max_area_ratio", 0.22))
        self.silver_specular_min_circularity = float(cfg.get("silver_specular_min_circularity", 0.55))
        self.silver_conf_threshold = float(cfg.get("silver_conf_threshold", 0.62))
        self.silver_candidate_threshold = float(
            cfg.get("silver_candidate_threshold", max(0.28, self.silver_conf_threshold * 0.72))
        )
        self.silver_max_candidates = max(1, int(cfg.get("silver_max_candidates", 3)))
        self.silver_candidate_min_center_dist = float(
            cfg.get("silver_candidate_min_center_dist", max(14.0, self.silver_hough_min_distance * 0.45))
        )
        self.silver_candidate_iou_nms = float(cfg.get("silver_candidate_iou_nms", 0.35))

        self.dead_white_kernel = np.ones((9, 9), np.uint8)
        self.dead_black_kernel = np.ones((25, 25), np.uint8)
        self.dead_white_threshold = tuple(cfg.get("dead_white_threshold", [160, 160, 160]))
        self.dead_black_threshold = tuple(cfg.get("dead_black_threshold", [60, 60, 60]))
        self.dead_min_black_area = int(cfg.get("dead_min_black_area", 300))
        self.dead_min_y = int(cfg.get("dead_min_y", 40))
        self.dead_radius_y_min = int(cfg.get("dead_radius_y_min", -50))
        self.dead_hough_dp = float(cfg.get("dead_hough_dp", 1))
        self.dead_hough_min_distance = int(cfg.get("dead_hough_min_distance", 200))
        self.dead_hough_param1 = int(cfg.get("dead_hough_param1", 50))
        self.dead_hough_param2 = int(cfg.get("dead_hough_param2", 30))
        self.dead_hough_min_radius = int(cfg.get("dead_hough_min_radius", 5))
        self.dead_hough_max_radius = int(cfg.get("dead_hough_max_radius", 150))
        self.black_conf_threshold = float(cfg.get("black_conf_threshold", 0.60))
        self.silver_black_overlap_px = int(cfg.get("silver_black_overlap_px", 24))

        self.last_live_circle: tuple[int, int, int] | None = None
        self.last_dead_circle: tuple[int, int, int] | None = None
        self.last_live_confidence = 0.0
        self.last_dead_confidence = 0.0
        self.last_live_bbox: dict[str, int] | None = None
        self.last_dead_bbox: dict[str, int] | None = None
        self.last_live_origin = "none"
        self.last_dead_origin = "none"

    def live(self, image: np.ndarray, last_x: int | None) -> int | None:
        out = self.live_detection(image, last_x)
        return int(out["x"]) if out["found"] else None

    def dead(self, image: np.ndarray, last_x: int | None) -> int | None:
        out = self.dead_detection(image, last_x)
        return int(out["x"]) if out["found"] else None

    def build_dead_mask(self, image: np.ndarray, last_x: int | None = None) -> tuple[np.ndarray, int]:
        if image is None or image.size == 0:
            return _empty_mask(0, 0), 0

        if last_x is not None:
            x0 = max(0, int(last_x) - self.crop_size)
            x1 = min(image.shape[1], int(last_x) + self.crop_size)
            working = image[:, x0:x1].copy()
        else:
            x0 = 0
            working = image.copy()

        white = cv2.inRange(working, self.dead_white_threshold, (255, 255, 255))
        white = cv2.erode(white, self.dead_white_kernel, iterations=1)
        white = cv2.dilate(white, self.dead_white_kernel, iterations=1)
        working[white > 0] = [160, 160, 160]

        black = cv2.inRange(working, (0, 0, 0), self.dead_black_threshold)
        contours, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(black)
        for contour in contours:
            if cv2.contourArea(contour) >= self.dead_min_black_area:
                cv2.fillPoly(filtered, [contour], 255)
        black = cv2.dilate(filtered, self.dead_black_kernel, iterations=1)
        return black, int(x0)

    def live_detection(self, image: np.ndarray, last_x: int | None) -> dict[str, Any]:
        candidates = self.live_detections(image, last_x)
        if not candidates:
            self._reset_live()
            return self._empty_result()
        return dict(candidates[0])

    def live_detections(self, image: np.ndarray, last_x: int | None) -> list[dict[str, Any]]:
        if image is None or image.size == 0:
            self._reset_live()
            return []

        if last_x is not None:
            x0 = max(0, int(last_x) - self.crop_size)
            x1 = min(image.shape[1], int(last_x) + self.crop_size)
            working = image[:, x0:x1]
        else:
            x0 = 0
            working = image

        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        ksize = max(3, self.silver_blur + (1 - self.silver_blur % 2))
        gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            self.silver_hough_dp,
            self.silver_hough_min_distance,
            param1=self.silver_hough_param1,
            param2=self.silver_hough_param2,
            minRadius=self.silver_hough_min_radius,
            maxRadius=self.silver_hough_max_radius,
        )

        candidates: list[dict[str, Any]] = []
        if circles is not None:
            for cx, cy, radius in np.round(circles[0, :]).astype("int"):
                if radius < 2:
                    continue
                conf = self._silver_circle_confidence(working, int(cx), int(cy), int(radius))
                if last_x is not None:
                    abs_x = int(cx) + x0
                    proximity = 1.0 - (abs(float(abs_x - last_x)) / max(1.0, float(image.shape[1])))
                    conf = _clamp01(0.90 * conf + 0.10 * _clamp01(proximity))
                if conf < self.silver_candidate_threshold:
                    continue
                abs_x = int(cx) + x0
                bbox = _bbox_from_circle(abs_x, int(cy), int(radius), image.shape)
                if bbox is None:
                    continue
                candidates.append(
                    {
                        "found": bool(conf >= self.silver_conf_threshold),
                        "x": int(abs_x),
                        "confidence": float(_clamp01(conf)),
                        "bbox": bbox,
                        "circle": {"x": int(abs_x), "y": int(cy), "r": int(radius)},
                        "origin": "heuristic_hough",
                    }
                )

        if not candidates:
            fallback = self._fallback_live_circle(working, last_x=last_x, x_offset=x0, frame_width=image.shape[1])
            if fallback is not None:
                fx, fy, fr, fconf = fallback
                abs_x = int(fx) + x0
                bbox = _bbox_from_circle(abs_x, int(fy), int(fr), image.shape)
                if bbox is not None:
                    conf = float(_clamp01(fconf))
                    candidates.append(
                        {
                            "found": bool(conf >= self.silver_conf_threshold),
                            "x": int(abs_x),
                            "confidence": conf,
                            "bbox": bbox,
                            "circle": {"x": int(abs_x), "y": int(fy), "r": int(fr)},
                            "origin": "heuristic_specular",
                        }
                    )

        if not candidates:
            self._reset_live()
            return []

        deduped = self._deduplicate_live_candidates(candidates)
        deduped.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        deduped = deduped[: self.silver_max_candidates]
        if not deduped:
            self._reset_live()
            return []

        top = deduped[0]
        top_circle = top.get("circle")
        if isinstance(top_circle, Mapping):
            self.last_live_circle = (
                int(top_circle.get("x", 0)),
                int(top_circle.get("y", 0)),
                int(top_circle.get("r", 0)),
            )
        else:
            self.last_live_circle = None
        self.last_live_confidence = float(_clamp01(float(top.get("confidence", 0.0))))
        self.last_live_bbox = top.get("bbox") if isinstance(top.get("bbox"), Mapping) else None
        self.last_live_origin = str(top.get("origin", "none"))
        return deduped

    def dead_detection(self, image: np.ndarray, last_x: int | None) -> dict[str, Any]:
        if image is None or image.size == 0:
            self._reset_dead()
            return self._empty_result()

        black, x0 = self.build_dead_mask(image, last_x)
        working = image[:, x0 : x0 + black.shape[1]].copy() if black.size > 0 else image.copy()

        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, black)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            self.dead_hough_dp,
            self.dead_hough_min_distance,
            param1=self.dead_hough_param1,
            param2=self.dead_hough_param2,
            minRadius=self.dead_hough_min_radius,
            maxRadius=self.dead_hough_max_radius,
        )
        if circles is None:
            self._reset_dead()
            return self._empty_result()

        best: tuple[int, int, int] | None = None
        best_conf = -1.0
        for x, y, radius in np.round(circles[0, :]).astype("int"):
            if y <= self.dead_min_y or (y - radius) <= self.dead_radius_y_min:
                continue
            conf = self._black_circle_confidence(working, black, int(x), int(y), int(radius))
            if last_x is not None:
                abs_x = int(x) + x0
                proximity = 1.0 - (abs(float(abs_x - last_x)) / max(1.0, float(image.shape[1])))
                conf = _clamp01(0.90 * conf + 0.10 * _clamp01(proximity))
            if conf > best_conf:
                best_conf = conf
                best = (int(x), int(y), int(radius))

        if best is None or best_conf < self.black_conf_threshold:
            self._reset_dead()
            return self._empty_result()

        cx = int(best[0] + x0)
        cy = int(best[1])
        radius = int(best[2])
        bbox = _bbox_from_circle(cx, cy, radius, image.shape)
        self.last_dead_circle = (cx, cy, radius)
        self.last_dead_confidence = _clamp01(best_conf)
        self.last_dead_bbox = bbox
        self.last_dead_origin = "heuristic_black"
        return {
            "found": True,
            "x": int(cx),
            "confidence": float(self.last_dead_confidence),
            "bbox": bbox,
            "circle": {"x": int(cx), "y": int(cy), "r": int(radius)},
            "origin": "heuristic_black",
        }

    def _fallback_live_circle(
        self,
        working: np.ndarray,
        *,
        last_x: int | None,
        x_offset: int,
        frame_width: int,
    ) -> tuple[int, int, int, float] | None:
        if working is None or working.size == 0:
            return None
        hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            (0, 0, self.silver_specular_v_min),
            (180, self.silver_specular_s_max, 255),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        img_area = float(working.shape[0] * working.shape[1])
        max_area = img_area * max(0.02, self.silver_specular_max_area_ratio)
        best_score = -1.0
        best_circle: tuple[int, int, int, float] | None = None

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.silver_specular_min_area or area > max_area:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
            if circularity < self.silver_specular_min_circularity:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < 3.0:
                continue
            conf = self._silver_circle_confidence(working, int(x), int(y), int(radius))
            if last_x is not None:
                abs_x = int(x) + int(x_offset)
                proximity = 1.0 - (abs(float(abs_x - last_x)) / max(1.0, float(frame_width)))
                conf = _clamp01(0.90 * conf + 0.10 * _clamp01(proximity))
            score = conf * (0.35 + min(1.0, circularity))
            if score > best_score:
                best_score = score
                best_circle = (int(x), int(y), int(radius), float(conf))
        return best_circle

    def _deduplicate_live_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        sorted_candidates = sorted(candidates, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        kept: list[dict[str, Any]] = []
        min_dist_base = max(4.0, float(self.silver_candidate_min_center_dist))
        for candidate in sorted_candidates:
            circle = candidate.get("circle")
            if not isinstance(circle, Mapping):
                continue
            cx = float(circle.get("x", 0.0))
            cy = float(circle.get("y", 0.0))
            radius = max(1.0, float(circle.get("r", 1.0)))
            reject = False
            for current in kept:
                current_circle = current.get("circle")
                if not isinstance(current_circle, Mapping):
                    continue
                ox = float(current_circle.get("x", 0.0))
                oy = float(current_circle.get("y", 0.0))
                oradius = max(1.0, float(current_circle.get("r", 1.0)))
                min_dist = max(min_dist_base, 0.45 * (radius + oradius))
                if ((cx - ox) * (cx - ox) + (cy - oy) * (cy - oy)) <= (min_dist * min_dist):
                    reject = True
                    break
                if _bbox_iou(candidate.get("bbox"), current.get("bbox")) >= self.silver_candidate_iou_nms:
                    reject = True
                    break
            if not reject:
                kept.append(candidate)
        return kept

    def _silver_circle_confidence(self, image: np.ndarray, cx: int, cy: int, radius: int) -> float:
        if image is None or image.size == 0 or radius <= 1:
            return 0.0
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(radius), 255, thickness=-1)
        pixels = image[mask > 0]
        if pixels.size == 0:
            return 0.0
        hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        sat = float(np.mean(hsv_pixels[:, 1]))
        val = float(np.mean(hsv_pixels[:, 2]))
        sat_score = _clamp01(1.0 - (sat / 255.0))
        val_score = _clamp01(val / 255.0)
        spec_ratio = float(
            np.mean(
                (hsv_pixels[:, 1] <= self.silver_specular_s_max) & (hsv_pixels[:, 2] >= self.silver_specular_v_min)
            )
        )
        spec_score = _clamp01(spec_ratio * 2.6)
        gray_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).reshape(-1)
        texture_score = _clamp01(float(np.std(gray_pixels)) / 70.0)
        conf = 0.36 * sat_score + 0.28 * val_score + 0.22 * spec_score + 0.14 * texture_score
        return _clamp01(conf)

    def _black_circle_confidence(
        self,
        image: np.ndarray,
        black_mask: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
    ) -> float:
        if image is None or image.size == 0 or radius <= 1:
            return 0.0
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(radius), 255, thickness=-1)
        pixel_idx = mask > 0
        pixels = image[pixel_idx]
        if pixels.size == 0:
            return 0.0
        hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        val_mean = float(np.mean(hsv_pixels[:, 2]))
        sat_mean = float(np.mean(hsv_pixels[:, 1]))
        dark_ratio = float(np.mean(hsv_pixels[:, 2] <= (self.dead_black_threshold[2] + 25)))
        mask_ratio = float(np.mean(black_mask[pixel_idx] > 0)) if black_mask.size > 0 else 0.0
        conf = (
            0.50 * _clamp01(dark_ratio)
            + 0.25 * _clamp01(1.0 - (val_mean / 255.0))
            + 0.15 * _clamp01(mask_ratio)
            + 0.10 * _clamp01(1.0 - (sat_mean / 255.0))
        )
        return _clamp01(conf)

    def _empty_result(self) -> dict[str, Any]:
        return {"found": False, "x": None, "confidence": 0.0, "bbox": None, "circle": None, "origin": "none"}

    def _reset_live(self) -> None:
        self.last_live_circle = None
        self.last_live_confidence = 0.0
        self.last_live_bbox = None
        self.last_live_origin = "none"

    def _reset_dead(self) -> None:
        self.last_dead_circle = None
        self.last_dead_confidence = 0.0
        self.last_dead_bbox = None
        self.last_dead_origin = "none"


class ColorMarkerDetector:
    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        cfg = dict(settings or {})

        self.green_h_min = int(cfg.get("green_h_min", 35))
        self.green_h_max = int(cfg.get("green_h_max", 90))
        self.green_s_min = int(cfg.get("green_s_min", 70))
        self.green_v_min = int(cfg.get("green_v_min", 50))
        self.green_min_area = float(cfg.get("green_min_area", 180))
        self.green_min_aspect = float(cfg.get("green_min_aspect", 0.35))
        self.green_max_aspect = float(cfg.get("green_max_aspect", 3.0))
        self.green_max_area_ratio = float(cfg.get("green_max_area_ratio", 0.20))
        self.green_min_solidity = float(cfg.get("green_min_solidity", 0.72))
        self.green_min_extent = float(cfg.get("green_min_extent", 0.20))
        self.green_max_extent = float(cfg.get("green_max_extent", 0.97))
        self.green_min_short_side = float(cfg.get("green_min_short_side", 8.0))
        self.green_min_line_pixels_nearby = int(cfg.get("green_min_line_pixels_nearby", 110))
        self.green_pair_min_separation_ratio = float(cfg.get("green_pair_min_separation_ratio", 0.24))
        self.green_pair_max_vertical_delta_ratio = float(cfg.get("green_pair_max_vertical_delta_ratio", 0.45))
        self.green_pair_min_area_balance = float(cfg.get("green_pair_min_area_balance", 0.18))
        # A real route marker remains chromatically distinct from the floor
        # even when the LED or automatic exposure makes it dark.  The wider
        # recovery masks must therefore prove both sustained saturation inside
        # the square and a visible local boundary.  This rejects gray/cyan
        # floor casts and reflections without narrowing the accepted hue range.
        self.green_region_inner_margin_ratio = float(
            cfg.get("green_region_inner_margin_ratio", 0.15)
        )
        self.green_region_min_saturation_p25 = float(
            cfg.get("green_region_min_saturation_p25", 50.0)
        )
        self.green_region_min_local_contrast = float(
            cfg.get("green_region_min_local_contrast", 8.0)
        )
        # The USB camera shifts illuminated green tape towards cyan in isolated
        # frames (OpenCV HSV H ~= 100-105). Keep the primary mask conservative
        # for single-green instructions; use this wider mask only for a pair
        # that also passes the two-squares-plus-black-T geometry check.
        self.green_pair_h_min = int(cfg.get("green_pair_h_min", self.green_h_min))
        self.green_pair_h_max = int(cfg.get("green_pair_h_max", max(self.green_h_max, 105)))
        self.green_pair_s_min = int(cfg.get("green_pair_s_min", self.green_s_min))
        self.green_pair_v_min = int(cfg.get("green_pair_v_min", self.green_v_min))
        self.green_pair_max_area_ratio = float(cfg.get("green_pair_max_area_ratio", 0.25))
        # The illuminated USB-camera feed can render the dark-green square with
        # S/V below the conservative primary thresholds.  A separate recovery
        # mask is allowed only for one square that is geometrically adjacent to
        # a black line, so the proven two-square detector remains unchanged.
        self.green_single_recovery_h_min = int(cfg.get("green_single_recovery_h_min", 55))
        self.green_single_recovery_h_max = int(cfg.get("green_single_recovery_h_max", 110))
        self.green_single_recovery_s_min = int(cfg.get("green_single_recovery_s_min", 40))
        self.green_single_recovery_v_min = int(cfg.get("green_single_recovery_v_min", 30))
        self.green_single_recovery_max_area_ratio = float(
            cfg.get("green_single_recovery_max_area_ratio", 0.25)
        )
        self.green_single_recovery_min_line_pixels = int(
            cfg.get("green_single_recovery_min_line_pixels", 110)
        )
        # With the current USB-camera mounting, smaller image Y is physically
        # before the transverse line.  Keep the convention configurable so the
        # legacy camera and synthetic detector tests retain their orientation.
        self.green_before_is_above = bool(cfg.get("green_before_is_above", False))
        self.green_corner_conf_threshold = float(cfg.get("green_corner_conf_threshold", 0.62))
        self.green_corner_min_area = float(cfg.get("green_corner_min_area", 520))
        self.green_corner_max_area_ratio = float(cfg.get("green_corner_max_area_ratio", 0.55))
        self.green_corner_min_short_side = float(cfg.get("green_corner_min_short_side", 18.0))
        self.green_corner_min_aspect = float(cfg.get("green_corner_min_aspect", 0.30))
        self.green_corner_max_aspect = float(cfg.get("green_corner_max_aspect", 4.2))
        self.green_corner_min_extent = float(cfg.get("green_corner_min_extent", 0.36))
        self.green_corner_min_solidity = float(cfg.get("green_corner_min_solidity", 0.55))
        self.green_corner_border_margin = int(cfg.get("green_corner_border_margin", 12))

        self.red_h1_min = int(cfg.get("red_h1_min", 0))
        self.red_h1_max = int(cfg.get("red_h1_max", 12))
        self.red_h2_min = int(cfg.get("red_h2_min", 165))
        self.red_h2_max = int(cfg.get("red_h2_max", 179))
        self.red_s_min = int(cfg.get("red_s_min", 120))
        self.red_v_min = int(cfg.get("red_v_min", 80))
        self.red_min_area = float(cfg.get("red_min_area", 300))
        self.red_min_ratio = float(cfg.get("red_min_ratio", 3.5))
        self.red_min_long_side = float(cfg.get("red_min_long_side", 30))

        self.green_row_margin = int(cfg.get("green_row_margin", 4))
        self.color_erode_iter = int(cfg.get("color_erode_iter", 1))
        self.color_dilate_iter = int(cfg.get("color_dilate_iter", 2))
        self.color_kernel = int(cfg.get("color_kernel", 3))

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel_size = max(1, self.color_kernel)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        if self.color_erode_iter > 0:
            mask = cv2.erode(mask, kernel, iterations=self.color_erode_iter)
        if self.color_dilate_iter > 0:
            mask = cv2.dilate(mask, kernel, iterations=self.color_dilate_iter)
        return mask

    def build_green_mask(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return _empty_mask(0, 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = (int(self.green_h_min), int(self.green_s_min), int(self.green_v_min))
        upper = (int(self.green_h_max), 255, 255)
        return self._clean_mask(cv2.inRange(hsv, lower, upper))

    def build_green_pair_mask(self, image: np.ndarray) -> np.ndarray:
        """Build the LED-tolerant mask used only for a validated two-square T."""
        if image is None or image.size == 0:
            return _empty_mask(0, 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = (
            int(self.green_pair_h_min),
            int(self.green_pair_s_min),
            int(self.green_pair_v_min),
        )
        upper = (int(self.green_pair_h_max), 255, 255)
        return self._clean_mask(cv2.inRange(hsv, lower, upper))

    def build_green_single_recovery_mask(self, image: np.ndarray) -> np.ndarray:
        """Build the dark/LED-tolerant mask used by strict one-square recovery."""
        if image is None or image.size == 0:
            return _empty_mask(0, 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = (
            int(self.green_single_recovery_h_min),
            int(self.green_single_recovery_s_min),
            int(self.green_single_recovery_v_min),
        )
        upper = (int(self.green_single_recovery_h_max), 255, 255)
        return self._clean_mask(cv2.inRange(hsv, lower, upper))

    def build_red_mask(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return _empty_mask(0, 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(
            hsv,
            (self.red_h1_min, self.red_s_min, self.red_v_min),
            (self.red_h1_max, 255, 255),
        )
        red2 = cv2.inRange(
            hsv,
            (self.red_h2_min, self.red_s_min, self.red_v_min),
            (self.red_h2_max, 255, 255),
        )
        return self._clean_mask(cv2.bitwise_or(red1, red2))

    def detect_green_instruction(self, image: np.ndarray, black_mask: np.ndarray | None) -> dict[str, Any]:
        green = self.detect_green_side(image)
        contours = list(green["contours"])
        marker_contours = list(green.get("marker_contours", []))
        marker_bboxes = list(green.get("marker_bboxes", []))
        marker_count = int(green.get("marker_count", 0) or 0)
        pair_quality = self._green_pair_quality(
            marker_bboxes,
            black_mask,
            image.shape,
            image=image,
        )
        segmentation_source = "primary" if marker_count > 0 else "none"
        single_recovery_quality: dict[str, Any] = {
            "valid": False,
            "confidence": 0.0,
            "line_pixels": 0,
            "reason": "not_attempted",
        }
        primary_pair_valid = bool(
            green.get("side") == "BOTH"
            and marker_count == 2
            and len(marker_contours) == 2
            and pair_quality.get("valid", False)
        )
        if not primary_pair_valid:
            recovered = self._detect_green_side_from_mask(
                image,
                self.build_green_pair_mask(image),
                max_area_ratio=float(self.green_pair_max_area_ratio),
            )
            recovered_contours = list(recovered.get("marker_contours", []))
            recovered_bboxes = list(recovered.get("marker_bboxes", []))
            recovered_quality = self._green_pair_quality(
                recovered_bboxes,
                black_mask,
                image.shape,
                image=image,
            )
            if (
                recovered.get("side") == "BOTH"
                and int(recovered.get("marker_count", 0) or 0) == 2
                and len(recovered_contours) == 2
                and bool(recovered_quality.get("valid", False))
            ):
                green = recovered
                contours = list(recovered.get("contours", []))
                marker_contours = recovered_contours
                marker_bboxes = recovered_bboxes
                marker_count = 2
                pair_quality = recovered_quality
                segmentation_source = "pair_recovery"
        if (
            green.get("side") == "BOTH"
            and marker_count == 2
            and len(marker_contours) == 2
            and bool(pair_quality.get("valid", False))
        ):
            merged = self._merge_contour_bboxes(marker_contours, image.shape)
            return {
                "found": True,
                "side": "BOTH",
                "instruction": "VERDE MEIA VOLTA",
                "ref_y": None,
                "center_y": None,
                "relation_delta_y": None,
                "relation_confidence": 1.0,
                "contours": contours,
                "marker_count": 2,
                "marker_bboxes": marker_bboxes,
                "confidence": float(pair_quality.get("confidence", 0.95)),
                "bbox": merged,
                "pair_quality": pair_quality,
                "single_recovery_quality": single_recovery_quality,
                "segmentation_source": segmentation_source,
            }

        # The primary mask often sees only a small flickering fragment of the
        # dark square.  Prefer a substantially larger recovered square, but
        # only when exactly one side is present and nearby black-line support
        # proves that it is a route marker rather than a colored reflection.
        recovered_single = self._detect_green_side_from_mask(
            image,
            self.build_green_single_recovery_mask(image),
            max_area_ratio=float(self.green_single_recovery_max_area_ratio),
        )
        recovered_single_contours = list(recovered_single.get("marker_contours", []))
        recovered_single_bboxes = list(recovered_single.get("marker_bboxes", []))
        recovered_single_count = int(recovered_single.get("marker_count", 0) or 0)
        if (
            recovered_single.get("side") in {"LEFT", "RIGHT"}
            and recovered_single_count == 1
            and len(recovered_single_contours) == 1
            and len(recovered_single_bboxes) == 1
        ):
            single_recovery_quality = self._green_single_recovery_quality(
                recovered_single_bboxes[0],
                black_mask,
                image.shape,
                image=image,
            )
            primary_area = max(
                (float(cv2.contourArea(item)) for item in marker_contours),
                default=0.0,
            )
            recovered_area = float(cv2.contourArea(recovered_single_contours[0]))
            recovered_is_better = bool(
                marker_count == 0 or recovered_area >= max(1.0, primary_area * 1.20)
            )
            if bool(single_recovery_quality.get("valid", False)) and recovered_is_better:
                green = recovered_single
                contours = list(recovered_single.get("contours", []))
                marker_contours = recovered_single_contours
                marker_bboxes = recovered_single_bboxes
                marker_count = 1
                segmentation_source = "single_recovery"

        if marker_count == 0 or len(marker_contours) == 0:
            return {
                "found": False,
                "side": "NONE",
                "instruction": "NO GREEN",
                "ref_y": None,
                "center_y": None,
                "relation_delta_y": None,
                "relation_confidence": 0.0,
                "contours": [],
                "marker_count": 0,
                "marker_bboxes": [],
                "confidence": 0.0,
                "bbox": None,
                "pair_quality": pair_quality,
                "single_recovery_quality": single_recovery_quality,
                "segmentation_source": "none",
            }

        contour = max(marker_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        selected_bbox = _safe_bbox(x, y, w, h, image.shape)
        # Every single green marker, including one found by the primary HSV
        # mask, must be physically adjacent to a real black track.  Previously
        # only the wider single-recovery mask had this guard, so blue/teal
        # objects in an otherwise line-less frame could be promoted to a route
        # instruction and classified against the image midpoint.
        single_recovery_quality = self._green_single_recovery_quality(
            selected_bbox or {"x": x, "y": y, "w": w, "h": h},
            black_mask,
            image.shape,
            image=image,
        )
        if not bool(single_recovery_quality.get("valid", False)):
            return {
                "found": False,
                "side": "NONE",
                "instruction": "NO GREEN",
                "ref_y": None,
                "center_y": None,
                "relation_delta_y": None,
                "relation_confidence": 0.0,
                "contours": [],
                "marker_count": 0,
                "marker_bboxes": [],
                "confidence": 0.0,
                "bbox": None,
                "pair_quality": pair_quality,
                "single_recovery_quality": single_recovery_quality,
                "segmentation_source": "none",
            }
        marker_side = "LEFT" if (x + (w / 2.0)) < (image.shape[1] / 2.0) else "RIGHT"
        green_center_y = y + (h / 2.0)

        ref_y = self._horizontal_reference_y(
            black_mask,
            # A narrow window can contain only the vertical branch of an L and
            # incorrectly report row zero as the transverse line.  Include a
            # wider neighbourhood and constrain the search around the marker.
            x0=x - (w * 2.0),
            x1=x + (w * 3.0),
            y0=y - h,
            y1=y + (h * 2.0),
        )
        if ref_y is None:
            # Defensive consistency with the validation above.  A single
            # marker is never classified relative to an arbitrary image row.
            return {
                "found": False,
                "side": "NONE",
                "instruction": "NO GREEN",
                "ref_y": None,
                "center_y": None,
                "relation_delta_y": None,
                "relation_confidence": 0.0,
                "contours": [],
                "marker_count": 0,
                "marker_bboxes": [],
                "confidence": 0.0,
                "bbox": None,
                "pair_quality": pair_quality,
                "single_recovery_quality": single_recovery_quality,
                "segmentation_source": "none",
            }

        margin = int(self.green_row_margin)
        relation_delta_y = float(green_center_y - float(ref_y))
        if relation_delta_y <= -float(margin):
            instruction = "VERDE ANTES" if self.green_before_is_above else "VERDE DEPOIS"
        elif relation_delta_y >= float(margin):
            instruction = "VERDE DEPOIS" if self.green_before_is_above else "VERDE ANTES"
        else:
            mask_one = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.drawContours(mask_one, [contour], -1, 255, thickness=cv2.FILLED)
            upper = np.count_nonzero(mask_one[: int(ref_y), :])
            lower = np.count_nonzero(mask_one[int(ref_y) :, :])
            marker_is_above = upper >= lower
            if self.green_before_is_above:
                instruction = "VERDE ANTES" if marker_is_above else "VERDE DEPOIS"
            else:
                instruction = "VERDE DEPOIS" if marker_is_above else "VERDE ANTES"

        # The controller must be able to distinguish a square clearly located
        # on one side of the transverse line from a perspective overlap.  The
        # latter is still reported for observability, but cannot command motion.
        relation_scale = max(6.0, float(h) * 0.30)
        relation_confidence = _clamp01(
            (abs(relation_delta_y) - float(margin)) / relation_scale
        )

        area = float(cv2.contourArea(contour))
        img_area = float(image.shape[0] * image.shape[1])
        conf_area = _clamp01(area / max(float(self.green_min_area), img_area * 0.03, 1.0))
        conf_geom = 1.0 if green["side"] in {"LEFT", "RIGHT", "BOTH"} else 0.0
        confidence = _clamp01(0.65 * conf_geom + 0.35 * conf_area)
        return {
            "found": True,
            "side": marker_side,
            "instruction": instruction,
            "ref_y": int(ref_y),
            "center_y": round(float(green_center_y), 2),
            "relation_delta_y": round(float(relation_delta_y), 2),
            "relation_confidence": round(float(relation_confidence), 4),
            "contours": contours,
            "marker_count": 1,
            "marker_bboxes": [selected_bbox] if selected_bbox is not None else [],
            "confidence": float(confidence),
            "bbox": selected_bbox,
            "pair_quality": pair_quality,
            "single_recovery_quality": single_recovery_quality,
            "segmentation_source": segmentation_source,
        }

    def _green_single_recovery_quality(
        self,
        marker_bbox: Mapping[str, Any],
        black_mask: np.ndarray | None,
        frame_shape: tuple[int, ...],
        *,
        image: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Require recovered dark green to sit next to a real black track."""
        if black_mask is None or black_mask.size == 0:
            return {"valid": False, "confidence": 0.0, "line_pixels": 0, "reason": "no_line_mask"}

        frame_h = int(frame_shape[0])
        frame_w = int(frame_shape[1])
        x = max(0, int(marker_bbox.get("x", 0)))
        y = max(0, int(marker_bbox.get("y", 0)))
        w = max(1, int(marker_bbox.get("w", 1)))
        h = max(1, int(marker_bbox.get("h", 1)))
        pad_x = max(10, int(round(w * 0.55)))
        pad_y = max(10, int(round(h * 0.55)))
        x0 = max(0, x - pad_x)
        x1 = min(frame_w, x + w + pad_x)
        y0 = max(0, y - pad_y)
        y1 = min(frame_h, y + h + pad_y)
        nearby = black_mask[y0:y1, x0:x1].copy()
        if nearby.size == 0:
            return {"valid": False, "confidence": 0.0, "line_pixels": 0, "reason": "empty_nearby"}

        mx0 = max(0, x - x0)
        mx1 = min(nearby.shape[1], x + w - x0)
        my0 = max(0, y - y0)
        my1 = min(nearby.shape[0], y + h - y0)
        nearby[my0:my1, mx0:mx1] = 0
        line_pixels = int(np.count_nonzero(nearby))
        ref_y = self._horizontal_reference_y(
            black_mask,
            x0=x - (w * 2.0),
            x1=x + (w * 3.0),
            y0=y - h,
            y1=y + (h * 2.0),
        )
        color_quality = self._green_region_color_quality(image, marker_bbox)
        valid = bool(
            ref_y is not None
            and line_pixels >= int(self.green_single_recovery_min_line_pixels)
            and bool(color_quality.get("valid", False))
        )
        confidence = _clamp01(
            line_pixels / max(1.0, float(self.green_single_recovery_min_line_pixels) * 2.0)
        )
        return {
            "valid": valid,
            "confidence": round(float(confidence), 4),
            "line_pixels": line_pixels,
            "reference_y": ref_y,
            "color_quality": color_quality,
            "reason": (
                "ok"
                if valid
                else (
                    "insufficient_color_quality"
                    if not bool(color_quality.get("valid", False))
                    else "insufficient_line_support"
                )
            ),
        }

    def _green_region_color_quality(
        self,
        image: np.ndarray | None,
        marker_bbox: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Distinguish a physical green square from a global cyan/gray cast."""
        if image is None or image.size == 0 or image.ndim < 2:
            return {
                "valid": False,
                "saturation_p25": 0.0,
                "local_contrast": 0.0,
                "reason": "no_image",
            }

        frame_h, frame_w = image.shape[:2]
        x = max(0, min(frame_w - 1, int(marker_bbox.get("x", 0))))
        y = max(0, min(frame_h - 1, int(marker_bbox.get("y", 0))))
        w = max(1, min(frame_w - x, int(marker_bbox.get("w", 1))))
        h = max(1, min(frame_h - y, int(marker_bbox.get("h", 1))))
        margin_ratio = max(0.0, min(0.35, float(self.green_region_inner_margin_ratio)))
        margin_x = max(1, int(round(w * margin_ratio))) if w >= 4 else 0
        margin_y = max(1, int(round(h * margin_ratio))) if h >= 4 else 0
        inner_x0 = x + margin_x
        inner_x1 = x + w - margin_x
        inner_y0 = y + margin_y
        inner_y1 = y + h - margin_y
        if inner_x1 <= inner_x0 or inner_y1 <= inner_y0:
            inner_x0, inner_x1 = x, x + w
            inner_y0, inner_y1 = y, y + h

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        inner_saturation = hsv[inner_y0:inner_y1, inner_x0:inner_x1, 1]
        if inner_saturation.size == 0:
            return {
                "valid": False,
                "saturation_p25": 0.0,
                "local_contrast": 0.0,
                "reason": "empty_inner_region",
            }
        saturation_p25 = float(np.percentile(inner_saturation, 25))

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        inner_lab = lab[inner_y0:inner_y1, inner_x0:inner_x1]
        ring_pad = max(10, int(round(max(w, h) * 0.35)))
        ring_x0 = max(0, x - ring_pad)
        ring_x1 = min(frame_w, x + w + ring_pad)
        ring_y0 = max(0, y - ring_pad)
        ring_y1 = min(frame_h, y + h + ring_pad)
        ring_lab = lab[ring_y0:ring_y1, ring_x0:ring_x1]
        ring_mask = np.ones(ring_lab.shape[:2], dtype=bool)
        box_x0 = max(0, x - ring_x0)
        box_x1 = min(ring_mask.shape[1], x + w - ring_x0)
        box_y0 = max(0, y - ring_y0)
        box_y1 = min(ring_mask.shape[0], y + h - ring_y0)
        ring_mask[box_y0:box_y1, box_x0:box_x1] = False
        surrounding_lab = ring_lab[ring_mask]
        if inner_lab.size == 0 or surrounding_lab.size == 0:
            local_contrast = 0.0
        else:
            inner_median = np.median(inner_lab.reshape(-1, 3), axis=0)
            surrounding_median = np.median(surrounding_lab.reshape(-1, 3), axis=0)
            local_contrast = float(np.linalg.norm(inner_median - surrounding_median))

        valid = bool(
            saturation_p25 >= float(self.green_region_min_saturation_p25)
            and local_contrast >= float(self.green_region_min_local_contrast)
        )
        return {
            "valid": valid,
            "saturation_p25": round(saturation_p25, 2),
            "local_contrast": round(local_contrast, 2),
            "reason": "ok" if valid else "weak_chroma_or_local_contrast",
        }

    def _green_pair_quality(
        self,
        marker_bboxes: list[dict[str, int]],
        black_mask: np.ndarray | None,
        image_shape: tuple[int, ...],
        *,
        image: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Validate that two green blobs form the two-square T marker."""
        if len(marker_bboxes) != 2 or len(image_shape) < 2:
            return {"valid": False, "confidence": 0.0, "reason": "marker_count"}

        frame_h, frame_w = int(image_shape[0]), int(image_shape[1])
        ordered = sorted(marker_bboxes, key=lambda item: int(item.get("x", 0)))
        left, right = ordered
        left_cx = float(left["x"]) + (float(left["w"]) / 2.0)
        right_cx = float(right["x"]) + (float(right["w"]) / 2.0)
        left_cy = float(left["y"]) + (float(left["h"]) / 2.0)
        right_cy = float(right["y"]) + (float(right["h"]) / 2.0)
        separation_ratio = abs(right_cx - left_cx) / float(max(1, frame_w))
        vertical_delta_ratio = abs(right_cy - left_cy) / float(max(1, frame_h))
        left_area = float(max(1, int(left["w"]) * int(left["h"])))
        right_area = float(max(1, int(right["w"]) * int(right["h"])))
        area_balance = min(left_area, right_area) / max(left_area, right_area)
        color_qualities = [
            self._green_region_color_quality(image, marker_bbox)
            for marker_bbox in ordered
        ]
        color_valid = bool(
            len(color_qualities) == 2
            and all(bool(item.get("valid", False)) for item in color_qualities)
        )

        line_pixels = 0
        if isinstance(black_mask, np.ndarray) and black_mask.size > 0:
            x0 = max(0, min(int(left["x"]), int(right["x"])))
            x1 = min(frame_w, max(int(left["x"] + left["w"]), int(right["x"] + right["w"])))
            y0_raw = min(int(left["y"]), int(right["y"]))
            y1_raw = max(int(left["y"] + left["h"]), int(right["y"] + right["h"]))
            pad_y = max(8, int(round(max(int(left["h"]), int(right["h"])) * 0.18)))
            y0 = max(0, y0_raw - pad_y)
            y1 = min(frame_h, y1_raw + pad_y)
            if x1 > x0 and y1 > y0:
                line_pixels = int(np.count_nonzero(black_mask[y0:y1, x0:x1]))

        valid = bool(
            separation_ratio >= float(self.green_pair_min_separation_ratio)
            and vertical_delta_ratio <= float(self.green_pair_max_vertical_delta_ratio)
            and area_balance >= float(self.green_pair_min_area_balance)
            and line_pixels >= int(self.green_min_line_pixels_nearby)
            and color_valid
        )
        separation_score = _clamp01(
            separation_ratio / max(float(self.green_pair_min_separation_ratio) * 1.8, 1e-6)
        )
        alignment_score = _clamp01(
            1.0 - (vertical_delta_ratio / max(float(self.green_pair_max_vertical_delta_ratio), 1e-6))
        )
        balance_score = _clamp01(area_balance / max(float(self.green_pair_min_area_balance) * 2.5, 1e-6))
        line_score = _clamp01(line_pixels / max(float(self.green_min_line_pixels_nearby) * 2.0, 1.0))
        confidence = 0.90 + (0.10 * min(separation_score, alignment_score, balance_score, line_score)) if valid else 0.0
        return {
            "valid": valid,
            "confidence": round(float(confidence), 4),
            "separation_ratio": round(float(separation_ratio), 4),
            "vertical_delta_ratio": round(float(vertical_delta_ratio), 4),
            "area_balance": round(float(area_balance), 4),
            "line_pixels": int(line_pixels),
            "color_qualities": color_qualities,
            "reason": "ok" if valid else "geometry_or_line",
        }

    def detect_green_side(self, image: np.ndarray) -> dict[str, Any]:
        if image is None or image.size == 0:
            return {"found": False, "side": "NONE", "contours": []}

        green_mask = self.build_green_mask(image)

        return self._detect_green_side_from_mask(
            image,
            green_mask,
            max_area_ratio=float(self.green_max_area_ratio),
        )

    def _detect_green_side_from_mask(
        self,
        image: np.ndarray,
        green_mask: np.ndarray,
        *,
        max_area_ratio: float,
    ) -> dict[str, Any]:
        """Extract at most one square candidate on each side from a mask."""

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours: list[np.ndarray] = []
        half_x = image.shape[1] / 2.0

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < float(self.green_min_area):
                continue
            img_area = float(image.shape[0] * image.shape[1])
            if area > img_area * max(0.05, float(max_area_ratio)):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h <= 0:
                continue
            aspect = w / float(h)
            if aspect < float(self.green_min_aspect) or aspect > float(self.green_max_aspect):
                continue
            if min(w, h) < float(self.green_min_short_side):
                continue
            extent = area / float(max(1, w * h))
            if extent < float(self.green_min_extent):
                continue
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = area / max(1.0, hull_area)
            if solidity < float(self.green_min_solidity):
                continue
            valid_contours.append(contour)

        valid_contours.sort(key=cv2.contourArea, reverse=True)
        left_contours = [
            contour
            for contour in valid_contours
            if (cv2.boundingRect(contour)[0] + (cv2.boundingRect(contour)[2] / 2.0)) < half_x
        ]
        right_contours = [
            contour
            for contour in valid_contours
            if (cv2.boundingRect(contour)[0] + (cv2.boundingRect(contour)[2] / 2.0)) >= half_x
        ]
        marker_contours: list[np.ndarray] = []
        if left_contours:
            marker_contours.append(left_contours[0])
        if right_contours:
            marker_contours.append(right_contours[0])
        marker_bboxes = [
            _safe_bbox(*cv2.boundingRect(contour), image.shape)
            for contour in marker_contours
        ]
        marker_bboxes = [bbox for bbox in marker_bboxes if bbox is not None]

        if left_contours and right_contours:
            side = "BOTH"
        elif left_contours:
            side = "LEFT"
        elif right_contours:
            side = "RIGHT"
        else:
            side = "NONE"
        return {
            "found": side != "NONE",
            "side": side,
            "contours": valid_contours,
            "marker_contours": marker_contours,
            "marker_count": len(marker_contours),
            "marker_bboxes": marker_bboxes,
        }

    def detect_green_corner(self, image: np.ndarray) -> dict[str, Any]:
        if image is None or image.size == 0:
            return {"found": False, "confidence": 0.0, "bbox": None, "contours": []}

        green_mask = self.build_green_mask(image)

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = float(max(1, image.shape[0] * image.shape[1]))
        max_area = img_area * max(0.08, self.green_corner_max_area_ratio)
        margin = max(2, int(self.green_corner_border_margin))

        best_conf = 0.0
        best_bbox: dict[str, int] | None = None
        best_contour: np.ndarray | None = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.green_corner_min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            short = float(min(w, h))
            long = float(max(w, h))
            if short < self.green_corner_min_short_side:
                continue
            aspect = long / max(1.0, short)
            if aspect < self.green_corner_min_aspect or aspect > self.green_corner_max_aspect:
                continue

            extent = area / float(max(1, w * h))
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / hull_area if hull_area > 1e-6 else 0.0
            if extent < self.green_corner_min_extent or solidity < self.green_corner_min_solidity:
                continue

            touches_border = (
                x <= margin
                or y <= margin
                or (x + w) >= (image.shape[1] - margin)
                or (y + h) >= (image.shape[0] - margin)
            )
            area_score = _clamp01(area / max(self.green_corner_min_area * 3.0, 1.0))
            shape_score = _clamp01(
                0.45 * _clamp01(extent / max(self.green_corner_min_extent, 1e-6))
                + 0.35 * _clamp01(solidity / max(self.green_corner_min_solidity, 1e-6))
                + 0.20 * _clamp01(short / max(self.green_corner_min_short_side * 1.5, 1.0))
            )
            border_score = 1.0 if touches_border else 0.35
            confidence = _clamp01(0.48 * shape_score + 0.34 * area_score + 0.18 * border_score)
            if confidence > best_conf:
                best_conf = confidence
                best_bbox = _safe_bbox(x, y, w, h, image.shape)
                best_contour = contour

        found = bool(best_bbox is not None and best_conf >= self.green_corner_conf_threshold)
        return {
            "found": found,
            "confidence": float(best_conf if found else 0.0),
            "bbox": best_bbox if found else None,
            "contours": [best_contour] if (found and best_contour is not None) else [],
        }

    @staticmethod
    def _horizontal_reference_y(
        black_mask: np.ndarray | None,
        x0: int | float,
        x1: int | float,
        y0: int | float | None = None,
        y1: int | float | None = None,
    ) -> int | None:
        if black_mask is None or black_mask.size == 0:
            return None
        start_x = max(0, int(x0))
        end_x = min(black_mask.shape[1], int(x1))
        if end_x <= start_x:
            return None
        roi = black_mask[:, start_x:end_x]
        if roi.size == 0:
            return None
        row_counts = np.count_nonzero(roi, axis=1).astype(np.float32)
        if row_counts.max() <= 0:
            return None
        smooth = cv2.GaussianBlur(row_counts.reshape(-1, 1), (1, 9), 0).reshape(-1)
        start_y = max(0, int(y0)) if y0 is not None else 0
        end_y = min(black_mask.shape[0], int(y1)) if y1 is not None else black_mask.shape[0]
        if end_y <= start_y:
            return int(np.argmax(smooth))
        local = smooth[start_y:end_y]
        if local.size == 0 or float(local.max()) <= 0.0:
            return None
        return int(start_y + int(np.argmax(local)))

    @staticmethod
    def _merge_contour_bboxes(contours: list[np.ndarray], frame_shape: tuple[int, ...]) -> dict[str, int] | None:
        if not contours:
            return None
        xs: list[int] = []
        ys: list[int] = []
        x2s: list[int] = []
        y2s: list[int] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            xs.append(int(x))
            ys.append(int(y))
            x2s.append(int(x + w))
            y2s.append(int(y + h))
        if not xs:
            return None
        return _safe_bbox(min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys), frame_shape)

    def detect_red_presence(self, image: np.ndarray) -> dict[str, Any]:
        red = self.build_red_mask(image)
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = 0.0
        valid: list[np.ndarray] = []
        best_conf = 0.0
        best_bbox: dict[str, int] | None = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.red_min_area:
                continue
            rect = cv2.minAreaRect(contour)
            width = float(rect[1][0])
            height = float(rect[1][1])
            short = min(width, height)
            long = max(width, height)
            if short <= 0:
                continue
            ratio = long / short
            if ratio < self.red_min_ratio or long < self.red_min_long_side:
                continue
            valid.append(contour)
            total_area += area
            x, y, w, h = cv2.boundingRect(contour)
            bbox = _safe_bbox(x, y, w, h, image.shape)
            if bbox is None:
                continue
            ratio_score = _clamp01((ratio - self.red_min_ratio) / max(0.5, self.red_min_ratio))
            area_score = _clamp01(area / max(float(self.red_min_area), 1.0) / 4.0)
            conf = _clamp01(0.65 * ratio_score + 0.35 * area_score)
            if conf > best_conf:
                best_conf = conf
                best_bbox = bbox
        return {
            "found": len(valid) > 0,
            "area": int(total_area),
            "contours": valid,
            "confidence": float(best_conf) if valid else 0.0,
            "bbox": best_bbox if valid else None,
            "mask": red,
        }


class SilverBallRuntime:
    def __init__(self, enabled: bool, model_path: Path | None, threshold: float, input_size: int = 64) -> None:
        self.enabled = bool(enabled)
        self.available = False
        self.error: str | None = None
        self.threshold = float(threshold)
        self.input_size = max(24, int(input_size))
        self._model_path = model_path
        self._initialized = False
        self._model: Any = None
        if not self.enabled:
            self.error = "disabled"

    def _ensure_loaded(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        if not self.enabled:
            self.error = "disabled"
            return
        if self._model_path is None or not self._model_path.exists():
            self.error = "model_not_found"
            return
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            self.error = str(exc)
            return
        try:
            safe_model_path = self._filesystem_safe_path(self._model_path)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self._model = torch.jit.load(safe_model_path, map_location="cpu")
                self._model.eval()
            self.available = True
        except Exception as exc:  # pragma: no cover
            self.error = str(exc)

    @staticmethod
    def _filesystem_safe_path(path: Path) -> str:
        value = str(path)
        if os.name != "nt":
            return value
        try:
            import ctypes

            initial = 260
            buffer = ctypes.create_unicode_buffer(initial)
            result = ctypes.windll.kernel32.GetShortPathNameW(value, buffer, initial)
            if result == 0:
                return value
            if result > initial:
                buffer = ctypes.create_unicode_buffer(result)
                result = ctypes.windll.kernel32.GetShortPathNameW(value, buffer, result)
                if result == 0:
                    return value
            short_path = buffer.value
            return short_path if short_path else value
        except Exception:
            return value

    def _prepare_patch(self, image: np.ndarray, bbox: Mapping[str, Any] | None) -> np.ndarray:
        if bbox is not None:
            x = int(bbox.get("x", 0))
            y = int(bbox.get("y", 0))
            w = int(bbox.get("w", 0))
            h = int(bbox.get("h", 0))
            if w > 0 and h > 0:
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(image.shape[1], x + w)
                y1 = min(image.shape[0], y + h)
                if x1 > x0 and y1 > y0:
                    patch = image[y0:y1, x0:x1]
                else:
                    patch = image
            else:
                patch = image
        else:
            patch = image
        patch = cv2.resize(patch, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        return patch

    def predict(self, image: np.ndarray, bbox: Mapping[str, Any] | None) -> tuple[bool, float, dict[str, Any]]:
        was_initialized = self._initialized
        self._ensure_loaded()
        just_initialized = not was_initialized
        if not self.available or self._model is None:
            return False, 0.0, {"available": False, "error": self.error or "unavailable", "just_initialized": just_initialized}

        try:
            import torch

            patch = self._prepare_patch(image, bbox)
            rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)
            with torch.no_grad():
                logits = self._model(tensor)
            out = logits.reshape(-1).detach().cpu().numpy().astype(np.float32)
            if out.size == 0:
                score = 0.0
            elif out.size == 1:
                score = float(out[0])
                if score < 0.0 or score > 1.0:
                    score = float(1.0 / (1.0 + np.exp(-score)))
            else:
                shifted = out - np.max(out)
                exp = np.exp(shifted)
                denom = float(np.sum(exp))
                probs = exp / denom if denom > 0 else np.zeros_like(exp)
                score = float(probs[1] if probs.size > 1 else probs[0])
            score = _clamp01(score)
            return score >= self.threshold, score, {"available": True, "just_initialized": just_initialized}
        except Exception as exc:  # pragma: no cover
            return False, 0.0, {"available": False, "error": str(exc), "just_initialized": just_initialized}


class SilverLineRuntime:
    def __init__(
        self,
        enabled: bool,
        model_path: Path | None,
        threshold: float,
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        cfg = dict(settings or {})
        self.available = False
        self.error: str | None = None
        self.threshold = float(threshold)
        self.candidate_threshold = float(cfg.get("candidate_threshold", max(0.55, self.threshold * 0.82)))
        self.mode = str(cfg.get("mode", "assist")).strip().lower() or "assist"
        self.heuristic_enabled = bool(cfg.get("heuristic_enabled", True))
        self.heuristic_threshold = float(cfg.get("heuristic_threshold", max(0.55, self.candidate_threshold)))
        self.decision_policy = str(cfg.get("decision_policy", "model_or_heuristic")).strip().lower() or "model_or_heuristic"
        self.stability_window = max(1, int(cfg.get("stability_window", 4)))
        self.required_votes = max(1, min(self.stability_window, int(cfg.get("required_votes", 3))))
        self.specular_v_min = int(cfg.get("specular_v_min", 180))
        self.specular_s_max = int(cfg.get("specular_s_max", 80))
        self.min_area = float(cfg.get("min_area", 260.0))
        self.min_width_ratio = float(cfg.get("min_width_ratio", 0.32))
        self.min_aspect_ratio = float(cfg.get("min_aspect_ratio", 2.2))
        self.max_area_ratio = float(cfg.get("max_area_ratio", 0.48))
        self.top_clear_band_ratio = float(cfg.get("top_clear_band_ratio", 0.14))
        self.top_black_ratio_max = float(cfg.get("top_black_ratio_max", 0.035))
        self.center_tolerance_ratio = float(cfg.get("center_tolerance_ratio", 0.48))
        self.focus_height_ratio = float(cfg.get("focus_height_ratio", 0.60))
        self.adaptive_v_percentile = float(cfg.get("adaptive_v_percentile", 80.0))
        self.kernel_size = max(3, int(cfg.get("kernel_size", 5)))
        self.mask_open_iterations = max(0, int(cfg.get("mask_open_iterations", 1)))
        self.mask_close_iterations = max(0, int(cfg.get("mask_close_iterations", 1)))
        self._enabled = bool(enabled)
        self._model_path = model_path
        self._initialized = False
        self._detector: Any = None
        self._vote_window: deque[int] = deque(maxlen=self.stability_window)
        self._confidence_window: deque[float] = deque(maxlen=self.stability_window)
        self.last_mask: np.ndarray | None = None
        self.last_bbox: dict[str, int] | None = None

    def _ensure_loaded(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        if not self._enabled:
            self.error = "disabled"
            return
        if self._model_path is None or not self._model_path.exists():
            self.error = "model_not_found"
            return
        try:
            intl_root = _REPO_ROOT / "1_international"
            if str(intl_root) not in sys.path:
                sys.path.insert(0, str(intl_root))
            module = importlib.import_module("behaviours.silver_detection")
            detector_cls = getattr(module, "SilverLineDetector")
            model_arg = self._filesystem_safe_path(self._model_path)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                self._detector = detector_cls(model_arg)
            self.available = True
        except Exception as exc:  # pragma: no cover
            self.error = str(exc)

    @staticmethod
    def _filesystem_safe_path(path: Path) -> str:
        value = str(path)
        if os.name != "nt":
            return value
        try:
            import ctypes

            initial = 260
            buffer = ctypes.create_unicode_buffer(initial)
            result = ctypes.windll.kernel32.GetShortPathNameW(value, buffer, initial)
            if result == 0:
                return value
            if result > initial:
                buffer = ctypes.create_unicode_buffer(result)
                result = ctypes.windll.kernel32.GetShortPathNameW(value, buffer, result)
                if result == 0:
                    return value
            short_path = buffer.value
            return short_path if short_path else value
        except Exception:
            return value

    def _predict_model(self, image: np.ndarray) -> tuple[bool, float, dict[str, Any]]:
        was_initialized = self._initialized
        self._ensure_loaded()
        just_initialized = not was_initialized
        if not self.available or self._detector is None:
            return (
                False,
                0.0,
                {
                    "available": False,
                    "error": self.error or "unavailable",
                    "just_initialized": just_initialized,
                },
            )
        try:
            out = self._detector.predict(image)
            if isinstance(out, tuple):
                out = out[0]
            confidence = float(out.get("confidence", 0.0))
            found = int(out.get("prediction", 0)) == 1 and confidence >= self.threshold
            return (
                found,
                confidence,
                {
                    "available": True,
                    "just_initialized": just_initialized,
                    "prediction": int(out.get("prediction", 0)),
                    "class_name": str(out.get("class_name", "")),
                },
            )
        except Exception as exc:  # pragma: no cover
            return (
                False,
                0.0,
                {
                    "available": False,
                    "error": str(exc),
                    "just_initialized": just_initialized,
                },
            )

    def _build_heuristic_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        focus_limit = max(1, min(hsv.shape[0], int(round(hsv.shape[0] * self.focus_height_ratio))))
        dynamic_v_min = max(0, self.specular_v_min)
        candidate_values = hsv[:focus_limit, :, 2][hsv[:focus_limit, :, 1] <= max(0, self.specular_s_max)]
        if candidate_values.size >= 64:
            dynamic_v_min = max(
                dynamic_v_min,
                int(
                    round(
                        float(
                            np.percentile(
                                candidate_values.astype(np.float32),
                                float(np.clip(self.adaptive_v_percentile, 0.0, 100.0)),
                            )
                        )
                    )
                ),
            )
        mask = cv2.inRange(
            hsv,
            (0, 0, dynamic_v_min),
            (179, max(0, self.specular_s_max), 255),
        )
        mask[focus_limit:, :] = 0
        kernel_size = self.kernel_size + (1 - self.kernel_size % 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        if self.mask_open_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.mask_open_iterations)
        if self.mask_close_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.mask_close_iterations)
        return mask

    def _heuristic_predict(
        self,
        image: np.ndarray,
        *,
        black_mask: np.ndarray | None,
        green_found: bool,
    ) -> tuple[bool, float, dict[str, Any]]:
        if image is None or image.size == 0:
            self.last_mask = _empty_mask(0, 0)
            self.last_bbox = None
            return False, 0.0, {"found": False, "suppressed_reason": "empty_image", "bbox": None}
        if not self.heuristic_enabled:
            self.last_mask = _empty_mask(image.shape[0], image.shape[1])
            self.last_bbox = None
            return False, 0.0, {"found": False, "suppressed_reason": "heuristic_disabled", "bbox": None}

        mask = self._build_heuristic_mask(image)
        self.last_mask = mask
        self.last_bbox = None

        top_band_h = max(1, int(round(image.shape[0] * self.top_clear_band_ratio)))
        top_black_ratio = _mask_ratio(black_mask[:top_band_h, :]) if black_mask is not None and black_mask.size else 0.0
        suppressed_reason = ""
        if green_found:
            suppressed_reason = "green_marker"
        elif top_black_ratio > self.top_black_ratio_max:
            suppressed_reason = "top_black_ratio"

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(max(1, image.shape[0] * image.shape[1]))
        best_conf = 0.0
        best_bbox: dict[str, int] | None = None
        best_stats: dict[str, Any] = {}

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > (frame_area * self.max_area_ratio):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            width_ratio = float(w / max(1, image.shape[1]))
            if width_ratio < self.min_width_ratio:
                continue
            aspect_ratio = float(w / max(1, h))
            if aspect_ratio < self.min_aspect_ratio:
                continue
            center_x = x + (w / 2.0)
            center_offset = abs(center_x - (image.shape[1] / 2.0)) / max(1.0, image.shape[1] / 2.0)
            if center_offset > self.center_tolerance_ratio:
                continue

            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            sat_mean = float(np.mean(hsv[:, :, 1][contour_mask > 0])) if np.any(contour_mask > 0) else 255.0
            val_mean = float(np.mean(hsv[:, :, 2][contour_mask > 0])) if np.any(contour_mask > 0) else 0.0

            sat_score = _clamp01(1.0 - (sat_mean / max(1.0, float(self.specular_s_max) * 1.6)))
            val_score = _clamp01((val_mean - float(self.specular_v_min)) / max(1.0, 255.0 - float(self.specular_v_min)))
            width_score = _clamp01(width_ratio / max(1e-6, self.min_width_ratio * 1.45))
            aspect_score = _clamp01(aspect_ratio / max(1e-6, self.min_aspect_ratio * 1.8))
            clear_score = _clamp01(1.0 - (top_black_ratio / max(1e-6, self.top_black_ratio_max)))
            confidence = _clamp01(
                0.26 * sat_score
                + 0.26 * val_score
                + 0.20 * width_score
                + 0.16 * aspect_score
                + 0.12 * clear_score
            )
            if confidence <= best_conf:
                continue

            best_conf = confidence
            best_bbox = _safe_bbox(x, y, w, h, image.shape)
            best_stats = {
                "area": float(area),
                "width_ratio": float(width_ratio),
                "aspect_ratio": float(aspect_ratio),
                "center_offset": float(center_offset),
                "sat_mean": float(sat_mean),
                "val_mean": float(val_mean),
            }

        self.last_bbox = best_bbox
        if best_bbox is None and not suppressed_reason:
            suppressed_reason = "no_candidate"

        found = bool(best_bbox is not None and best_conf >= self.heuristic_threshold and not suppressed_reason)
        return (
            found,
            float(best_conf),
            {
                "found": bool(found),
                "confidence": float(best_conf),
                "bbox": best_bbox,
                "mask_ratio": float(_mask_ratio(mask)),
                "top_black_ratio": float(top_black_ratio),
                "suppressed_reason": suppressed_reason,
                **best_stats,
            },
        )

    def predict(
        self,
        image: np.ndarray,
        *,
        heuristic_image: np.ndarray | None = None,
        black_mask: np.ndarray | None = None,
        green_found: bool = False,
    ) -> tuple[bool, float, dict[str, Any]]:
        model_found, model_conf, model_meta = self._predict_model(image)
        heuristic_found, heuristic_conf, heuristic_meta = self._heuristic_predict(
            heuristic_image if isinstance(heuristic_image, np.ndarray) and heuristic_image.size > 0 else image,
            black_mask=black_mask,
            green_found=green_found,
        )

        if self.decision_policy == "heuristic_only":
            raw_found = heuristic_found
        elif self.decision_policy == "model_only":
            raw_found = model_found
        elif self.decision_policy == "consensus":
            raw_found = heuristic_found and model_found if model_meta.get("available", False) else heuristic_found
        else:
            raw_found = heuristic_found or model_found

        raw_confidence = float(max(model_conf, heuristic_conf))
        vote = 1 if raw_found and raw_confidence >= self.candidate_threshold else 0
        self._vote_window.append(vote)
        self._confidence_window.append(raw_confidence if vote else 0.0)
        votes = int(sum(self._vote_window))
        stable_conf = raw_confidence
        if votes > 0:
            positive = [score for score in self._confidence_window if score > 0.0]
            if positive:
                stable_conf = float(sum(positive) / len(positive))
        validated_found = votes >= self.required_votes
        public_found = bool(validated_found and self.mode != "manual")

        return (
            public_found,
            float(_clamp01(stable_conf)),
            {
                "available": bool(model_meta.get("available", False) or self.heuristic_enabled),
                "just_initialized": bool(model_meta.get("just_initialized", False)),
                "error": model_meta.get("error"),
                "mode": self.mode,
                "bbox": heuristic_meta.get("bbox"),
                "model": {
                    "found": bool(model_found),
                    "confidence": float(_clamp01(model_conf)),
                    **model_meta,
                },
                "heuristic": {
                    **heuristic_meta,
                    "confidence": float(_clamp01(heuristic_conf)),
                },
                "decision": {
                    "policy": self.decision_policy,
                    "raw_found": bool(raw_found),
                    "validated_found": bool(validated_found),
                    "votes": int(votes),
                    "required_votes": int(self.required_votes),
                    "window": int(self.stability_window),
                    "confidence": float(_clamp01(stable_conf)),
                },
            },
        )


class DeadVictimRuntime:
    def __init__(self, enabled: bool, model_path: Path | None, threshold: float) -> None:
        self.enabled = bool(enabled)
        self.available = False
        self.error: str | None = None
        self.threshold = float(threshold)
        self._enabled = self.enabled
        self._model_path = model_path
        self._initialized = False
        self._interpreter: Any = None
        self._input: dict[str, Any] | None = None
        self._output: dict[str, Any] | None = None
        if not self._enabled:
            self.error = "disabled"

    def _load(self, model_path: Path) -> None:
        interpreter_cls: Any | None = None
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore

            interpreter_cls = Interpreter
        except Exception:
            try:
                from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

                interpreter_cls = Interpreter
            except Exception as exc:  # pragma: no cover
                self.error = str(exc)
                return
        try:
            safe_model_path = SilverLineRuntime._filesystem_safe_path(model_path)
            self._interpreter = interpreter_cls(model_path=safe_model_path)
            self._interpreter.allocate_tensors()
            self._input = self._interpreter.get_input_details()[0]
            self._output = self._interpreter.get_output_details()[0]
            self.available = True
        except Exception as exc:  # pragma: no cover
            self.error = str(exc)

    def _ensure_loaded(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        if not self._enabled:
            self.error = "disabled"
            return
        if self._model_path is None or not self._model_path.exists():
            self.error = "model_not_found"
            return
        self._load(self._model_path)

    @staticmethod
    def _softmax(vector: np.ndarray) -> np.ndarray:
        shifted = vector - np.max(vector)
        exp = np.exp(shifted)
        denom = np.sum(exp)
        return exp / denom if denom > 0 else np.zeros_like(vector)

    def _prepare_input(self, image: np.ndarray) -> np.ndarray:
        if self._input is None:
            raise RuntimeError("missing tflite input details")
        shape = [int(v) for v in self._input["shape"]]
        dtype = self._input["dtype"]
        if len(shape) != 4:
            raise RuntimeError(f"unsupported input shape: {shape}")
        _, d1, d2, d3 = shape
        if d3 in (1, 3):
            h, w, c = d1, d2, d3
            is_nhwc = True
        else:
            c, h, w = d1, d2, d3
            is_nhwc = False
        if c not in (1, 3):
            raise RuntimeError(f"unsupported channel count: {c}")

        tensor = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        if c == 1:
            tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2GRAY)[:, :, None]
        if not is_nhwc:
            tensor = np.transpose(tensor, (2, 0, 1))
        tensor = tensor.astype(np.float32)
        if np.max(tensor) > 1.0:
            tensor = tensor / 255.0
        tensor = tensor[None, ...]

        if dtype == np.uint8:
            quant = self._input.get("quantization", (0.0, 0))
            scale = float(quant[0]) if quant else 0.0
            zero = int(quant[1]) if quant else 0
            if scale > 0:
                tensor = np.clip(np.round(tensor / scale + zero), 0, 255).astype(np.uint8)
            else:
                tensor = np.clip(np.round(tensor * 255.0), 0, 255).astype(np.uint8)
        else:
            tensor = tensor.astype(dtype)
        return tensor

    def predict(self, image: np.ndarray) -> tuple[bool, float, dict[str, Any]]:
        was_initialized = self._initialized
        self._ensure_loaded()
        just_initialized = not was_initialized
        if not self.available or self._interpreter is None or self._input is None or self._output is None:
            return False, 0.0, {"available": False, "error": self.error or "unavailable", "just_initialized": just_initialized}
        try:
            tensor = self._prepare_input(image)
            self._interpreter.set_tensor(self._input["index"], tensor)
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output["index"]).reshape(-1).astype(np.float32)
            if output.size == 0:
                return False, 0.0, {"available": True, "just_initialized": just_initialized}
            if output.size == 1:
                score = float(output[0])
                if score < 0 or score > 1:
                    score = float(1.0 / (1.0 + np.exp(-score)))
            else:
                probs = self._softmax(output)
                score = float(probs[1] if probs.size > 1 else probs[0])
            return score >= self.threshold, score, {"available": True, "just_initialized": just_initialized}
        except Exception as exc:  # pragma: no cover
            return False, 0.0, {"available": False, "error": str(exc), "just_initialized": just_initialized}


@dataclass(slots=True)
class PipelineOutput:
    event: VisionDetectionEvent
    processed_frame: np.ndarray
    debug_views: dict[str, np.ndarray] = field(default_factory=dict)


class VisionPipelineManager:
    DEFAULT_DEBUG_VIEWS: tuple[str, ...] = (
        "raw",
        "processed",
        "line_mask",
        "green_mask",
        "red_mask",
        "victim_mask",
        "silver_line_mask",
        "composite",
    )

    def __init__(self, config: VisionConfig, *, debug_artifacts_enabled: bool | None = None) -> None:
        self.config = config
        self.preprocessor = VisionPreprocessor(config.preprocessor)

        line_resize = self._profile_resize("line", default_w=320, default_h=200)
        rescue_resize = self._profile_resize("rescue", default_w=320, default_h=240)
        self.line_detector = LineDetector(line_resize[0], line_resize[1], config.detector("line"))
        self.ball_detector = BallDetector(rescue_resize[0], rescue_resize[1], config.detector("ball"))
        color_cfg = config.detector("color")
        self.color_detector = ColorMarkerDetector(color_cfg)
        self.green_pair_hold_frames = max(0, min(5, int(color_cfg.get("green_pair_hold_frames", 2))))
        ball_cfg = config.detector("ball")
        self.silver_ball = SilverBallRuntime(
            enabled=bool(ball_cfg.get("silver_model_enabled", True)),
            model_path=config.model_path("silver_ball"),
            threshold=float(ball_cfg.get("silver_model_threshold", 0.62)),
            input_size=int(ball_cfg.get("silver_model_input_size", 64)),
        )

        silver_cfg = config.detector("silver_line")
        self.silver_every_n = max(1, int(silver_cfg.get("run_every_n_frames", 2)))
        self.silver = SilverLineRuntime(
            enabled=bool(silver_cfg.get("enabled", True)),
            model_path=config.model_path("silver_line"),
            threshold=float(silver_cfg.get("confidence_threshold", 0.95)),
            settings=silver_cfg,
        )
        dead_cfg = config.detector("dead_victim")
        self.dead_every_n = max(1, int(dead_cfg.get("run_every_n_frames", 3)))
        self.dead = DeadVictimRuntime(
            enabled=bool(dead_cfg.get("enabled", True)),
            model_path=config.model_path("dead_victim_tflite"),
            threshold=float(dead_cfg.get("confidence_threshold", 0.55)),
        )

        benchmark_cfg = config.benchmark()
        self.baseline_monolithic_ms = float(benchmark_cfg.get("baseline_monolithic_ms", 0.0))
        self.expected_by_state = benchmark_cfg.get("state_expected_ms", {})
        runtime_cfg = config.data.get("runtime", {}) if isinstance(config.data.get("runtime"), Mapping) else {}
        history_size = max(10, int(runtime_cfg.get("history_size", 120)))
        self.corner_stability_window = max(3, int(runtime_cfg.get("corner_stability_window", 5)))
        self.corner_on_votes = max(1, int(runtime_cfg.get("corner_on_votes", 3)))
        self.corner_off_votes = max(0, int(runtime_cfg.get("corner_off_votes", 1)))
        if debug_artifacts_enabled is None:
            self.debug_artifacts_enabled = bool(runtime_cfg.get("debug_artifacts_enabled", False))
        else:
            self.debug_artifacts_enabled = bool(debug_artifacts_enabled)
        requested_views = runtime_cfg.get("debug_views", list(self.DEFAULT_DEBUG_VIEWS))
        if isinstance(requested_views, list):
            normalized = [str(item).strip().lower() for item in requested_views if str(item).strip()]
            self.debug_views = tuple(normalized or self.DEFAULT_DEBUG_VIEWS)
        else:
            self.debug_views = self.DEFAULT_DEBUG_VIEWS
        self.latency_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=history_size))

        self.frame_counter = 0
        self.last_live_x: int | None = None
        self.last_dead_x: int | None = None
        self.silver_cache = (False, 0.0)
        self.dead_cache = (False, 0.0, {"available": False})
        self.dead_cache_ready = False
        self._warmup_seen: set[str] = set()
        self._green_pair_hold_remaining = 0
        self._last_green_pair: dict[str, Any] = {}

    def _stabilize_green_pair(self, current: Mapping[str, Any]) -> dict[str, Any]:
        """Hold a confirmed two-square label across very short contour dropouts."""
        result = dict(current)
        live_pair = bool(
            result.get("found", False)
            and str(result.get("side", "NONE")).upper() == "BOTH"
            and str(result.get("instruction", "")).upper().replace("_", " ") == "VERDE MEIA VOLTA"
            and int(result.get("marker_count", 0) or 0) == 2
            and bool((result.get("pair_quality") or {}).get("valid", False))
        )
        if live_pair:
            self._green_pair_hold_remaining = int(self.green_pair_hold_frames)
            self._last_green_pair = {
                "marker_bboxes": [dict(item) for item in result.get("marker_bboxes", []) if isinstance(item, Mapping)],
                "bbox": dict(result["bbox"]) if isinstance(result.get("bbox"), Mapping) else None,
                "confidence": float(result.get("confidence", 0.95) or 0.95),
                "pair_quality": dict(result.get("pair_quality", {})),
                "segmentation_source": str(result.get("segmentation_source", "primary")),
            }
            result["pair_live"] = True
            result["pair_stable"] = True
            result["pair_source"] = "live"
            result["pair_hold_remaining"] = int(self._green_pair_hold_remaining)
            return result

        if self._green_pair_hold_remaining <= 0 or not self._last_green_pair:
            result["pair_live"] = False
            result["pair_stable"] = False
            result["pair_source"] = "none"
            result["pair_hold_remaining"] = 0
            return result

        self._green_pair_hold_remaining -= 1
        last_boxes = [dict(item) for item in self._last_green_pair.get("marker_bboxes", [])]
        current_boxes = [dict(item) for item in result.get("marker_bboxes", []) if isinstance(item, Mapping)]
        frame_mid = self.line_detector.width / 2.0
        for box in current_boxes:
            center_x = float(box.get("x", 0)) + (float(box.get("w", 0)) / 2.0)
            target_index = 0 if center_x < frame_mid else 1
            if target_index < len(last_boxes):
                last_boxes[target_index] = box

        result.update(
            {
                "found": True,
                "side": "BOTH",
                "instruction": "VERDE MEIA VOLTA",
                "marker_count": 2,
                "marker_bboxes": last_boxes,
                "bbox": self._last_green_pair.get("bbox"),
                "confidence": max(0.90, float(self._last_green_pair.get("confidence", 0.95)) * 0.98),
                "pair_quality": {
                    **dict(self._last_green_pair.get("pair_quality", {})),
                    "temporal_hold": True,
                },
                "pair_live": False,
                "pair_stable": True,
                "pair_source": "temporal_hold",
                "segmentation_source": "temporal_hold",
                "pair_hold_remaining": int(self._green_pair_hold_remaining),
            }
        )
        return result

    def run(self, state: RobotState | str, frame_bgr: np.ndarray) -> PipelineOutput:
        state_name = _state_name(state)
        self.frame_counter += 1

        pipeline_cfg = self.config.pipeline(state_name)
        profile = str(pipeline_cfg.get("profile", self._default_profile(state_name)))
        detectors = self._active_detectors(pipeline_cfg.get("detectors"))

        started = time.perf_counter()
        prepared = self.preprocessor.prepare(frame_bgr, profile=profile)
        processed = prepared.frame
        overlay_frame = processed.copy()

        line = False
        balls = 0
        green = False
        red = False
        victims = 0
        ignore_benchmark_sample = False
        contour: np.ndarray | None = None
        green_contours: list[np.ndarray] = []
        red_contours: list[np.ndarray] = []
        frame_h = int(processed.shape[0]) if processed.ndim >= 2 else 0
        frame_w = int(processed.shape[1]) if processed.ndim >= 2 else 0
        green_mask = _empty_mask(frame_h, frame_w)
        red_mask = _empty_mask(frame_h, frame_w)
        victim_mask = _empty_mask(frame_h, frame_w)
        silver_line_mask = _empty_mask(frame_h, frame_w)

        metadata: dict[str, Any] = {
            "pipeline_profile": profile,
            "active_detectors": detectors,
            "preprocessor": prepared.metadata,
            "runtime": {
                "corner_stability_window": int(self.corner_stability_window),
                "corner_on_votes": int(self.corner_on_votes),
                "corner_off_votes": int(self.corner_off_votes),
            },
            "silver_ball_found": False,
            "silver_ball_confidence": 0.0,
            "silver_ball_bbox": None,
            "silver_ball_origin": "none",
            "silver_ball_candidates": [],
            "silver_ball_count": 0,
            "black_ball_found": False,
            "black_ball_confidence": 0.0,
            "black_ball_bbox": None,
            "black_ball_origin": "none",
            "green_side": "NONE",
            "green_instruction": "NO GREEN",
            "green_marker_found": False,
            "green_marker_confidence": 0.0,
            "green_marker_bbox": None,
            "green_marker_count": 0,
            "green_marker_bboxes": [],
            "green_reference_y": None,
            "green_center_y": None,
            "green_relation_delta_y": None,
            "green_relation_confidence": 0.0,
            "green_segmentation_source": "none",
            "green_rejected_without_line": False,
            "green_corner_found": False,
            "green_corner_confidence": 0.0,
            "green_corner_bbox": None,
            "red_corner_found": False,
            "red_corner_confidence": 0.0,
            "red_corner_bbox": None,
            "silver_ball_model": {"found": False, "confidence": 0.0, "available": False},
            "debug_views_available": [],
            "silver_ball_black_overlap_suppressed": False,
            "line_angle_deg": 90,
            "line_gap_frames": 0,
            "line_center_x": None,
            "line_center_y": None,
            "line_offset_norm": 0.0,
            "line_offset_source": "none",
            "line_confidence": 0.0,
            "line_bbox": None,
            "line_mask_ratio": 0.0,
            "line_candidate_reason": "not_evaluated",
            "line_geometry": {},
        }

        black_mask: np.ndarray | None = None
        if "line" in detectors:
            contour, black_mask = self.line_detector.black_mask(processed)
            line = contour is not None
            angle, gap = self.line_detector.calculate_angle(contour)
            metadata["line_angle_deg"] = int(angle)
            metadata["line_gap_frames"] = int(gap)
            metadata["line_mask_ratio"] = float(_mask_ratio(black_mask))
            metadata["line_candidate_reason"] = str(self.line_detector.last_rejection_reason)
            metadata["line_geometry"] = dict(self.line_detector.last_geometry)
            if contour is not None:
                x, y, w, h = cv2.boundingRect(contour)
                area = float(cv2.contourArea(contour))
                center_x = float(x + (w / 2.0))
                center_y = float(y + (h / 2.0))
                offset_norm = 0.0
                if frame_w > 1:
                    offset_norm = float((center_x - (frame_w / 2.0)) / max(1.0, frame_w / 2.0))
                geometry = metadata.get("line_geometry")
                if isinstance(geometry, Mapping) and geometry.get("path_center_offset_norm") is not None:
                    # The full contour bounding box includes the far end of a
                    # diagonal/curve. Steering must use the lower ground path
                    # that the robot is about to reach, not that far endpoint.
                    path_offset = geometry.get("path_center_offset_norm")
                    offset_norm = float(path_offset)
                    center_x = (frame_w / 2.0) * (1.0 + offset_norm)
                    metadata["line_offset_source"] = "ground_path"
                else:
                    metadata["line_offset_source"] = "bbox"
                area_ratio = area / float(max(1, frame_w * frame_h))
                confidence = _clamp01(max(float(metadata["line_mask_ratio"]), area_ratio * 8.0))
                if gap > 0:
                    confidence = _clamp01(confidence * 0.6)
                metadata["line_center_x"] = round(center_x, 2)
                metadata["line_center_y"] = round(center_y, 2)
                metadata["line_offset_norm"] = round(max(-1.0, min(1.0, offset_norm)), 4)
                metadata["line_confidence"] = round(confidence, 4)
                metadata["line_bbox"] = _safe_bbox(x, y, w, h, processed.shape)

        if "green" in detectors:
            green_mask = self.color_detector.build_green_mask(processed)
            green_out = self.color_detector.detect_green_instruction(processed, black_mask)
            if bool(green_out.get("found", False)) and not line:
                # A route marker is meaningful only beside a line contour that
                # passed the same frame's geometry checks.  Color pixels near a
                # rejected floor/shadow contour must not start a new maneuver.
                # Feeding an empty observation to the pair stabilizer still
                # permits its short hold for a pair proven on an earlier frame.
                metadata["green_rejected_without_line"] = True
                green_out = {
                    "found": False,
                    "side": "NONE",
                    "instruction": "NO GREEN",
                    "ref_y": None,
                    "center_y": None,
                    "relation_delta_y": None,
                    "relation_confidence": 0.0,
                    "contours": [],
                    "marker_count": 0,
                    "marker_bboxes": [],
                    "confidence": 0.0,
                    "bbox": None,
                    "pair_quality": dict(green_out.get("pair_quality", {})),
                    "single_recovery_quality": dict(
                        green_out.get("single_recovery_quality", {})
                    ),
                    "segmentation_source": "line_gate",
                }
            green_out = self._stabilize_green_pair(green_out)
            green = bool(green_out.get("found", False))
            green_contours = list(green_out.get("contours", []))
            metadata["green_side"] = green_out.get("side", "NONE")
            metadata["green_instruction"] = green_out.get("instruction", "NO GREEN")
            metadata["green_marker_found"] = bool(green)
            metadata["green_marker_confidence"] = float(green_out.get("confidence", 0.0) if green else 0.0)
            metadata["green_marker_bbox"] = green_out.get("bbox") if green else None
            metadata["green_marker_count"] = int(green_out.get("marker_count", 0) if green else 0)
            metadata["green_marker_bboxes"] = list(green_out.get("marker_bboxes", []) if green else [])
            metadata["green_reference_y"] = green_out.get("ref_y") if green else None
            metadata["green_center_y"] = green_out.get("center_y") if green else None
            metadata["green_relation_delta_y"] = (
                green_out.get("relation_delta_y") if green else None
            )
            metadata["green_relation_confidence"] = float(
                green_out.get("relation_confidence", 0.0) if green else 0.0
            )
            metadata["green_pair_quality"] = dict(green_out.get("pair_quality", {}))
            metadata["green_single_recovery_quality"] = dict(
                green_out.get("single_recovery_quality", {})
            )
            metadata["green_pair_live"] = bool(green_out.get("pair_live", False))
            metadata["green_pair_stable"] = bool(green_out.get("pair_stable", False))
            metadata["green_pair_source"] = str(green_out.get("pair_source", "none"))
            metadata["green_segmentation_source"] = str(green_out.get("segmentation_source", "none"))
            metadata["green_pair_hold_remaining"] = int(green_out.get("pair_hold_remaining", 0) or 0)
        else:
            self._green_pair_hold_remaining = 0
            self._last_green_pair = {}

        if "green_corner" in detectors:
            if green_mask.size == 0:
                green_mask = self.color_detector.build_green_mask(processed)
            green_corner_out = self.color_detector.detect_green_corner(processed)
            metadata["green_corner_found"] = bool(green_corner_out.get("found", False))
            metadata["green_corner_confidence"] = float(
                green_corner_out.get("confidence", 0.0) if bool(green_corner_out.get("found", False)) else 0.0
            )
            metadata["green_corner_bbox"] = green_corner_out.get("bbox") if bool(green_corner_out.get("found", False)) else None

        if "red" in detectors or "red_zone" in detectors:
            red_out = self.color_detector.detect_red_presence(processed)
            red = bool(red_out.get("found", False))
            red_contours = list(red_out.get("contours", []))
            red_mask = red_out.get("mask") if isinstance(red_out.get("mask"), np.ndarray) else red_mask
            metadata["red_area"] = int(red_out.get("area", 0))
            metadata["red_corner_found"] = bool(red)
            metadata["red_corner_confidence"] = float(red_out.get("confidence", 0.0) if red else 0.0)
            metadata["red_corner_bbox"] = red_out.get("bbox") if red else None

        if "silver_line" in detectors:
            if self.frame_counter % self.silver_every_n == 0:
                silver_input = (
                    prepared.source_frame
                    if isinstance(prepared.source_frame, np.ndarray) and prepared.source_frame.size > 0
                    else processed
                )
                silver_found, silver_conf, silver_meta = self.silver.predict(
                    processed,
                    heuristic_image=silver_input,
                    black_mask=black_mask,
                    green_found=green,
                )
                self.silver_cache = (silver_found, silver_conf)
                silver_meta["cached"] = False
            else:
                silver_found, silver_conf = self.silver_cache
                silver_meta = {"available": self.silver.available, "cached": True}
            if bool(silver_meta.get("just_initialized", False)):
                ignore_benchmark_sample = True
            if isinstance(self.silver.last_mask, np.ndarray):
                silver_line_mask = self.silver.last_mask.copy()
            metadata["silver_line"] = {
                "found": bool(silver_found),
                "confidence": float(silver_conf),
                "bbox": silver_meta.get("bbox"),
                **silver_meta,
            }

        silver_candidates_found: list[dict[str, Any]] = []
        if "balls" in detectors:
            raw_silver_candidates = self.ball_detector.live_detections(processed, self.last_live_x)
            tuned_candidates: list[dict[str, Any]] = []
            primary_model_found = False
            primary_model_conf = 0.0
            primary_model_meta: dict[str, Any] = {"available": False}

            for idx, raw_candidate in enumerate(raw_silver_candidates):
                candidate = dict(raw_candidate)
                heur_conf = float(candidate.get("confidence", 0.0))
                model_conf = 0.0
                model_meta: dict[str, Any] = {"available": False}
                if self.silver_ball.enabled:
                    model_found, model_conf, model_meta = self.silver_ball.predict(
                        processed,
                        candidate.get("bbox") if isinstance(candidate.get("bbox"), Mapping) else None,
                    )
                    if bool(model_meta.get("just_initialized", False)):
                        ignore_benchmark_sample = True
                    if bool(model_meta.get("available", False)):
                        combined = _clamp01(0.55 * heur_conf + 0.45 * float(model_conf))
                        candidate["confidence"] = float(combined)
                        candidate["found"] = bool(combined >= max(self.ball_detector.silver_conf_threshold, 0.48))
                        if bool(candidate.get("found", False)):
                            candidate["origin"] = "hybrid_model"
                    if idx == 0:
                        primary_model_found = bool(model_found)
                        primary_model_conf = float(model_conf)
                        primary_model_meta = dict(model_meta)
                candidate["model_confidence"] = float(model_conf)
                tuned_candidates.append(candidate)

            silver_candidates_found = [
                candidate
                for candidate in tuned_candidates
                if bool(candidate.get("found", False)) and isinstance(candidate.get("bbox"), Mapping)
            ]
            silver_candidates_found.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
            balls = len(silver_candidates_found)
            self.last_live_x = int(silver_candidates_found[0].get("x", 0)) if silver_candidates_found else None
            metadata["silver_ball_model"] = {
                "found": bool(primary_model_found),
                "confidence": float(primary_model_conf),
                **primary_model_meta,
            }

        black_overlap_suppressed = False

        black_out: dict[str, Any] = self.ball_detector._empty_result()
        if "victims" in detectors:
            victim_mask, _ = self.ball_detector.build_dead_mask(processed, self.last_dead_x)
            black_out = self.ball_detector.dead_detection(processed, self.last_dead_x)
            if bool(black_out.get("found", False)):
                self.last_dead_x = int(black_out["x"])
            else:
                self.last_dead_x = None
            metadata["dead_ball_circle"] = black_out.get("circle")
            metadata["black_ball_found"] = bool(black_out.get("found", False))
            metadata["black_ball_confidence"] = float(black_out.get("confidence", 0.0))
            metadata["black_ball_bbox"] = black_out.get("bbox")
            metadata["black_ball_origin"] = str(black_out.get("origin", "none"))
            metadata["dead_ball_x"] = int(black_out["x"]) if black_out.get("x") is not None else None
        else:
            metadata["dead_ball_circle"] = None
            metadata["dead_ball_x"] = None

        if silver_candidates_found and bool(black_out.get("found", False)):
            overlap_px = max(0, int(self.ball_detector.silver_black_overlap_px))
            black_x = int(black_out["x"]) if black_out.get("x") is not None else None
            black_bbox = black_out.get("bbox") if isinstance(black_out.get("bbox"), Mapping) else None
            filtered_candidates: list[dict[str, Any]] = []
            for candidate in silver_candidates_found:
                silver_x = candidate.get("x")
                silver_bbox = candidate.get("bbox") if isinstance(candidate.get("bbox"), Mapping) else None
                x_overlap = (
                    black_x is not None
                    and silver_x is not None
                    and abs(int(silver_x) - int(black_x)) <= overlap_px
                )
                iou_overlap = _bbox_iou(silver_bbox, black_bbox) >= 0.25
                if x_overlap or iou_overlap:
                    black_overlap_suppressed = True
                    continue
                filtered_candidates.append(candidate)
            silver_candidates_found = filtered_candidates
            balls = len(silver_candidates_found)
            metadata["silver_ball_black_overlap_suppressed"] = bool(black_overlap_suppressed)

        if "balls" in detectors:
            metadata["silver_ball_candidates"] = [
                {
                    "x": int(candidate.get("x", 0)),
                    "confidence": float(_clamp01(float(candidate.get("confidence", 0.0)))),
                    "bbox": dict(candidate.get("bbox", {})) if isinstance(candidate.get("bbox"), Mapping) else None,
                    "circle": dict(candidate.get("circle", {})) if isinstance(candidate.get("circle"), Mapping) else None,
                    "origin": str(candidate.get("origin", "none")),
                }
                for candidate in silver_candidates_found
            ]
            metadata["silver_ball_count"] = int(len(silver_candidates_found))
            if silver_candidates_found:
                primary_silver = silver_candidates_found[0]
                metadata["silver_ball_x"] = int(primary_silver["x"]) if primary_silver.get("x") is not None else None
                metadata["silver_ball_circle"] = primary_silver.get("circle")
                metadata["silver_ball_found"] = True
                metadata["silver_ball_confidence"] = float(primary_silver.get("confidence", 0.0))
                metadata["silver_ball_bbox"] = primary_silver.get("bbox")
                metadata["silver_ball_origin"] = str(primary_silver.get("origin", "none"))
            else:
                metadata["silver_ball_x"] = None
                metadata["silver_ball_circle"] = None
                metadata["silver_ball_found"] = False
                metadata["silver_ball_confidence"] = 0.0
                metadata["silver_ball_bbox"] = None
                metadata["silver_ball_origin"] = (
                    "suppressed_by_black_overlap" if black_overlap_suppressed else "none"
                )
        else:
            metadata["silver_ball_x"] = None
            metadata["silver_ball_circle"] = None

        if "victims" in detectors:
            if self.dead.enabled:
                should_infer = (self.frame_counter % self.dead_every_n == 0) or (not self.dead_cache_ready)
                if should_infer:
                    found, confidence, victim_meta = self.dead.predict(processed)
                    victim_meta["cached"] = False
                    self.dead_cache = (found, confidence, victim_meta)
                    self.dead_cache_ready = True
                else:
                    found, confidence, victim_meta = self.dead_cache
                    victim_meta = dict(victim_meta)
                    victim_meta["cached"] = True
                    victim_meta["just_initialized"] = False

                if bool(victim_meta.get("just_initialized", False)):
                    ignore_benchmark_sample = True

                merged_found = bool(found) or bool(black_out.get("found", False))
                merged_conf = max(float(confidence), float(black_out.get("confidence", 0.0)))
                if merged_found:
                    victims += 1

                if bool(victim_meta.get("available", False)):
                    metadata["dead_victim"] = {
                        "found": bool(merged_found),
                        "confidence": float(merged_conf),
                        "available": True,
                        "fallback": "black_heuristic" if black_out.get("found", False) and not found else "none",
                        "dead_ball_x": metadata.get("dead_ball_x"),
                        "heuristic_black_found": bool(black_out.get("found", False)),
                        **victim_meta,
                    }
                else:
                    metadata["dead_victim"] = {
                        "found": bool(black_out.get("found", False)),
                        "confidence": float(black_out.get("confidence", 0.0)),
                        "available": False,
                        "fallback": "ball_detector_dead",
                        "dead_ball_x": metadata.get("dead_ball_x"),
                        "error": victim_meta.get("error"),
                    }
            else:
                if bool(black_out.get("found", False)):
                    victims += 1
                metadata["dead_victim"] = {
                    "found": bool(black_out.get("found", False)),
                    "confidence": float(black_out.get("confidence", 0.0)),
                    "available": False,
                    "fallback": "ball_detector_dead",
                    "dead_ball_x": metadata.get("dead_ball_x"),
                }

        status_text = self._status_text(
            state_name=state_name,
            line=line,
            green=green,
            red=red,
            balls=balls,
            victims=victims,
            metadata=metadata,
        )
        metadata["status_text"] = status_text
        self._draw_overlay(
            overlay_frame,
            state_name=state_name,
            contour=contour,
            green_contours=green_contours,
            red_contours=red_contours,
            status_text=status_text,
            metadata=metadata,
        )

        latency_ms = (time.perf_counter() - started) * 1000.0
        metadata["benchmark"] = self._update_benchmark(
            state_name,
            latency_ms,
            ignore_sample=ignore_benchmark_sample,
        )
        event = VisionDetectionEvent(
            timestamp=time.time(),
            state=state_name,
            line=bool(line),
            balls=int(balls),
            green=bool(green),
            red=bool(red),
            victims=int(victims),
            latency_ms=float(latency_ms),
            metadata=metadata,
        )
        debug_views = self._build_debug_views(
            raw_frame=frame_bgr,
            processed_frame=processed,
            black_mask=black_mask,
            green_mask=green_mask,
            red_mask=red_mask,
            victim_mask=victim_mask,
            silver_line_mask=silver_line_mask,
        )
        event.metadata["debug_views_available"] = list(debug_views.keys())
        return PipelineOutput(event=event, processed_frame=overlay_frame, debug_views=debug_views)

    @staticmethod
    def _draw_overlay(
        frame: np.ndarray,
        *,
        state_name: str,
        contour: np.ndarray | None,
        green_contours: list[np.ndarray],
        red_contours: list[np.ndarray],
        status_text: str,
        metadata: Mapping[str, Any],
    ) -> None:
        if frame is None or frame.size == 0:
            return

        if contour is not None and len(contour) >= 3:
            cv2.drawContours(frame, [contour], -1, (255, 255, 0), 2)
            rect = cv2.minAreaRect(contour)
            box = np.intp(cv2.boxPoints(rect))
            cv2.polylines(frame, [box], True, (255, 0, 0), 2)

        for item in green_contours:
            if item is None or len(item) < 3:
                continue
            cv2.drawContours(frame, [item], -1, (0, 220, 0), 2)
            rect = cv2.minAreaRect(item)
            box = np.intp(cv2.boxPoints(rect))
            cv2.polylines(frame, [box], True, (0, 60, 220), 1)

        for item in red_contours:
            if item is None or len(item) < 3:
                continue
            x, y, w, h = cv2.boundingRect(item)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x - 2, y - 2), (x + w + 2, y + h + 2), (40, 220, 40), 1)

        silver_candidates = metadata.get("silver_ball_candidates")
        if isinstance(silver_candidates, list) and silver_candidates:
            for candidate in silver_candidates:
                if not isinstance(candidate, Mapping):
                    continue
                VisionPipelineManager._draw_labeled_bbox(
                    frame,
                    candidate.get("bbox"),
                    label="SILVER",
                    confidence=float(candidate.get("confidence", 0.0)),
                    color=(255, 229, 0),  # #00E5FF
                )
        else:
            VisionPipelineManager._draw_labeled_bbox(
                frame,
                metadata.get("silver_ball_bbox"),
                label="SILVER",
                confidence=float(metadata.get("silver_ball_confidence", 0.0)),
                color=(255, 229, 0),  # #00E5FF
            )
        VisionPipelineManager._draw_labeled_bbox(
            frame,
            metadata.get("black_ball_bbox"),
            label="BLACK",
            confidence=float(metadata.get("black_ball_confidence", 0.0)),
            color=(0, 179, 255),  # #FFB300
        )
        green_marker_bboxes = metadata.get("green_marker_bboxes")
        green_marker_count = int(metadata.get("green_marker_count", 0) or 0)
        if isinstance(green_marker_bboxes, list) and green_marker_bboxes:
            for index, marker_bbox in enumerate(green_marker_bboxes, start=1):
                label = (
                    f"GREEN {index}/{green_marker_count}"
                    if green_marker_count > 1
                    else "GREEN"
                )
                VisionPipelineManager._draw_labeled_bbox(
                    frame,
                    marker_bbox,
                    label=label,
                    confidence=float(metadata.get("green_marker_confidence", 0.0)),
                    color=(30, 240, 80),
                )
        else:
            VisionPipelineManager._draw_labeled_bbox(
                frame,
                metadata.get("green_marker_bbox"),
                label="GREEN",
                confidence=float(metadata.get("green_marker_confidence", 0.0)),
                color=(30, 240, 80),
            )
        VisionPipelineManager._draw_labeled_bbox(
            frame,
            metadata.get("green_corner_bbox"),
            label="GREEN CORNER",
            confidence=float(metadata.get("green_corner_confidence", 0.0)),
            color=(102, 255, 0),  # #00FF66
        )
        VisionPipelineManager._draw_labeled_bbox(
            frame,
            metadata.get("red_corner_bbox"),
            label="RED CORNER",
            confidence=float(metadata.get("red_corner_confidence", 0.0)),
            color=(48, 59, 255),  # #FF3B30
        )

        cv2.putText(
            frame,
            state_name,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (90, 220, 120),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            status_text,
            (10, frame.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_labeled_bbox(
        frame: np.ndarray,
        bbox: Any,
        *,
        label: str,
        confidence: float,
        color: tuple[int, int, int],
    ) -> None:
        if not isinstance(bbox, Mapping):
            return
        x = int(bbox.get("x", -1))
        y = int(bbox.get("y", -1))
        w = int(bbox.get("w", -1))
        h = int(bbox.get("h", -1))
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return
        x2 = min(frame.shape[1] - 1, x + w)
        y2 = min(frame.shape[0] - 1, y + h)
        if x2 <= x or y2 <= y:
            return

        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
        text = f"{label} {int(round(_clamp01(float(confidence)) * 100.0)):02d}%"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tx = max(2, x)
        ty = max(th + 4, y - 4)
        bx0, by0 = tx - 2, ty - th - 3
        bx1, by1 = tx + tw + 2, ty + baseline + 2
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), color, thickness=-1)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)

    @staticmethod
    def _status_text(
        *,
        state_name: str,
        line: bool,
        green: bool,
        red: bool,
        balls: int,
        victims: int,
        metadata: Mapping[str, Any],
    ) -> str:
        silver_ball_found = bool(metadata.get("silver_ball_found", False))
        black_ball_found = bool(metadata.get("black_ball_found", False))
        red_corner_found = bool(metadata.get("red_corner_found", False))
        green_corner_found = bool(metadata.get("green_corner_found", False))
        if state_name == RobotState.SEARCHING_LINE.value:
            return "... Searching line ..."
        if state_name == RobotState.FOLLOWING_LINE.value:
            silver_meta = metadata.get("silver_line")
            if isinstance(silver_meta, Mapping) and bool(silver_meta.get("found", False)):
                return "... Validating silver line ..."
            if green:
                return "... Green marker detected ..."
            return "... Following Line ..."
        if state_name == RobotState.VALIDATING_GAP.value:
            return "... Validating gap ..."
        if state_name == RobotState.CROSSING_GAP.value:
            return "... Crossing gap ..."
        if state_name == RobotState.VICTIM_FOUND.value:
            if victims > 0 or black_ball_found:
                return "... Picking up alive victim ..."
            if balls > 0 or silver_ball_found:
                return "... Tracking victim ..."
            return "... Searching victim ..."
        if state_name == RobotState.RESCUE_ZONE_DETECTED.value:
            if balls > 0 or silver_ball_found:
                return "... Tracking silver ball ..."
            if victims > 0 or black_ball_found:
                return "... Picking up alive victim ..."
            if red or red_corner_found:
                return "... Driving to red corner ..."
            if green_corner_found:
                return "... Driving to green corner ..."
            return "... Searching for red corner ..."
        if not line:
            return "... Searching line ..."
        return "... Following Line ..."

    def _build_debug_views(
        self,
        *,
        raw_frame: np.ndarray,
        processed_frame: np.ndarray,
        black_mask: np.ndarray | None,
        green_mask: np.ndarray | None,
        red_mask: np.ndarray | None,
        victim_mask: np.ndarray | None,
        silver_line_mask: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        if not self.debug_artifacts_enabled:
            return {}
        if processed_frame is None or processed_frame.size == 0:
            return {}

        views: dict[str, np.ndarray] = {}
        requested = set(self.debug_views)
        h, w = processed_frame.shape[:2]
        black_mask = black_mask if isinstance(black_mask, np.ndarray) and black_mask.size > 0 else _empty_mask(h, w)
        green_mask = green_mask if isinstance(green_mask, np.ndarray) and green_mask.size > 0 else _empty_mask(h, w)
        red_mask = red_mask if isinstance(red_mask, np.ndarray) and red_mask.size > 0 else _empty_mask(h, w)
        victim_mask = victim_mask if isinstance(victim_mask, np.ndarray) and victim_mask.size > 0 else _empty_mask(h, w)
        silver_line_mask = (
            silver_line_mask if isinstance(silver_line_mask, np.ndarray) and silver_line_mask.size > 0 else _empty_mask(h, w)
        )

        if "raw" in requested and raw_frame is not None and raw_frame.size > 0:
            views["raw"] = raw_frame.copy()
        if "processed" in requested:
            views["processed"] = processed_frame.copy()
        if "line_mask" in requested:
            views["line_mask"] = black_mask.copy()
        if "green_mask" in requested:
            views["green_mask"] = green_mask.copy()
        if "red_mask" in requested:
            views["red_mask"] = red_mask.copy()
        if "victim_mask" in requested:
            views["victim_mask"] = victim_mask.copy()
        if "silver_line_mask" in requested:
            views["silver_line_mask"] = silver_line_mask.copy()
        if "composite" in requested:
            composite = processed_frame.copy()
            composite = self._blend_mask(composite, black_mask, (220, 220, 220))
            composite = self._blend_mask(composite, green_mask, (40, 220, 90))
            composite = self._blend_mask(composite, red_mask, (70, 70, 240))
            composite = self._blend_mask(composite, victim_mask, (30, 180, 255))
            composite = self._blend_mask(composite, silver_line_mask, (255, 229, 0))
            views["composite"] = composite
        return views

    @staticmethod
    def _blend_mask(base: np.ndarray, mask: np.ndarray, tint: tuple[int, int, int]) -> np.ndarray:
        if base is None or base.size == 0 or mask is None or mask.size == 0:
            return base
        colored = np.zeros_like(base)
        colored[mask > 0] = tint
        return cv2.addWeighted(base, 0.82, colored, 0.55, 0.0)

    def benchmark_snapshot(self) -> dict[str, dict[str, float]]:
        snapshot: dict[str, dict[str, float]] = {}
        for state_name, values in self.latency_history.items():
            if not values:
                continue
            array = np.array(values, dtype=np.float32)
            avg_ms = float(np.mean(array))
            p95_ms = float(np.percentile(array, 95))
            reduction = 0.0
            if self.baseline_monolithic_ms > 0:
                reduction = (self.baseline_monolithic_ms - avg_ms) / self.baseline_monolithic_ms * 100.0
            snapshot[state_name] = {
                "samples": float(len(array)),
                "avg_ms": avg_ms,
                "p95_ms": p95_ms,
                "baseline_monolithic_ms": float(self.baseline_monolithic_ms),
                "load_reduction_pct": float(reduction),
            }
        return snapshot

    def _update_benchmark(self, state_name: str, latency_ms: float, *, ignore_sample: bool = False) -> dict[str, float]:
        expected_ms = 0.0
        if isinstance(self.expected_by_state, Mapping):
            expected_ms = float(self.expected_by_state.get(state_name, 0.0))

        if state_name not in self._warmup_seen or ignore_sample:
            self._warmup_seen.add(state_name)
            reason = "state_warmup" if not ignore_sample else "detector_init_warmup"
            return {
                "samples": 0.0,
                "avg_ms": float(latency_ms),
                "p95_ms": float(latency_ms),
                "baseline_monolithic_ms": float(self.baseline_monolithic_ms),
                "expected_state_ms": expected_ms,
                "load_reduction_pct": 0.0,
                "warmup_ignored": 1.0,
                "warmup_reason": reason,
            }

        history = self.latency_history[state_name]
        history.append(float(latency_ms))
        array = np.array(history, dtype=np.float32)
        avg_ms = float(np.mean(array))
        p95_ms = float(np.percentile(array, 95)) if len(array) > 1 else avg_ms
        reduction = 0.0
        if self.baseline_monolithic_ms > 0:
            reduction = (self.baseline_monolithic_ms - avg_ms) / self.baseline_monolithic_ms * 100.0
        return {
            "samples": float(len(array)),
            "avg_ms": avg_ms,
            "p95_ms": p95_ms,
            "baseline_monolithic_ms": float(self.baseline_monolithic_ms),
            "expected_state_ms": expected_ms,
            "load_reduction_pct": float(reduction),
        }

    def _profile_resize(self, profile: str, *, default_w: int, default_h: int) -> tuple[int, int]:
        payload = self.config.preprocessor.get(profile, {})
        if not isinstance(payload, Mapping):
            return default_w, default_h
        resize = payload.get("resize", {})
        if not isinstance(resize, Mapping):
            return default_w, default_h
        return int(resize.get("width", default_w)), int(resize.get("height", default_h))

    @staticmethod
    def _active_detectors(payload: Any) -> list[str]:
        if isinstance(payload, list):
            out = [str(item).strip().lower() for item in payload if str(item).strip()]
            if out:
                return out
        return ["line", "red"]

    @staticmethod
    def _default_profile(state_name: str) -> str:
        if state_name in (RobotState.RESCUE_ZONE_DETECTED.value, RobotState.VICTIM_FOUND.value):
            return "rescue"
        return "line"


_MANAGER_CACHE: dict[str, VisionPipelineManager] = {}


def _coerce_config(config: VisionConfig | Mapping[str, Any] | str | Path | None) -> VisionConfig:
    if config is None:
        return load_vision_config()
    if isinstance(config, VisionConfig):
        return config
    if isinstance(config, (str, Path)):
        return load_vision_config(config)
    if isinstance(config, Mapping):
        return VisionConfig(
            data=dict(config),
            config_path=DEFAULT_CONFIG_PATH,
            project_root=_REPO_ROOT,
            cache_key=f"inline:{id(config)}",
        )
    raise TypeError(f"unsupported vision config type: {type(config)!r}")


def get_pipeline_manager(config: VisionConfig | Mapping[str, Any] | str | Path | None = None) -> VisionPipelineManager:
    resolved = _coerce_config(config)
    manager = _MANAGER_CACHE.get(resolved.cache_key)
    if manager is None:
        manager = VisionPipelineManager(resolved)
        _MANAGER_CACHE[resolved.cache_key] = manager
    return manager


def switch_pipeline(
    state: RobotState | str,
    frame_bgr: np.ndarray,
    config: VisionConfig | Mapping[str, Any] | str | Path | None,
) -> VisionDetectionEvent:
    manager = get_pipeline_manager(config)
    return manager.run(state, frame_bgr).event
