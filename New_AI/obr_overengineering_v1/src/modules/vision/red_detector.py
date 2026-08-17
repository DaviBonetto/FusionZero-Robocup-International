from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


_FILE = Path(__file__).resolve()
_OBR_ROOT = _FILE.parents[3]
_REPO_ROOT = _FILE.parents[5]
_DEFAULT_CONFIG_PATH = _OBR_ROOT / "configs" / "hsv_red.json"

RED_LINE_FINISH = "red_line_finish"
RESCUE_ZONE_BORDER = "rescue_zone_border"
UNKNOWN_RED = "unknown_red"


@dataclass(slots=True)
class RedContourContext:
    label: str
    area: float
    perimeter: float
    aspect_ratio: float
    long_side: float
    short_side: float
    convexity_defects: int
    solidity: float
    extent: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "label": self.label,
            "area": float(self.area),
            "perimeter": float(self.perimeter),
            "aspect_ratio": float(self.aspect_ratio),
            "long_side": float(self.long_side),
            "short_side": float(self.short_side),
            "convexity_defects": int(self.convexity_defects),
            "solidity": float(self.solidity),
            "extent": float(self.extent),
        }


@dataclass(slots=True)
class RedDetectionResult:
    found: bool
    total_area: int
    primary_context: str
    mask: np.ndarray
    contours: list[np.ndarray]
    contexts: list[RedContourContext]

    def context_counts(self) -> dict[str, int]:
        counts = {
            RED_LINE_FINISH: 0,
            RESCUE_ZONE_BORDER: 0,
            UNKNOWN_RED: 0,
        }
        for context in self.contexts:
            counts[context.label] = counts.get(context.label, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": bool(self.found),
            "total_area": int(self.total_area),
            "primary_context": self.primary_context,
            "mask": self.mask,
            "contours": self.contours,
            "contexts": [ctx.to_dict() for ctx in self.contexts],
            "counts": self.context_counts(),
        }


class RedDetector:
    def __init__(self, config: Mapping[str, Any], *, config_path: Path | None = None) -> None:
        self.config = dict(config)
        self.config_path = config_path

        hsv_cfg = _required_mapping(self.config, "hsv_ranges")
        red_1_cfg = _required_mapping(hsv_cfg, "red_1")
        red_2_cfg = _required_mapping(hsv_cfg, "red_2")

        self.lower_1 = _as_triplet(red_1_cfg, "lower")
        self.upper_1 = _as_triplet(red_1_cfg, "upper")
        self.lower_2 = _as_triplet(red_2_cfg, "lower")
        self.upper_2 = _as_triplet(red_2_cfg, "upper")

        morph_cfg = _required_mapping(self.config, "morphology")
        self.kernel_shape = _kernel_shape_from_name(str(morph_cfg.get("kernel_shape", "rect")))
        self.kernel_size = int(_required_value(morph_cfg, "kernel_size"))
        self.open_iterations = int(_required_value(morph_cfg, "open_iterations"))
        self.close_iterations = int(_required_value(morph_cfg, "close_iterations"))

        filters_cfg = _required_mapping(self.config, "filters")
        self.min_contour_area = float(_required_value(filters_cfg, "min_contour_area"))
        self.min_contour_perimeter = float(_required_value(filters_cfg, "min_contour_perimeter"))
        self.min_total_area = float(_required_value(filters_cfg, "min_total_area"))

        defects_cfg = _required_mapping(self.config, "convexity_defects")
        self.defects_epsilon_ratio = float(_required_value(defects_cfg, "epsilon_ratio"))
        self.defects_min_depth = float(_required_value(defects_cfg, "min_depth"))

        classifier_cfg = _required_mapping(self.config, "classifier")
        line_cfg = _required_mapping(classifier_cfg, RED_LINE_FINISH)
        rescue_cfg = _required_mapping(classifier_cfg, RESCUE_ZONE_BORDER)

        self.line_min_aspect_ratio = float(_required_value(line_cfg, "min_aspect_ratio"))
        self.line_min_long_side = float(_required_value(line_cfg, "min_long_side"))
        self.line_max_defects = int(_required_value(line_cfg, "max_defects"))
        self.line_min_solidity = float(_required_value(line_cfg, "min_solidity"))
        self.line_min_extent = float(_required_value(line_cfg, "min_extent"))

        self.rescue_min_defects = int(_required_value(rescue_cfg, "min_defects"))
        self.rescue_max_aspect_ratio = float(_required_value(rescue_cfg, "max_aspect_ratio"))
        self.rescue_min_long_side = float(_required_value(rescue_cfg, "min_long_side"))
        self.rescue_shape_only_max_defects = int(_required_value(rescue_cfg, "shape_only_max_defects"))
        self.rescue_max_solidity = float(_required_value(rescue_cfg, "max_solidity"))
        self.rescue_max_extent = float(_required_value(rescue_cfg, "max_extent"))

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> "RedDetector":
        resolved = _resolve_config_path(config_path)
        with resolved.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if not isinstance(payload, Mapping):
            raise ValueError(f"invalid red detector config: {resolved}")
        return cls(payload, config_path=resolved)

    def build_red_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_1, self.upper_1)
        mask2 = cv2.inRange(hsv, self.lower_2, self.upper_2)
        mask = cv2.bitwise_or(mask1, mask2)
        return self._clean_mask(mask)

    def detect(self, frame_bgr: np.ndarray) -> RedDetectionResult:
        if frame_bgr is None or frame_bgr.size == 0:
            return RedDetectionResult(
                found=False,
                total_area=0,
                primary_context=UNKNOWN_RED,
                mask=np.zeros((0, 0), dtype=np.uint8),
                contours=[],
                contexts=[],
            )

        mask = self.build_red_mask(frame_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours: list[np.ndarray] = []
        contexts: list[RedContourContext] = []
        total_area = 0.0

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_contour_area:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter < self.min_contour_perimeter:
                continue

            context = self.identify_red_context(contour)
            valid_contours.append(contour)
            contexts.append(context)
            total_area += area

        primary_context = UNKNOWN_RED
        if contexts:
            primary_context = max(contexts, key=lambda item: item.area).label

        found = bool(valid_contours) and total_area >= self.min_total_area
        return RedDetectionResult(
            found=found,
            total_area=int(total_area),
            primary_context=primary_context,
            mask=mask,
            contours=valid_contours,
            contexts=contexts,
        )

    def identify_red_context(self, contour: np.ndarray) -> RedContourContext:
        metrics = self._contour_metrics(contour)

        line_match = (
            metrics["aspect_ratio"] >= self.line_min_aspect_ratio
            and metrics["long_side"] >= self.line_min_long_side
            and metrics["convexity_defects"] <= self.line_max_defects
            and metrics["solidity"] >= self.line_min_solidity
            and metrics["extent"] >= self.line_min_extent
        )

        if line_match:
            label = RED_LINE_FINISH
        else:
            rescue_shape = (
                metrics["aspect_ratio"] <= self.rescue_max_aspect_ratio
                and metrics["long_side"] >= self.rescue_min_long_side
            )
            rescue_by_defects = (
                rescue_shape and metrics["convexity_defects"] >= self.rescue_min_defects
            )
            rescue_by_fallback = (
                rescue_shape
                and (
                    metrics["solidity"] <= self.rescue_max_solidity
                    or metrics["extent"] <= self.rescue_max_extent
                )
            )
            rescue_by_shape_only = (
                rescue_shape and metrics["convexity_defects"] <= self.rescue_shape_only_max_defects
            )
            if rescue_by_defects or rescue_by_fallback or rescue_by_shape_only:
                label = RESCUE_ZONE_BORDER
            else:
                label = UNKNOWN_RED

        return RedContourContext(
            label=label,
            area=float(metrics["area"]),
            perimeter=float(metrics["perimeter"]),
            aspect_ratio=float(metrics["aspect_ratio"]),
            long_side=float(metrics["long_side"]),
            short_side=float(metrics["short_side"]),
            convexity_defects=int(metrics["convexity_defects"]),
            solidity=float(metrics["solidity"]),
            extent=float(metrics["extent"]),
        )

    def detect_dict(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        return self.detect(frame_bgr).to_dict()

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel_size = max(1, int(self.kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(self.kernel_shape, (kernel_size, kernel_size))

        if self.open_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(self.open_iterations))
        if self.close_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(self.close_iterations))
        return mask

    def _contour_metrics(self, contour: np.ndarray) -> dict[str, float | int]:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))

        rect = cv2.minAreaRect(contour)
        width = float(rect[1][0])
        height = float(rect[1][1])
        long_side = max(width, height)
        short_side = min(width, height)
        aspect_ratio = long_side / short_side if short_side > 0 else 0.0

        x, y, w, h = cv2.boundingRect(contour)
        bound_area = float(w * h)
        extent = area / bound_area if bound_area > 0 else 0.0

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 0.0

        convexity_defects = self._count_convexity_defects(contour, perimeter)

        return {
            "area": area,
            "perimeter": perimeter,
            "aspect_ratio": aspect_ratio,
            "long_side": long_side,
            "short_side": short_side,
            "convexity_defects": convexity_defects,
            "solidity": solidity,
            "extent": extent,
        }

    def _count_convexity_defects(self, contour: np.ndarray, perimeter: float) -> int:
        epsilon = max(0.0, self.defects_epsilon_ratio) * max(0.0, perimeter)
        simplified = contour
        if epsilon > 0.0 and len(contour) >= 4:
            simplified = cv2.approxPolyDP(contour, epsilon, True)

        if simplified is None or len(simplified) < 4:
            return 0

        try:
            hull_indices = cv2.convexHull(simplified, returnPoints=False)
            if hull_indices is None or len(hull_indices) < 4:
                return 0

            defects = cv2.convexityDefects(simplified, hull_indices)
            if defects is None:
                return 0

            count = 0
            for idx in range(defects.shape[0]):
                depth = float(defects[idx, 0, 3]) / 256.0
                if depth >= self.defects_min_depth:
                    count += 1
            return count
        except cv2.error:
            return 0


def load_red_config(config_path: str | Path | None = None) -> dict[str, Any]:
    resolved = _resolve_config_path(config_path)
    with resolved.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid red detector config: {resolved}")
    return dict(payload)


def _resolve_config_path(config_path: str | Path | None) -> Path:
    if config_path is None:
        return _DEFAULT_CONFIG_PATH.resolve()

    path = Path(config_path)
    if path.is_absolute():
        return path

    repo_candidate = (_REPO_ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate

    obr_candidate = (_OBR_ROOT / path).resolve()
    return obr_candidate


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping key: {key}")
    return value


def _required_value(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"missing required config key: {key}")
    return payload[key]


def _as_triplet(payload: Mapping[str, Any], key: str) -> tuple[int, int, int]:
    value = _required_value(payload, key)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"expected list[3] for key: {key}")
    return (int(value[0]), int(value[1]), int(value[2]))


def _kernel_shape_from_name(name: str) -> int:
    normalized = name.strip().lower()
    mapping = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    return mapping.get(normalized, cv2.MORPH_RECT)


__all__ = [
    "RED_LINE_FINISH",
    "RESCUE_ZONE_BORDER",
    "UNKNOWN_RED",
    "RedContourContext",
    "RedDetectionResult",
    "RedDetector",
    "load_red_config",
]
