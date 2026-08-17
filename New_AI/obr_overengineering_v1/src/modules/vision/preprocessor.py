from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np


@dataclass(slots=True)
class PreprocessedFrame:
    frame: np.ndarray
    source_frame: np.ndarray
    metadata: dict[str, Any]


class VisionPreprocessor:
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._config = dict(config or {})
        self._default_profile = str(self._config.get("default_profile", "line"))

    def prepare(self, frame_bgr: np.ndarray, *, profile: str | None = None) -> PreprocessedFrame:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame_bgr is empty")

        profile_name = profile or self._default_profile
        profile_cfg = self._profile(profile_name)

        roi_cfg = profile_cfg.get("roi", {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0})
        roi_frame, roi_pixels = self._apply_roi(frame_bgr, roi_cfg)

        resize_cfg = profile_cfg.get("resize", {})
        resized = self._resize_if_needed(roi_frame, resize_cfg)

        source_frame = resized.copy()
        current = resized
        luma_cfg = profile_cfg.get("luma", {})
        luma_gain = 1.0
        input_luma_mean = float(self._luma_mean(current))
        if bool(luma_cfg.get("enabled", False)):
            current, luma_gain = self._adjust_luma(current, luma_cfg)

        clahe_cfg = profile_cfg.get("clahe", {})
        if bool(clahe_cfg.get("enabled", False)):
            current = self._apply_clahe(current, clahe_cfg)

        morphology_cfg = profile_cfg.get("morphology", {})
        if bool(morphology_cfg.get("enabled", False)):
            current = self._apply_morphology(current, morphology_cfg)

        metadata = {
            "profile": profile_name,
            "input_shape": {"width": int(frame_bgr.shape[1]), "height": int(frame_bgr.shape[0])},
            "roi_pixels": roi_pixels,
            "output_shape": {"width": int(current.shape[1]), "height": int(current.shape[0])},
            "luma_gain": float(luma_gain),
            "input_luma_mean": input_luma_mean,
            "output_luma_mean": float(self._luma_mean(current)),
        }
        return PreprocessedFrame(frame=current, source_frame=source_frame, metadata=metadata)

    def _profile(self, profile_name: str) -> Mapping[str, Any]:
        raw = self._config.get(profile_name)
        if isinstance(raw, Mapping):
            return raw
        if profile_name != self._default_profile:
            fallback = self._config.get(self._default_profile)
            if isinstance(fallback, Mapping):
                return fallback
        return {}

    @staticmethod
    def _apply_roi(frame_bgr: np.ndarray, roi_cfg: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, int]]:
        h, w = frame_bgr.shape[:2]
        x = float(roi_cfg.get("x", 0.0))
        y = float(roi_cfg.get("y", 0.0))
        roi_w = float(roi_cfg.get("w", 1.0))
        roi_h = float(roi_cfg.get("h", 1.0))

        x0 = int(max(0, min(w - 1, round(x * w))))
        y0 = int(max(0, min(h - 1, round(y * h))))
        x1 = int(max(x0 + 1, min(w, round((x + roi_w) * w))))
        y1 = int(max(y0 + 1, min(h, round((y + roi_h) * h))))

        cropped = frame_bgr[y0:y1, x0:x1]
        return cropped, {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

    @staticmethod
    def _resize_if_needed(frame_bgr: np.ndarray, resize_cfg: Mapping[str, Any]) -> np.ndarray:
        width = int(resize_cfg.get("width", frame_bgr.shape[1]))
        height = int(resize_cfg.get("height", frame_bgr.shape[0]))
        if width <= 0 or height <= 0:
            return frame_bgr
        if width == frame_bgr.shape[1] and height == frame_bgr.shape[0]:
            return frame_bgr
        return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _adjust_luma(frame_bgr: np.ndarray, luma_cfg: Mapping[str, Any]) -> tuple[np.ndarray, float]:
        target_mean = float(luma_cfg.get("target_mean", 128.0))
        min_gain = float(luma_cfg.get("min_gain", 0.75))
        max_gain = float(luma_cfg.get("max_gain", 1.35))

        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        luma = ycrcb[:, :, 0].astype(np.float32)
        current_mean = float(np.mean(luma))

        if current_mean <= 1e-6:
            gain = 1.0
        else:
            gain = target_mean / current_mean

        gain = float(np.clip(gain, min_gain, max_gain))
        ycrcb[:, :, 0] = np.clip(luma * gain, 0, 255).astype(np.uint8)
        adjusted = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return adjusted, gain

    @staticmethod
    def _apply_clahe(frame_bgr: np.ndarray, clahe_cfg: Mapping[str, Any]) -> np.ndarray:
        clip_limit = float(clahe_cfg.get("clip_limit", 2.0))
        grid = clahe_cfg.get("tile_grid_size", [8, 8])
        grid_x = max(2, int(grid[0] if isinstance(grid, list) and len(grid) > 0 else 8))
        grid_y = max(2, int(grid[1] if isinstance(grid, list) and len(grid) > 1 else 8))

        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_x, grid_y))
        l_channel = clahe.apply(l_channel)
        merged = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _apply_morphology(frame_bgr: np.ndarray, morphology_cfg: Mapping[str, Any]) -> np.ndarray:
        kernel_size = int(morphology_cfg.get("kernel_size", 3))
        kernel_size = max(1, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        open_iterations = int(morphology_cfg.get("open_iterations", 1))
        close_iterations = int(morphology_cfg.get("close_iterations", 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        value_channel = hsv[:, :, 2]
        if open_iterations > 0:
            value_channel = cv2.morphologyEx(
                value_channel,
                cv2.MORPH_OPEN,
                kernel,
                iterations=open_iterations,
            )
        if close_iterations > 0:
            value_channel = cv2.morphologyEx(
                value_channel,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=close_iterations,
            )
        hsv[:, :, 2] = value_channel
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def _luma_mean(frame_bgr: np.ndarray) -> float:
        if frame_bgr is None or frame_bgr.size == 0:
            return 0.0
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        return float(np.mean(ycrcb[:, :, 0]))
