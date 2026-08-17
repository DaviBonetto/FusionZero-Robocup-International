from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.state_machine import RobotState
from modules.vision import pipelines as pipelines_module
from modules.vision.pipelines import switch_pipeline
from modules.vision.preprocessor import VisionPreprocessor
from modules.vision.red_detector import RED_LINE_FINISH, RedDetector, load_red_config


@pytest.fixture(autouse=True)
def clear_pipeline_cache() -> None:
    pipelines_module._MANAGER_CACHE.clear()
    yield
    pipelines_module._MANAGER_CACHE.clear()


@pytest.fixture
def inline_vision_config() -> dict:
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
                "erode_ksize": 3,
                "dilate_ksize": 3,
                "min_ground_span_ratio": 0.38,
                "min_vertical_support_ratio": 0.55,
                "min_line_aspect": 1.25,
                "max_line_width_ratio": 0.52,
                "max_line_row_occupancy": 0.48,
                "max_bottom_row_occupancy": 0.38,
                "wide_corridor_enabled": True,
                "wide_corridor_min_height_ratio": 0.85,
                "wide_corridor_max_width_ratio": 0.90,
                "wide_corridor_min_side_gap_ratio": 0.08,
                "wide_corridor_min_side_support_ratio": 0.60,
                "wide_corridor_max_center_range_ratio": 0.45,
                "wide_corridor_max_row_occupancy": 0.82,
                "wide_corridor_max_bottom_row_occupancy": 0.82,
                "turn_corridor_enabled": True,
                "turn_corridor_min_area_ratio": 0.30,
                "turn_corridor_min_height_ratio": 0.50,
                "turn_corridor_max_width_ratio": 1.0,
                "turn_corridor_min_side_support_ratio": 0.45,
                "turn_corridor_min_center_range_ratio": 0.12,
                "turn_corridor_max_median_row_occupancy": 0.96,
                "turn_corridor_max_bottom_row_occupancy": 1.0,
                "turn_corridor_max_extent_ratio": 0.98,
                "right_angle_corridor_enabled": True,
                "right_angle_corridor_min_area_ratio": 0.30,
                "right_angle_corridor_min_height_ratio": 0.65,
                "right_angle_corridor_min_width_ratio": 0.75,
                "right_angle_corridor_min_center_range_ratio": 0.08,
                "right_angle_corridor_max_extent_ratio": 0.995,
                "turn_continuation_enabled": True,
                "turn_continuation_min_area_ratio": 0.75,
                "turn_continuation_min_height_ratio": 0.85,
                "turn_continuation_min_width_ratio": 0.82,
                "turn_continuation_min_extent_ratio": 0.90,
                "turn_continuation_max_extent_ratio": 0.999,
                "turn_continuation_max_gap_frames": 1,
                "turn_continuation_max_frames": 18,
                "turn_continuation_min_fragment_area_ratio": 0.005,
                "turn_continuation_min_fragment_height_ratio": 0.18,
                "turn_continuation_max_fragment_width_ratio": 0.50,
                "turn_continuation_min_fragment_aspect": 1.25,
                "turn_continuation_min_wide_area_ratio": 0.05,
                "turn_continuation_min_wide_width_ratio": 0.50,
                "turn_continuation_min_wide_aspect": 1.80,
                "turn_continuation_min_bend_range_ratio": 0.08,
            },
            "color": {
                "green_h_min": 35,
                "green_h_max": 90,
                "green_s_min": 70,
                "green_v_min": 50,
                "green_min_area": 100,
                "green_min_aspect": 0.2,
                "green_max_aspect": 4.0,
                "red_h1_min": 0,
                "red_h1_max": 12,
                "red_h2_min": 165,
                "red_h2_max": 179,
                "red_s_min": 90,
                "red_v_min": 80,
                "red_min_area": 120,
                "red_min_ratio": 2.0,
                "red_min_long_side": 18,
                "green_row_margin": 4,
                "color_erode_iter": 1,
                "color_dilate_iter": 2,
                "color_kernel": 3,
                "green_corner_conf_threshold": 0.55,
            },
            "ball": {
                "silver_conf_threshold": 0.45,
                "black_conf_threshold": 0.45,
                "silver_model_enabled": False,
            },
            "silver_line": {"enabled": False, "confidence_threshold": 0.95, "run_every_n_frames": 2},
            "dead_victim": {"enabled": False, "confidence_threshold": 0.55, "run_every_n_frames": 3},
        },
        "pipelines": {
            "SEARCHING_LINE": {"profile": "line", "detectors": ["line", "red"]},
            "FOLLOWING_LINE": {"profile": "line", "detectors": ["line", "green", "red", "silver_line"]},
            "VALIDATING_GAP": {"profile": "line", "detectors": ["line", "red"]},
            "CROSSING_GAP": {"profile": "line", "detectors": ["line"]},
            "VICTIM_FOUND": {"profile": "rescue", "detectors": ["balls", "victims", "red_zone", "green_corner"]},
            "RESCUE_ZONE_DETECTED": {
                "profile": "rescue",
                "detectors": ["balls", "victims", "red_zone", "green_corner"],
            },
            "default": {"profile": "line", "detectors": ["line", "red"]},
        },
    }


def _build_track_frame() -> np.ndarray:
    frame = np.full((480, 640, 3), 235, dtype=np.uint8)
    cv2.rectangle(frame, (300, 140), (340, 479), (0, 0, 0), thickness=-1)
    cv2.rectangle(frame, (420, 210), (620, 285), (0, 0, 255), thickness=-1)
    return frame


def _state_cases() -> list[tuple[RobotState, str, list[str]]]:
    return [
        (RobotState.SEARCHING_LINE, "line", ["line", "red"]),
        (RobotState.FOLLOWING_LINE, "line", ["line", "green", "red", "silver_line"]),
        (RobotState.VALIDATING_GAP, "line", ["line", "red"]),
        (RobotState.CROSSING_GAP, "line", ["line"]),
        (RobotState.VICTIM_FOUND, "rescue", ["balls", "victims", "red_zone", "green_corner"]),
        (RobotState.RESCUE_ZONE_DETECTED, "rescue", ["balls", "victims", "red_zone", "green_corner"]),
    ]


@pytest.mark.parametrize(("state", "expected_profile", "expected_detectors"), _state_cases())
def test_switch_pipeline_selects_profile_and_detectors_by_state(
    inline_vision_config: dict,
    state: RobotState,
    expected_profile: str,
    expected_detectors: list[str],
) -> None:
    frame = _build_track_frame()
    event = switch_pipeline(state, frame, inline_vision_config)

    assert event.state == state.value
    assert event.metadata["pipeline_profile"] == expected_profile
    assert event.metadata["active_detectors"] == expected_detectors
    assert event.latency_ms >= 0.0
    for key in (
        "silver_ball_found",
        "silver_ball_confidence",
        "silver_ball_bbox",
        "black_ball_found",
        "black_ball_confidence",
        "black_ball_bbox",
        "green_corner_found",
        "green_corner_confidence",
        "green_corner_bbox",
        "green_marker_found",
        "green_marker_confidence",
        "green_marker_bbox",
        "silver_ball_candidates",
        "silver_ball_count",
        "red_corner_found",
        "red_corner_confidence",
        "red_corner_bbox",
        "line_center_x",
        "line_center_y",
        "line_offset_norm",
        "line_confidence",
        "line_bbox",
        "line_mask_ratio",
    ):
        assert key in event.metadata

    if "line" in expected_detectors:
        assert event.line is True
    if "red" in expected_detectors or "red_zone" in expected_detectors:
        assert event.red is True
    else:
        assert event.red is False


def test_preprocessor_applies_clahe_and_luma_gain() -> None:
    preprocessor = VisionPreprocessor(
        {
            "default_profile": "line",
            "line": {
                "roi": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
                "resize": {"width": 160, "height": 120},
                "luma": {"enabled": True, "target_mean": 130.0, "min_gain": 0.8, "max_gain": 1.3},
                "clahe": {"enabled": True, "clip_limit": 2.5, "tile_grid_size": [8, 8]},
                "morphology": {"enabled": False},
            }
        }
    )

    gradient = np.tile(np.linspace(35, 55, 160, dtype=np.uint8), (120, 1))
    frame = np.dstack((gradient, gradient, gradient))
    cv2.rectangle(frame, (40, 30), (120, 95), (48, 48, 48), thickness=-1)

    output = preprocessor.prepare(frame, profile="line")
    assert output.metadata["luma_gain"] >= 1.0
    assert output.metadata["output_luma_mean"] >= output.metadata["input_luma_mean"]
    assert not np.array_equal(output.frame, frame)


def test_red_detector_detects_finish_line_context() -> None:
    config_path = Path("New_AI/obr_overengineering_v1/configs/hsv_red.json")
    detector = RedDetector(load_red_config(config_path))

    frame = np.full((220, 320, 3), 230, dtype=np.uint8)
    cv2.rectangle(frame, (30, 120), (295, 155), (0, 0, 255), thickness=-1)

    result = detector.detect(frame)
    assert result.found is True
    assert result.total_area >= int(detector.min_total_area)
    assert result.primary_context == RED_LINE_FINISH


def test_red_detector_returns_not_found_on_non_red_frame() -> None:
    config_path = Path("New_AI/obr_overengineering_v1/configs/hsv_red.json")
    detector = RedDetector(load_red_config(config_path))

    frame = np.full((220, 320, 3), 220, dtype=np.uint8)
    result = detector.detect(frame)
    assert result.found is False
    assert result.total_area == 0


def test_line_detector_does_not_confuse_black_ball_with_line(inline_vision_config: dict) -> None:
    frame = np.full((480, 640, 3), 235, dtype=np.uint8)
    cv2.circle(frame, (320, 300), 58, (0, 0, 0), thickness=-1)

    event = switch_pipeline(RobotState.FOLLOWING_LINE, frame, inline_vision_config)
    assert event.line is False
    assert int(event.metadata.get("line_gap_frames", 0)) >= 1


def test_line_detector_rejects_black_table_band_that_reaches_the_bottom(inline_vision_config: dict) -> None:
    frame = np.full((480, 640, 3), 235, dtype=np.uint8)
    cv2.rectangle(frame, (0, 380), (639, 479), (0, 0, 0), thickness=-1)

    event = switch_pipeline(RobotState.FOLLOWING_LINE, frame, inline_vision_config)

    assert event.line is False
    assert event.metadata["line_candidate_reason"] in {
        "insufficient_vertical_span",
        "line_too_wide",
        "rows_too_wide",
        "bottom_band_too_wide",
    }


def test_line_detector_rejects_black_shorts_like_shape(inline_vision_config: dict) -> None:
    frame = np.full((480, 640, 3), 235, dtype=np.uint8)
    cv2.rectangle(frame, (205, 250), (435, 360), (0, 0, 0), thickness=-1)
    cv2.rectangle(frame, (205, 350), (300, 479), (0, 0, 0), thickness=-1)
    cv2.rectangle(frame, (340, 350), (435, 479), (0, 0, 0), thickness=-1)

    event = switch_pipeline(RobotState.FOLLOWING_LINE, frame, inline_vision_config)

    assert event.line is False
    assert event.metadata["line_candidate_reason"] in {
        "not_line_like_aspect",
        "line_too_wide",
        "rows_too_wide",
        "bottom_band_too_wide",
    }


def test_line_detector_accepts_a_curved_ground_line(inline_vision_config: dict) -> None:
    frame = np.full((480, 640, 3), 235, dtype=np.uint8)
    points = np.array([(320, 479), (335, 400), (300, 320), (260, 240), (280, 150)], dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=False, color=(0, 0, 0), thickness=24)

    event = switch_pipeline(RobotState.FOLLOWING_LINE, frame, inline_vision_config)

    assert event.line is True
    assert event.metadata["line_candidate_reason"] == "accepted"


def test_line_detector_accepts_a_wide_ground_corridor_seen_by_close_pi_camera(inline_vision_config: dict) -> None:
    frame = np.full((480, 640, 3), 185, dtype=np.uint8)
    points = np.array([(185, 479), (205, 400), (235, 320), (285, 240), (310, 145)], dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=False, color=(0, 0, 0), thickness=150)

    event = switch_pipeline(RobotState.FOLLOWING_LINE, frame, inline_vision_config)

    assert event.line is True
    assert event.metadata["line_candidate_reason"] == "accepted"
    assert float(event.metadata["line_geometry"]["side_gap_support_ratio"]) >= 0.60


def test_line_detector_accepts_a_tight_turn_corridor_before_it_reaches_bottom(
    inline_vision_config: dict,
) -> None:
    # This is the post-resize geometry seen in the failed capture: the bend
    # enters from the top, sweeps across the frame, and ends before the bottom
    # edge. It must be accepted as a turn, but not as a flat black band.
    frame = np.full((200, 320, 3), 185, dtype=np.uint8)
    turn = np.array([(90, 0), (319, 0), (319, 94), (250, 108), (165, 120), (90, 116)], dtype=np.int32)
    cv2.fillPoly(frame, [turn], (0, 0, 0))

    from modules.vision.pipelines import LineDetector

    detector = LineDetector(320, 200, inline_vision_config["detectors"]["line"])
    contour, _ = detector.black_mask(frame)

    assert contour is not None
    assert detector.last_rejection_reason == "accepted"
    assert detector.last_geometry["turn_corridor"] == 1.0


def test_line_detector_accepts_a_ground_anchored_right_angle_corridor(
    inline_vision_config: dict,
) -> None:
    # At 90 degrees the close camera sees the black track as a diagonal band
    # touching both vertical edges and the bottom. Its center must be taken
    # from the lower rows, not from the full-width bounding box.
    frame = np.full((200, 320, 3), 185, dtype=np.uint8)
    turn = np.array([(0, 0), (218, 0), (319, 100), (319, 199), (73, 199)], dtype=np.int32)
    cv2.fillPoly(frame, [turn], (0, 0, 0))

    event = switch_pipeline(RobotState.FOLLOWING_LINE, frame, inline_vision_config)

    assert event.line is True
    assert event.metadata["line_candidate_reason"] == "accepted"
    assert event.metadata["line_geometry"]["turn_corridor"] == 1.0
    assert event.metadata["line_geometry"]["right_angle_corridor"] == 1.0
    assert float(event.metadata["line_geometry"]["path_bend_delta_norm"]) > 0.10
    assert float(event.metadata["line_offset_norm"]) > 0.15
    assert 0.0 <= float(event.metadata["line_geometry"]["dominant_row_y_ratio"]) <= 1.0


def test_line_detector_keeps_geometry_paired_with_selected_contour(
    inline_vision_config: dict,
) -> None:
    # Multiple line-like black components can coexist in a real frame.  The
    # selected contour and its path offset must always describe the same
    # object; otherwise the overlay looks correct while steering uses noise.
    frame = np.full((200, 320, 3), 185, dtype=np.uint8)
    cv2.rectangle(frame, (20, 0), (32, 199), (0, 0, 0), thickness=-1)
    cv2.rectangle(frame, (150, 0), (164, 199), (0, 0, 0), thickness=-1)
    cv2.rectangle(frame, (280, 0), (290, 199), (0, 0, 0), thickness=-1)

    from modules.vision.pipelines import LineDetector

    detector = LineDetector(320, 200, inline_vision_config["detectors"]["line"])
    contour, _ = detector.black_mask(frame)

    assert contour is not None
    x, _, width, _ = cv2.boundingRect(contour)
    selected_offset = (float(x) + (float(width) / 2.0) - 160.0) / 160.0
    assert abs(float(detector.last_geometry["path_center_offset_norm"]) - selected_offset) < 0.03
    assert detector.last_geometry["candidate_count"] == 3.0


def test_line_detector_bridges_a_full_corridor_only_after_a_valid_line(
    inline_vision_config: dict,
) -> None:
    from modules.vision.pipelines import LineDetector

    detector = LineDetector(320, 200, inline_vision_config["detectors"]["line"])
    valid = np.full((200, 320, 3), 185, dtype=np.uint8)
    cv2.rectangle(valid, (125, 0), (205, 199), (0, 0, 0), thickness=-1)
    contour, _ = detector.black_mask(valid)
    assert contour is not None
    detector.calculate_angle(contour)

    full = np.zeros((200, 320, 3), dtype=np.uint8)
    contour, _ = detector.black_mask(full)

    assert contour is not None
    assert detector.last_rejection_reason == "accepted"
    assert detector.last_geometry["turn_continuation"] == 1.0


def test_line_detector_bridges_usb_camera_top_fragment_only_after_a_valid_line(
    inline_vision_config: dict,
) -> None:
    from modules.vision.pipelines import LineDetector

    cfg = dict(inline_vision_config["detectors"]["line"])
    detector = LineDetector(320, 200, cfg)
    fragment = np.full((200, 320, 3), 185, dtype=np.uint8)
    cv2.rectangle(fragment, (0, 0), (58, 92), (0, 0, 0), thickness=-1)

    contour, _ = detector.black_mask(fragment)
    assert contour is None
    assert detector.last_rejection_reason == "not_ground_anchored"

    valid = np.full((200, 320, 3), 185, dtype=np.uint8)
    cv2.rectangle(valid, (42, 0), (82, 199), (0, 0, 0), thickness=-1)
    contour, _ = detector.black_mask(valid)
    assert contour is not None
    detector.calculate_angle(contour)

    contour, _ = detector.black_mask(fragment)
    assert contour is not None
    assert detector.last_geometry["turn_continuation"] == 1.0
    assert detector.last_geometry["turn_continuation_compact_top"] == 1.0


def test_line_detector_bridges_usb_camera_horizontal_corner_fragment(
    inline_vision_config: dict,
) -> None:
    from modules.vision.pipelines import LineDetector

    detector = LineDetector(320, 200, inline_vision_config["detectors"]["line"])
    valid = np.full((200, 320, 3), 185, dtype=np.uint8)
    cv2.rectangle(valid, (135, 0), (185, 199), (0, 0, 0), thickness=-1)
    contour, _ = detector.black_mask(valid)
    assert contour is not None
    detector.calculate_angle(contour)

    corner = np.full((200, 320, 3), 185, dtype=np.uint8)
    cv2.rectangle(corner, (35, 0), (285, 42), (0, 0, 0), thickness=-1)
    contour, _ = detector.black_mask(corner)

    assert contour is not None
    assert detector.last_geometry["turn_continuation_wide_top"] == 1.0


def test_line_detector_usb_camera_continuation_expires(inline_vision_config: dict) -> None:
    from modules.vision.pipelines import LineDetector

    cfg = dict(inline_vision_config["detectors"]["line"])
    cfg["turn_continuation_max_frames"] = 2
    detector = LineDetector(320, 200, cfg)
    valid = np.full((200, 320, 3), 185, dtype=np.uint8)
    cv2.rectangle(valid, (42, 0), (82, 199), (0, 0, 0), thickness=-1)
    contour, _ = detector.black_mask(valid)
    assert contour is not None
    detector.calculate_angle(contour)

    fragment = np.full((200, 320, 3), 185, dtype=np.uint8)
    cv2.rectangle(fragment, (0, 0), (58, 92), (0, 0, 0), thickness=-1)
    for _ in range(2):
        contour, _ = detector.black_mask(fragment)
        assert contour is not None
        detector.calculate_angle(contour)

    contour, _ = detector.black_mask(fragment)
    assert contour is None
    assert detector.last_rejection_reason == "not_ground_anchored"


def test_line_detector_does_not_bridge_a_round_blob_after_a_valid_line(
    inline_vision_config: dict,
) -> None:
    from modules.vision.pipelines import LineDetector

    detector = LineDetector(320, 200, inline_vision_config["detectors"]["line"])
    valid = np.full((200, 320, 3), 185, dtype=np.uint8)
    cv2.rectangle(valid, (135, 0), (185, 199), (0, 0, 0), thickness=-1)
    contour, _ = detector.black_mask(valid)
    assert contour is not None
    detector.calculate_angle(contour)

    blob = np.full((200, 320, 3), 185, dtype=np.uint8)
    cv2.circle(blob, (160, 42), 38, (0, 0, 0), thickness=-1)
    contour, _ = detector.black_mask(blob)

    assert contour is None


def test_line_detector_rejects_a_full_width_black_ground_band(inline_vision_config: dict) -> None:
    frame = np.full((480, 640, 3), 185, dtype=np.uint8)
    cv2.rectangle(frame, (0, 145), (639, 479), (0, 0, 0), thickness=-1)

    event = switch_pipeline(RobotState.FOLLOWING_LINE, frame, inline_vision_config)

    assert event.line is False
    assert event.metadata["line_candidate_reason"] in {
        "not_line_like_aspect",
        "line_too_wide",
        "rows_too_wide",
        "bottom_band_too_wide",
    }


def test_follow_line_metadata_exposes_normalized_line_control_signals(inline_vision_config: dict) -> None:
    frame = _build_track_frame()

    event = switch_pipeline(RobotState.FOLLOWING_LINE, frame, inline_vision_config)

    assert event.line is True
    assert isinstance(event.metadata.get("line_bbox"), dict)
    assert abs(float(event.metadata.get("line_offset_norm", 1.0))) < 0.2
    assert float(event.metadata.get("line_confidence", 0.0)) > 0.15
    assert float(event.metadata.get("line_mask_ratio", 0.0)) > 0.0


def test_green_detector_rejects_large_false_patch(inline_vision_config: dict) -> None:
    frame = np.full((480, 640, 3), 235, dtype=np.uint8)
    cv2.rectangle(frame, (300, 140), (340, 479), (0, 0, 0), thickness=-1)
    cv2.rectangle(frame, (80, 210), (560, 420), (0, 220, 0), thickness=-1)

    event = switch_pipeline(RobotState.FOLLOWING_LINE, frame, inline_vision_config)
    assert event.green is False
    assert event.metadata.get("green_instruction", "NO GREEN") == "NO GREEN"
    assert event.metadata.get("green_corner_found") is False
    assert float(event.metadata.get("green_corner_confidence", 0.0)) <= 0.01


def test_rescue_pipeline_detects_silver_ball(inline_vision_config: dict) -> None:
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.circle(frame, (330, 280), 52, (235, 235, 235), thickness=-1)
    cv2.circle(frame, (330, 280), 30, (210, 210, 210), thickness=2)

    event = switch_pipeline(RobotState.RESCUE_ZONE_DETECTED, frame, inline_vision_config)
    assert event.balls >= 1
    assert event.metadata.get("silver_ball_circle") is not None
    assert event.metadata.get("silver_ball_found") is True
    assert 0.0 <= float(event.metadata.get("silver_ball_confidence", 0.0)) <= 1.0
    assert isinstance(event.metadata.get("silver_ball_bbox"), dict)


def test_rescue_pipeline_detects_multiple_silver_balls(inline_vision_config: dict) -> None:
    frame = np.full((480, 640, 3), 185, dtype=np.uint8)
    cv2.circle(frame, (230, 280), 42, (240, 240, 240), thickness=-1)
    cv2.circle(frame, (230, 280), 24, (205, 205, 205), thickness=2)
    cv2.circle(frame, (420, 280), 46, (236, 236, 236), thickness=-1)
    cv2.circle(frame, (420, 280), 28, (198, 198, 198), thickness=2)

    event = switch_pipeline(RobotState.RESCUE_ZONE_DETECTED, frame, inline_vision_config)
    candidates = event.metadata.get("silver_ball_candidates")
    assert isinstance(candidates, list)
    assert len(candidates) >= 2
    assert int(event.metadata.get("silver_ball_count", 0)) >= 2
    assert event.balls >= 2


def test_green_corner_does_not_block_ball_detection(inline_vision_config: dict) -> None:
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.rectangle(frame, (0, 60), (220, 240), (0, 220, 0), thickness=-1)
    cv2.circle(frame, (430, 300), 50, (238, 238, 238), thickness=-1)
    cv2.circle(frame, (430, 300), 30, (200, 200, 200), thickness=2)

    event = switch_pipeline(RobotState.RESCUE_ZONE_DETECTED, frame, inline_vision_config)
    assert event.metadata.get("green_corner_found") is True
    assert event.metadata.get("silver_ball_found") is True
    assert event.balls >= 1
