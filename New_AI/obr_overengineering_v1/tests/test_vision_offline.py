from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.event_bus import EventBus
from core.state_machine import RobotState
from modules.vision import pipelines as pipelines_module
from modules.vision.pipelines import ColorMarkerDetector, get_pipeline_manager
from modules.vision.vision_node import VisionNode
from tools.vision_edge_dataset import EdgeDatasetWriter
from tools.vision_replay import VisionReplayRunner


@pytest.fixture(autouse=True)
def clear_pipeline_cache() -> None:
    pipelines_module._MANAGER_CACHE.clear()
    yield
    pipelines_module._MANAGER_CACHE.clear()


def _vision_config(*, debug_enabled: bool = False) -> dict:
    return {
        "paths": {"model_root": "."},
        "runtime": {
            "history_size": 120,
            "corner_stability_window": 5,
            "corner_on_votes": 3,
            "corner_off_votes": 1,
            "debug_artifacts_enabled": debug_enabled,
            "debug_views": [
                "raw",
                "processed",
                "line_mask",
                "green_mask",
                "red_mask",
                "victim_mask",
                "silver_line_mask",
                "composite",
            ],
        },
        "offline_ops": {
            "replay": {
                "output_root": "artifacts/test_replay",
                "default_state": "FOLLOWING_LINE",
                "save_overlay_frames": True,
                "save_debug_views": True,
            },
            "edge_dataset": {
                "output_root": "dataset/edge_cases",
                "metadata_file": "metadata.jsonl",
                "save_debug_views": ["processed", "silver_line_mask", "composite"],
            },
        },
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
            "silver_line": {
                "enabled": True,
                "confidence_threshold": 0.72,
                "candidate_threshold": 0.58,
                "run_every_n_frames": 1,
                "heuristic_enabled": True,
                "heuristic_threshold": 0.58,
                "decision_policy": "model_or_heuristic",
                "stability_window": 3,
                "required_votes": 2,
                "specular_v_min": 178,
                "specular_s_max": 70,
                "min_area": 240,
                "min_width_ratio": 0.30,
                "min_aspect_ratio": 2.0,
                "top_clear_band_ratio": 0.14,
                "top_black_ratio_max": 0.035,
                "center_tolerance_ratio": 0.48,
                "kernel_size": 5,
                "mask_open_iterations": 1,
                "mask_close_iterations": 1,
            },
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


def _silver_line_frame(*, top_black: bool = False) -> np.ndarray:
    frame = np.full((480, 640, 3), 205, dtype=np.uint8)
    cv2.rectangle(frame, (0, 174), (639, 220), (236, 236, 236), thickness=-1)
    cv2.rectangle(frame, (42, 182), (598, 212), (246, 246, 246), thickness=-1)
    if top_black:
        cv2.rectangle(frame, (304, 144), (336, 320), (0, 0, 0), thickness=-1)
    else:
        cv2.rectangle(frame, (304, 280), (336, 479), (0, 0, 0), thickness=-1)
    return frame


def _green_marker_frame() -> np.ndarray:
    frame = np.full((480, 640, 3), 235, dtype=np.uint8)
    cv2.rectangle(frame, (300, 140), (340, 479), (0, 0, 0), thickness=-1)
    cv2.rectangle(frame, (160, 198), (460, 214), (0, 0, 0), thickness=-1)
    cv2.rectangle(frame, (170, 215), (260, 295), (0, 220, 0), thickness=-1)
    return frame


def _green_corner_frame() -> np.ndarray:
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.rectangle(frame, (0, 40), (210, 250), (0, 220, 0), thickness=-1)
    return frame


def test_green_half_turn_requires_one_valid_square_on_each_side() -> None:
    detector = ColorMarkerDetector(
        {
            "green_min_area": 180,
            "green_min_aspect": 0.35,
            "green_max_aspect": 3.0,
            "green_min_short_side": 8,
            "green_min_extent": 0.20,
            "green_min_solidity": 0.72,
        }
    )
    black_mask = np.zeros((200, 320), dtype=np.uint8)
    cv2.rectangle(black_mask, (0, 168), (319, 190), 255, thickness=-1)
    cv2.rectangle(black_mask, (150, 60), (170, 199), 255, thickness=-1)

    opposite_sides = np.full((200, 320, 3), 235, dtype=np.uint8)
    cv2.rectangle(opposite_sides, (38, 108), (102, 172), (0, 220, 0), thickness=-1)
    cv2.rectangle(opposite_sides, (218, 108), (282, 172), (0, 220, 0), thickness=-1)
    both = detector.detect_green_instruction(opposite_sides, black_mask)

    assert both["side"] == "BOTH"
    assert both["instruction"] == "VERDE MEIA VOLTA"
    assert both["marker_count"] == 2
    assert len(both["marker_bboxes"]) == 2
    assert both["pair_quality"]["valid"] is True
    assert both["pair_quality"]["line_pixels"] >= 110

    same_side = np.full((200, 320, 3), 235, dtype=np.uint8)
    cv2.rectangle(same_side, (18, 108), (70, 172), (0, 220, 0), thickness=-1)
    cv2.rectangle(same_side, (92, 108), (144, 172), (0, 220, 0), thickness=-1)
    one_side = detector.detect_green_instruction(same_side, black_mask)

    assert one_side["side"] == "LEFT"
    assert one_side["instruction"] != "VERDE MEIA VOLTA"
    assert one_side["marker_count"] == 1

    without_t = detector.detect_green_instruction(opposite_sides, np.zeros_like(black_mask))
    assert without_t["instruction"] != "VERDE MEIA VOLTA"
    assert without_t["pair_quality"]["valid"] is False


def test_single_green_classifies_before_and_after_against_transverse_line() -> None:
    detector = ColorMarkerDetector(
        {
            "green_min_area": 180,
            "green_min_aspect": 0.35,
            "green_max_aspect": 3.0,
            "green_min_short_side": 8,
            "green_min_extent": 0.20,
            "green_min_solidity": 0.72,
            "green_row_margin": 4,
        }
    )
    black_mask = np.zeros((200, 320), dtype=np.uint8)
    cv2.rectangle(black_mask, (0, 88), (319, 108), 255, thickness=-1)
    cv2.rectangle(black_mask, (148, 88), (172, 199), 255, thickness=-1)

    before = np.full((200, 320, 3), 235, dtype=np.uint8)
    cv2.rectangle(before, (220, 116), (276, 172), (0, 220, 0), thickness=-1)
    before_out = detector.detect_green_instruction(before, black_mask)

    after = np.full((200, 320, 3), 235, dtype=np.uint8)
    cv2.rectangle(after, (220, 24), (276, 80), (0, 220, 0), thickness=-1)
    after_out = detector.detect_green_instruction(after, black_mask)

    assert before_out["instruction"] == "VERDE ANTES"
    assert before_out["relation_delta_y"] > 0
    assert before_out["relation_confidence"] >= 0.90
    assert after_out["instruction"] == "VERDE DEPOIS"
    assert after_out["relation_delta_y"] < 0
    assert after_out["relation_confidence"] >= 0.90


def test_primary_green_without_adjacent_black_track_is_rejected() -> None:
    detector = ColorMarkerDetector(
        {
            "green_min_area": 180,
            "green_min_aspect": 0.35,
            "green_max_aspect": 3.0,
            "green_min_short_side": 8,
            "green_min_extent": 0.20,
            "green_min_solidity": 0.72,
        }
    )
    frame = np.full((200, 320, 3), 205, dtype=np.uint8)
    cv2.rectangle(frame, (220, 116), (276, 172), (0, 150, 40), thickness=-1)

    rejected = detector.detect_green_instruction(
        frame,
        np.zeros((200, 320), dtype=np.uint8),
    )

    assert rejected["found"] is False
    assert rejected["side"] == "NONE"
    assert rejected["instruction"] == "NO GREEN"
    assert rejected["marker_count"] == 0
    assert rejected["single_recovery_quality"]["valid"] is False


def test_low_saturation_cast_next_to_line_is_not_a_green_marker() -> None:
    detector = ColorMarkerDetector(
        {
            "green_min_area": 180,
            "green_single_recovery_s_min": 40,
            "green_region_min_saturation_p25": 80,
            "green_region_min_local_contrast": 8,
        }
    )
    background = tuple(
        int(value)
        for value in cv2.cvtColor(
            np.uint8([[[91, 68, 90]]]),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
    )
    weak_cast = tuple(
        int(value)
        for value in cv2.cvtColor(
            np.uint8([[[89, 75, 90]]]),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
    )
    frame = np.full((200, 320, 3), background, dtype=np.uint8)
    cv2.rectangle(frame, (210, 112), (276, 174), weak_cast, thickness=-1)
    black_mask = np.zeros((200, 320), dtype=np.uint8)
    cv2.rectangle(black_mask, (0, 88), (319, 108), 255, thickness=-1)
    cv2.rectangle(black_mask, (148, 88), (172, 199), 255, thickness=-1)

    rejected = detector.detect_green_instruction(frame, black_mask)

    assert rejected["found"] is False
    assert rejected["instruction"] == "NO GREEN"
    quality = rejected["single_recovery_quality"]["color_quality"]
    assert quality["valid"] is False
    assert quality["saturation_p25"] < 80


def test_unbalanced_colored_objects_do_not_form_half_turn_marker() -> None:
    detector = ColorMarkerDetector(
        {
            "green_min_area": 120,
            "green_pair_min_area_balance": 0.18,
            "green_pair_min_separation_ratio": 0.20,
        }
    )
    frame = np.full((200, 320, 3), 235, dtype=np.uint8)
    cv2.rectangle(frame, (42, 105), (60, 139), (0, 180, 40), thickness=-1)
    cv2.rectangle(frame, (205, 90), (286, 148), (0, 180, 40), thickness=-1)
    black_mask = np.zeros((200, 320), dtype=np.uint8)
    cv2.rectangle(black_mask, (0, 154), (319, 180), 255, thickness=-1)
    cv2.rectangle(black_mask, (150, 60), (170, 199), 255, thickness=-1)

    rejected = detector.detect_green_instruction(frame, black_mask)

    assert rejected["instruction"] != "VERDE MEIA VOLTA"
    assert rejected["pair_quality"]["valid"] is False
    assert rejected["pair_quality"]["area_balance"] < 0.18


def test_green_candidate_is_rejected_when_line_geometry_is_invalid() -> None:
    config = _vision_config()
    config["preprocessor"]["line"]["roi"] = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    manager = get_pipeline_manager(config)
    frame = np.full((200, 320, 3), 235, dtype=np.uint8)
    cv2.rectangle(frame, (220, 24), (276, 80), (0, 220, 0), thickness=-1)
    # Nearby black pixels make the color detector's marker plausible, but this
    # floating horizontal band is not a valid ground-anchored line contour.
    cv2.rectangle(frame, (0, 88), (319, 108), (0, 0, 0), thickness=-1)

    result = manager.run(RobotState.FOLLOWING_LINE, frame).event

    assert result.line is False
    assert result.green is False
    assert result.metadata["green_instruction"] == "NO GREEN"
    assert result.metadata["green_rejected_without_line"] is True


def test_usb_dark_single_green_recovers_and_uses_inverted_track_relation() -> None:
    detector = ColorMarkerDetector(
        {
            "green_h_min": 35,
            "green_h_max": 90,
            "green_s_min": 75,
            "green_v_min": 55,
            "green_min_area": 180,
            "green_single_recovery_h_min": 55,
            "green_single_recovery_h_max": 110,
            "green_single_recovery_s_min": 40,
            "green_single_recovery_v_min": 30,
            "green_single_recovery_min_line_pixels": 80,
            "green_before_is_above": True,
        }
    )
    dark_led_green = tuple(
        int(value)
        for value in cv2.cvtColor(
            np.uint8([[[80, 55, 58]]]),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
    )
    frame = np.full((200, 320, 3), 235, dtype=np.uint8)
    cv2.rectangle(frame, (28, 24), (112, 82), dark_led_green, thickness=-1)
    black_mask = np.zeros((200, 320), dtype=np.uint8)
    cv2.rectangle(black_mask, (0, 94), (319, 116), 255, thickness=-1)
    cv2.rectangle(black_mask, (150, 94), (172, 199), 255, thickness=-1)

    recovered = detector.detect_green_instruction(frame, black_mask)

    assert recovered["found"] is True
    assert recovered["side"] == "LEFT"
    assert recovered["instruction"] == "VERDE ANTES"
    assert recovered["relation_delta_y"] < 0
    assert recovered["segmentation_source"] == "single_recovery"
    assert recovered["single_recovery_quality"]["valid"] is True

    without_track = detector.detect_green_instruction(frame, np.zeros_like(black_mask))
    assert without_track["found"] is False
    assert without_track["instruction"] == "NO GREEN"


def test_green_half_turn_and_line_adjacent_single_recover_led_shifted_green() -> None:
    detector = ColorMarkerDetector(
        {
            "green_h_max": 90,
            "green_pair_h_max": 105,
            "green_pair_max_area_ratio": 0.25,
        }
    )
    shifted_green = tuple(
        int(value)
        for value in cv2.cvtColor(
            np.uint8([[[100, 190, 105]]]),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
    )
    frame = np.full((200, 320, 3), 235, dtype=np.uint8)
    cv2.rectangle(frame, (24, 92), (132, 184), shifted_green, thickness=-1)
    cv2.rectangle(frame, (206, 96), (319, 188), shifted_green, thickness=-1)
    black_mask = np.zeros((200, 320), dtype=np.uint8)
    cv2.rectangle(black_mask, (0, 168), (319, 192), 255, thickness=-1)
    cv2.rectangle(black_mask, (150, 45), (170, 199), 255, thickness=-1)

    recovered = detector.detect_green_instruction(frame, black_mask)

    assert recovered["instruction"] == "VERDE MEIA VOLTA"
    assert recovered["side"] == "BOTH"
    assert recovered["marker_count"] == 2
    assert recovered["pair_quality"]["valid"] is True
    assert recovered["segmentation_source"] == "pair_recovery"

    single = np.full((200, 320, 3), 235, dtype=np.uint8)
    cv2.rectangle(single, (24, 92), (132, 184), shifted_green, thickness=-1)
    single_out = detector.detect_green_instruction(single, black_mask)
    assert single_out["found"] is True
    assert single_out["marker_count"] == 1
    assert single_out["segmentation_source"] == "single_recovery"
    assert single_out["single_recovery_quality"]["valid"] is True

    isolated_single = detector.detect_green_instruction(single, np.zeros_like(black_mask))
    assert isolated_single["found"] is False
    assert isolated_single["instruction"] == "NO GREEN"

    without_t = detector.detect_green_instruction(frame, np.zeros_like(black_mask))
    assert without_t["instruction"] != "VERDE MEIA VOLTA"
    assert without_t["pair_quality"]["valid"] is False


def test_green_pair_temporal_hold_stabilizes_two_short_dropouts() -> None:
    manager = get_pipeline_manager(_vision_config(debug_enabled=False))
    pair = {
        "found": True,
        "side": "BOTH",
        "instruction": "VERDE MEIA VOLTA",
        "marker_count": 2,
        "marker_bboxes": [
            {"x": 35, "y": 105, "w": 65, "h": 60},
            {"x": 220, "y": 106, "w": 64, "h": 60},
        ],
        "bbox": {"x": 35, "y": 105, "w": 249, "h": 61},
        "confidence": 0.96,
        "pair_quality": {"valid": True, "line_pixels": 500},
        "contours": [],
    }
    single = {
        "found": True,
        "side": "LEFT",
        "instruction": "VERDE ANTES",
        "marker_count": 1,
        "marker_bboxes": [{"x": 36, "y": 106, "w": 65, "h": 60}],
        "confidence": 0.9,
        "pair_quality": {"valid": False},
        "contours": [],
    }

    live = manager._stabilize_green_pair(pair)
    held_one = manager._stabilize_green_pair(single)
    held_two = manager._stabilize_green_pair(single)
    released = manager._stabilize_green_pair(single)

    assert live["pair_source"] == "live"
    assert held_one["instruction"] == "VERDE MEIA VOLTA"
    assert held_two["instruction"] == "VERDE MEIA VOLTA"
    assert held_two["pair_source"] == "temporal_hold"
    assert released["instruction"] == "VERDE ANTES"
    assert released["pair_stable"] is False


def test_silver_line_heuristic_detects_realistic_band() -> None:
    config = _vision_config(debug_enabled=False)
    manager = get_pipeline_manager(config)
    frame = _silver_line_frame(top_black=False)

    manager.run(RobotState.FOLLOWING_LINE, frame)
    event = manager.run(RobotState.FOLLOWING_LINE, frame).event

    silver = event.metadata["silver_line"]
    assert silver["found"] is True
    assert silver["heuristic"]["found"] is True
    assert silver["decision"]["votes"] >= 2
    assert isinstance(silver["bbox"], dict)


def test_silver_line_heuristic_rejects_top_black_conflict() -> None:
    config = _vision_config(debug_enabled=False)
    manager = get_pipeline_manager(config)
    frame = _silver_line_frame(top_black=True)

    manager.run(RobotState.FOLLOWING_LINE, frame)
    event = manager.run(RobotState.FOLLOWING_LINE, frame).event

    silver = event.metadata["silver_line"]
    assert silver["found"] is False
    assert silver["heuristic"]["found"] is False
    assert silver["heuristic"]["suppressed_reason"] == "top_black_ratio"


def test_black_ball_overlap_suppresses_silver_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _vision_config(debug_enabled=False)
    manager = get_pipeline_manager(config)
    frame = np.full((480, 640, 3), 160, dtype=np.uint8)

    monkeypatch.setattr(
        manager.ball_detector,
        "live_detections",
        lambda image, last_x: [
            {
                "found": True,
                "x": 210,
                "confidence": 0.92,
                "bbox": {"x": 170, "y": 180, "w": 80, "h": 80},
                "circle": {"x": 210, "y": 220, "r": 40},
                "origin": "heuristic_hough",
            }
        ],
    )
    monkeypatch.setattr(
        manager.ball_detector,
        "dead_detection",
        lambda image, last_x: {
            "found": True,
            "x": 214,
            "confidence": 0.88,
            "bbox": {"x": 174, "y": 182, "w": 80, "h": 80},
            "circle": {"x": 214, "y": 222, "r": 40},
            "origin": "heuristic_dead",
        },
    )

    event = manager.run(RobotState.RESCUE_ZONE_DETECTED, frame).event
    assert event.metadata["black_ball_found"] is True
    assert event.metadata["silver_ball_found"] is False
    assert event.metadata["silver_ball_count"] == 0
    assert event.metadata["silver_ball_origin"] == "suppressed_by_black_overlap"


def test_green_marker_and_green_corner_remain_separate() -> None:
    config = _vision_config(debug_enabled=False)
    manager = get_pipeline_manager(config)

    line_event = manager.run(RobotState.FOLLOWING_LINE, _green_marker_frame()).event
    assert line_event.green is True
    assert line_event.metadata["green_marker_found"] is True
    assert line_event.metadata["green_corner_found"] is False

    rescue_event = manager.run(RobotState.RESCUE_ZONE_DETECTED, _green_corner_frame()).event
    assert rescue_event.metadata["green_corner_found"] is True
    assert rescue_event.metadata["silver_ball_found"] is False


def test_vision_node_exposes_official_debug_bundle() -> None:
    config = _vision_config(debug_enabled=True)
    frame = _silver_line_frame(top_black=False)
    bus = EventBus(max_queue_size=128, drop_oldest=False)
    node = VisionNode(
        bus,
        config=config,
        publish_raw_frame=False,
        publish_processed_frame=False,
        debug_artifacts=True,
    )

    try:
        node.process_frame(frame, frame_id=7, timestamp=123.4)
        bundle = node.get_last_debug_bundle()
        assert bundle is not None
        assert bundle["frame_id"] == 7
        assert bundle["state"] == RobotState.SEARCHING_LINE.value
        assert {"processed", "line_mask", "green_mask", "red_mask", "victim_mask", "silver_line_mask", "composite"} <= set(
            bundle["views"].keys()
        )
        assert bundle["views"]["processed"].shape[:2] == bundle["views"]["line_mask"].shape[:2]
    finally:
        node.close()
        bus.stop()


def test_replay_runner_exports_overlays_and_edge_dataset(tmp_path: Path) -> None:
    config = _vision_config(debug_enabled=True)
    config_path = tmp_path / "vision_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(2):
        frame = _silver_line_frame(top_black=False)
        cv2.imwrite(str(frames_dir / f"frame_{idx:03d}.jpg"), frame)

    runner = VisionReplayRunner(
        config_path=config_path,
        output_root=tmp_path / "replay_output",
        debug_artifacts=True,
    )
    writer = EdgeDatasetWriter(tmp_path / "edge_dataset", debug_views=("processed", "silver_line_mask"))
    try:
        report = runner.run(
            source=frames_dir,
            source_type="frames_dir",
            state=RobotState.FOLLOWING_LINE.value,
            max_frames=2,
            save_overlay_frames=True,
            save_debug_views=True,
            dataset_writer=writer,
            dataset_label="silver_line_candidate",
        )
    finally:
        runner.close()

    assert report.frames_processed == 2
    assert report.overlay_dir.exists()
    assert report.debug_dir.exists()
    assert report.events_path.exists()
    assert report.dataset_count == 2

    lines = [json.loads(line) for line in report.events_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert all("silver_line" in item["metadata"] for item in lines)

    dataset_meta = tmp_path / "edge_dataset" / "metadata.jsonl"
    assert dataset_meta.exists()
    samples = [json.loads(line) for line in dataset_meta.read_text(encoding="utf-8").splitlines()]
    assert len(samples) == 2
    assert samples[0]["label"] == "silver_line_candidate"


def test_replay_runner_uses_offline_config_defaults(tmp_path: Path) -> None:
    config = _vision_config(debug_enabled=False)
    config["offline_ops"]["replay"]["output_root"] = str(tmp_path / "cfg_replay")
    config["offline_ops"]["replay"]["save_debug_views"] = True
    config_path = tmp_path / "vision_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frames_dir / "frame_000.jpg"), _silver_line_frame(top_black=False))

    runner = VisionReplayRunner(config_path=config_path)
    try:
        report = runner.run(
            source=frames_dir,
            source_type="frames_dir",
            max_frames=1,
        )
    finally:
        runner.close()

    assert runner.output_root == (tmp_path / "cfg_replay")
    assert report.frames_processed == 1
    assert report.debug_dir.exists()
