from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PROJECT_ROOT.parents[1]
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "vision_config.json"


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _dict_copy(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True, slots=True)
class OpsProfile:
    name: str
    description: str
    camera: dict[str, int]
    tuning: dict[str, int | float]
    recording: dict[str, Any]
    notes: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "camera": dict(self.camera),
            "tuning": dict(self.tuning),
            "recording": dict(self.recording),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class OpsProfileCatalog:
    config_path: Path
    default_camera: dict[str, int]
    default_tuning: dict[str, int | float]
    default_recording: dict[str, Any]
    default_profile_name: str | None
    profiles: dict[str, OpsProfile]

    def get(self, name: str | None) -> OpsProfile | None:
        if not name:
            return None
        return self.profiles.get(str(name).strip().lower())

    def available_payload(self) -> list[dict[str, Any]]:
        return [profile.to_payload() for profile in self.profiles.values()]


def _default_tuning_from_config(data: Mapping[str, Any]) -> dict[str, int | float]:
    detectors = data.get("detectors", {})
    line = detectors.get("line", {}) if isinstance(detectors, Mapping) else {}
    color = detectors.get("color", {}) if isinstance(detectors, Mapping) else {}
    ball = detectors.get("ball", {}) if isinstance(detectors, Mapping) else {}
    silver_line = detectors.get("silver_line", {}) if isinstance(detectors, Mapping) else {}
    return {
        "line.black_v_max": _safe_int(line.get("black_v_max"), 70),
        "line.black_s_max": _safe_int(line.get("black_s_max"), 255),
        "line.min_area": _safe_int(line.get("min_black_area"), 50),
        "line.erode_iter": _safe_int(line.get("erode_iter"), 3),
        "line.dilate_iter": _safe_int(line.get("dilate_iter"), 4),
        "green.h_min": _safe_int(color.get("green_h_min"), 35),
        "green.h_max": _safe_int(color.get("green_h_max"), 90),
        "green.s_min": _safe_int(color.get("green_s_min"), 70),
        "green.v_min": _safe_int(color.get("green_v_min"), 50),
        "green.min_area": _safe_int(color.get("green_min_area"), 180),
        "red.h1_min": _safe_int(color.get("red_h1_min"), 0),
        "red.h1_max": _safe_int(color.get("red_h1_max"), 12),
        "red.h2_min": _safe_int(color.get("red_h2_min"), 165),
        "red.h2_max": _safe_int(color.get("red_h2_max"), 179),
        "red.s_min": _safe_int(color.get("red_s_min"), 120),
        "red.v_min": _safe_int(color.get("red_v_min"), 80),
        "red.min_area": _safe_int(color.get("red_min_area"), 300),
        "silver.conf": _safe_float(silver_line.get("confidence_threshold"), 0.95),
        "silver.blur": _safe_int(ball.get("silver_blur"), 7),
        "dead.black_v_max": _safe_int((ball.get("dead_black_threshold") or [60])[0], 60),
    }


def _default_camera_from_config(data: Mapping[str, Any]) -> dict[str, int]:
    dashboard_ops = data.get("dashboard_ops", {})
    camera = dashboard_ops.get("default_camera", {}) if isinstance(dashboard_ops, Mapping) else {}
    return {
        "index": _safe_int(camera.get("index"), 0),
        "width": _safe_int(camera.get("width"), 640),
        "height": _safe_int(camera.get("height"), 480),
        "fps": max(1, _safe_int(camera.get("fps"), 30)),
    }


def _default_recording_from_config(data: Mapping[str, Any]) -> dict[str, Any]:
    dashboard_ops = data.get("dashboard_ops", {})
    recording = dashboard_ops.get("default_recording", {}) if isinstance(dashboard_ops, Mapping) else {}
    return {
        "auto_start": bool(recording.get("auto_start", False)),
        "include_raw": bool(recording.get("include_raw", False)),
        "include_processed": bool(recording.get("include_processed", True)),
        "every_n_frames": max(1, _safe_int(recording.get("every_n_frames"), 20)),
        "jpeg_quality": max(35, min(95, _safe_int(recording.get("jpeg_quality"), 70))),
    }


def load_ops_profile_catalog(config_path: str | Path | None = None) -> OpsProfileCatalog:
    resolved = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
    if not resolved.is_absolute():
        project_candidate = (_PROJECT_ROOT / resolved).resolve()
        repo_candidate = (_REPO_ROOT / resolved).resolve()
        cwd_candidate = (Path.cwd() / resolved).resolve()
        if resolved.exists():
            resolved = resolved.resolve()
        elif project_candidate.exists():
            resolved = project_candidate
        elif repo_candidate.exists():
            resolved = repo_candidate
        elif cwd_candidate.exists():
            resolved = cwd_candidate
        else:
            resolved = project_candidate

    with resolved.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    default_camera = _default_camera_from_config(data)
    default_tuning = _default_tuning_from_config(data)
    default_recording = _default_recording_from_config(data)

    dashboard_ops = data.get("dashboard_ops", {})
    raw_profiles = dashboard_ops.get("profiles", {}) if isinstance(dashboard_ops, Mapping) else {}
    default_profile_name = None
    if isinstance(dashboard_ops, Mapping):
        candidate = str(dashboard_ops.get("default_profile", "")).strip().lower()
        default_profile_name = candidate or None

    profiles: dict[str, OpsProfile] = {}
    if isinstance(raw_profiles, Mapping):
        for raw_name, payload in raw_profiles.items():
            if not isinstance(payload, Mapping):
                continue
            name = str(raw_name).strip().lower()
            if not name:
                continue

            camera = dict(default_camera)
            camera.update({key: _safe_int(value, camera.get(str(key), 0)) for key, value in _dict_copy(payload.get("camera")).items()})

            tuning = dict(default_tuning)
            for key, value in _dict_copy(payload.get("tuning")).items():
                if key not in tuning:
                    continue
                if isinstance(tuning[key], float):
                    tuning[key] = _safe_float(value, float(tuning[key]))
                else:
                    tuning[key] = _safe_int(value, int(tuning[key]))

            recording = dict(default_recording)
            recording.update(_dict_copy(payload.get("recording")))
            recording["every_n_frames"] = max(1, _safe_int(recording.get("every_n_frames"), default_recording["every_n_frames"]))
            recording["jpeg_quality"] = max(35, min(95, _safe_int(recording.get("jpeg_quality"), default_recording["jpeg_quality"])))
            recording["include_raw"] = bool(recording.get("include_raw", default_recording["include_raw"]))
            recording["include_processed"] = bool(recording.get("include_processed", default_recording["include_processed"]))
            recording["auto_start"] = bool(recording.get("auto_start", default_recording["auto_start"]))

            notes_payload = payload.get("notes", [])
            notes = tuple(str(item).strip() for item in notes_payload if str(item).strip()) if isinstance(notes_payload, list) else ()
            profiles[name] = OpsProfile(
                name=name,
                description=str(payload.get("description", name.replace("_", " ").title())).strip(),
                camera=camera,
                tuning=tuning,
                recording=recording,
                notes=notes,
            )

    return OpsProfileCatalog(
        config_path=resolved,
        default_camera=default_camera,
        default_tuning=default_tuning,
        default_recording=default_recording,
        default_profile_name=default_profile_name,
        profiles=profiles,
    )
