from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _normalize_token(value: object, *, fallback: str = "NONE") -> str:
    text = str(value or "").strip().upper().replace(" ", "_")
    return text or fallback


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


@dataclass(frozen=True, slots=True)
class LineAssist:
    found: bool
    offset_norm: float
    angle_deg: float
    confidence: float
    gap_frames: int
    source: str = "vision"


@dataclass(frozen=True, slots=True)
class GreenAssist:
    found: bool
    instruction: str
    side: str
    confidence: float
    hold_ms: int
    source: str = "vision"


@dataclass(frozen=True, slots=True)
class ObstacleAssist:
    state: str
    confidence: float
    hold_ms: int
    source: str = "vision"


def encode_line_assist(payload: LineAssist) -> str:
    return (
        "ASST LINE "
        f"found={1 if payload.found else 0} "
        f"offset={_format_float(max(-1.0, min(1.0, float(payload.offset_norm))))} "
        f"angle={_format_float(float(payload.angle_deg))} "
        f"conf={_format_float(_clamp01(float(payload.confidence)))} "
        f"gap={max(0, int(payload.gap_frames))} "
        f"source={_normalize_token(payload.source, fallback='vision').lower()}"
    )


def encode_green_assist(payload: GreenAssist) -> str:
    return (
        "ASST GREEN "
        f"found={1 if payload.found else 0} "
        f"instruction={_normalize_token(payload.instruction)} "
        f"side={_normalize_token(payload.side)} "
        f"conf={_format_float(_clamp01(float(payload.confidence)))} "
        f"hold_ms={max(0, int(payload.hold_ms))} "
        f"source={_normalize_token(payload.source, fallback='vision').lower()}"
    )


def encode_obstacle_assist(payload: ObstacleAssist) -> str:
    return (
        "ASST OBSTACLE "
        f"state={_normalize_token(payload.state, fallback='CLEAR')} "
        f"conf={_format_float(_clamp01(float(payload.confidence)))} "
        f"hold_ms={max(0, int(payload.hold_ms))} "
        f"source={_normalize_token(payload.source, fallback='vision').lower()}"
    )


def _coerce_scalar(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    try:
        if any(marker in lowered for marker in (".", "e")):
            return float(lowered)
        return int(lowered)
    except Exception:
        return value.strip()


def decode_telemetry_line(line: str) -> dict[str, Any]:
    text = str(line or "").strip()
    if not text:
        return {}
    if text.upper().startswith("TLM "):
        text = text[4:].strip()
    elif text.upper().startswith("STAT "):
        text = text[5:].strip()

    payload: dict[str, Any] = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, raw_value = token.split("=", 1)
        key = key.strip().lower()
        if not key:
            continue
        payload[key] = _coerce_scalar(raw_value)
    return payload
