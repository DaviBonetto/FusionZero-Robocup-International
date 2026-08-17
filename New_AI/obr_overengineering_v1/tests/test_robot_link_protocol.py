from __future__ import annotations

from src.modules.control.robot_link_protocol import (
    GreenAssist,
    LineAssist,
    ObstacleAssist,
    decode_telemetry_line,
    encode_green_assist,
    encode_line_assist,
    encode_obstacle_assist,
)


def test_protocol_encodes_high_level_assists_as_short_text_lines() -> None:
    assert (
        encode_line_assist(
            LineAssist(
                found=True,
                offset_norm=0.25,
                angle_deg=108.0,
                confidence=0.84,
                gap_frames=0,
                source="vision",
            )
        )
        == "ASST LINE found=1 offset=0.250 angle=108.000 conf=0.840 gap=0 source=vision"
    )
    assert (
        encode_green_assist(
            GreenAssist(
                found=True,
                instruction="VERDE DEPOIS",
                side="LEFT",
                confidence=0.91,
                hold_ms=900,
                source="vision",
            )
        )
        == "ASST GREEN found=1 instruction=VERDE_DEPOIS side=LEFT conf=0.910 hold_ms=900 source=vision"
    )
    assert (
        encode_obstacle_assist(
            ObstacleAssist(
                state="AHEAD",
                confidence=0.78,
                hold_ms=1200,
                source="dashboard",
            )
        )
        == "ASST OBSTACLE state=AHEAD conf=0.780 hold_ms=1200 source=dashboard"
    )


def test_protocol_parses_telemetry_types_and_tokens() -> None:
    payload = decode_telemetry_line(
        "TLM mode=FOLLOW_LINE line_error=-0.125 pid=18.75 front=320 left=450 right=410 "
        "yaw=3.5 roll=-0.7 pitch=1.2 failsafe=0 obstacle=AHEAD green=VERDE_DEPOIS"
    )

    assert payload["mode"] == "FOLLOW_LINE"
    assert payload["line_error"] == -0.125
    assert payload["pid"] == 18.75
    assert payload["front"] == 320
    assert payload["left"] == 450
    assert payload["right"] == 410
    assert payload["yaw"] == 3.5
    assert payload["roll"] == -0.7
    assert payload["pitch"] == 1.2
    assert payload["failsafe"] is False
    assert payload["obstacle"] == "AHEAD"
    assert payload["green"] == "VERDE_DEPOIS"
