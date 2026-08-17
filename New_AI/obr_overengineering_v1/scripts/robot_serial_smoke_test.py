from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modules.control.robot_link_protocol import (  # noqa: E402
    GreenAssist,
    LineAssist,
    ObstacleAssist,
    encode_green_assist,
    encode_line_assist,
    encode_obstacle_assist,
)


def _print_lines(prefix: str, lines: Iterable[str]) -> None:
    for line in lines:
        print(f"{prefix} {line}")


def _drain_lines(transport: Any, duration_s: float) -> list[str]:
    deadline = time.monotonic() + max(0.0, float(duration_s))
    lines: list[str] = []
    while time.monotonic() < deadline:
        raw = transport.readline()
        if not raw:
            continue
        text = raw.decode("ascii", errors="ignore").strip()
        if text:
            lines.append(text)
    return lines


def _send_and_expect(
    transport: Any,
    *,
    command: str,
    expected_prefixes: tuple[str, ...],
    timeout_s: float,
) -> str:
    print(f"[TX] {command}")
    transport.write(f"{command}\n".encode("ascii"))
    transport.flush()

    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() < deadline:
        raw = transport.readline()
        if not raw:
            continue

        line = raw.decode("ascii", errors="ignore").strip()
        if not line:
            continue

        print(f"[RX] {line}")
        if line.startswith("TLM ") or line.startswith("EVENT ") or line == "READY FUSIONZERO":
            continue
        if line.startswith("ERR "):
            raise RuntimeError(line)
        if any(line.startswith(prefix) for prefix in expected_prefixes):
            return line

    expected = " or ".join(expected_prefixes)
    raise TimeoutError(f"timeout waiting for {expected}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test for FusionZero Arduino serial protocol.")
    parser.add_argument("--port", required=True, help="Serial port, for example COM5.")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate.")
    parser.add_argument("--timeout", type=float, default=1.5, help="ACK/PONG timeout in seconds.")
    parser.add_argument("--warmup-seconds", type=float, default=0.8, help="Initial passive read window.")
    parser.add_argument("--post-telemetry-seconds", type=float, default=1.0, help="Telemetry capture after commands.")
    parser.add_argument("--forward-ms", type=int, default=0, help="Optional manual forward test. Use 0 to skip.")
    parser.add_argument("--line-offset", type=float, default=0.25)
    parser.add_argument("--line-angle", type=float, default=108.0)
    parser.add_argument("--line-conf", type=float, default=0.84)
    parser.add_argument("--line-gap", type=int, default=0)
    parser.add_argument("--skip-green", action="store_true", help="Skip green assist step.")
    parser.add_argument("--green-instruction", default="VERDE_DEPOIS")
    parser.add_argument("--green-side", default="LEFT")
    parser.add_argument("--green-conf", type=float, default=0.91)
    parser.add_argument("--green-hold-ms", type=int, default=900)
    parser.add_argument("--skip-obstacle", action="store_true", help="Skip obstacle assist step.")
    parser.add_argument("--obstacle-state", default="TEST")
    parser.add_argument("--obstacle-conf", type=float, default=1.0)
    parser.add_argument("--obstacle-hold-ms", type=int, default=1200)
    parser.add_argument("--no-stop", action="store_true", help="Do not send CMD STOP 0 at the end.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        import serial
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pyserial is required for robot_serial_smoke_test.py") from exc

    print(f"[INFO] Opening {args.port} at {args.baud} baud")
    with serial.Serial(port=args.port, baudrate=int(args.baud), timeout=0.10, write_timeout=0.25) as transport:
        time.sleep(0.2)
        transport.reset_input_buffer()
        transport.reset_output_buffer()

        warmup_lines = _drain_lines(transport, args.warmup_seconds)
        if warmup_lines:
            _print_lines("[BOOT]", warmup_lines)

        _send_and_expect(transport, command="PING", expected_prefixes=("PONG",), timeout_s=args.timeout)

        if int(args.forward_ms) > 0:
            _send_and_expect(
                transport,
                command=f"CMD FORWARD {int(args.forward_ms)}",
                expected_prefixes=(f"ACK FORWARD {int(args.forward_ms)}",),
                timeout_s=args.timeout,
            )

        line_command = encode_line_assist(
            LineAssist(
                found=True,
                offset_norm=float(args.line_offset),
                angle_deg=float(args.line_angle),
                confidence=float(args.line_conf),
                gap_frames=int(args.line_gap),
                source="smoke",
            )
        )
        _send_and_expect(transport, command=line_command, expected_prefixes=("ACK ASST LINE",), timeout_s=args.timeout)

        if not args.skip_green:
            green_command = encode_green_assist(
                GreenAssist(
                    found=True,
                    instruction=str(args.green_instruction),
                    side=str(args.green_side),
                    confidence=float(args.green_conf),
                    hold_ms=int(args.green_hold_ms),
                    source="smoke",
                )
            )
            _send_and_expect(
                transport,
                command=green_command,
                expected_prefixes=("ACK ASST GREEN",),
                timeout_s=args.timeout,
            )

        if not args.skip_obstacle:
            obstacle_command = encode_obstacle_assist(
                ObstacleAssist(
                    state=str(args.obstacle_state),
                    confidence=float(args.obstacle_conf),
                    hold_ms=int(args.obstacle_hold_ms),
                    source="smoke",
                )
            )
            _send_and_expect(
                transport,
                command=obstacle_command,
                expected_prefixes=("ACK ASST OBSTACLE",),
                timeout_s=args.timeout,
            )

        if not args.no_stop:
            _send_and_expect(transport, command="CMD STOP 0", expected_prefixes=("ACK STOP",), timeout_s=args.timeout)

        telemetry_lines = _drain_lines(transport, args.post_telemetry_seconds)
        if telemetry_lines:
            _print_lines("[POST]", telemetry_lines)

    print("[OK] Smoke test flow completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
