from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SRC_ROOT = Path(__file__).resolve().parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from src.core.event_bus import EventBus
    from src.remote_dashboard import RemoteDashboardClient
    from src.ui_overengineering.dashboard import run_dashboard
else:
    from .core.event_bus import EventBus
    from .remote_dashboard import RemoteDashboardClient
    from .ui_overengineering.dashboard import run_dashboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FusionZero dashboard on the PC and consume AI data from the Raspberry Pi")
    parser.add_argument("--host", required=True, help="Raspberry Pi IP on the same Wi-Fi network")
    parser.add_argument("--port", type=int, default=8765, help="TCP port exposed by the Raspberry Pi relay")
    parser.add_argument("--reconnect-interval", type=float, default=1.5, help="Seconds between reconnect attempts")
    parser.add_argument("--heartbeat-interval", type=float, default=1.0, help="Seconds between heartbeat pings")
    parser.add_argument("--server-timeout", type=float, default=6.0, help="Seconds without server activity before reconnecting")
    parser.add_argument("--camera-index", type=int, default=0, help="Local camera index (kept idle when remote bus is active)")
    parser.add_argument("--camera-width", type=int, default=640, help="Preview width")
    parser.add_argument("--camera-height", type=int, default=480, help="Preview height")
    parser.add_argument("--camera-fps", type=int, default=30, help="Preview FPS")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bus = EventBus(max_queue_size=2048, drop_oldest=True)
    client = RemoteDashboardClient(
        bus,
        host=args.host,
        port=args.port,
        reconnect_interval=args.reconnect_interval,
        heartbeat_interval=args.heartbeat_interval,
        server_timeout=args.server_timeout,
    )
    client.start()
    try:
        return run_dashboard(
            event_bus=bus,
            camera_index=args.camera_index,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
            mock=False,
        )
    finally:
        client.stop()
        bus.stop()


if __name__ == "__main__":
    raise SystemExit(main())
