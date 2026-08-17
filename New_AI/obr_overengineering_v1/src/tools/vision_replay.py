from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import cv2
import numpy as np

if __package__ in (None, "", "tools"):
    import sys

    SRC_ROOT = Path(__file__).resolve().parents[1]
    PROJECT_ROOT = SRC_ROOT.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from core.event_bus import EventBus  # type: ignore
    from modules.vision.pipelines import VisionConfig, load_vision_config  # type: ignore
    from modules.vision.vision_node import VisionNode  # type: ignore
else:
    from ..core.event_bus import EventBus
    from ..modules.vision.pipelines import VisionConfig, load_vision_config
    from ..modules.vision.vision_node import VisionNode


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(slots=True)
class ReplayFrame:
    frame_id: int
    timestamp: float
    state: str
    frame: np.ndarray
    source_path: str


@dataclass(slots=True)
class ReplayRunReport:
    output_dir: Path
    events_path: Path
    overlay_dir: Path
    debug_dir: Path
    frames_processed: int
    dataset_count: int
    source_type: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _detect_source_type(path: Path) -> str:
    if path.is_dir():
        if (path / "events.jsonl").exists():
            return "session_dir"
        return "frames_dir"
    if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        return "video"
    return "frames_dir"


def _iter_frames_dir(path: Path, *, state: str) -> Iterator[ReplayFrame]:
    for index, image_path in enumerate(sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES), start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        yield ReplayFrame(
            frame_id=index,
            timestamp=time.time(),
            state=state,
            frame=frame,
            source_path=str(image_path),
        )


def _iter_session_dir(path: Path, *, state: str, frame_kind: str = "processed") -> Iterator[ReplayFrame]:
    events_path = path / "events.jsonl"
    current_state = state
    if not events_path.exists():
        return
    with events_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            item = json.loads(line)
            event_type = str(item.get("event_type", ""))
            fields = item.get("fields", {})
            if event_type == "StateSnapshotEvent" and isinstance(fields, Mapping):
                current_state = str(fields.get("state", current_state))
                continue
            if event_type != "FrameSample" or not isinstance(fields, Mapping):
                continue
            if str(fields.get("kind", "processed")).strip().lower() != frame_kind:
                continue
            rel_path = str(fields.get("path", "")).strip()
            if not rel_path:
                continue
            image_path = path / rel_path
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            yield ReplayFrame(
                frame_id=int(fields.get("frame_id", 0)),
                timestamp=float(fields.get("timestamp", time.time())),
                state=current_state,
                frame=frame,
                source_path=str(image_path),
            )


def _iter_video(path: Path, *, state: str) -> Iterator[ReplayFrame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                return
            frame_id += 1
            yield ReplayFrame(
                frame_id=frame_id,
                timestamp=time.time(),
                state=state,
                frame=frame,
                source_path=str(path),
            )
    finally:
        cap.release()


def _iter_source(
    source: Path,
    *,
    source_type: str,
    state: str,
    frame_kind: str = "processed",
) -> Iterable[ReplayFrame]:
    normalized = source_type.strip().lower()
    if normalized == "auto":
        normalized = _detect_source_type(source)
    if normalized == "frames_dir":
        return _iter_frames_dir(source, state=state)
    if normalized == "session_dir":
        return _iter_session_dir(source, state=state, frame_kind=frame_kind)
    if normalized == "video":
        return _iter_video(source, state=state)
    raise ValueError(f"unsupported source_type: {source_type}")


class VisionReplayRunner:
    def __init__(
        self,
        *,
        config_path: VisionConfig | str | Path | None = None,
        output_root: str | Path | None = None,
        debug_artifacts: bool = False,
    ) -> None:
        self._config = config_path if isinstance(config_path, VisionConfig) else load_vision_config(config_path)
        replay_cfg = self._offline_replay_cfg(self._config)
        configured_root = replay_cfg.get("output_root")
        if output_root is not None:
            resolved_root = Path(output_root)
        elif configured_root:
            resolved_root = Path(str(configured_root))
            if not resolved_root.is_absolute():
                resolved_root = (self._config.project_root / resolved_root).resolve()
        else:
            resolved_root = (self._config.project_root / "New_AI/obr_overengineering_v1/artifacts/vision_replay").resolve()
        self.output_root = resolved_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.default_state = str(replay_cfg.get("default_state", "FOLLOWING_LINE"))
        self.default_frame_kind = str(replay_cfg.get("frame_kind", "raw"))
        self.default_save_overlay_frames = bool(replay_cfg.get("save_overlay_frames", True))
        self.default_save_debug_views = bool(replay_cfg.get("save_debug_views", False))
        self.debug_artifacts = bool(debug_artifacts or self.default_save_debug_views)
        self._bus = EventBus(max_queue_size=256, drop_oldest=True)
        self._node = VisionNode(
            self._bus,
            config=self._config,
            publish_raw_frame=False,
            publish_processed_frame=False,
            debug_artifacts=self.debug_artifacts,
        )

    def close(self) -> None:
        self._node.close()
        self._bus.stop()

    def run(
        self,
        *,
        source: str | Path,
        source_type: str = "auto",
        state: str | None = None,
        frame_kind: str | None = None,
        max_frames: int | None = None,
        save_overlay_frames: bool | None = None,
        save_debug_views: bool | None = None,
        dataset_writer: Any | None = None,
        dataset_label: str | None = None,
    ) -> ReplayRunReport:
        source_path = Path(source)
        resolved_state = str(state or self.default_state)
        resolved_frame_kind = str(frame_kind or self.default_frame_kind)
        resolved_save_overlay = self.default_save_overlay_frames if save_overlay_frames is None else bool(save_overlay_frames)
        resolved_save_debug = self.default_save_debug_views if save_debug_views is None else bool(save_debug_views)
        run_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        run_dir = self.output_root / f"replay_{run_stamp}_{int(time.time() * 1000) % 1000:03d}"
        overlay_dir = run_dir / "overlays"
        debug_dir = run_dir / "debug"
        run_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
        events_path = run_dir / "detections.jsonl"

        frames_processed = 0
        dataset_count = 0
        iterable = _iter_source(
            source_path,
            source_type=source_type,
            state=resolved_state,
            frame_kind=resolved_frame_kind,
        )
        normalized_source_type = source_type.strip().lower()
        if normalized_source_type == "auto":
            normalized_source_type = _detect_source_type(source_path)

        with events_path.open("a", encoding="utf-8") as event_fp:
            for replay_frame in iterable:
                if max_frames is not None and frames_processed >= int(max_frames):
                    break
                self._node.set_state(replay_frame.state)
                event = self._node.process_frame(
                    replay_frame.frame,
                    frame_id=replay_frame.frame_id,
                    timestamp=replay_frame.timestamp,
                )
                processed = self._node.get_last_processed_frame(copy=False)
                debug_bundle = self._node.get_last_debug_bundle(copy=False)

                payload = {
                    "frame_id": int(replay_frame.frame_id),
                    "timestamp": float(event.timestamp),
                    "source_path": replay_frame.source_path,
                    "state": str(event.state),
                    "line": bool(event.line),
                    "balls": int(event.balls),
                    "green": bool(event.green),
                    "red": bool(event.red),
                    "victims": int(event.victims),
                    "latency_ms": float(event.latency_ms),
                    "metadata": _json_safe(dict(event.metadata or {})),
                }
                event_fp.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")

                if resolved_save_overlay and isinstance(processed, np.ndarray) and processed.size > 0:
                    overlay_path = overlay_dir / f"overlay_{replay_frame.frame_id:06d}.jpg"
                    self._write_image(overlay_path, processed)

                if resolved_save_debug and isinstance(debug_bundle, Mapping):
                    views = debug_bundle.get("views", {})
                    if isinstance(views, Mapping):
                        for name, frame in views.items():
                            if not isinstance(frame, np.ndarray) or frame.size == 0:
                                continue
                            view_dir = debug_dir / str(name)
                            view_dir.mkdir(parents=True, exist_ok=True)
                            self._write_image(view_dir / f"{replay_frame.frame_id:06d}.jpg", frame)

                if dataset_writer is not None and dataset_label:
                    dataset_writer.write_sample(
                        label=dataset_label,
                        frame=replay_frame.frame,
                        event=type(
                            "ReplayDetectionEvent",
                            (),
                            {
                                "timestamp": event.timestamp,
                                "frame_id": replay_frame.frame_id,
                                "state": event.state,
                                "line": event.line,
                                "balls": event.balls,
                                "green": event.green,
                                "red": event.red,
                                "victims": event.victims,
                                "latency_ms": event.latency_ms,
                                "metadata": dict(event.metadata or {}),
                            },
                        )(),
                        debug_bundle=debug_bundle,
                        source_path=replay_frame.source_path,
                    )
                    dataset_count += 1

                frames_processed += 1

        return ReplayRunReport(
            output_dir=run_dir,
            events_path=events_path,
            overlay_dir=overlay_dir,
            debug_dir=debug_dir,
            frames_processed=frames_processed,
            dataset_count=dataset_count,
            source_type=normalized_source_type,
        )

    @staticmethod
    def _write_image(path: Path, frame: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError(f"failed to encode image: {path}")
        path.write_bytes(encoded.tobytes())

    @staticmethod
    def _offline_replay_cfg(config: VisionConfig) -> Mapping[str, Any]:
        offline_ops = config.data.get("offline_ops", {})
        if not isinstance(offline_ops, Mapping):
            return {}
        replay_cfg = offline_ops.get("replay", {})
        return replay_cfg if isinstance(replay_cfg, Mapping) else {}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the official vision pipeline offline on recorded frames/video")
    parser.add_argument("--source", type=Path, required=True, help="Video file, session dir, or frames dir")
    parser.add_argument("--source-type", default="auto", choices=["auto", "video", "frames_dir", "session_dir"])
    parser.add_argument("--frame-kind", default=None, choices=["processed", "raw"])
    parser.add_argument("--state", default=None, help="State to use when the source has no FSM events")
    parser.add_argument("--config", dest="config_path", type=Path, default=None, help="Vision config path")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-debug-views", action="store_true", default=None)
    parser.add_argument("--disable-overlay-save", dest="save_overlay_frames", action="store_false", default=None)
    parser.add_argument("--debug-artifacts", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    runner = VisionReplayRunner(
        config_path=args.config_path,
        output_root=args.output_root,
        debug_artifacts=bool(args.debug_artifacts or args.save_debug_views),
    )
    try:
        report = runner.run(
            source=args.source,
            source_type=args.source_type,
            state=str(args.state) if args.state else None,
            frame_kind=args.frame_kind,
            max_frames=args.max_frames,
            save_overlay_frames=args.save_overlay_frames,
            save_debug_views=args.save_debug_views,
        )
    finally:
        runner.close()

    print(
        json.dumps(
            {
                "output_dir": str(report.output_dir),
                "events_path": str(report.events_path),
                "overlay_dir": str(report.overlay_dir),
                "debug_dir": str(report.debug_dir),
                "frames_processed": int(report.frames_processed),
                "dataset_count": int(report.dataset_count),
                "source_type": report.source_type,
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
