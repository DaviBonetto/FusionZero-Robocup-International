from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


def _resolve_config_payload(config: Mapping[str, Any] | str | Path | None) -> tuple[Mapping[str, Any], Path | None]:
    if config is None:
        return {}, None
    if isinstance(config, Mapping):
        return config, None
    path = Path(config)
    if not path.exists():
        return {}, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    base_root = path.resolve().parents[1] if len(path.resolve().parents) > 1 else path.resolve().parent
    return (payload if isinstance(payload, Mapping) else {}), base_root


@dataclass(slots=True)
class EdgeDatasetRecord:
    image_path: Path
    metadata_path: Path
    debug_paths: dict[str, Path]


class EdgeDatasetWriter:
    def __init__(
        self,
        output_root: str | Path,
        *,
        metadata_file: str = "metadata.jsonl",
        debug_views: Sequence[str] = (),
        jpeg_quality: int = 92,
    ) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_root / str(metadata_file)
        self.debug_views = tuple(str(view).strip().lower() for view in debug_views if str(view).strip())
        self.jpeg_quality = max(35, min(100, int(jpeg_quality)))

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | str | Path | None,
        *,
        output_root: str | Path | None = None,
        metadata_file: str | None = None,
        debug_views: Sequence[str] | None = None,
        jpeg_quality: int = 92,
    ) -> "EdgeDatasetWriter":
        payload, base_root = _resolve_config_payload(config)
        offline_ops = payload.get("offline_ops", {}) if isinstance(payload, Mapping) else {}
        edge_cfg = offline_ops.get("edge_dataset", {}) if isinstance(offline_ops, Mapping) else {}
        if not isinstance(edge_cfg, Mapping):
            edge_cfg = {}
        resolved_root = output_root or edge_cfg.get("output_root") or "New_AI/obr_overengineering_v1/dataset/edge_cases"
        if base_root is not None:
            resolved_root_path = Path(str(resolved_root))
            if not resolved_root_path.is_absolute():
                resolved_root = (base_root / resolved_root_path).resolve()
        resolved_metadata = metadata_file or str(edge_cfg.get("metadata_file", "metadata.jsonl"))
        resolved_debug_views = debug_views if debug_views is not None else edge_cfg.get("debug_views", ())
        if isinstance(resolved_debug_views, (str, bytes)):
            resolved_debug_views = ()
        return cls(
            resolved_root,
            metadata_file=resolved_metadata,
            debug_views=resolved_debug_views if isinstance(resolved_debug_views, Sequence) else (),
            jpeg_quality=jpeg_quality,
        )

    def write_sample(
        self,
        *,
        label: str,
        frame: np.ndarray,
        event: Any,
        debug_bundle: Mapping[str, Any] | None = None,
        source_path: str = "",
        tags: Sequence[str] = (),
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> EdgeDatasetRecord:
        if frame is None or frame.size == 0:
            raise ValueError("frame is empty")

        event_metadata = event.metadata if isinstance(getattr(event, "metadata", None), Mapping) else {}
        frame_id = int(getattr(event, "frame_id", 0) or 0)
        timestamp = float(getattr(event, "timestamp", 0.0) or time.time())
        state = str(getattr(event, "state", "") or "")
        label_name = str(label).strip().lower() or "unlabeled"
        slug = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
        millis = int((timestamp - int(timestamp)) * 1000.0)
        filename = f"{slug}_{millis:03d}_{frame_id:06d}.jpg"

        image_dir = self.output_root / "images" / label_name
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / filename
        self._write_image(image_path, frame)

        debug_paths: dict[str, Path] = {}
        views = debug_bundle.get("views", {}) if isinstance(debug_bundle, Mapping) else {}
        if isinstance(views, Mapping):
            for view_name in self.debug_views:
                view = views.get(view_name)
                if not isinstance(view, np.ndarray) or view.size == 0:
                    continue
                view_dir = self.output_root / "debug" / view_name / label_name
                view_dir.mkdir(parents=True, exist_ok=True)
                view_path = view_dir / filename
                self._write_image(view_path, view)
                debug_paths[view_name] = view_path

        payload = {
            "label": label_name,
            "timestamp": float(timestamp),
            "frame_id": int(frame_id),
            "state": state,
            "source_path": str(source_path),
            "image_path": image_path.relative_to(self.output_root).as_posix(),
            "debug_paths": {name: path.relative_to(self.output_root).as_posix() for name, path in debug_paths.items()},
            "tags": [str(tag) for tag in tags],
            "event": {
                "line": bool(getattr(event, "line", False)),
                "balls": int(getattr(event, "balls", 0)),
                "green": bool(getattr(event, "green", False)),
                "red": bool(getattr(event, "red", False)),
                "victims": int(getattr(event, "victims", 0)),
                "latency_ms": float(getattr(event, "latency_ms", 0.0)),
            },
            "metadata": dict(event_metadata),
        }
        if extra_metadata:
            payload["extra_metadata"] = dict(extra_metadata)

        with self.metadata_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")

        return EdgeDatasetRecord(
            image_path=image_path,
            metadata_path=self.metadata_path,
            debug_paths=debug_paths,
        )

    def _write_image(self, path: Path, frame: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = frame
        if frame.ndim == 2:
            image = frame
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
        if not ok:
            raise RuntimeError(f"failed to encode image: {path}")
        path.write_bytes(encoded.tobytes())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persist replay/live frames into an edge-case dataset")
    parser.add_argument("--output-root", type=Path, required=True, help="Dataset output root")
    parser.add_argument("--label", required=True, help="Dataset label")
    parser.add_argument("--image", type=Path, required=True, help="Single image to append")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional JSON file with extra metadata")
    parser.add_argument("--debug-view", action="append", default=[], help="Debug views to persist if present")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"failed to read image: {args.image}")
    extra = None
    if args.metadata is not None and args.metadata.exists():
        extra = json.loads(args.metadata.read_text(encoding="utf-8"))

    writer = EdgeDatasetWriter(args.output_root, debug_views=args.debug_view)
    writer.write_sample(
        label=args.label,
        frame=frame,
        event=type("ReplayEvent", (), {"timestamp": time.time(), "frame_id": 0, "state": "", "metadata": {}})(),
        source_path=str(args.image),
        extra_metadata=extra,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
