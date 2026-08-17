from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

try:
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyQt6 is required for this tool: {exc}") from exc


class SilverCaptureWindow(QMainWindow):
    def __init__(self, output_root: Path, camera_index: int, width: int, height: int, fps: int) -> None:
        super().__init__()
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "silver").mkdir(parents=True, exist_ok=True)
        (self.output_root / "no_silver").mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_root / "metadata.csv"
        self.jsonl_path = self.output_root / "metadata.jsonl"

        self._capture = None
        self._frame = None
        self._saved_count = 0

        self.setWindowTitle("Silver Ball Capture")
        self.resize(980, 720)
        self._build_ui(camera_index, width, height, fps)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)
        self._connect_camera()

    def _build_ui(self, camera_index: int, width: int, height: int, fps: int) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        controls = QWidget(root)
        form = QFormLayout(controls)
        form.setContentsMargins(0, 0, 0, 0)

        self.camera_index_spin = QSpinBox(controls)
        self.camera_index_spin.setRange(0, 12)
        self.camera_index_spin.setValue(int(camera_index))

        self.width_spin = QSpinBox(controls)
        self.width_spin.setRange(160, 3840)
        self.width_spin.setValue(int(width))

        self.height_spin = QSpinBox(controls)
        self.height_spin.setRange(120, 2160)
        self.height_spin.setValue(int(height))

        self.fps_spin = QSpinBox(controls)
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(int(fps))

        self.label_combo = QComboBox(controls)
        self.label_combo.addItems(["silver", "no_silver"])

        self.burst_spin = QSpinBox(controls)
        self.burst_spin.setRange(1, 40)
        self.burst_spin.setValue(8)

        form.addRow("Camera index", self.camera_index_spin)
        form.addRow("Width", self.width_spin)
        form.addRow("Height", self.height_spin)
        form.addRow("FPS", self.fps_spin)
        form.addRow("Label", self.label_combo)
        form.addRow("Burst size", self.burst_spin)
        layout.addWidget(controls)

        buttons = QHBoxLayout()
        self.reconnect_btn = QPushButton("Reconnect Camera", root)
        self.reconnect_btn.clicked.connect(self._connect_camera)
        self.capture_btn = QPushButton("Capture", root)
        self.capture_btn.clicked.connect(lambda: self._capture_images(1))
        self.burst_btn = QPushButton("Capture Burst", root)
        self.burst_btn.clicked.connect(lambda: self._capture_images(int(self.burst_spin.value())))
        buttons.addWidget(self.reconnect_btn)
        buttons.addWidget(self.capture_btn)
        buttons.addWidget(self.burst_btn)
        layout.addLayout(buttons)

        self.preview = QLabel("No camera frame", root)
        self.preview.setMinimumHeight(520)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#101216;border:1px solid #3A3F48;border-radius:8px;color:#E8EAF0;")
        layout.addWidget(self.preview, 1)

        self.status = QLabel("Ready", root)
        self.count = QLabel("Saved: 0", root)
        layout.addWidget(self.status)
        layout.addWidget(self.count)

    def _connect_camera(self) -> None:
        self._release_camera()
        idx = int(self.camera_index_spin.value())
        width = int(self.width_spin.value())
        height = int(self.height_spin.value())
        fps = int(self.fps_spin.value())

        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(idx)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if cap is None or not cap.isOpened():
            self.status.setText(f"camera {idx} not available")
            self._capture = None
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        self._capture = cap
        self.status.setText(f"camera {idx} connected ({width}x{height}@{fps})")

    def _release_camera(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
        self._capture = None

    def _tick(self) -> None:
        if self._capture is None:
            return
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return
        self._frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(image).scaled(
            self.preview.width(),
            self.preview.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pix)

    def _capture_images(self, count: int) -> None:
        if self._frame is None:
            self.status.setText("no frame to save")
            return
        label = str(self.label_combo.currentText()).strip().lower()
        if label not in {"silver", "no_silver"}:
            self.status.setText("invalid label")
            return

        saved = 0
        for idx in range(max(1, int(count))):
            frame = self._frame.copy()
            ts = time.time()
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
            ms = int((ts - int(ts)) * 1000.0)
            name = f"{stamp}_{ms:03d}_{label}_{idx:02d}.jpg"
            rel = Path(label) / name
            path = self.output_root / rel
            ok = cv2.imwrite(str(path), frame)
            if not ok:
                continue
            saved += 1
            self._saved_count += 1
            self._append_metadata(ts, label, rel, frame.shape[1], frame.shape[0])

        self.count.setText(f"Saved: {self._saved_count}")
        self.status.setText(f"saved {saved}/{count} frame(s) in class '{label}'")

    def _append_metadata(self, ts: float, label: str, rel_path: Path, width: int, height: int) -> None:
        row = {
            "timestamp": f"{ts:.6f}",
            "label": label,
            "path": str(rel_path).replace("\\", "/"),
            "camera_index": int(self.camera_index_spin.value()),
            "width": int(width),
            "height": int(height),
            "fps": int(self.fps_spin.value()),
        }

        csv_exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
            if not csv_exists:
                writer.writeheader()
            writer.writerow(row)

        with self.jsonl_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self._timer.stop()
        self._release_camera()
        super().closeEvent(event)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture labeled silver-ball images for quick training")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("New_AI/obr_overengineering_v1/dataset/silver_ball"),
        help="Output dataset root (creates silver/ and no_silver/)",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--fps", type=int, default=30, help="Capture fps")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = QApplication.instance()
    owns = app is None
    if app is None:
        app = QApplication(sys.argv)
    win = SilverCaptureWindow(
        output_root=args.output_root,
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    win.show()
    if owns:
        return app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
