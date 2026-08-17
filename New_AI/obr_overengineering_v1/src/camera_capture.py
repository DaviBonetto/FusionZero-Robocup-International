"""Camera capture adapters used by the live Raspberry Pi runner.

The CSI IMX219 is exposed through libcamera/Picamera2, not as a normal V4L2
capture device.  This module keeps that detail at the edge of the runtime and
returns the same ``read``/``release`` interface used by OpenCV captures.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


class Picamera2Capture:
    """Small OpenCV-compatible adapter for a Picamera2 video stream."""

    def __init__(self, camera_index: int, width: int, height: int, fps: int) -> None:
        if int(camera_index) < 0:
            raise ValueError("camera index must be non-negative")

        try:
            from picamera2 import Picamera2
        except Exception as exc:  # pragma: no cover - depends on Raspberry Pi OS
            raise RuntimeError(f"Picamera2 is unavailable: {exc}") from exc

        self._camera: Any = None
        self._started = False
        self._camera = Picamera2(camera_num=int(camera_index))
        try:
            try:
                configuration = self._camera.create_video_configuration(
                    main={
                        "size": (max(1, int(width)), max(1, int(height))),
                        "format": "BGR888",
                    },
                    controls={"FrameRate": float(max(1, int(fps)))},
                )
            except Exception:
                # Some sensor/driver combinations do not expose FrameRate as a
                # configurable control. The stream itself remains valid.
                configuration = self._camera.create_video_configuration(
                    main={
                        "size": (max(1, int(width)), max(1, int(height))),
                        "format": "BGR888",
                    }
                )

            self._camera.configure(configuration)
            self._camera.start()
            self._started = True
            # Let auto-exposure and the first completed request settle before
            # the vision loop consumes a frame.
            time.sleep(0.15)
        except Exception:
            self.release()
            raise

    def isOpened(self) -> bool:
        return self._camera is not None and self._started

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.isOpened():
            return False, None
        try:
            frame = self._camera.capture_array("main")
        except Exception:
            return False, None
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False, None
        return True, np.ascontiguousarray(frame)

    def release(self) -> None:
        camera = self._camera
        self._camera = None
        was_started = self._started
        self._started = False
        if camera is None:
            return
        if was_started:
            try:
                camera.stop()
            except Exception:
                pass
        try:
            camera.close()
        except Exception:
            pass

