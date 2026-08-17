from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def wait_until(
    condition: Callable[[], bool],
    *,
    timeout: float = 1.5,
    interval: float = 0.005,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return bool(condition())


@pytest.fixture
def event_bus():
    from core.event_bus import EventBus

    bus = EventBus(max_queue_size=1024)
    try:
        yield bus
    finally:
        bus.stop()
