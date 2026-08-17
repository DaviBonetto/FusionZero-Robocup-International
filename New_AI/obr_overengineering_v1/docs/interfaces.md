# Interfaces v1 (Frozen)

Contract scope: `New_AI/obr_overengineering_v1`  
Status: `v1 frozen for Wave 1 integration`  
Owner: `Agent_StateMachine`  
Date: `2026-03-02`

## 1) Official FSM contract

### 1.1 States
- `SEARCHING_LINE`
- `FOLLOWING_LINE`
- `VALIDATING_GAP`
- `CROSSING_GAP`
- `VICTIM_FOUND`
- `RESCUE_ZONE_DETECTED`

### 1.2 Events
- `ON_GAP`
- `ON_LINE_FOUND`
- `ON_LINE_LOST`
- `ON_VICTIM_DETECTED`
- `ON_RESCUE_RED_DETECTED`
- `ON_INTERSECTION`
- `ON_TIMEOUT`
- `ON_RESET`

Legacy event aliases accepted by implementation:
- `ON_VICTIM` -> `ON_VICTIM_DETECTED`
- `ON_INTERSECT` -> `ON_INTERSECTION`
- `ON_RESCUE_RED` -> `ON_RESCUE_RED_DETECTED`

### 1.3 Complete transition table (deterministic)
| Current state | ON_GAP | ON_LINE_FOUND | ON_LINE_LOST | ON_VICTIM_DETECTED | ON_RESCUE_RED_DETECTED | ON_INTERSECTION | ON_TIMEOUT | ON_RESET |
|---|---|---|---|---|---|---|---|---|
| `SEARCHING_LINE` | `SEARCHING_LINE` | `FOLLOWING_LINE` | `SEARCHING_LINE` | `VICTIM_FOUND` | `RESCUE_ZONE_DETECTED` | `SEARCHING_LINE` | `SEARCHING_LINE` | `SEARCHING_LINE` |
| `FOLLOWING_LINE` | `VALIDATING_GAP` | `FOLLOWING_LINE` | `SEARCHING_LINE` | `VICTIM_FOUND` | `RESCUE_ZONE_DETECTED` | `FOLLOWING_LINE` | `SEARCHING_LINE` | `SEARCHING_LINE` |
| `VALIDATING_GAP` | `CROSSING_GAP` | `FOLLOWING_LINE` | `SEARCHING_LINE` | `VICTIM_FOUND` | `RESCUE_ZONE_DETECTED` | `FOLLOWING_LINE` | `SEARCHING_LINE` | `SEARCHING_LINE` |
| `CROSSING_GAP` | `CROSSING_GAP` | `FOLLOWING_LINE` | `CROSSING_GAP` | `VICTIM_FOUND` | `RESCUE_ZONE_DETECTED` | `CROSSING_GAP` | `SEARCHING_LINE` | `SEARCHING_LINE` |
| `VICTIM_FOUND` | `VICTIM_FOUND` | `FOLLOWING_LINE` | `SEARCHING_LINE` | `VICTIM_FOUND` | `RESCUE_ZONE_DETECTED` | `VICTIM_FOUND` | `FOLLOWING_LINE` | `SEARCHING_LINE` |
| `RESCUE_ZONE_DETECTED` | `RESCUE_ZONE_DETECTED` | `FOLLOWING_LINE` | `RESCUE_ZONE_DETECTED` | `VICTIM_FOUND` | `RESCUE_ZONE_DETECTED` | `RESCUE_ZONE_DETECTED` | `FOLLOWING_LINE` | `SEARCHING_LINE` |

Transition log format (mandatory):
- `TIMESTAMP [STATE] msg`
- Example: `2026-03-02T18:52:10.123+00:00 [FOLLOWING_LINE] SEARCHING_LINE --ON_LINE_FOUND--> FOLLOWING_LINE | line detected`

## 2) Official Event Bus topics

| Topic | Payload type | Producer(s) | Consumer(s) |
|---|---|---|---|
| `vision.raw_frame` | `FrameEvent` | Vision Node | UI, Recorder |
| `vision.processed_frame` | `FrameEvent` | Vision Node | UI |
| `vision.detections` | `VisionDetectionEvent` | Vision Node | FSM, UI, Test |
| `fsm.state` | `StateSnapshotEvent` | FSM | Vision, UI, Nav, Test |
| `fsm.transition` | `StateTransitionEvent` | FSM | UI, Logger, Test |
| `nav.pose` | `PoseEvent` | Navigation Node | Path tracker, UI |
| `nav.path` | `PathEvent` | Navigation Node | UI |
| `ui.command` | `UICommandEvent` | UI | Orchestrator, FSM |
| `system.health` | `HealthEvent` | Orchestrator, Perf monitor | UI, Logger, Test |
| `system.log` | `LogEvent` | All nodes | UI, Logger, Test |

## 3) Payload schemas (v1)

### 3.1 BaseEvent
```json
{
  "timestamp": "float (unix seconds)"
}
```

### 3.2 StateTransitionEvent
```json
{
  "timestamp": "float",
  "old_state": "str",
  "new_state": "str",
  "trigger": "str",
  "reason": "str"
}
```

### 3.3 StateSnapshotEvent
```json
{
  "timestamp": "float",
  "state": "str"
}
```

### 3.4 VisionDetectionEvent
```json
{
  "timestamp": "float",
  "state": "str",
  "line": "bool",
  "balls": "int",
  "green": "bool",
  "red": "bool",
  "victims": "int",
  "latency_ms": "float",
  "metadata": "dict[str, any]"
}
```

Additional metadata fields (non-breaking, optional for consumers):
- `silver_ball_found: bool`
- `silver_ball_confidence: float`
- `silver_ball_bbox: {x:int,y:int,w:int,h:int} | null`
- `black_ball_found: bool`
- `black_ball_confidence: float`
- `black_ball_bbox: {x:int,y:int,w:int,h:int} | null`
- `green_corner_found: bool`
- `green_corner_confidence: float`
- `green_corner_bbox: {x:int,y:int,w:int,h:int} | null`
- `red_corner_found: bool`
- `red_corner_confidence: float`
- `red_corner_bbox: {x:int,y:int,w:int,h:int} | null`
- `debug_views_available: list[str]`
- `silver_ball_black_overlap_suppressed: bool`
- `silver_line: {found:bool, confidence:float, bbox:{x:int,y:int,w:int,h:int}|null, mode:str, model:dict, heuristic:dict, decision:dict}`

Important integration rule:
- `VisionDetectionEvent.metadata` must remain JSON-serializable.
- Numpy masks and intermediate frames are exposed only via `VisionNode.get_last_debug_bundle()`, never embedded directly in `vision.detections`.

### 3.5 PoseEvent
```json
{
  "timestamp": "float",
  "x": "float",
  "y": "float",
  "theta": "float"
}
```

### 3.6 PathEvent
```json
{
  "timestamp": "float",
  "poses": "list[PoseEvent]"
}
```

### 3.7 HealthEvent
```json
{
  "timestamp": "float",
  "cpu_percent": "float",
  "fps_capture": "float",
  "fps_process": "float",
  "fps_ui": "float",
  "queue_depth": "int"
}
```

### 3.8 UICommandEvent
```json
{
  "timestamp": "float",
  "command": "str",
  "params": "dict[str, any]"
}
```

### 3.9 FrameEvent
```json
{
  "timestamp": "float",
  "frame_id": "int",
  "width": "int",
  "height": "int",
  "encoding": "str",
  "data": "bytes"
}
```

### 3.10 LogEvent
```json
{
  "timestamp": "float",
  "level": "str",
  "message": "str",
  "source": "str",
  "state": "str"
}
```

## 4) Public APIs frozen in v1
- `EventBus.publish(topic: str, message: BaseEvent) -> None`
- `EventBus.subscribe(topic: str, handler: Callable[[BaseEvent], None]) -> Subscription`
- `StateMachine.handle(event: RobotEvent, payload: dict | None = None) -> RobotState`
- `switch_pipeline(state: RobotState, frame_bgr: np.ndarray, config: VisionConfig) -> VisionDetectionEvent`
- `PathTracker.push(x: float, y: float, theta: float, timestamp: float) -> None`
- `PathTracker.snapshot() -> list[PoseEvent]`

Local-only debug API for calibration/replay:
- `VisionNode.get_last_processed_frame(copy: bool = True) -> np.ndarray | None`
- `VisionNode.get_last_debug_bundle(copy: bool = True) -> dict[str, Any] | None`

`get_last_debug_bundle()` schema:
```json
{
  "frame_id": "int",
  "timestamp": "float",
  "state": "str",
  "metadata": "dict[str, any]",
  "views": "dict[str, np.ndarray]"
}
```

## 5) Error handling and retry rules
- EventBus queue is bounded (`max_queue_size`, default `512`).
- `publish(block=False)` on full queue raises `EventBusFullError` (no silent drop by default).
- Optional policy: instantiate `EventBus(drop_oldest=True)` only for non-critical streams (for example, video frames).
- Subscriber exceptions are isolated; one failing subscriber does not block others.
- Unknown event names in FSM raise `InvalidTransitionError` immediately.
- Recommended retry policy for critical publishers:
  1. Retry up to `3` times.
  2. Backoff `5ms`, `10ms`, `20ms`.
  3. If still failing, emit degraded `system.health` and one `system.log` error.

## 6) Integration notes for builders
- UI should subscribe to: `fsm.state`, `fsm.transition`, `vision.raw_frame`, `vision.processed_frame`, `nav.path`, `system.health`, `system.log`.
- Vision should subscribe to: `fsm.state`; publish `vision.*` only.
- Path module should subscribe to: `nav.pose`; publish `nav.path`.
- TestValidation should assert both `fsm.state` and `fsm.transition` sequences per scenario.

## 7) Handoff
Ready for:
- `Agent_UI_Refactor`
- `Agent_Vision_Optimization`
- `Agent_Path_Render`
- `Agent_RedColorCalibration`
- `Agent_TestValidation`
