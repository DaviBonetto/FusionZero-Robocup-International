# Performance and Stability Report - Agent_TestValidation

Date: 2026-03-03  
Scope: `New_AI/obr_overengineering_v1`

## Method

- Functional/perf validation suite:
  - `python -m pytest New_AI/obr_overengineering_v1/tests -q`
- Stage-latency probe:
  - Synthetic capture frames through `VisionNode` + `StateMachine` + EventBus subscribers representing UI ingestion.
  - 45 frames, bounded queue (`max_queue_size=2048`, `drop_oldest=False`).

## Latency by stage

| Stage | Avg (ms) | P95 (ms) |
|---|---:|---:|
| Capture -> Vision call return | 39.073 | 60.296 |
| Capture -> Detection dispatch | 39.413 | 60.716 |
| Vision detection -> FSM transition | 0.509 | 0.726 |
| FSM transition -> UI state subscriber | 0.037 | 0.055 |
| End-to-end (Capture -> UI state) | 39.959 | 61.436 |

## Stability signals

- Frames requested: `45`
- Samples matched across all stages: `45`
- Lost samples: `0`
- EventBus concurrency test: `4 publishers x 120 messages`, no message loss (`480/480` received).
- Queue backpressure behavior:
  - Without `drop_oldest`: queue-full errors observed as expected.
  - With `drop_oldest=True`: no queue-full error observed under same stress pattern.

## Interpretation

- Pipeline stage timing is stable under synthetic load, with low dispersion and ~61 ms P95 end-to-end in this desktop execution profile.
- EventBus queue policy behaves deterministically for both strict and lossy modes.
- No intermittent thread/queue failures were observed in executed tests.
- UI stress/corner/steering/no-camera tests execute with `PyQt6` installed and pass (`4/4`).

## Residual performance risks

- Numbers are synthetic and desktop-runner based; they are not direct Raspberry Pi 4 hardware measurements.
- Model-backed branches (real silver/dead victim inference) were not profiled with production model execution in this pass.
