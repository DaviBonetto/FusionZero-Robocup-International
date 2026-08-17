from __future__ import annotations

import argparse
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe PCA9685 channel smoke test for FusionZero.")
    parser.add_argument("--address", type=lambda value: int(str(value), 0), default=0x40)
    parser.add_argument("--frequency-hz", type=int, default=50)
    parser.add_argument("--left-channel", type=int, default=0)
    parser.add_argument("--right-channel", type=int, default=1)
    parser.add_argument("--neutral-us", type=int, default=1500)
    parser.add_argument("--test-offset-us", type=int, default=120)
    parser.add_argument("--step-seconds", type=float, default=0.55)
    parser.add_argument("--left-inverted", action="store_true")
    parser.add_argument("--right-inverted", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class Pca9685Driver:
    def __init__(self, *, address: int, frequency_hz: int) -> None:
        import board  # type: ignore
        import busio  # type: ignore
        from adafruit_pca9685 import PCA9685  # type: ignore

        self.frequency_hz = int(frequency_hz)
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c, address=int(address))
        self.pca.frequency = self.frequency_hz

    def set_us(self, channel: int, pulse_us: int) -> None:
        period_us = 1_000_000.0 / float(self.frequency_hz)
        duty_cycle = int(max(0, min(0xFFFF, round((float(pulse_us) / period_us) * 0xFFFF))))
        self.pca.channels[int(channel)].duty_cycle = duty_cycle

    def close(self) -> None:
        self.pca.deinit()


def main() -> int:
    args = _parse_args()
    offset = abs(int(args.test_offset_us))
    left_forward = int(args.neutral_us) + (-offset if args.left_inverted else offset)
    right_forward = int(args.neutral_us) + (-offset if args.right_inverted else offset)

    driver = Pca9685Driver(address=args.address, frequency_hz=args.frequency_hz)
    try:
        steps = [
            ("neutral", args.neutral_us, args.neutral_us),
            ("left_forward_only", left_forward, args.neutral_us),
            ("neutral", args.neutral_us, args.neutral_us),
            ("right_forward_only", args.neutral_us, right_forward),
            ("neutral", args.neutral_us, args.neutral_us),
            ("both_forward", left_forward, right_forward),
            ("neutral", args.neutral_us, args.neutral_us),
        ]
        for name, left_us, right_us in steps:
            print(f"{name}: left={left_us}us right={right_us}us", flush=True)
            driver.set_us(args.left_channel, int(left_us))
            driver.set_us(args.right_channel, int(right_us))
            time.sleep(max(0.1, float(args.step_seconds)))
    finally:
        try:
            driver.set_us(args.left_channel, int(args.neutral_us))
            driver.set_us(args.right_channel, int(args.neutral_us))
        finally:
            driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
