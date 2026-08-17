# Pi 5 Line Robot Run

This run keeps the official vision pipeline unchanged and adds only the hardware
outputs around it:

- PCA9685 I2C address `0x40`
- right side channels `0,1` (rear/front)
- left side channels `4,5` (rear/front)
- LED1 GPIO `18` (physical pin 12)
- LED2 GPIO `23` (physical pin 16)
- remote dashboard TCP port `8765`

The calibrated PCA9685 profile is:

- all four channels stop at `1600 us`
- left channels `4,5`: forward is above neutral (`1700 us` weak, `1900 us` strong)
- right channels `0,1`: forward below neutral (`1500 us` weak, `1300 us` strong)
- calibrated straight drive: left `1900 us`, right `1400 us`

The line detector also applies a ground-line guard before publishing
`FOLLOWING_LINE`: the black candidate must be anchored in the lower camera
band, span enough of the image vertically, and remain narrow across its rows.
Wide black bands, compact patches, and background objects are rejected, so the
robot receives the calibrated neutral pulse when the frame is ambiguous.

Green-square maneuvers remain outside this line-only validation phase.

## Safety sequence

1. Keep the wheels raised, or disconnect motor power, for the first real-PCA
   start. The adapter sends neutral PWM at startup, but the camera may quickly
   produce a drive command.
2. Confirm every LED has a series resistor. Do not enable the LED outputs with
   bare LEDs connected directly to GPIO.
3. Keep a physical battery/power cutoff within reach.
4. Run the dry-run command first. Only then use the real-PCA command.

## Raspberry Pi

```bash
cd ~/FusionZero-Robocup-International/New_AI/obr_overengineering_v1
source .venv/bin/activate

# Safe camera + dashboard validation; no motor PWM is driven.
python src/live_dashboard_runner.py \
  --headless \
  --camera-index 0 \
  --camera-backend picamera2 \
  --line-only \
  --robot-backend pca9685 \
  --robot-dry-run \
   --pca9685-left-channels 4,5 \
   --pca9685-right-channels 0,1 \
   --no-pca9685-left-inverted \
   --pca9685-right-inverted \
   --pca9685-line-steering-inverted \
  --pca9685-sharp-curve-threshold 0.55 \
  --pca9685-sharp-curve-hold-ms 900 \
  --pca9685-sharp-curve-outer-scale 1.0 \
  --pca9685-sharp-curve-inner-reverse-scale 0.55 \
  --pca9685-sharp-curve-visible-commit-ms 2100 \
  --pca9685-sharp-curve-finish-inner-reverse-scale 0.35 \
  --pca9685-left-base-throttle-us 300 \
  --pca9685-right-base-throttle-us 200 \
  --pca9685-neutral-us 1600 \
  --enable-leds \
  --remote-bind 0.0.0.0 \
  --remote-port 8765
```

## PC dashboard

Run this in a separate PowerShell window from the repository root. Replace the
IP if the Pi receives a new DHCP lease:

```powershell
python .\New_AI\obr_overengineering_v1\src\remote_dashboard_client.py --host 192.168.2.153 --port 8765
```

The dashboard's LED buttons send `leds.on`, `leds.off`, `led1.toggle`, and
`led2.toggle` to the Raspberry Pi. The dry-run mode makes the robot status
visible without energizing motor outputs.

## Real PCA test

Stop the dry-run runner with `Ctrl+C`. Keep the wheels raised and run:

```bash
cd ~/FusionZero-Robocup-International/New_AI/obr_overengineering_v1
source .venv/bin/activate
python src/live_dashboard_runner.py \
  --headless \
  --camera-index 0 \
  --camera-backend picamera2 \
  --line-only \
  --robot-backend pca9685 \
   --pca9685-left-channels 4,5 \
   --pca9685-right-channels 0,1 \
   --no-pca9685-left-inverted \
   --pca9685-right-inverted \
   --pca9685-line-steering-inverted \
  --pca9685-sharp-curve-threshold 0.55 \
  --pca9685-sharp-curve-hold-ms 900 \
  --pca9685-sharp-curve-outer-scale 1.0 \
  --pca9685-sharp-curve-inner-reverse-scale 0.55 \
  --pca9685-sharp-curve-visible-commit-ms 2100 \
  --pca9685-sharp-curve-finish-inner-reverse-scale 0.35 \
  --pca9685-left-base-throttle-us 300 \
  --pca9685-right-base-throttle-us 200 \
  --pca9685-neutral-us 1600 \
  --enable-leds \
  --remote-bind 0.0.0.0 \
  --remote-port 8765
```

Use `STOP` or `Force STOP` in the dashboard. `Force STOP` latches the
software emergency stop. In `line-only`, rescue/victim/gap/intersection
events cannot change the FSM; a lost/low-confidence line stops the PCA
outputs instead of leaving the previous command active. The sole exception is
a detector-confirmed 90-degree curve: the inside side reverses at a bounded
fraction, the outside side remains forward, and the pivot keeps a strong
minimum of `900 ms` before accepting the outgoing straight. If the line leaves
the camera, the same direction continues at reduced strength up to an absolute
`2100 ms` bound. On `Ctrl+C`, the
   runner keeps all four channels at the calibrated neutral pulse (`1600 us`) and
  closes the PCA driver.

For repeatable Pi setup, copy `deploy/fusionzero.pi5.line.env.example` to a
Pi-local env file and run:

```bash
cp deploy/fusionzero.pi5.line.env.example /tmp/fusionzero.pi5.line.env
# Keep FZ_ROBOT_DRY_RUN=1 for the first run.
bash scripts/run_pi_headless.sh --env-file /tmp/fusionzero.pi5.line.env
```

## Shutdown

1. Press `Force STOP` in the dashboard.
2. Stop the Raspberry process with `Ctrl+C`.
3. Verify that the runner printed its shutdown/neutralization message.
4. Run `sudo poweroff` on the Raspberry Pi and wait until the display/network
   disappear before removing power.
