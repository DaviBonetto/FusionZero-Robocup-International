/*
  FusionZero serial motor bridge template

  Protocol:
    PING
    CMD FORWARD <duration_ms>
    CMD STOP 0
    CMD ESTOP 0
    CMD RESET_ESTOP 0

  Responses:
    READY FUSIONZERO
    PONG
    ACK FORWARD <duration_ms>
    ACK STOP
    ACK ESTOP
    ACK RESET_ESTOP
    ERR <reason>
    EVENT WATCHDOG_STOP

  Safety:
    - Raspberry sends heartbeat PINGs while idle or while a motion is active.
    - If the heartbeat disappears, this sketch stops both motors automatically.
    - CMD ESTOP latches an emergency stop until CMD RESET_ESTOP 0 is received.

  Edit the pin mapping below to match the motor driver actually used on the robot.
*/

const long SERIAL_BAUD = 115200;
const unsigned long HEARTBEAT_TIMEOUT_MS = 1500;

const int LEFT_PWM_PIN = 5;
const int LEFT_IN1_PIN = 7;
const int LEFT_IN2_PIN = 8;
const int RIGHT_PWM_PIN = 6;
const int RIGHT_IN1_PIN = 9;
const int RIGHT_IN2_PIN = 10;

const int DEFAULT_SPEED = 170;

String lineBuffer;
unsigned long forwardUntilMs = 0;
unsigned long watchdogDeadlineMs = 0;
bool estopLatched = false;
bool watchdogStopAnnounced = false;

void setup() {
  Serial.begin(SERIAL_BAUD);
  lineBuffer.reserve(64);

  pinMode(LEFT_PWM_PIN, OUTPUT);
  pinMode(LEFT_IN1_PIN, OUTPUT);
  pinMode(LEFT_IN2_PIN, OUTPUT);
  pinMode(RIGHT_PWM_PIN, OUTPUT);
  pinMode(RIGHT_IN1_PIN, OUTPUT);
  pinMode(RIGHT_IN2_PIN, OUTPUT);

  stopMotors();
  refreshWatchdog();
  Serial.println("READY FUSIONZERO");
}

void loop() {
  while (Serial.available() > 0) {
    char ch = (char)Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (lineBuffer.length() > 0) {
        handleCommand(lineBuffer);
        lineBuffer = "";
      }
    } else {
      lineBuffer += ch;
    }
  }

  if (!estopLatched && forwardUntilMs > 0 && millis() >= forwardUntilMs) {
    stopMotors();
    forwardUntilMs = 0;
    Serial.println("ACK STOP");
  }

  if (!estopLatched && watchdogDeadlineMs > 0 && millis() >= watchdogDeadlineMs) {
    stopMotors();
    forwardUntilMs = 0;
    watchdogDeadlineMs = 0;
    if (!watchdogStopAnnounced) {
      watchdogStopAnnounced = true;
      Serial.println("EVENT WATCHDOG_STOP");
    }
  }
}

void handleCommand(const String& rawCommand) {
  String command = rawCommand;
  command.trim();

  if (command == "PING") {
    refreshWatchdog();
    Serial.println("PONG");
    return;
  }

  if (command == "CMD ESTOP 0") {
    stopMotors();
    forwardUntilMs = 0;
    estopLatched = true;
    watchdogDeadlineMs = 0;
    watchdogStopAnnounced = false;
    Serial.println("ACK ESTOP");
    return;
  }

  if (command == "CMD RESET_ESTOP 0") {
    estopLatched = false;
    refreshWatchdog();
    Serial.println("ACK RESET_ESTOP");
    return;
  }

  if (estopLatched) {
    Serial.println("ERR estop_latched");
    return;
  }

  if (command.startsWith("CMD FORWARD ")) {
    long durationMs = command.substring(12).toInt();
    if (durationMs <= 0) {
      Serial.println("ERR invalid_duration");
      return;
    }
    driveForward(DEFAULT_SPEED);
    forwardUntilMs = millis() + (unsigned long)durationMs;
    refreshWatchdog();
    Serial.print("ACK FORWARD ");
    Serial.println(durationMs);
    return;
  }

  if (command == "CMD STOP 0") {
    stopMotors();
    forwardUntilMs = 0;
    refreshWatchdog();
    Serial.println("ACK STOP");
    return;
  }

  Serial.println("ERR unknown_command");
}

void refreshWatchdog() {
  watchdogDeadlineMs = millis() + HEARTBEAT_TIMEOUT_MS;
  watchdogStopAnnounced = false;
}

void driveForward(int pwm) {
  digitalWrite(LEFT_IN1_PIN, HIGH);
  digitalWrite(LEFT_IN2_PIN, LOW);
  digitalWrite(RIGHT_IN1_PIN, HIGH);
  digitalWrite(RIGHT_IN2_PIN, LOW);
  analogWrite(LEFT_PWM_PIN, pwm);
  analogWrite(RIGHT_PWM_PIN, pwm);
}

void stopMotors() {
  analogWrite(LEFT_PWM_PIN, 0);
  analogWrite(RIGHT_PWM_PIN, 0);
  digitalWrite(LEFT_IN1_PIN, LOW);
  digitalWrite(LEFT_IN2_PIN, LOW);
  digitalWrite(RIGHT_IN1_PIN, LOW);
  digitalWrite(RIGHT_IN2_PIN, LOW);
}
