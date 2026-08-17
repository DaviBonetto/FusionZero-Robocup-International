/*
  FusionZero line follower controller (Arduino main motion brain)

  Responsibilities:
    - read line sensors at deterministic rate
    - run local PID + heading stabilization
    - control motors
    - keep a watchdog against Raspberry/serial loss
    - accept only high-level assists from Raspberry Pi

  Supported protocol:
    PING
    CMD FORWARD <duration_ms>
    CMD STOP 0
    CMD ESTOP 0
    CMD RESET_ESTOP 0
    ASST LINE found=<0|1> offset=<float> angle=<float> conf=<float> gap=<int> source=<token>
    ASST GREEN found=<0|1> instruction=<VERDE_ANTES|VERDE_DEPOIS|VERDE_MEIA_VOLTA> side=<LEFT|RIGHT|BOTH> conf=<float> hold_ms=<int> source=<token>
    ASST OBSTACLE state=<CLEAR|AHEAD|TEST> conf=<float> hold_ms=<int> source=<token>

  Responses:
    READY FUSIONZERO
    PONG
    ACK FORWARD <duration_ms>
    ACK STOP
    ACK ESTOP
    ACK RESET_ESTOP
    ACK ASST LINE
    ACK ASST GREEN
    ACK ASST OBSTACLE
    ERR <reason>
    EVENT WATCHDOG_STOP
    TLM ...

  Hardware gaps intentionally left configurable:
    - IMU reading is a stub until the real gyro is wired
    - local color sensor hook is a stub until the real sensor is wired
    - front obstacle sensors are optional and can be disabled with pin = -1
*/

#include <string.h>

const long SERIAL_BAUD = 115200;
const unsigned long CONTROL_PERIOD_MS = 10;
const unsigned long TELEMETRY_PERIOD_MS = 100;
const unsigned long HEARTBEAT_TIMEOUT_MS = 1400;
const unsigned long LINE_ASSIST_VALID_MS = 450;

const int LEFT_PWM_PIN = 2;
const int LEFT_IN1_PIN = 4;
const int LEFT_IN2_PIN = 8;
const int RIGHT_PWM_PIN = 7;
const int RIGHT_IN1_PIN = 3;
const int RIGHT_IN2_PIN = 5;

// Flip these on the first bench test if a track spins backwards.
const bool LEFT_MOTOR_INVERTED = false;
const bool RIGHT_MOTOR_INVERTED = true;
const bool CAMERA_ASSIST_ONLY_MODE = true;

const int LINE_SENSOR_COUNT = 5;
const int LINE_SENSOR_PINS[LINE_SENSOR_COUNT] = {A0, A1, A2, A3, A4};
const float LINE_SENSOR_WEIGHTS[LINE_SENSOR_COUNT] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
const int LINE_SENSOR_CAL_MIN[LINE_SENSOR_COUNT] = {140, 140, 140, 140, 140};
const int LINE_SENSOR_CAL_MAX[LINE_SENSOR_COUNT] = {900, 900, 900, 900, 900};
const bool LINE_BLACK_IS_LOW = true;
const float LINE_ACTIVE_THRESHOLD = 0.34f;

const int FRONT_SENSOR_PIN = -1;
const int FRONT_LEFT_SENSOR_PIN = -1;
const int FRONT_RIGHT_SENSOR_PIN = -1;
const bool FRONT_SENSOR_ACTIVE_LOW = true;

const bool GYRO_ENABLED = false;
const bool LOCAL_COLOR_SENSOR_ENABLED = false;

struct PIDConfig {
  float kp;
  float ki;
  float kd;
  float smoothingAlpha;
  float integralLimit;
  float derivativeLimit;
  float maxTurn;
  float headingKp;
};

struct SpeedConfig {
  int straight;
  int medium;
  int curve;
  int cautious;
  int reverse;
  int turn;
};

struct PIDRuntime {
  float filteredError;
  float lastFilteredError;
  float integral;
  float derivative;
  float output;
};

struct LineReading {
  bool found;
  float error;
  float confidence;
  int activeCount;
};

struct ObstacleReading {
  bool frontBlocked;
  bool leftBlocked;
  bool rightBlocked;
  int frontMm;
  int leftMm;
  int rightMm;
};

struct LineAssist {
  bool valid;
  bool found;
  float offsetNorm;
  float angleDeg;
  float confidence;
  int gapFrames;
  unsigned long expiresAtMs;
};

struct GreenAssist {
  bool active;
  char instruction[20];
  char side[12];
  float confidence;
  unsigned long expiresAtMs;
};

struct ObstacleAssist {
  bool active;
  char state[12];
  float confidence;
  unsigned long expiresAtMs;
};

enum ControlMode {
  MODE_SAFE_STOP = 0,
  MODE_FOLLOW_LINE = 1,
  MODE_GREEN_MANEUVER = 2,
  MODE_OBSTACLE_AVOID = 3,
  MODE_MANUAL_FORWARD = 4,
  MODE_ESTOP = 5
};

PIDConfig pidCfg = {1.45f, 0.18f, 0.42f, 0.45f, 0.70f, 2.40f, 145.0f, 18.0f};
SpeedConfig speedCfg = {165, 140, 115, 95, 85, 120};
PIDRuntime pidState = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
LineAssist lineAssist = {false, false, 0.0f, 90.0f, 0.0f, 0, 0};
GreenAssist greenAssist = {false, "NO_GREEN", "NONE", 0.0f, 0};
ObstacleAssist obstacleAssist = {false, "CLEAR", 0.0f, 0};

String lineBuffer;
unsigned long lastControlAtMs = 0;
unsigned long lastTelemetryAtMs = 0;
unsigned long lastHeartbeatAtMs = 0;
unsigned long manualForwardUntilMs = 0;
unsigned long maneuverStartedAtMs = 0;
unsigned long obstacleStartedAtMs = 0;
bool estopLatched = false;
bool watchdogEventSent = false;
bool failsafeSerialLoss = false;

ControlMode currentMode = MODE_SAFE_STOP;
int lastLeftPwm = 0;
int lastRightPwm = 0;
LineReading lastLine = {false, 0.0f, 0.0f, 0};
ObstacleReading lastObstacle = {false, false, false, 1200, 1200, 1200};

float fAbs(float value);
float clampFloat(float value, float low, float high);
float normalizeLineSensor(int rawValue, int minValue, int maxValue);
LineReading readLineSensors();
ObstacleReading readObstacleSensors();
float readGyroYawDeg();
bool localColorSuggestsGreen();
void handleCommand(const String& rawCommand);
String valueForKey(const String& line, const String& key, const String& fallback);
float floatForKey(const String& line, const String& key, float fallback);
long longForKey(const String& line, const String& key, long fallback);
bool boolForKey(const String& line, const String& key, bool fallback);
void copyTokenValue(const String& value, char* target, size_t targetSize);
bool tokenEquals(const char* value, const char* expected);
void refreshHeartbeat();
bool hasFreshLineAssist();
void updateControl(unsigned long nowMs);
void executeFollowLine(unsigned long nowMs);
void executeGreenManeuver(unsigned long nowMs);
void executeObstacleAvoidance(unsigned long nowMs);
void setMode(ControlMode nextMode, unsigned long nowMs);
void resetPid();
void applyMotorCommand(int leftPwm, int rightPwm);
void stopMotors();
void emitTelemetry(unsigned long nowMs);
const char* modeName(ControlMode mode);

void setup() {
  Serial.begin(SERIAL_BAUD);
  lineBuffer.reserve(160);

  pinMode(LEFT_PWM_PIN, OUTPUT);
  pinMode(LEFT_IN1_PIN, OUTPUT);
  pinMode(LEFT_IN2_PIN, OUTPUT);
  pinMode(RIGHT_PWM_PIN, OUTPUT);
  pinMode(RIGHT_IN1_PIN, OUTPUT);
  pinMode(RIGHT_IN2_PIN, OUTPUT);

  if (!CAMERA_ASSIST_ONLY_MODE) {
    for (int i = 0; i < LINE_SENSOR_COUNT; ++i) {
      pinMode(LINE_SENSOR_PINS[i], INPUT);
    }
  }
  if (FRONT_SENSOR_PIN >= 0) pinMode(FRONT_SENSOR_PIN, INPUT_PULLUP);
  if (FRONT_LEFT_SENSOR_PIN >= 0) pinMode(FRONT_LEFT_SENSOR_PIN, INPUT_PULLUP);
  if (FRONT_RIGHT_SENSOR_PIN >= 0) pinMode(FRONT_RIGHT_SENSOR_PIN, INPUT_PULLUP);

  stopMotors();
  refreshHeartbeat();
  setMode(MODE_SAFE_STOP, millis());
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
    } else if (lineBuffer.length() < 159) {
      lineBuffer += ch;
    }
  }

  unsigned long nowMs = millis();
  if ((nowMs - lastControlAtMs) >= CONTROL_PERIOD_MS) {
    lastControlAtMs = nowMs;
    updateControl(nowMs);
  }
  if ((nowMs - lastTelemetryAtMs) >= TELEMETRY_PERIOD_MS) {
    lastTelemetryAtMs = nowMs;
    emitTelemetry(nowMs);
  }
}

void handleCommand(const String& rawCommand) {
  String command = rawCommand;
  command.trim();
  if (command.length() == 0) {
    return;
  }

  if (command == "PING") {
    refreshHeartbeat();
    Serial.println("PONG");
    return;
  }

  if (command == "CMD ESTOP 0") {
    estopLatched = true;
    setMode(MODE_ESTOP, millis());
    stopMotors();
    Serial.println("ACK ESTOP");
    return;
  }

  if (command == "CMD RESET_ESTOP 0") {
    estopLatched = false;
    failsafeSerialLoss = false;
    watchdogEventSent = false;
    resetPid();
    refreshHeartbeat();
    setMode(MODE_SAFE_STOP, millis());
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
    manualForwardUntilMs = millis() + (unsigned long)durationMs;
    refreshHeartbeat();
    setMode(MODE_MANUAL_FORWARD, millis());
    Serial.print("ACK FORWARD ");
    Serial.println(durationMs);
    return;
  }

  if (command == "CMD STOP 0") {
    manualForwardUntilMs = 0;
    lineAssist.valid = false;
    lineAssist.found = false;
    greenAssist.active = false;
    obstacleAssist.active = false;
    strcpy(obstacleAssist.state, "CLEAR");
    failsafeSerialLoss = false;
    refreshHeartbeat();
    resetPid();
    stopMotors();
    setMode(MODE_SAFE_STOP, millis());
    Serial.println("ACK STOP");
    return;
  }

  if (command.startsWith("ASST LINE ")) {
    lineAssist.valid = true;
    lineAssist.found = boolForKey(command, "found", true);
    lineAssist.offsetNorm = clampFloat(floatForKey(command, "offset", 0.0f), -1.0f, 1.0f);
    lineAssist.angleDeg = floatForKey(command, "angle", 90.0f);
    lineAssist.confidence = clampFloat(floatForKey(command, "conf", 0.0f), 0.0f, 1.0f);
    lineAssist.gapFrames = (int)longForKey(command, "gap", 0);
    lineAssist.expiresAtMs = millis() + LINE_ASSIST_VALID_MS;
    refreshHeartbeat();
    Serial.println("ACK ASST LINE");
    return;
  }

  if (command.startsWith("ASST GREEN ")) {
    if (!boolForKey(command, "found", true)) {
      greenAssist.active = false;
      strcpy(greenAssist.instruction, "NO_GREEN");
      strcpy(greenAssist.side, "NONE");
      refreshHeartbeat();
      Serial.println("ACK ASST GREEN");
      return;
    }
    copyTokenValue(valueForKey(command, "instruction", "NO_GREEN"), greenAssist.instruction, sizeof(greenAssist.instruction));
    copyTokenValue(valueForKey(command, "side", "NONE"), greenAssist.side, sizeof(greenAssist.side));
    greenAssist.confidence = clampFloat(floatForKey(command, "conf", 0.0f), 0.0f, 1.0f);
    greenAssist.expiresAtMs = millis() + (unsigned long)max(0L, longForKey(command, "hold_ms", 900L));
    greenAssist.active = greenAssist.confidence >= 0.20f;
    if (greenAssist.active) {
      setMode(MODE_GREEN_MANEUVER, millis());
    }
    refreshHeartbeat();
    Serial.println("ACK ASST GREEN");
    return;
  }

  if (command.startsWith("ASST OBSTACLE ")) {
    copyTokenValue(valueForKey(command, "state", "CLEAR"), obstacleAssist.state, sizeof(obstacleAssist.state));
    obstacleAssist.confidence = clampFloat(floatForKey(command, "conf", 0.0f), 0.0f, 1.0f);
    obstacleAssist.expiresAtMs = millis() + (unsigned long)max(0L, longForKey(command, "hold_ms", 1200L));
    obstacleAssist.active = !tokenEquals(obstacleAssist.state, "CLEAR");
    if (obstacleAssist.active) {
      setMode(MODE_OBSTACLE_AVOID, millis());
    }
    refreshHeartbeat();
    Serial.println("ACK ASST OBSTACLE");
    return;
  }

  Serial.println("ERR unknown_command");
}

String valueForKey(const String& line, const String& key, const String& fallback) {
  String token = key + "=";
  int start = line.indexOf(token);
  if (start < 0) {
    return fallback;
  }
  start += token.length();
  int end = line.indexOf(' ', start);
  if (end < 0) {
    end = line.length();
  }
  String value = line.substring(start, end);
  value.trim();
  return value.length() > 0 ? value : fallback;
}

float floatForKey(const String& line, const String& key, float fallback) {
  return valueForKey(line, key, String(fallback, 3)).toFloat();
}

long longForKey(const String& line, const String& key, long fallback) {
  return valueForKey(line, key, String(fallback)).toInt();
}

bool boolForKey(const String& line, const String& key, bool fallback) {
  String value = valueForKey(line, key, fallback ? "1" : "0");
  value.toUpperCase();
  return value == "1" || value == "TRUE" || value == "YES" || value == "ON";
}

void copyTokenValue(const String& value, char* target, size_t targetSize) {
  if (target == NULL || targetSize == 0) {
    return;
  }
  value.toCharArray(target, targetSize);
  target[targetSize - 1] = '\0';
}

bool tokenEquals(const char* value, const char* expected) {
  if (value == NULL || expected == NULL) {
    return false;
  }
  return strcmp(value, expected) == 0;
}

void refreshHeartbeat() {
  lastHeartbeatAtMs = millis();
  watchdogEventSent = false;
  failsafeSerialLoss = false;
}

bool hasFreshLineAssist() {
  return lineAssist.valid && lineAssist.found && lineAssist.confidence >= 0.25f;
}

float fAbs(float value) {
  return value >= 0.0f ? value : -value;
}

float clampFloat(float value, float low, float high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

float normalizeLineSensor(int rawValue, int minValue, int maxValue) {
  if (maxValue <= minValue) {
    return 0.0f;
  }
  float normalized = float(rawValue - minValue) / float(maxValue - minValue);
  normalized = clampFloat(normalized, 0.0f, 1.0f);
  return LINE_BLACK_IS_LOW ? (1.0f - normalized) : normalized;
}

LineReading readLineSensors() {
  if (CAMERA_ASSIST_ONLY_MODE) {
    LineReading out = {false, 0.0f, 0.0f, 0};
    return out;
  }

  float weighted = 0.0f;
  float total = 0.0f;
  int activeCount = 0;

  for (int i = 0; i < LINE_SENSOR_COUNT; ++i) {
    int raw = analogRead(LINE_SENSOR_PINS[i]);
    float blackStrength = normalizeLineSensor(raw, LINE_SENSOR_CAL_MIN[i], LINE_SENSOR_CAL_MAX[i]);
    weighted += blackStrength * LINE_SENSOR_WEIGHTS[i];
    total += blackStrength;
    if (blackStrength >= LINE_ACTIVE_THRESHOLD) {
      activeCount += 1;
    }
  }

  LineReading out;
  out.found = total > 0.06f && activeCount > 0;
  out.error = 0.0f;
  out.confidence = clampFloat(total / (LINE_SENSOR_COUNT * 0.65f), 0.0f, 1.0f);
  out.activeCount = activeCount;
  if (out.found && total > 0.001f) {
    out.error = clampFloat(weighted / total, -1.0f, 1.0f);
  }
  return out;
}

ObstacleReading readObstacleSensors() {
  ObstacleReading out;
  out.frontBlocked = false;
  out.leftBlocked = false;
  out.rightBlocked = false;
  out.frontMm = 1200;
  out.leftMm = 1200;
  out.rightMm = 1200;

  if (FRONT_SENSOR_PIN >= 0) {
    bool blocked = digitalRead(FRONT_SENSOR_PIN) == (FRONT_SENSOR_ACTIVE_LOW ? LOW : HIGH);
    out.frontBlocked = blocked;
    out.frontMm = blocked ? 140 : 1200;
  }
  if (FRONT_LEFT_SENSOR_PIN >= 0) {
    bool blocked = digitalRead(FRONT_LEFT_SENSOR_PIN) == (FRONT_SENSOR_ACTIVE_LOW ? LOW : HIGH);
    out.leftBlocked = blocked;
    out.leftMm = blocked ? 180 : 1200;
  }
  if (FRONT_RIGHT_SENSOR_PIN >= 0) {
    bool blocked = digitalRead(FRONT_RIGHT_SENSOR_PIN) == (FRONT_SENSOR_ACTIVE_LOW ? LOW : HIGH);
    out.rightBlocked = blocked;
    out.rightMm = blocked ? 180 : 1200;
  }
  return out;
}

float readGyroYawDeg() {
  if (!GYRO_ENABLED) {
    return 0.0f;
  }
  return 0.0f;  // TODO: replace with real IMU integration.
}

bool localColorSuggestsGreen() {
  if (!LOCAL_COLOR_SENSOR_ENABLED) {
    return false;
  }
  return false;  // TODO: replace with real color sensor confirmation.
}

void updateControl(unsigned long nowMs) {
  if (lineAssist.valid && nowMs > lineAssist.expiresAtMs) {
    lineAssist.valid = false;
  }
  if (greenAssist.active && nowMs > greenAssist.expiresAtMs) {
    greenAssist.active = false;
  }
  if (obstacleAssist.active && nowMs > obstacleAssist.expiresAtMs) {
    obstacleAssist.active = false;
    strcpy(obstacleAssist.state, "CLEAR");
  }

  lastLine = readLineSensors();
  lastObstacle = readObstacleSensors();
  bool lineAssistReady = hasFreshLineAssist();

  bool serialHealthy = (nowMs - lastHeartbeatAtMs) <= HEARTBEAT_TIMEOUT_MS;
  if (!serialHealthy) {
    failsafeSerialLoss = true;
    if (!watchdogEventSent) {
      Serial.println("EVENT WATCHDOG_STOP");
      watchdogEventSent = true;
    }
  }

  if (failsafeSerialLoss && CAMERA_ASSIST_ONLY_MODE) {
    setMode(MODE_SAFE_STOP, nowMs);
    stopMotors();
    return;
  }

  if (estopLatched) {
    setMode(MODE_ESTOP, nowMs);
    stopMotors();
    return;
  }

  bool obstacleTriggered = lastObstacle.frontBlocked || obstacleAssist.active || tokenEquals(obstacleAssist.state, "TEST");
  if (obstacleTriggered) {
    setMode(MODE_OBSTACLE_AVOID, nowMs);
  }

  if (!obstacleTriggered && (greenAssist.active || localColorSuggestsGreen())) {
    setMode(MODE_GREEN_MANEUVER, nowMs);
  }

  if (currentMode == MODE_MANUAL_FORWARD) {
    if (failsafeSerialLoss || obstacleTriggered) {
      manualForwardUntilMs = 0;
      setMode(obstacleTriggered ? MODE_OBSTACLE_AVOID : MODE_SAFE_STOP, nowMs);
    } else if (nowMs >= manualForwardUntilMs) {
      manualForwardUntilMs = 0;
      setMode(MODE_SAFE_STOP, nowMs);
    } else {
      applyMotorCommand(speedCfg.straight, speedCfg.straight);
      return;
    }
  }

  if (failsafeSerialLoss && !lastLine.found && !lineAssistReady) {
    setMode(MODE_SAFE_STOP, nowMs);
    stopMotors();
    return;
  }

  switch (currentMode) {
    case MODE_GREEN_MANEUVER:
      executeGreenManeuver(nowMs);
      break;
    case MODE_OBSTACLE_AVOID:
      executeObstacleAvoidance(nowMs);
      break;
    case MODE_FOLLOW_LINE:
      executeFollowLine(nowMs);
      break;
    case MODE_ESTOP:
      stopMotors();
      break;
    case MODE_SAFE_STOP:
    default:
      if (lastLine.found || lineAssistReady) {
        setMode(MODE_FOLLOW_LINE, nowMs);
        executeFollowLine(nowMs);
      } else {
        stopMotors();
      }
      break;
  }
}

void executeFollowLine(unsigned long nowMs) {
  (void)nowMs;
  float fusedError = lastLine.error;
  float fusedConfidence = lastLine.confidence;
  bool lineAssistReady = hasFreshLineAssist();

  if (lineAssistReady) {
    float visionWeight = clampFloat(lineAssist.confidence * 0.35f, 0.0f, 0.35f);
    if (lastLine.found) {
      fusedError = ((1.0f - visionWeight) * lastLine.error) + (visionWeight * lineAssist.offsetNorm);
    } else if (lineAssist.confidence >= 0.25f) {
      fusedError = lineAssist.offsetNorm;
      fusedConfidence = lineAssist.confidence;
    }
  }

  if (!lastLine.found && !lineAssistReady) {
    stopMotors();
    return;
  }

  float dt = CONTROL_PERIOD_MS / 1000.0f;
  float safeDt = dt < 0.001f ? 0.001f : dt;
  pidState.filteredError += pidCfg.smoothingAlpha * (fusedError - pidState.filteredError);
  float derivative = (pidState.filteredError - pidState.lastFilteredError) / safeDt;
  derivative = clampFloat(derivative, -pidCfg.derivativeLimit, pidCfg.derivativeLimit);

  float tentativeIntegral = clampFloat(
    pidState.integral + (pidState.filteredError * dt),
    -pidCfg.integralLimit,
    pidCfg.integralLimit
  );
  float headingTarget = 0.0f;
  if (lineAssistReady) {
    headingTarget = clampFloat((lineAssist.angleDeg - 90.0f) / 90.0f, -1.0f, 1.0f) * 18.0f;
  }
  float headingCorrection = GYRO_ENABLED ? ((headingTarget - readGyroYawDeg()) * pidCfg.headingKp) : 0.0f;
  float unsaturated = (
    (pidCfg.kp * pidState.filteredError) +
    (pidCfg.ki * tentativeIntegral) +
    (pidCfg.kd * derivative) +
    headingCorrection
  );
  float saturated = clampFloat(unsaturated, -pidCfg.maxTurn, pidCfg.maxTurn);
  if ((saturated == unsaturated) || ((saturated > 0.0f) && (pidState.filteredError < 0.0f)) || ((saturated < 0.0f) && (pidState.filteredError > 0.0f))) {
    pidState.integral = tentativeIntegral;
  }

  int baseSpeed = speedCfg.straight;
  float severity = fAbs(fusedError);
  if (severity > 0.55f) {
    baseSpeed = speedCfg.curve;
  } else if (severity > 0.25f) {
    baseSpeed = speedCfg.medium;
  }
  if (fusedConfidence < 0.35f || failsafeSerialLoss) {
    baseSpeed = speedCfg.cautious;
  }

  pidState.derivative = derivative;
  pidState.output = saturated;
  pidState.lastFilteredError = pidState.filteredError;

  int left = (int)(baseSpeed - saturated);
  int right = (int)(baseSpeed + saturated);
  applyMotorCommand(left, right);
  currentMode = MODE_FOLLOW_LINE;
}

void executeGreenManeuver(unsigned long nowMs) {
  unsigned long elapsed = nowMs - maneuverStartedAtMs;

  if (!greenAssist.active) {
    setMode(MODE_FOLLOW_LINE, nowMs);
    return;
  }

  if (tokenEquals(greenAssist.instruction, "VERDE_MEIA_VOLTA")) {
    if (elapsed < 780) {
      applyMotorCommand(-speedCfg.turn, speedCfg.turn);
      return;
    }
  } else if (tokenEquals(greenAssist.instruction, "VERDE_ANTES")) {
    if (elapsed < 430) {
      applyMotorCommand(-speedCfg.turn, speedCfg.turn);
      return;
    }
    if (elapsed < 850) {
      applyMotorCommand(speedCfg.cautious, speedCfg.cautious);
      return;
    }
  } else if (tokenEquals(greenAssist.instruction, "VERDE_DEPOIS")) {
    if (elapsed < 430) {
      applyMotorCommand(speedCfg.turn, -speedCfg.turn);
      return;
    }
    if (elapsed < 850) {
      applyMotorCommand(speedCfg.cautious, speedCfg.cautious);
      return;
    }
  } else {
    greenAssist.active = false;
  }

  greenAssist.active = false;
  setMode(MODE_FOLLOW_LINE, nowMs);
}

void executeObstacleAvoidance(unsigned long nowMs) {
  unsigned long elapsed = nowMs - obstacleStartedAtMs;
  bool preferLeft = lastObstacle.rightBlocked && !lastObstacle.leftBlocked;
  bool preferRight = lastObstacle.leftBlocked && !lastObstacle.rightBlocked;
  int turnDir = preferLeft ? -1 : 1;
  if (preferRight) {
    turnDir = 1;
  }

  if (tokenEquals(obstacleAssist.state, "CLEAR") && !lastObstacle.frontBlocked && elapsed > 200) {
    obstacleAssist.active = false;
    setMode(MODE_FOLLOW_LINE, nowMs);
    return;
  }

  if (elapsed < 180) {
    applyMotorCommand(-speedCfg.reverse, -speedCfg.reverse);
    return;
  }
  if (elapsed < 620) {
    applyMotorCommand(-turnDir * speedCfg.turn, turnDir * speedCfg.turn);
    return;
  }
  if (elapsed < 1220) {
    applyMotorCommand(speedCfg.cautious, speedCfg.cautious);
    return;
  }
  if (elapsed < 1580) {
    applyMotorCommand(turnDir * speedCfg.turn, -turnDir * speedCfg.turn);
    return;
  }

  obstacleAssist.active = false;
  strcpy(obstacleAssist.state, "CLEAR");
  setMode(MODE_FOLLOW_LINE, nowMs);
}

void setMode(ControlMode nextMode, unsigned long nowMs) {
  if (currentMode == nextMode) {
    return;
  }
  currentMode = nextMode;
  if (nextMode == MODE_GREEN_MANEUVER) {
    maneuverStartedAtMs = nowMs;
  }
  if (nextMode == MODE_OBSTACLE_AVOID) {
    obstacleStartedAtMs = nowMs;
  }
  if (nextMode == MODE_SAFE_STOP || nextMode == MODE_ESTOP) {
    resetPid();
    stopMotors();
  }
}

void resetPid() {
  pidState.filteredError = 0.0f;
  pidState.lastFilteredError = 0.0f;
  pidState.integral = 0.0f;
  pidState.derivative = 0.0f;
  pidState.output = 0.0f;
}

void applyMotorCommand(int leftPwm, int rightPwm) {
  leftPwm = constrain(leftPwm, -255, 255);
  rightPwm = constrain(rightPwm, -255, 255);
  lastLeftPwm = leftPwm;
  lastRightPwm = rightPwm;

  bool leftForward = leftPwm >= 0;
  bool rightForward = rightPwm >= 0;
  int leftMagnitude = abs(leftPwm);
  int rightMagnitude = abs(rightPwm);

  if (LEFT_MOTOR_INVERTED) {
    leftForward = !leftForward;
  }
  if (RIGHT_MOTOR_INVERTED) {
    rightForward = !rightForward;
  }

  digitalWrite(LEFT_IN1_PIN, leftForward ? HIGH : LOW);
  digitalWrite(LEFT_IN2_PIN, leftForward ? LOW : HIGH);
  digitalWrite(RIGHT_IN1_PIN, rightForward ? HIGH : LOW);
  digitalWrite(RIGHT_IN2_PIN, rightForward ? LOW : HIGH);
  analogWrite(LEFT_PWM_PIN, leftMagnitude);
  analogWrite(RIGHT_PWM_PIN, rightMagnitude);
}

void stopMotors() {
  lastLeftPwm = 0;
  lastRightPwm = 0;
  analogWrite(LEFT_PWM_PIN, 0);
  analogWrite(RIGHT_PWM_PIN, 0);
  digitalWrite(LEFT_IN1_PIN, LOW);
  digitalWrite(LEFT_IN2_PIN, LOW);
  digitalWrite(RIGHT_IN1_PIN, LOW);
  digitalWrite(RIGHT_IN2_PIN, LOW);
}

void emitTelemetry(unsigned long nowMs) {
  (void)nowMs;
  lastLine = readLineSensors();
  lastObstacle = readObstacleSensors();
  bool lineAssistReady = hasFreshLineAssist();
  float telemetryConfidence = lastLine.confidence;
  if (CAMERA_ASSIST_ONLY_MODE && !lastLine.found && lineAssistReady) {
    telemetryConfidence = lineAssist.confidence;
  }
  bool failsafe = failsafeSerialLoss || estopLatched;
  Serial.print("TLM mode=");
  Serial.print(modeName(currentMode));
  Serial.print(" line_error=");
  Serial.print(pidState.filteredError, 3);
  Serial.print(" pid=");
  Serial.print(pidState.output, 3);
  Serial.print(" confidence=");
  Serial.print(telemetryConfidence, 3);
  Serial.print(" front=");
  Serial.print(lastObstacle.frontMm);
  Serial.print(" left=");
  Serial.print(lastObstacle.leftMm);
  Serial.print(" right=");
  Serial.print(lastObstacle.rightMm);
  Serial.print(" yaw=");
  Serial.print(readGyroYawDeg(), 2);
  Serial.print(" roll=0.00 pitch=0.00");
  Serial.print(" gripper=90");
  Serial.print(" green=");
  Serial.print(greenAssist.active ? greenAssist.instruction : "NO_GREEN");
  Serial.print(" obstacle=");
  Serial.print(obstacleAssist.active ? obstacleAssist.state : "CLEAR");
  Serial.print(" failsafe=");
  Serial.print(failsafe ? 1 : 0);
  Serial.print(" left_pwm=");
  Serial.print(lastLeftPwm);
  Serial.print(" right_pwm=");
  Serial.println(lastRightPwm);
}

const char* modeName(ControlMode mode) {
  switch (mode) {
    case MODE_FOLLOW_LINE: return "FOLLOW_LINE";
    case MODE_GREEN_MANEUVER: return "GREEN";
    case MODE_OBSTACLE_AVOID: return "OBSTACLE";
    case MODE_MANUAL_FORWARD: return "MANUAL";
    case MODE_ESTOP: return "ESTOP";
    case MODE_SAFE_STOP:
    default:
      return "SAFE_STOP";
  }
}
