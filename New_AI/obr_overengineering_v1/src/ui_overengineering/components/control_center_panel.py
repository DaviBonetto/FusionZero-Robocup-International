from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from PyQt6.QtCore import QUrl, Qt, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .base import CardFrame


class ControlCenterPanel(CardFrame):
    """Compact home for the controls used during every field run."""

    robot_command_requested = pyqtSignal(str, dict)
    recording_command_requested = pyqtSignal(str, dict)
    zoom_requested = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Controle principal", parent)
        self.setObjectName("ControlCenterCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumHeight(154)
        self._last_recording_session_dir = ""

        sections = QHBoxLayout()
        sections.setContentsMargins(0, 0, 0, 0)
        sections.setSpacing(10)
        self.content_layout.addLayout(sections)

        robot_section = self._section(self)
        robot_layout = robot_section.layout()
        assert isinstance(robot_layout, QVBoxLayout)
        robot_header = QHBoxLayout()
        robot_header.setContentsMargins(0, 0, 0, 0)
        self._section_title(robot_header, "MOTORES", robot_section)
        self.robot_status_label = self._status_label("AGUARDANDO", robot_section)
        robot_header.addWidget(self.robot_status_label, 0, Qt.AlignmentFlag.AlignRight)
        robot_layout.addLayout(robot_header)

        robot_actions = QHBoxLayout()
        robot_actions.setContentsMargins(0, 0, 0, 0)
        robot_actions.setSpacing(6)
        self.robot_start_button = QPushButton("START ROBÔ", robot_section)
        self.robot_start_button.setObjectName("PrimaryStartButton")
        self.robot_start_button.setToolTip("Liga os LEDs e inicia o controle dos motores")
        self.robot_start_button.clicked.connect(
            lambda: self.robot_command_requested.emit("system.start", {})
        )
        robot_actions.addWidget(self.robot_start_button, 2)

        self.robot_stop_button = QPushButton("PARAR", robot_section)
        self.robot_stop_button.setObjectName("PrimaryStopButton")
        self.robot_stop_button.setToolTip("Para os motores e desliga os LEDs")
        self.robot_stop_button.clicked.connect(
            lambda: self.robot_command_requested.emit("system.stop", {})
        )
        robot_actions.addWidget(self.robot_stop_button, 1)

        self.clear_estop_button = QPushButton("LIMPAR ESTOP", robot_section)
        self.clear_estop_button.setToolTip("Libera o ESTOP depois de conferir a area")
        self.clear_estop_button.clicked.connect(
            lambda: self.robot_command_requested.emit("robot.clear_estop", {})
        )
        robot_actions.addWidget(self.clear_estop_button, 1)
        robot_layout.addLayout(robot_actions)
        sections.addWidget(robot_section, 5)

        capture_section = self._section(self)
        capture_layout = capture_section.layout()
        assert isinstance(capture_layout, QVBoxLayout)
        capture_header = QHBoxLayout()
        capture_header.setContentsMargins(0, 0, 0, 0)
        self._section_title(capture_header, "CAPTURA", capture_section)
        self.record_raw_checkbox = QCheckBox("Raw", capture_section)
        self.record_raw_checkbox.setChecked(True)
        self.record_raw_checkbox.setToolTip("Salvar frames brutos")
        capture_header.addWidget(self.record_raw_checkbox)
        self.record_processed_checkbox = QCheckBox("Processado", capture_section)
        self.record_processed_checkbox.setChecked(True)
        self.record_processed_checkbox.setToolTip("Salvar frames processados pela IA")
        capture_header.addWidget(self.record_processed_checkbox)
        every_label = QLabel("a cada", capture_section)
        every_label.setObjectName("QuickHint")
        capture_header.addWidget(every_label)
        self.record_every_spin = QSpinBox(capture_section)
        self.record_every_spin.setRange(1, 600)
        self.record_every_spin.setValue(5)
        self.record_every_spin.setSuffix(" frames")
        self.record_every_spin.setToolTip("Intervalo entre frames salvos")
        self.record_every_spin.setMaximumWidth(104)
        capture_header.addWidget(self.record_every_spin)
        self.recording_status_label = self._status_label("PRONTA", capture_section)
        capture_header.addWidget(self.recording_status_label)
        capture_layout.addLayout(capture_header)

        capture_actions = QHBoxLayout()
        capture_actions.setContentsMargins(0, 0, 0, 0)
        capture_actions.setSpacing(6)
        self.recording_start_button = QPushButton("INICIAR CAPTURA", capture_section)
        self.recording_start_button.setObjectName("CaptureStartButton")
        self.recording_start_button.clicked.connect(self._emit_recording_start)
        capture_actions.addWidget(self.recording_start_button, 2)

        self.recording_stop_button = QPushButton("PARAR CAPTURA", capture_section)
        self.recording_stop_button.setObjectName("CaptureStopButton")
        self.recording_stop_button.clicked.connect(
            lambda: self.recording_command_requested.emit("stop", {})
        )
        self.recording_stop_button.setEnabled(False)
        capture_actions.addWidget(self.recording_stop_button, 2)

        self.recording_apply_button = QPushButton("APLICAR", capture_section)
        self.recording_apply_button.setToolTip("Aplicar as opcoes sem iniciar uma captura")
        self.recording_apply_button.clicked.connect(
            lambda: self.recording_command_requested.emit(
                "configure", self.current_recording_options()
            )
        )
        capture_actions.addWidget(self.recording_apply_button, 1)

        self.recording_open_button = QPushButton("ABRIR PASTA", capture_section)
        self.recording_open_button.clicked.connect(self._open_recording_folder)
        self.recording_open_button.setEnabled(False)
        capture_actions.addWidget(self.recording_open_button, 1)
        capture_layout.addLayout(capture_actions)
        sections.addWidget(capture_section, 6)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(6)
        self.robot_detail_label = QLabel(
            "Aguardando dados do robo",
            self,
        )
        self.robot_detail_label.setObjectName("RobotRuntimeDetail")
        self.robot_detail_label.setWordWrap(True)
        self.robot_detail_label.setTextFormat(Qt.TextFormat.PlainText)
        self.robot_detail_label.setToolTip("Estado completo recebido do controlador")
        footer.addWidget(self.robot_detail_label, 1)

        self.leds_on_button = QPushButton("LED ON", self)
        self.leds_on_button.setObjectName("LedControlButton")
        self.leds_on_button.setToolTip("Ligar os LEDs do robo")
        self.leds_on_button.clicked.connect(
            lambda: self.robot_command_requested.emit("leds.on", {})
        )
        footer.addWidget(self.leds_on_button)

        self.leds_off_button = QPushButton("LED OFF", self)
        self.leds_off_button.setObjectName("LedControlButton")
        self.leds_off_button.setToolTip("Desligar os LEDs do robo")
        self.leds_off_button.clicked.connect(
            lambda: self.robot_command_requested.emit("leds.off", {})
        )
        footer.addWidget(self.leds_off_button)

        zoom_title = QLabel("ZOOM", self)
        zoom_title.setObjectName("QuickHint")
        footer.addWidget(zoom_title)
        self.zoom_out_button = QPushButton("-", self)
        self.zoom_out_button.setObjectName("ZoomButton")
        self.zoom_out_button.setAccessibleName("Diminuir zoom do dashboard")
        self.zoom_out_button.setToolTip("Diminuir o dashboard em 10%")
        self.zoom_out_button.clicked.connect(lambda: self.zoom_requested.emit(-10))
        footer.addWidget(self.zoom_out_button)
        self.zoom_value_label = QLabel("100%", self)
        self.zoom_value_label.setObjectName("ZoomValue")
        self.zoom_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.addWidget(self.zoom_value_label)
        self.zoom_in_button = QPushButton("+", self)
        self.zoom_in_button.setObjectName("ZoomButton")
        self.zoom_in_button.setAccessibleName("Aumentar zoom do dashboard")
        self.zoom_in_button.setToolTip("Aumentar o dashboard em 10%")
        self.zoom_in_button.clicked.connect(lambda: self.zoom_requested.emit(10))
        footer.addWidget(self.zoom_in_button)
        self.content_layout.addLayout(footer)

        # Let the two action rows contract with the dashboard zoom instead of
        # forcing the central workspace to grow a horizontal scrollbar.
        for button in (
            self.robot_start_button,
            self.robot_stop_button,
            self.clear_estop_button,
            self.recording_start_button,
            self.recording_stop_button,
            self.recording_apply_button,
            self.recording_open_button,
        ):
            policy = button.sizePolicy()
            policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
            button.setSizePolicy(policy)

    @staticmethod
    def _section(parent: QWidget) -> QFrame:
        section = QFrame(parent)
        section.setObjectName("QuickSection")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(7)
        return section

    @staticmethod
    def _section_title(layout: QHBoxLayout, text: str, parent: QWidget) -> None:
        label = QLabel(text, parent)
        label.setObjectName("QuickSectionTitle")
        layout.addWidget(label)
        layout.addStretch(1)

    @staticmethod
    def _status_label(text: str, parent: QWidget) -> QLabel:
        label = QLabel(text, parent)
        label.setObjectName("QuickStatus")
        label.setProperty("level", "neutral")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def current_recording_options(self) -> dict[str, Any]:
        return {
            "include_raw": bool(self.record_raw_checkbox.isChecked()),
            "include_processed": bool(self.record_processed_checkbox.isChecked()),
            "every_n_frames": int(self.record_every_spin.value()),
        }

    def _emit_recording_start(self) -> None:
        self.recording_command_requested.emit("start", self.current_recording_options())

    def update_recording_status(self, payload: Mapping[str, Any]) -> None:
        enabled = bool(payload.get("enabled", False))
        options = payload.get("options", {})
        if isinstance(options, Mapping):
            self.record_raw_checkbox.setChecked(
                bool(options.get("include_raw", self.record_raw_checkbox.isChecked()))
            )
            self.record_processed_checkbox.setChecked(
                bool(
                    options.get(
                        "include_processed",
                        self.record_processed_checkbox.isChecked(),
                    )
                )
            )
            self.record_every_spin.setValue(
                max(1, int(options.get("every_n_frames", self.record_every_spin.value())))
            )

        session_dir = str(payload.get("session_dir", "")).strip()
        if session_dir:
            self._last_recording_session_dir = session_dir

        if enabled:
            frames = int(payload.get("frame_count", 0))
            self._set_status(self.recording_status_label, f"GRAVANDO | {frames} FR", "ok")
        elif self._last_recording_session_dir:
            self._set_status(self.recording_status_label, "ULTIMA SESSAO PRONTA", "neutral")
        else:
            self._set_status(self.recording_status_label, "PRONTA", "neutral")

        self.recording_start_button.setEnabled(not enabled)
        self.recording_stop_button.setEnabled(enabled)
        self.recording_open_button.setEnabled(self._recording_session_dir_path() is not None)
        self.recording_open_button.setToolTip(self._last_recording_session_dir)

    def update_robot_status(self, payload: Mapping[str, Any]) -> None:
        state = str(payload.get("state", "")).strip().lower()
        armed = bool(payload.get("motor_armed", False))
        failsafe = bool(payload.get("failsafe", False))
        control_mode = str(payload.get("control_mode", "")).strip().upper()
        if failsafe:
            self._set_status(self.robot_status_label, "FAILSAFE", "error")
        elif armed:
            text = "ARMADO" if not control_mode else f"ARMADO | {control_mode}"
            self._set_status(self.robot_status_label, text, "ok")
        elif state in {"connected", "dry-run", "waiting"}:
            self._set_status(self.robot_status_label, "DESARMADO", "warn")
        elif state:
            self._set_status(self.robot_status_label, state.upper(), "error")
        else:
            self._set_status(self.robot_status_label, "AGUARDANDO", "neutral")

        detail_parts = ["ARMED" if armed else "DISARMED"]
        if control_mode:
            detail_parts.append(control_mode)
        obstacle_state = str(payload.get("obstacle_state", "")).strip().upper()
        if obstacle_state and obstacle_state not in {"CLEAR", "NONE"}:
            detail_parts.append(f"OBS {obstacle_state}")
        green_instruction = str(payload.get("green_instruction", "")).strip().upper()
        if green_instruction and green_instruction not in {"NO_GREEN", "NONE"}:
            detail_parts.append(f"G {green_instruction}")
        route = str(payload.get("green_route_decision", "")).strip().upper()
        if route:
            detail_parts.append(f"ROUTE {route}")
        master_switch = payload.get("master_switch", {})
        if isinstance(master_switch, Mapping) and bool(master_switch.get("enabled", False)):
            switch_state = str(master_switch.get("state", "")).strip().upper()
            switch_gpio = master_switch.get("gpio")
            if switch_state:
                switch_label = f"SW {switch_state}"
                if switch_gpio is not None:
                    switch_label = f"SW GPIO{int(switch_gpio)} {switch_state}"
                detail_parts.append(switch_label)
        if failsafe:
            detail_parts.append("FAILSAFE")
        line_error = self._safe_number(payload.get("line_error"))
        pid_output = self._safe_number(payload.get("pid_output"))
        if line_error is not None:
            detail_parts.append(f"E {line_error:+.2f}")
        if pid_output is not None:
            detail_parts.append(f"PID {pid_output:+.1f}")
        left_pwm = payload.get("left_pwm")
        right_pwm = payload.get("right_pwm")
        if left_pwm is not None and right_pwm is not None:
            try:
                detail_parts.append(f"PWM L {int(left_pwm)} | R {int(right_pwm)} us")
            except (TypeError, ValueError):
                pass
        detail = " | ".join(detail_parts)
        self.robot_detail_label.setText(detail)
        self.robot_detail_label.setToolTip(detail)

    def set_zoom_percent(self, percent: int) -> None:
        value = max(80, min(120, int(percent)))
        self.zoom_value_label.setText(f"{value}%")
        self.zoom_out_button.setEnabled(value > 80)
        self.zoom_in_button.setEnabled(value < 120)

    @staticmethod
    def _safe_number(value: Any) -> float | None:
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _set_status(label: QLabel, text: str, level: str) -> None:
        label.setText(text)
        label.setProperty("level", level)
        label.style().unpolish(label)
        label.style().polish(label)

    def _recording_session_dir_path(self) -> Path | None:
        if not self._last_recording_session_dir:
            return None
        try:
            candidate = Path(self._last_recording_session_dir)
        except Exception:
            return None
        return candidate if candidate.exists() else None

    def _open_recording_folder(self) -> None:
        session_dir = self._recording_session_dir_path()
        if session_dir is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(session_dir)))
