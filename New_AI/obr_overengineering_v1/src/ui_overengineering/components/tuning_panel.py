from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from PyQt6.QtCore import QEvent, QObject, QTimer, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QGuiApplication
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .base import CardFrame


class TuningPanel(CardFrame):
    parameter_changed = pyqtSignal(str, object)
    camera_reconnect_requested = pyqtSignal(dict)
    mock_mode_changed = pyqtSignal(bool)
    mode_switch_requested = pyqtSignal(str)
    robot_command_requested = pyqtSignal(str, dict)
    profile_apply_requested = pyqtSignal(str)
    recording_command_requested = pyqtSignal(str, dict)
    calibration_changed = pyqtSignal(dict)
    calibration_snapshot_requested = pyqtSignal(dict)
    corner_timing_apply_requested = pyqtSignal(dict)
    green_half_turn_apply_requested = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None, *, show_primary_controls: bool = True) -> None:
        super().__init__("Operations", parent)
        self._controls: dict[str, QSpinBox | QDoubleSpinBox] = {}
        self._defaults: dict[str, float | int] = {}
        self._profiles: dict[str, dict[str, Any]] = {}
        self._last_recording_session_dir = ""
        self._corner_timing_pending: tuple[int, int] | None = None
        self._corner_timing_loaded = False
        self._green_half_turn_pending: tuple[int, int, int] | None = None
        self._green_half_turn_loaded = False

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_content = QWidget(self.scroll_area)
        self._scroll_content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        root = QVBoxLayout(self._scroll_content)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        root.addWidget(self._build_profile_group(self._scroll_content))
        root.addWidget(self._build_camera_group(self._scroll_content))
        root.addWidget(self._build_calibration_group(self._scroll_content))
        self.recording_group = self._build_recording_group(self._scroll_content)
        root.addWidget(self.recording_group)
        root.addWidget(self._build_line_group(self._scroll_content))
        root.addWidget(self._build_pid_group(self._scroll_content))
        root.addWidget(self._build_corner_timing_group(self._scroll_content))
        root.addWidget(self._build_green_half_turn_group(self._scroll_content))
        root.addWidget(self._build_color_group(self._scroll_content))
        root.addWidget(self._build_ball_group(self._scroll_content))
        self.robot_group = self._build_robot_group(self._scroll_content)
        root.addWidget(self.robot_group)
        root.addWidget(self._build_actions_group(self._scroll_content))
        root.addStretch(1)

        self.recording_group.setVisible(show_primary_controls)
        self.robot_group.setVisible(show_primary_controls)

        self.scroll_area.setWidget(self._scroll_content)
        self.scroll_area.viewport().installEventFilter(self)
        self.content_layout.addWidget(self.scroll_area, 1)
        QTimer.singleShot(0, self._sync_scroll_content_width)

    def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:  # type: ignore[name-defined]
        if watched is self.scroll_area.viewport() and event is not None and event.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self._sync_scroll_content_width)
        return super().eventFilter(watched, event)

    def _build_profile_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Profiles", parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.profile_combo = QComboBox(group)
        self.profile_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.profile_combo.setMinimumContentsLength(14)
        layout.addWidget(self.profile_combo)

        self.apply_profile_button = QPushButton("Apply profile", group)
        self.apply_profile_button.clicked.connect(self._emit_profile_apply)
        layout.addWidget(self.apply_profile_button)

        self.profile_status_label = self._build_detail_label(group, "Using manual base config")
        layout.addWidget(self.profile_status_label)
        return self._finalize_group(group)

    def _build_camera_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Camera", parent)
        layout = QVBoxLayout(group)
        form = QFormLayout()
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)

        self._add_spin(form, "camera.index", "Index", 0, 8, 0)
        self._add_spin(form, "camera.width", "Width", 160, 1920, 640, step=16)
        self._add_spin(form, "camera.height", "Height", 120, 1080, 480, step=16)
        self._add_spin(form, "camera.fps", "FPS", 1, 120, 30)
        layout.addLayout(form)

        self.mock_checkbox = QCheckBox("Mock mode", group)
        self.mock_checkbox.setChecked(False)
        self.mock_checkbox.toggled.connect(self.mock_mode_changed.emit)
        layout.addWidget(self.mock_checkbox)

        reconnect = QPushButton("Reconnect camera", group)
        reconnect.clicked.connect(self._emit_camera_reconnect)
        layout.addWidget(reconnect)
        return self._finalize_group(group)

    def _build_calibration_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Calibration", parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.calibration_view_combo = QComboBox(group)
        self.calibration_view_combo.addItem("Raw frame", "raw")
        self.calibration_view_combo.addItem("Processed view", "processed")
        self.calibration_view_combo.addItem("Line mask", "line_mask")
        self.calibration_view_combo.addItem("Green mask", "green_mask")
        self.calibration_view_combo.addItem("Red mask", "red_mask")
        self.calibration_view_combo.addItem("Victim mask", "victim_mask")
        self.calibration_view_combo.addItem("Composite debug", "composite")
        self.calibration_view_combo.currentIndexChanged.connect(self._emit_calibration_changed)
        layout.addWidget(self.calibration_view_combo)

        self.freeze_checkbox = QCheckBox("Freeze current frame", group)
        self.freeze_checkbox.toggled.connect(self._emit_calibration_changed)
        layout.addWidget(self.freeze_checkbox)

        snapshot_button = QPushButton("Audit snapshot", group)
        snapshot_button.clicked.connect(
            lambda: self.calibration_snapshot_requested.emit(self.current_calibration_options())
        )
        layout.addWidget(snapshot_button)

        self.calibration_status_label = self._build_detail_label(group, "Raw frame live")
        layout.addWidget(self.calibration_status_label)
        return self._finalize_group(group)

    def _build_recording_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Session Recording", parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.record_raw_checkbox = QCheckBox("Sample raw frames", group)
        self.record_raw_checkbox.setChecked(True)
        layout.addWidget(self.record_raw_checkbox)

        self.record_processed_checkbox = QCheckBox("Sample processed frames", group)
        self.record_processed_checkbox.setChecked(True)
        layout.addWidget(self.record_processed_checkbox)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        self.record_every_spin = QSpinBox(group)
        self.record_every_spin.setMinimum(1)
        self.record_every_spin.setMaximum(600)
        self.record_every_spin.setValue(5)
        form.addRow("Every N frames", self.record_every_spin)
        layout.addLayout(form)

        start_row = QHBoxLayout()
        start_row.setContentsMargins(0, 0, 0, 0)
        start_row.setSpacing(6)
        self.recording_start_button = QPushButton("INICIAR CAPTURA", group)
        self.recording_start_button.setObjectName("CaptureStartButton")
        self.recording_start_button.setToolTip("Inicia a captura de frames brutos e processados da linha")
        self.recording_start_button.clicked.connect(
            lambda: self.recording_command_requested.emit("start", self.current_recording_options())
        )
        start_row.addWidget(self.recording_start_button)

        self.recording_apply_button = QPushButton("Apply options", group)
        self.recording_apply_button.clicked.connect(
            lambda: self.recording_command_requested.emit("configure", self.current_recording_options())
        )
        start_row.addWidget(self.recording_apply_button)
        layout.addLayout(start_row)

        stop_row = QHBoxLayout()
        stop_row.setContentsMargins(0, 0, 0, 0)
        stop_row.setSpacing(6)
        self.recording_stop_button = QPushButton("PARAR CAPTURA", group)
        self.recording_stop_button.setObjectName("CaptureStopButton")
        self.recording_stop_button.setToolTip("Finaliza a captura e fecha a sessão no Raspberry Pi")
        self.recording_stop_button.clicked.connect(lambda: self.recording_command_requested.emit("stop", {}))
        stop_row.addWidget(self.recording_stop_button)

        self.recording_open_button = QPushButton("Open folder", group)
        self.recording_open_button.clicked.connect(self._open_recording_folder)
        stop_row.addWidget(self.recording_open_button)
        layout.addLayout(stop_row)

        self.recording_copy_button = QPushButton("Copy path", group)
        self.recording_copy_button.clicked.connect(self._copy_recording_path)
        layout.addWidget(self.recording_copy_button)

        self.recording_status_label = self._build_detail_label(group, "Recorder idle")
        layout.addWidget(self.recording_status_label)

        self.recording_path_label = self._build_detail_label(group, "No session folder yet", selectable=True)
        layout.addWidget(self.recording_path_label)
        self._refresh_recording_buttons(active=False)
        return self._finalize_group(group)

    def _build_line_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Line Detector", parent)
        form = QFormLayout(group)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)
        self._add_spin(form, "line.black_v_max", "Black V max", 0, 255, 70)
        self._add_spin(form, "line.black_s_max", "Black S max", 0, 255, 255)
        self._add_spin(form, "line.min_area", "Min area", 1, 4000, 50)
        self._add_spin(form, "line.erode_iter", "Erode iter", 0, 8, 3)
        self._add_spin(form, "line.dilate_iter", "Dilate iter", 0, 8, 4)
        return self._finalize_group(group)

    def _build_pid_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Line PID / Steering", parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        self._add_double(form, "control.pid.kp_us", "Kp", 0.0, 2000.0, 320.0, 10.0)
        self._add_double(form, "control.pid.ki_us", "Ki", 0.0, 200.0, 0.0, 1.0)
        self._add_double(form, "control.pid.kd_us", "Kd", 0.0, 500.0, 12.0, 1.0)
        self._add_double(form, "control.pid.integral_limit", "I limit", 0.0, 1.0, 0.15, 0.01)
        self._add_double(form, "control.pid.derivative_filter", "Filtro D", 0.0, 0.99, 0.60, 0.05)
        self._add_spin(form, "control.pid.max_output_us", "Limite correcao", 0, 1000, 240, step=10)
        self._add_spin(form, "control.line_hold_ms", "Hold (ms)", 0, 2000, 120, step=10)
        self._add_spin(form, "control.base.left_us", "Base esquerda", 0, 500, 300, step=10)
        self._add_spin(form, "control.base.right_us", "Base direita", 0, 500, 200, step=10)
        self._add_double(form, "control.line.deadband", "Zona neutra", 0.0, 0.25, 0.025, 0.005)
        layout.addLayout(form)

        explanation = self._build_detail_label(
            group,
            "Kp: corrige imediatamente conforme a linha se afasta do centro.\n"
            "Ki: corrige um erro pequeno que permanece; mantenha baixo para nao acumular desvio.\n"
            "Kd: freia mudancas rapidas e reduz a oscilacao.\n"
            "Filtro D: maior suaviza o Kd; menor deixa a resposta mais rapida e sensivel.\n"
            "I limit: limita quanto o Ki pode acumular. Limite correcao: impede trancos.\n"
            "Hold: mantem por poucos ms a ultima correcao durante uma falha breve de frame.\n"
            "Base esquerda/direita: velocidade de frente de cada lado; ajuste de 10 em 10 para compensar drift.\n"
            "Zona neutra: ignora pequenos erros visuais no centro e manda frente reto.",
        )
        explanation.setToolTip("Os valores sao enviados ao Raspberry Pi assim que cada campo muda.")
        layout.addWidget(explanation)

        preset_label = self._build_detail_label(group, "Testes prontos: clique uma vez e os valores sao enviados ao Pi.")
        layout.addWidget(preset_label)
        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(6)
        presets = (
            (
                "Teste 1\ncalmo",
                {
                    "control.pid.kp_us": 240.0,
                    "control.pid.ki_us": 0.0,
                    "control.pid.kd_us": 24.0,
                    "control.pid.integral_limit": 0.10,
                    "control.pid.derivative_filter": 0.65,
                    "control.pid.max_output_us": 200,
                    "control.line_hold_ms": 100,
                    "control.base.left_us": 300,
                    "control.base.right_us": 200,
                    "control.line.deadband": 0.030,
                },
                "Kp 240 | Ki 0 | Kd 24 | filtro 0,65 | limite 200 | hold 100 ms",
            ),
            (
                "Teste 2\nequilibrado",
                {
                    "control.pid.kp_us": 300.0,
                    "control.pid.ki_us": 0.0,
                    "control.pid.kd_us": 22.0,
                    "control.pid.integral_limit": 0.12,
                    "control.pid.derivative_filter": 0.60,
                    "control.pid.max_output_us": 220,
                    "control.line_hold_ms": 120,
                    "control.base.left_us": 300,
                    "control.base.right_us": 200,
                    "control.line.deadband": 0.025,
                },
                "Kp 300 | Ki 0 | Kd 22 | filtro 0,60 | limite 220 | hold 120 ms",
            ),
            (
                "Teste 3\nrapido",
                {
                    "control.pid.kp_us": 380.0,
                    "control.pid.ki_us": 0.0,
                    "control.pid.kd_us": 28.0,
                    "control.pid.integral_limit": 0.15,
                    "control.pid.derivative_filter": 0.50,
                    "control.pid.max_output_us": 260,
                    "control.line_hold_ms": 120,
                    "control.base.left_us": 300,
                    "control.base.right_us": 200,
                    "control.line.deadband": 0.020,
                },
                "Kp 380 | Ki 0 | Kd 28 | filtro 0,50 | limite 260 | hold 120 ms",
            ),
        )
        for text, values, tooltip in presets:
            button = QPushButton(text, group)
            button.setMinimumHeight(42)
            button.setToolTip(tooltip)
            button.clicked.connect(
                lambda _checked=False, preset=values: self.apply_values(preset, emit_changes=True)
            )
            preset_row.addWidget(button, 1)
        layout.addLayout(preset_row)
        return self._finalize_group(group)

    def _build_corner_timing_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Curva de 90 graus - esquerda", parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        self.corner_advance_ms_spin = QSpinBox(group)
        self.corner_advance_ms_spin.setObjectName("CornerAdvanceMsSpin")
        self.corner_advance_ms_spin.setRange(0, 1000)
        self.corner_advance_ms_spin.setSingleStep(25)
        self.corner_advance_ms_spin.setValue(550)
        self._add_stepper_row(form, "Avanco esquerdo (ms)", self.corner_advance_ms_spin)

        self.corner_pivot_ms_spin = QSpinBox(group)
        self.corner_pivot_ms_spin.setObjectName("CornerPivotMsSpin")
        self.corner_pivot_ms_spin.setRange(250, 4000)
        self.corner_pivot_ms_spin.setSingleStep(50)
        self.corner_pivot_ms_spin.setValue(2100)
        self._add_stepper_row(form, "Giro esquerdo (ms)", self.corner_pivot_ms_spin)
        layout.addLayout(form)

        self.corner_timing_apply_button = QPushButton("APLICAR CURVA ESQUERDA", group)
        self.corner_timing_apply_button.setObjectName("CornerTimingApplyButton")
        self.corner_timing_apply_button.setMinimumHeight(40)
        self.corner_timing_apply_button.setToolTip(
            "Aplica somente os dois tempos da curva esquerda no Raspberry Pi."
        )
        self.corner_timing_apply_button.clicked.connect(self._emit_corner_timing_apply)
        layout.addWidget(self.corner_timing_apply_button)

        explanation = self._build_detail_label(
            group,
            "Avanco: deslocamento reto depois de confirmar o L e antes da parada.\n"
            "Giro: duracao do pivô no proprio eixo; vale para esquerda e direita.",
        )
        explanation.setText(
            "Avanco: movimento reto depois de confirmar o L e antes da parada.\n"
            "Giro: limite do pivo no proprio eixo, somente para a esquerda."
        )
        layout.addWidget(explanation)

        self.corner_timing_status_label = self._build_detail_label(
            group,
            "Aguardando telemetria do robo",
        )
        self.corner_timing_status_label.setObjectName("CornerTimingStatus")
        layout.addWidget(self.corner_timing_status_label)
        return self._finalize_group(group)

    def _build_green_half_turn_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Giro de 180 graus - dois verdes", parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        self.green_half_turn_first_ms_spin = QSpinBox(group)
        self.green_half_turn_first_ms_spin.setObjectName("GreenHalfTurnFirstMsSpin")
        self.green_half_turn_first_ms_spin.setRange(250, 4000)
        self.green_half_turn_first_ms_spin.setSingleStep(50)
        self.green_half_turn_first_ms_spin.setValue(1900)
        self._add_stepper_row(form, "Primeiro giro (ms)", self.green_half_turn_first_ms_spin)

        self.green_half_turn_reverse_ms_spin = QSpinBox(group)
        self.green_half_turn_reverse_ms_spin.setObjectName("GreenHalfTurnReverseMsSpin")
        self.green_half_turn_reverse_ms_spin.setRange(0, 1500)
        self.green_half_turn_reverse_ms_spin.setSingleStep(50)
        self.green_half_turn_reverse_ms_spin.setValue(550)
        self._add_stepper_row(form, "Re apos primeiro giro (ms)", self.green_half_turn_reverse_ms_spin)

        self.green_half_turn_second_ms_spin = QSpinBox(group)
        self.green_half_turn_second_ms_spin.setObjectName("GreenHalfTurnSecondMsSpin")
        self.green_half_turn_second_ms_spin.setRange(250, 4000)
        self.green_half_turn_second_ms_spin.setSingleStep(50)
        self.green_half_turn_second_ms_spin.setValue(2100)
        self._add_stepper_row(form, "Segundo giro (ms)", self.green_half_turn_second_ms_spin)
        layout.addLayout(form)

        self.green_half_turn_apply_button = QPushButton("APLICAR GIRO 180 GRAUS", group)
        self.green_half_turn_apply_button.setObjectName("GreenHalfTurnApplyButton")
        self.green_half_turn_apply_button.setMinimumHeight(40)
        self.green_half_turn_apply_button.setToolTip(
            "Aplica os dois giros e a pequena re intermediaria no Raspberry Pi."
        )
        self.green_half_turn_apply_button.clicked.connect(
            self._emit_green_half_turn_apply
        )
        layout.addWidget(self.green_half_turn_apply_button)

        explanation = self._build_detail_label(
            group,
            "Sequencia: parar, primeiro giro, re curta, parar, segundo giro e retomar a linha.",
        )
        layout.addWidget(explanation)

        self.green_half_turn_status_label = self._build_detail_label(
            group,
            "Aguardando telemetria do robo",
        )
        self.green_half_turn_status_label.setObjectName("GreenHalfTurnStatus")
        layout.addWidget(self.green_half_turn_status_label)
        return self._finalize_group(group)

    def _build_color_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Markers", parent)
        form = QFormLayout(group)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)
        self._add_spin(form, "green.h_min", "Green H min", 0, 179, 35)
        self._add_spin(form, "green.h_max", "Green H max", 0, 179, 90)
        self._add_spin(form, "green.s_min", "Green S min", 0, 255, 70)
        self._add_spin(form, "green.v_min", "Green V min", 0, 255, 50)
        self._add_spin(form, "green.min_area", "Green area", 1, 6000, 180)

        self._add_spin(form, "red.h1_min", "Red H1 min", 0, 179, 0)
        self._add_spin(form, "red.h1_max", "Red H1 max", 0, 179, 12)
        self._add_spin(form, "red.h2_min", "Red H2 min", 0, 179, 165)
        self._add_spin(form, "red.h2_max", "Red H2 max", 0, 179, 179)
        self._add_spin(form, "red.s_min", "Red S min", 0, 255, 120)
        self._add_spin(form, "red.v_min", "Red V min", 0, 255, 80)
        self._add_spin(form, "red.min_area", "Red area", 1, 6000, 300)
        return self._finalize_group(group)

    def _build_ball_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Balls / Victims", parent)
        form = QFormLayout(group)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)
        self._add_double(form, "silver.conf", "Silver conf", 0.0, 1.0, 0.95, 0.01)
        self._add_spin(form, "silver.blur", "Silver blur", 3, 31, 7, step=2)
        self._add_spin(form, "dead.black_v_max", "Black V max", 0, 120, 60)
        return self._finalize_group(group)

    def _build_robot_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Robot", parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        self.robot_forward_ms_spin = QSpinBox(group)
        self.robot_forward_ms_spin.setMinimum(100)
        self.robot_forward_ms_spin.setMaximum(15000)
        self.robot_forward_ms_spin.setSingleStep(100)
        self.robot_forward_ms_spin.setValue(1200)
        form.addRow("Forward ms", self.robot_forward_ms_spin)
        layout.addLayout(form)

        self.robot_start_button = QPushButton("START MOTORES", group)
        self.robot_start_button.setObjectName("robotStartButton")
        self.robot_start_button.setStyleSheet(
            "QPushButton { background-color: #b91c1c; color: white; "
            "font-weight: 700; padding: 8px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #dc2626; }"
            "QPushButton:pressed { background-color: #7f1d1d; }"
        )
        self.robot_start_button.clicked.connect(self._emit_robot_start)
        layout.addWidget(self.robot_start_button)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(6)

        forward_button = QPushButton("Forward test", group)
        forward_button.clicked.connect(self._emit_robot_forward_test)
        button_row.addWidget(forward_button)

        reverse_button = QPushButton("Reverse test", group)
        reverse_button.clicked.connect(self._emit_robot_reverse_test)
        button_row.addWidget(reverse_button)

        stop_button = QPushButton("STOP", group)
        stop_button.clicked.connect(self._emit_robot_stop)
        button_row.addWidget(stop_button)
        layout.addLayout(button_row)

        estop_row = QHBoxLayout()
        estop_row.setContentsMargins(0, 0, 0, 0)
        estop_row.setSpacing(6)

        force_stop_button = QPushButton("Force STOP", group)
        force_stop_button.clicked.connect(self._emit_robot_force_stop)
        estop_row.addWidget(force_stop_button)

        clear_estop_button = QPushButton("Clear ESTOP", group)
        clear_estop_button.clicked.connect(self._emit_robot_clear_estop)
        estop_row.addWidget(clear_estop_button)
        layout.addLayout(estop_row)

        obstacle_row = QHBoxLayout()
        obstacle_row.setContentsMargins(0, 0, 0, 0)
        obstacle_row.setSpacing(6)

        obstacle_test_button = QPushButton("Obstacle test", group)
        obstacle_test_button.clicked.connect(self._emit_robot_obstacle_test)
        obstacle_row.addWidget(obstacle_test_button)

        obstacle_clear_button = QPushButton("Clear obstacle", group)
        obstacle_clear_button.clicked.connect(self._emit_robot_obstacle_clear)
        obstacle_row.addWidget(obstacle_clear_button)
        layout.addLayout(obstacle_row)

        led_row = QHBoxLayout()
        led_row.setContentsMargins(0, 0, 0, 0)
        led_row.setSpacing(6)

        leds_on_button = QPushButton("LEDs ON", group)
        leds_on_button.clicked.connect(self._emit_leds_on)
        led_row.addWidget(leds_on_button)

        leds_off_button = QPushButton("LEDs OFF", group)
        leds_off_button.clicked.connect(self._emit_leds_off)
        led_row.addWidget(leds_off_button)

        led1_button = QPushButton("LED1", group)
        led1_button.clicked.connect(self._emit_led1_toggle)
        led_row.addWidget(led1_button)

        led2_button = QPushButton("LED2", group)
        led2_button.clicked.connect(self._emit_led2_toggle)
        led_row.addWidget(led2_button)
        layout.addLayout(led_row)

        self.robot_status_label = self._build_detail_label(
            group,
            "Robot commands and obstacle test triggers are forwarded as UI_COMMAND events to the live runner.",
        )
        layout.addWidget(self.robot_status_label)
        return self._finalize_group(group)

    def _build_actions_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Actions", parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(6)
        force_line = QPushButton("Force LINE", group)
        force_line.clicked.connect(lambda: self.mode_switch_requested.emit("line"))
        mode_row.addWidget(force_line)

        force_rescue = QPushButton("Force RESCUE", group)
        force_rescue.clicked.connect(lambda: self.mode_switch_requested.emit("rescue"))
        mode_row.addWidget(force_rescue)
        layout.addLayout(mode_row)

        reset = QPushButton("Reset defaults", group)
        reset.clicked.connect(self.reset_defaults)
        layout.addWidget(reset)
        return self._finalize_group(group)

    def _build_detail_label(self, parent: QWidget, text: str, *, selectable: bool = False) -> QLabel:
        label = QLabel(text, parent)
        label.setObjectName("HealthDetail")
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        if selectable:
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    @staticmethod
    def _finalize_group(group: QGroupBox) -> QGroupBox:
        group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        return group

    def _sync_scroll_content_width(self) -> None:
        viewport_width = self.scroll_area.viewport().width()
        if viewport_width <= 0:
            return
        if self._scroll_content.width() != viewport_width:
            self._scroll_content.setFixedWidth(viewport_width)

    def _add_spin(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        minimum: int,
        maximum: int,
        value: int,
        step: int = 1,
    ) -> None:
        spin = QSpinBox(form.parentWidget())
        spin.setMinimum(minimum)
        spin.setMaximum(maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.valueChanged.connect(lambda v, item=key: self.parameter_changed.emit(item, int(v)))
        self._add_stepper_row(form, label, spin)
        self._controls[key] = spin
        self._defaults[key] = int(value)

    def _add_double(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
    ) -> None:
        spin = QDoubleSpinBox(form.parentWidget())
        spin.setMinimum(minimum)
        spin.setMaximum(maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setDecimals(3 if step < 0.01 else 2)
        spin.valueChanged.connect(lambda v, item=key: self.parameter_changed.emit(item, float(v)))
        self._add_stepper_row(form, label, spin)
        self._controls[key] = spin
        self._defaults[key] = float(value)

    @staticmethod
    def _add_stepper_row(
        form: QFormLayout,
        label: str,
        spin: QSpinBox | QDoubleSpinBox,
    ) -> None:
        """Use large explicit +/- controls so touch/mouse clicks are reliable."""
        spin.setReadOnly(True)
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setMinimumHeight(36)
        spin.setToolTip("Use os botoes - e + para alterar este valor.")

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        minus = QPushButton("−", form.parentWidget())
        plus = QPushButton("+", form.parentWidget())
        for button in (minus, plus):
            button.setMinimumSize(44, 36)
            button.setMaximumWidth(52)
        minus.setToolTip("Diminuir")
        plus.setToolTip("Aumentar")

        def adjust(direction: int) -> None:
            next_value = spin.value() + (direction * spin.singleStep())
            if isinstance(spin, QDoubleSpinBox):
                spin.setValue(float(next_value))
            else:
                spin.setValue(int(next_value))

        minus.clicked.connect(lambda: adjust(-1))
        plus.clicked.connect(lambda: adjust(1))
        row.addWidget(spin, 1)
        row.addWidget(minus)
        row.addWidget(plus)
        container = QWidget(form.parentWidget())
        container.setLayout(row)
        form.addRow(label, container)

    def _emit_camera_reconnect(self) -> None:
        self.camera_reconnect_requested.emit(
            {
                "index": int(self.value("camera.index", 0)),
                "width": int(self.value("camera.width", 640)),
                "height": int(self.value("camera.height", 480)),
                "fps": int(self.value("camera.fps", 30)),
            }
        )

    def _emit_profile_apply(self) -> None:
        name = str(self.profile_combo.currentData() or "").strip().lower()
        if name:
            self.profile_apply_requested.emit(name)

    def _open_recording_folder(self) -> None:
        session_dir = self._recording_session_dir_path()
        if session_dir is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(session_dir)))

    def _copy_recording_path(self) -> None:
        session_dir = self._recording_session_dir_path()
        if session_dir is None:
            return
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(str(session_dir))

    def _emit_robot_forward_test(self) -> None:
        self.robot_command_requested.emit(
            "robot.forward_test",
            {"duration_ms": int(self.robot_forward_ms_spin.value())},
        )

    def _emit_corner_timing_apply(self) -> None:
        advance_ms = int(self.corner_advance_ms_spin.value())
        pivot_ms = int(self.corner_pivot_ms_spin.value())
        self._corner_timing_pending = (advance_ms, pivot_ms)
        self.corner_timing_status_label.setStyleSheet("color: #f0b429;")
        self.corner_timing_status_label.setText(
            f"Enviando esquerda: avanco {advance_ms} ms | giro {pivot_ms} ms"
        )
        self.corner_timing_apply_requested.emit(
            {"approach_left_min_ms": advance_ms, "pivot_left_ms": pivot_ms}
        )

    def _emit_green_half_turn_apply(self) -> None:
        first_ms = int(self.green_half_turn_first_ms_spin.value())
        reverse_ms = int(self.green_half_turn_reverse_ms_spin.value())
        second_ms = int(self.green_half_turn_second_ms_spin.value())
        self._green_half_turn_pending = (first_ms, reverse_ms, second_ms)
        self.green_half_turn_status_label.setStyleSheet("color: #f0b429;")
        self.green_half_turn_status_label.setText(
            f"Enviando: giro 1 {first_ms} ms | re {reverse_ms} ms | giro 2 {second_ms} ms"
        )
        self.green_half_turn_apply_requested.emit(
            {
                "first_ms": first_ms,
                "reverse_ms": reverse_ms,
                "second_ms": second_ms,
            }
        )

    def _emit_robot_start(self) -> None:
        self.robot_command_requested.emit("robot.start", {})

    def _emit_robot_reverse_test(self) -> None:
        self.robot_command_requested.emit(
            "robot.reverse_test",
            {"duration_ms": int(self.robot_forward_ms_spin.value())},
        )

    def _emit_robot_stop(self) -> None:
        self.robot_command_requested.emit("robot.stop", {})

    def _emit_robot_force_stop(self) -> None:
        self.robot_command_requested.emit("robot.force_stop", {})

    def _emit_robot_clear_estop(self) -> None:
        self.robot_command_requested.emit("robot.clear_estop", {})

    def _emit_robot_obstacle_test(self) -> None:
        self.robot_command_requested.emit("robot.obstacle_test", {})

    def _emit_robot_obstacle_clear(self) -> None:
        self.robot_command_requested.emit("robot.obstacle_clear", {})

    def _emit_leds_on(self) -> None:
        self.robot_command_requested.emit("leds.on", {})

    def _emit_leds_off(self) -> None:
        self.robot_command_requested.emit("leds.off", {})

    def _emit_led1_toggle(self) -> None:
        self.robot_command_requested.emit("led1.toggle", {})

    def _emit_led2_toggle(self) -> None:
        self.robot_command_requested.emit("led2.toggle", {})

    def _emit_calibration_changed(self) -> None:
        payload = self.current_calibration_options()
        view_name = str(self.calibration_view_combo.currentText()).strip()
        suffix = "frozen" if payload["freeze"] else "live"
        self.calibration_status_label.setText(f"{view_name} {suffix}")
        self.calibration_changed.emit(payload)

    def reset_defaults(self) -> None:
        self.apply_values(self._defaults, emit_changes=True)

    def set_mock_mode(self, enabled: bool) -> None:
        self.mock_checkbox.blockSignals(True)
        self.mock_checkbox.setChecked(bool(enabled))
        self.mock_checkbox.blockSignals(False)

    def set_profiles(self, profiles: list[Mapping[str, Any]], active_name: str | None = None) -> None:
        current_name = str(active_name or self.profile_combo.currentData() or "").strip().lower()
        self._profiles = {}
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for payload in profiles:
            if not isinstance(payload, Mapping):
                continue
            name = str(payload.get("name", "")).strip().lower()
            if not name:
                continue
            description = str(payload.get("description", name.replace("_", " ").title())).strip()
            self.profile_combo.addItem(name, name)
            self.profile_combo.setItemData(self.profile_combo.count() - 1, description, Qt.ItemDataRole.ToolTipRole)
            self._profiles[name] = dict(payload)
        self.profile_combo.blockSignals(False)

        if current_name:
            for idx in range(self.profile_combo.count()):
                if str(self.profile_combo.itemData(idx) or "").strip().lower() == current_name:
                    self.profile_combo.setCurrentIndex(idx)
                    break

    def set_active_profile(self, name: str, description: str = "") -> None:
        active = str(name or "custom").strip().lower() or "custom"
        for idx in range(self.profile_combo.count()):
            if str(self.profile_combo.itemData(idx) or "").strip().lower() == active:
                self.profile_combo.setCurrentIndex(idx)
                break
        text = description.strip() if description else active.replace("_", " ")
        self.profile_combo.setToolTip(text)
        self.profile_status_label.setText(f"Active profile: {active}\n{text}")
        self.profile_status_label.setToolTip(text)
        self._sync_scroll_content_width()

    def update_recording_status(self, payload: Mapping[str, Any]) -> None:
        enabled = bool(payload.get("enabled", False))
        options = payload.get("options", {})
        if isinstance(options, Mapping):
            self.record_raw_checkbox.setChecked(bool(options.get("include_raw", self.record_raw_checkbox.isChecked())))
            self.record_processed_checkbox.setChecked(
                bool(options.get("include_processed", self.record_processed_checkbox.isChecked()))
            )
            self.record_every_spin.setValue(max(1, int(options.get("every_n_frames", self.record_every_spin.value()))))

        session_dir = str(payload.get("session_dir", "")).strip()
        if session_dir:
            self._last_recording_session_dir = session_dir

        if enabled:
            event_count = int(payload.get("event_count", 0))
            frame_count = int(payload.get("frame_count", 0))
            summary = f"REC ON | {event_count} events | {frame_count} frames"
            dropped = int(payload.get("dropped_records", 0))
            if dropped > 0:
                summary = f"{summary} | dropped {dropped}"
        else:
            summary = "Recorder idle"
            if self._last_recording_session_dir:
                summary = f"{summary} | last session available"
        self.recording_status_label.setText(summary)
        if self._last_recording_session_dir:
            self.recording_path_label.setText(self._display_recording_path(self._last_recording_session_dir))
            self.recording_path_label.setToolTip(self._last_recording_session_dir)
        else:
            self.recording_path_label.setText("No session folder yet")
            self.recording_path_label.setToolTip("")
        self._refresh_recording_buttons(active=enabled)
        self._sync_scroll_content_width()

    def update_corner_timing_status(self, payload: Mapping[str, Any]) -> None:
        if payload.get("corner_approach_left_min_ms") is None:
            return
        if payload.get("corner_pivot_left_ms") is None:
            return
        advance_ms = int(payload.get("corner_approach_left_min_ms", 0))
        pivot_left_ms = int(payload.get("corner_pivot_left_ms", 0))

        if not self._corner_timing_loaded:
            self.corner_advance_ms_spin.setValue(advance_ms)
            self.corner_pivot_ms_spin.setValue(pivot_left_ms)
            self._corner_timing_loaded = True

        pending = self._corner_timing_pending
        if pending is not None:
            if advance_ms == pending[0] and pivot_left_ms == pending[1]:
                self._corner_timing_pending = None
                self.corner_advance_ms_spin.setValue(advance_ms)
                self.corner_pivot_ms_spin.setValue(pivot_left_ms)
                self.corner_timing_status_label.setStyleSheet("color: #35d07f;")
                self.corner_timing_status_label.setText(
                    f"APLICADO NA ESQUERDA: avanco {advance_ms} ms | giro {pivot_left_ms} ms"
                )
                return
            self.corner_timing_status_label.setStyleSheet("color: #f0b429;")
            self.corner_timing_status_label.setText(
                f"Aguardando confirmacao do Pi: {pending[0]} / {pending[1]} ms"
            )
            return

        self.corner_timing_status_label.setStyleSheet("")
        self.corner_timing_status_label.setText(
            f"Atual na esquerda: avanco {advance_ms} ms | giro {pivot_left_ms} ms"
        )

    def update_green_half_turn_status(self, payload: Mapping[str, Any]) -> None:
        if payload.get("green_half_turn_ms") is None:
            return
        duration_ms = int(payload.get("green_half_turn_ms", 0))
        left_us = payload.get("green_half_turn_left_us")
        right_us = payload.get("green_half_turn_right_us")
        first_ms = payload.get("green_half_turn_first_ms")
        second_ms = payload.get("green_half_turn_second_ms")
        reverse_ms = payload.get("green_half_turn_reverse_ms")
        balance_text = ""
        if left_us is not None and right_us is not None:
            balance_text = f" | pivo L {float(left_us):.0f} / R {float(right_us):.0f} us"
        if first_ms is not None and second_ms is not None and reverse_ms is not None:
            balance_text += (
                f" | giro 1 {int(first_ms)} ms | re {int(reverse_ms)} ms"
                f" | giro 2 {int(second_ms)} ms"
            )

        if not self._green_half_turn_loaded:
            if first_ms is not None:
                self.green_half_turn_first_ms_spin.setValue(int(first_ms))
            if reverse_ms is not None:
                self.green_half_turn_reverse_ms_spin.setValue(int(reverse_ms))
            if second_ms is not None:
                self.green_half_turn_second_ms_spin.setValue(int(second_ms))
            self._green_half_turn_loaded = True

        pending = self._green_half_turn_pending
        if pending is not None:
            current = (
                int(first_ms or 0),
                int(reverse_ms or 0),
                int(second_ms or 0),
            )
            if current == pending:
                self._green_half_turn_pending = None
                self.green_half_turn_first_ms_spin.setValue(current[0])
                self.green_half_turn_reverse_ms_spin.setValue(current[1])
                self.green_half_turn_second_ms_spin.setValue(current[2])
                self.green_half_turn_status_label.setStyleSheet("color: #35d07f;")
                self.green_half_turn_status_label.setText(
                    f"APLICADO NO PI{balance_text}"
                )
                return
            self.green_half_turn_status_label.setStyleSheet("color: #f0b429;")
            self.green_half_turn_status_label.setText(
                "Aguardando confirmacao do Pi: "
                f"giro 1 {pending[0]} | re {pending[1]} | giro 2 {pending[2]} ms"
            )
            return

        self.green_half_turn_status_label.setStyleSheet("")
        self.green_half_turn_status_label.setText(
            f"Atual no robo: total de giro {duration_ms} ms{balance_text}"
        )

    def current_recording_options(self) -> dict[str, Any]:
        return {
            "include_raw": bool(self.record_raw_checkbox.isChecked()),
            "include_processed": bool(self.record_processed_checkbox.isChecked()),
            "every_n_frames": int(self.record_every_spin.value()),
        }

    def _recording_session_dir_path(self) -> Path | None:
        session_dir = self._last_recording_session_dir.strip()
        if not session_dir:
            return None
        try:
            candidate = Path(session_dir)
        except Exception:
            return None
        return candidate if candidate.exists() else None

    def _refresh_recording_buttons(self, *, active: bool) -> None:
        has_known_path = bool(self._last_recording_session_dir.strip())
        local_path_available = self._recording_session_dir_path() is not None
        self.recording_start_button.setEnabled(not active)
        self.recording_apply_button.setEnabled(True)
        self.recording_stop_button.setEnabled(active)
        self.recording_open_button.setEnabled(local_path_available)
        self.recording_copy_button.setEnabled(has_known_path)

    @staticmethod
    def _display_recording_path(session_dir: str) -> str:
        cleaned = str(session_dir).strip()
        if not cleaned:
            return "No session folder yet"
        try:
            parts = [part for part in Path(cleaned).parts if part and part != Path(cleaned).anchor]
        except Exception:
            return cleaned
        if not parts:
            return cleaned
        tail = parts[-3:]
        head = "/".join(tail[:-1])
        if len(parts) > len(tail) and head:
            head = f".../{head}"
        elif len(parts) > len(tail):
            head = "..."
        if head:
            return f"{head}\n{tail[-1]}"
        return tail[-1]

    def current_calibration_options(self) -> dict[str, Any]:
        return {
            "view_mode": str(self.calibration_view_combo.currentData() or "raw"),
            "freeze": bool(self.freeze_checkbox.isChecked()),
        }

    def control_snapshot(self) -> dict[str, Any]:
        return {key: self.value(key, default) for key, default in self._defaults.items()}

    def apply_values(self, values: Mapping[str, Any], *, emit_changes: bool) -> None:
        for key, value in values.items():
            control = self._controls.get(str(key))
            if control is None:
                continue
            control.blockSignals(True)
            if isinstance(control, QDoubleSpinBox):
                control.setValue(float(value))
            else:
                control.setValue(int(value))
            control.blockSignals(False)
            if emit_changes:
                self.parameter_changed.emit(str(key), self.value(str(key), value))

    def set_camera_values(self, values: Mapping[str, Any]) -> None:
        camera_values = {
            "camera.index": int(values.get("index", self.value("camera.index", 0))),
            "camera.width": int(values.get("width", self.value("camera.width", 640))),
            "camera.height": int(values.get("height", self.value("camera.height", 480))),
            "camera.fps": int(values.get("fps", self.value("camera.fps", 30))),
        }
        self.apply_values(camera_values, emit_changes=False)

    def value(self, key: str, fallback: Any = None) -> Any:
        control = self._controls.get(key)
        if control is None:
            return fallback
        if isinstance(control, QDoubleSpinBox):
            return float(control.value())
        return int(control.value())
