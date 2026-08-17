from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Palette:
    bg: str = "#1F2126"
    surface: str = "#272A31"
    surface_alt: str = "#2D3037"
    border: str = "#3A3F48"
    text: str = "#E8EAF0"
    text_dim: str = "#AEB5C2"
    accent: str = "#33D17A"
    danger: str = "#E24E5D"
    warning: str = "#F4C95D"
    chip_bg: str = "#171A1F"


PALETTE = Palette()


def build_dashboard_stylesheet(scale: float = 1.0) -> str:
    p = PALETTE
    factor = max(0.8, min(1.2, float(scale)))

    def px(value: int) -> int:
        return max(1, int(round(value * factor)))

    return f"""
QWidget#DashboardRoot {{
    background-color: {p.bg};
    color: {p.text};
    font-family: "Segoe UI";
    font-size: {px(12)}px;
}}
QFrame#CardFrame {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: 10px;
}}
QFrame#ControlCenterCard {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: 10px;
}}
QFrame#QuickSection {{
    background-color: {p.chip_bg};
    border: 1px solid {p.border};
    border-radius: 7px;
}}
QLabel#QuickSectionTitle {{
    color: {p.text_dim};
    font-size: {px(11)}px;
    font-weight: 700;
}}
QLabel#QuickHint {{
    color: {p.text_dim};
    font-size: {px(11)}px;
}}
QLabel#QuickStatus {{
    color: {p.text_dim};
    background-color: #11151B;
    border: 1px solid {p.border};
    border-radius: 5px;
    padding: {px(3)}px {px(7)}px;
    font-size: {px(10)}px;
    font-weight: 700;
}}
QLabel#QuickStatus[level="ok"] {{
    color: {p.accent};
    border-color: {p.accent};
}}
QLabel#QuickStatus[level="warn"] {{
    color: {p.warning};
    border-color: {p.warning};
}}
QLabel#QuickStatus[level="error"] {{
    color: {p.danger};
    border-color: {p.danger};
}}
QLabel#RobotRuntimeDetail {{
    color: {p.text};
    background-color: #11151B;
    border: 1px solid {p.border};
    border-radius: 6px;
    padding: {px(5)}px {px(8)}px;
    font-family: "Cascadia Mono", "Consolas";
    font-size: {px(11)}px;
}}
QWidget#HealthTile {{
    background-color: {p.chip_bg};
    border: 1px solid {p.border};
    border-radius: 6px;
}}
QLabel#CardTitle {{
    color: {p.text_dim};
    font-size: {px(11)}px;
    font-weight: 600;
    letter-spacing: 0.5px;
}}
QFrame#MetricPill {{
    background-color: {p.chip_bg};
    border: 1px solid {p.border};
    border-radius: 6px;
}}
QLabel#MetricKey {{
    color: {p.text_dim};
    font-size: {px(11)}px;
    font-weight: 600;
}}
QLabel#MetricValue {{
    color: {p.text};
    font-size: {px(13)}px;
    font-weight: 700;
}}
QLabel#HealthKey {{
    color: {p.text_dim};
    font-size: {px(11)}px;
    font-weight: 600;
}}
QLabel#HealthValue {{
    color: {p.text};
    font-size: {px(12)}px;
    font-weight: 700;
}}
QLabel#HealthValue[level="ok"] {{
    color: {p.accent};
}}
QLabel#HealthValue[level="warn"] {{
    color: {p.warning};
}}
QLabel#HealthValue[level="error"] {{
    color: {p.danger};
}}
QLabel#HealthDetail {{
    color: {p.text_dim};
    font-size: {px(11)}px;
}}
QFrame#StatusBadge {{
    background-color: {p.chip_bg};
    border: 1px solid {p.border};
    border-radius: 6px;
}}
QFrame#StatusBadge[active="true"] {{
    border: 1px solid {p.accent};
}}
QLabel#StatusName {{
    color: {p.text_dim};
    font-size: {px(10)}px;
    font-weight: 600;
}}
QLabel#StatusValue {{
    color: {p.text};
    font-size: {px(11)}px;
    font-weight: 700;
}}
QLabel#StatusValue[active="true"] {{
    color: {p.accent};
}}
QLabel#StatusValue[active="false"] {{
    color: {p.danger};
}}
QLabel#VideoLabel {{
    background-color: #101216;
    border: 1px solid {p.border};
    border-radius: 8px;
}}
QLabel#OverlayText {{
    background-color: rgba(16, 18, 22, 180);
    color: {p.text};
    border: 1px solid {p.border};
    border-radius: 6px;
    padding: {px(3)}px {px(8)}px;
    font-weight: 600;
}}
QLabel#StateHeadline {{
    font-size: {px(18)}px;
    font-weight: 700;
    color: {p.text};
}}
QLabel#StateSummary {{
    color: {p.text_dim};
    font-size: {px(13)}px;
}}
QLabel#TimerMain {{
    font-size: {px(42)}px;
    font-weight: 700;
    color: {p.text};
}}
QLabel#TimerSecondary {{
    font-size: {px(26)}px;
    font-weight: 600;
    color: {p.text_dim};
}}
QListWidget#TransitionList {{
    background-color: #1A1D23;
    border: 1px solid {p.border};
    border-radius: 8px;
    color: {p.text};
}}
QGroupBox {{
    border: 1px solid {p.border};
    border-radius: 8px;
    margin-top: {px(10)}px;
    padding-top: {px(10)}px;
    font-weight: 600;
    color: {p.text_dim};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: {px(8)}px;
    padding: 0 {px(4)}px;
}}
QSpinBox, QDoubleSpinBox {{
    background-color: #151920;
    border: 1px solid {p.border};
    border-radius: 4px;
    padding: {px(3)}px {px(4)}px;
    color: {p.text};
}}
QPushButton {{
    background-color: #11151B;
    border: 1px solid {p.border};
    border-radius: 6px;
    padding: {px(5)}px {px(10)}px;
    color: {p.text};
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: #181D25;
}}
QToolButton#AdvancedToggleButton {{
    background-color: #11151B;
    border: 1px solid {p.border};
    border-radius: 6px;
    padding: {px(6)}px {px(10)}px;
    color: {p.text};
    font-weight: 700;
}}
QToolButton#AdvancedToggleButton:hover {{
    background-color: #181D25;
}}
QPushButton#PrimaryStartButton {{
    background-color: #B91C1C;
    border-color: #EF4444;
    color: #FFFFFF;
    min-height: {px(30)}px;
}}
QPushButton#PrimaryStartButton:hover {{
    background-color: #DC2626;
}}
QPushButton#PrimaryStopButton {{
    border-color: {p.danger};
    color: {p.danger};
    min-height: {px(30)}px;
}}
QPushButton#LedControlButton {{
    min-width: {px(62)}px;
    min-height: {px(26)}px;
}}
QPushButton#ZoomButton {{
    min-width: {px(30)}px;
    max-width: {px(34)}px;
    min-height: {px(26)}px;
    padding: {px(2)}px {px(6)}px;
}}
QLabel#ZoomValue {{
    color: {p.text};
    background-color: #11151B;
    border: 1px solid {p.border};
    border-radius: 5px;
    min-width: {px(42)}px;
    padding: {px(4)}px {px(6)}px;
    font-weight: 700;
}}
QPushButton#CaptureStartButton, QPushButton#CaptureStopButton {{
    background-color: #1769AA;
    border: 1px solid #64B5F6;
    color: #FFFFFF;
    min-height: {px(30)}px;
    font-weight: 700;
}}
QPushButton#CaptureStartButton:hover, QPushButton#CaptureStopButton:hover {{
    background-color: #1E88E5;
}}
QPushButton#CaptureStartButton:pressed, QPushButton#CaptureStopButton:pressed {{
    background-color: #0D47A1;
}}
QPushButton#CaptureStartButton:disabled, QPushButton#CaptureStopButton:disabled {{
    background-color: #26384A;
    border-color: #40566B;
    color: #9AAFC2;
}}
QCheckBox {{
    spacing: {px(6)}px;
}}
QCheckBox::indicator {{
    width: {px(14)}px;
    height: {px(14)}px;
}}
QCheckBox::indicator:unchecked {{
    border: 1px solid {p.border};
    background-color: #151920;
    border-radius: 3px;
}}
QCheckBox::indicator:checked {{
    border: 1px solid {p.accent};
    background-color: {p.accent};
    border-radius: 3px;
}}
QScrollArea {{
    border: none;
}}
QWidget#SteeringPanel {{
    background-color: transparent;
}}
QLabel#SteeringChip {{
    background-color: #12161C;
    border: 1px solid {p.border};
    border-radius: 4px;
    color: {p.text};
    min-height: {px(28)}px;
    max-height: {px(28)}px;
    font-size: {px(12)}px;
    font-weight: 700;
}}
QLabel#SteeringArrow {{
    background-color: #12161C;
    border: 1px solid {p.border};
    border-radius: 4px;
    color: {p.text};
    min-height: {px(28)}px;
    max-height: {px(28)}px;
    font-size: {px(16)}px;
    font-weight: 700;
}}
QLabel#SteeringText {{
    background-color: #12161C;
    border: 1px solid {p.border};
    border-radius: 4px;
    color: {p.text_dim};
    min-height: {px(24)}px;
    max-height: {px(24)}px;
    font-size: {px(11)}px;
    font-weight: 600;
}}
"""
