"""
Apple-inspired UI Widgets for PySide6 with Light and Dark Theme Support.
Strictly NO emojis. Clean typography, Apple Blue accents (#007AFF / #0A84FF),
monochrome surfaces (white/black), and fluid transitions.
"""

from PySide6.QtCore import Qt, QRectF, QPropertyAnimation, Property, Signal
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QPainterPath
from PySide6.QtWidgets import (
    QWidget, QAbstractButton, QHBoxLayout, QVBoxLayout,
    QLabel, QSlider, QPushButton, QFrame, QSizePolicy
)


class AppleSwitch(QAbstractButton):
    """
    iOS / macOS native style toggle switch with Light and Dark theme support.
    Uses Apple System Blue (#007AFF / #0A84FF) active fill.
    """
    def __init__(self, parent=None, checked=False, theme="dark"):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(44, 26)
        self._theme = theme

        # 0.0 = completely left (unchecked), 1.0 = completely right (checked)
        self._thumb_position = 1.0 if checked else 0.0

        self._anim = QPropertyAnimation(self, b"thumb_position", self)
        self._anim.setDuration(160)
        self.toggled.connect(self._animate_toggle)

    def set_theme(self, theme: str):
        self._theme = theme
        self.update()

    @Property(float)
    def thumb_position(self):
        return self._thumb_position

    @thumb_position.setter
    def thumb_position(self, pos):
        self._thumb_position = pos
        self.update()

    def _animate_toggle(self, checked):
        self._anim.stop()
        self._anim.setStartValue(self._thumb_position)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def setChecked(self, checked):
        super().setChecked(checked)
        self._thumb_position = 1.0 if checked else 0.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        radius = h / 2.0

        # Background track color interpolation
        # Off -> On: Apple Blue (#007AFF in Light, #0A84FF in Dark)
        t = self._thumb_position
        if self._theme == "light":
            r_off, g_off, b_off = 229, 229, 234  # #E5E5EA
            r_on, g_on, b_on = 0, 122, 255      # #007AFF Apple Blue
        else:
            r_off, g_off, b_off = 58, 58, 60     # #3A3A3C
            r_on, g_on, b_on = 10, 132, 255     # #0A84FF Apple Blue

        r = int(r_off + (r_on - r_off) * t)
        g = int(g_off + (g_on - g_off) * t)
        b = int(b_off + (b_on - b_off) * t)

        track_color = QColor(r, g, b)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(track_color))
        painter.drawRoundedRect(0, 0, w, h, radius, radius)

        # Subtle border for off state in light mode
        if self._theme == "light" and t < 0.99:
            border_alpha = int(255 * (1.0 - t) * 0.4)
            painter.setPen(QPen(QColor(190, 190, 195, border_alpha), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(0.5, 0.5, w - 1, h - 1, radius, radius)

        # Thumb calculation
        thumb_diameter = h - 4
        x_min = 2
        x_max = w - thumb_diameter - 2
        thumb_x = x_min + (x_max - x_min) * t
        thumb_y = 2

        # Thumb drop shadow
        shadow_path = QPainterPath()
        shadow_path.addEllipse(QRectF(thumb_x, thumb_y + 1, thumb_diameter, thumb_diameter))
        shadow_alpha = 40 if self._theme == "dark" else 30
        painter.fillPath(shadow_path, QColor(0, 0, 0, shadow_alpha))

        # White Thumb
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(QRectF(thumb_x, thumb_y, thumb_diameter, thumb_diameter))


class AppleSegmentedControl(QWidget):
    """
    macOS style segmented capsule control (pill buttons in a track).
    Supports dynamic Light and Dark mode switching.
    """
    currentIndexChanged = Signal(int)
    currentTextChanged = Signal(str)

    def __init__(self, items=None, parent=None, theme="dark"):
        super().__init__(parent)
        self._items = items or []
        self._current_index = 0
        self._buttons = []
        self._theme = theme

        self.setFixedHeight(34)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(3, 3, 3, 3)
        self._layout.setSpacing(2)

        if self._items:
            self.set_items(self._items)
        self._update_styles()

    def set_theme(self, theme: str):
        self._theme = theme
        self._update_styles()

    def set_items(self, items):
        for btn in self._buttons:
            self._layout.removeWidget(btn)
            btn.deleteLater()
        self._buttons.clear()
        self._items = items

        for idx, item_text in enumerate(items):
            btn = QPushButton(item_text)
            btn.setCheckable(True)
            btn.setChecked(idx == self._current_index)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn.clicked.connect(lambda _, i=idx: self.set_current_index(i))
            self._buttons.append(btn)
            self._layout.addWidget(btn)

        self._update_styles()

    def set_current_index(self, index):
        if 0 <= index < len(self._buttons) and index != self._current_index:
            self._current_index = index
            self._update_styles()
            self.currentIndexChanged.emit(self._current_index)
            self.currentTextChanged.emit(self._items[self._current_index])
        elif 0 <= index < len(self._buttons):
            self._current_index = index
            self._update_styles()

    def current_index(self):
        return self._current_index

    def current_text(self):
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index]
        return ""

    def _update_styles(self):
        if self._theme == "light":
            self.setStyleSheet("""
                AppleSegmentedControl {
                    background-color: #E5E5EA;
                    border-radius: 9px;
                    border: 1px solid #D1D1D6;
                }
            """)
            for idx, btn in enumerate(self._buttons):
                is_active = (idx == self._current_index)
                btn.setChecked(is_active)
                if is_active:
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: #FFFFFF;
                            color: #000000;
                            font-weight: 600;
                            font-size: 11px;
                            border: 1px solid #D1D1D6;
                            border-radius: 7px;
                            padding: 2px 8px;
                        }
                    """)
                else:
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: transparent;
                            color: #6E6E73;
                            font-weight: 500;
                            font-size: 11px;
                            border: none;
                            border-radius: 7px;
                            padding: 2px 8px;
                        }
                        QPushButton:hover {
                            color: #1D1D1F;
                            background-color: #D8D8DC;
                        }
                    """)
        else:
            self.setStyleSheet("""
                AppleSegmentedControl {
                    background-color: #242426;
                    border-radius: 9px;
                    border: 1px solid #333336;
                }
            """)
            for idx, btn in enumerate(self._buttons):
                is_active = (idx == self._current_index)
                btn.setChecked(is_active)
                if is_active:
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: #505054;
                            color: #FFFFFF;
                            font-weight: 600;
                            font-size: 11px;
                            border: 1px solid #636366;
                            border-radius: 7px;
                            padding: 2px 8px;
                        }
                    """)
                else:
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: transparent;
                            color: #98989D;
                            font-weight: 500;
                            font-size: 11px;
                            border: none;
                            border-radius: 7px;
                            padding: 2px 8px;
                        }
                        QPushButton:hover {
                            color: #E5E5EA;
                            background-color: #2C2C2E;
                        }
                    """)


class AppleCard(QFrame):
    """
    macOS System Settings style grouped inset card.
    Rounded corners, subtle borders, adapts to Light and Dark modes.
    """
    def __init__(self, title=None, parent=None, theme="dark"):
        super().__init__(parent)
        self._theme = theme
        self._title = title
        self._lbl_title = None

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(14, 12, 14, 12)
        self._main_layout.setSpacing(10)

        if title:
            self._lbl_title = QLabel(title.upper())
            self._main_layout.addWidget(self._lbl_title)

        self._apply_style()

    def set_theme(self, theme: str):
        self._theme = theme
        self._apply_style()

    def _apply_style(self):
        if self._theme == "light":
            self.setStyleSheet("""
                AppleCard {
                    background-color: #FFFFFF;
                    border: 1px solid #E5E5EA;
                    border-radius: 12px;
                }
            """)
            if self._lbl_title:
                self._lbl_title.setStyleSheet("""
                    color: #6E6E73;
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 0.8px;
                    margin-bottom: 2px;
                """)
        else:
            self.setStyleSheet("""
                AppleCard {
                    background-color: #1C1C1E;
                    border: 1px solid #2C2C2E;
                    border-radius: 12px;
                }
            """)
            if self._lbl_title:
                self._lbl_title.setStyleSheet("""
                    color: #8E8E93;
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 0.8px;
                    margin-bottom: 2px;
                """)

    def add_widget(self, widget):
        self._main_layout.addWidget(widget)

    def add_layout(self, layout):
        self._main_layout.addLayout(layout)

    def content_layout(self):
        return self._main_layout


class AppleRow(QWidget):
    """
    Standard Apple System Settings row:
    Left: Label and subtitle
    Right: Interactive Control
    """
    def __init__(self, label_text, widget, subtitle=None, parent=None, theme="dark"):
        super().__init__(parent)
        self._theme = theme
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(10)

        text_container = QWidget()
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(1)

        self.label = QLabel(label_text)
        text_layout.addWidget(self.label)

        self.sub_label = None
        if subtitle:
            self.sub_label = QLabel(subtitle)
            text_layout.addWidget(self.sub_label)

        layout.addWidget(text_container, stretch=1)
        layout.addWidget(widget, stretch=0, alignment=Qt.AlignRight | Qt.AlignVCenter)
        self._apply_style()

    def set_theme(self, theme: str):
        self._theme = theme
        self._apply_style()

    def _apply_style(self):
        if self._theme == "light":
            self.label.setStyleSheet("color: #1D1D1F; font-size: 13px; font-weight: 500;")
            if self.sub_label:
                self.sub_label.setStyleSheet("color: #6E6E73; font-size: 11px;")
        else:
            self.label.setStyleSheet("color: #F2F2F7; font-size: 13px; font-weight: 500;")
            if self.sub_label:
                self.sub_label.setStyleSheet("color: #8E8E93; font-size: 11px;")


class AppleSliderRow(QWidget):
    """
    Horizontal slider row:
    Label on the left, thin sleek Apple Blue slider in center, numeric badge on right.
    """
    valueChanged = Signal(int)

    def __init__(self, label_text, min_val, max_val, default_val, unit="", formatter=None, parent=None, theme="dark"):
        super().__init__(parent)
        self._unit = unit
        self._formatter = formatter
        self._theme = theme

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(10)

        self.lbl_title = QLabel(label_text)
        self.lbl_title.setFixedWidth(105)
        layout.addWidget(self.lbl_title)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(default_val)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider, stretch=1)

        self.lbl_val = QLabel()
        self.lbl_val.setFixedWidth(58)
        self.lbl_val.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_val)

        self._update_val_label(default_val)
        self._apply_style()

    def set_theme(self, theme: str):
        self._theme = theme
        self._apply_style()

    def _apply_style(self):
        if self._theme == "light":
            self.lbl_title.setStyleSheet("color: #1D1D1F; font-size: 12px; font-weight: 500;")
            self.slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    height: 4px;
                    background: #E5E5EA;
                    border-radius: 2px;
                }
                QSlider::sub-page:horizontal {
                    background: #007AFF;
                    border-radius: 2px;
                }
                QSlider::handle:horizontal {
                    background: #FFFFFF;
                    border: 1px solid #D1D1D6;
                    width: 16px;
                    margin-top: -6px;
                    margin-bottom: -6px;
                    border-radius: 8px;
                }
                QSlider::handle:horizontal:hover {
                    background: #F2F2F7;
                }
            """)
            self.lbl_val.setStyleSheet("""
                QLabel {
                    background-color: #F2F2F7;
                    color: #1D1D1F;
                    font-size: 11px;
                    font-weight: 600;
                    border-radius: 5px;
                    padding: 2px 4px;
                    border: 1px solid #D1D1D6;
                }
            """)
        else:
            self.lbl_title.setStyleSheet("color: #E5E5EA; font-size: 12px; font-weight: 500;")
            self.slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    height: 4px;
                    background: #3A3A3C;
                    border-radius: 2px;
                }
                QSlider::sub-page:horizontal {
                    background: #0A84FF;
                    border-radius: 2px;
                }
                QSlider::handle:horizontal {
                    background: #FFFFFF;
                    width: 16px;
                    margin-top: -6px;
                    margin-bottom: -6px;
                    border-radius: 8px;
                }
                QSlider::handle:horizontal:hover {
                    background: #F2F2F7;
                }
            """)
            self.lbl_val.setStyleSheet("""
                QLabel {
                    background-color: #2C2C2E;
                    color: #8E8E93;
                    font-size: 11px;
                    font-weight: 600;
                    border-radius: 5px;
                    padding: 2px 4px;
                    border: 1px solid #3A3A3C;
                }
            """)

    def _on_slider_changed(self, val):
        self._update_val_label(val)
        self.valueChanged.emit(val)

    def _update_val_label(self, val):
        if self._formatter:
            text = self._formatter(val)
        else:
            text = f"{val}{self._unit}"
        self.lbl_val.setText(text)

    def value(self):
        return self.slider.value()

    def setValue(self, val):
        self.slider.setValue(val)


class AppleDisclosureSection(QWidget):
    """
    macOS collapsible disclosure group with chevron header.
    """
    def __init__(self, title, parent=None, theme="dark"):
        super().__init__(parent)
        self._is_expanded = False
        self._raw_title = title
        self._theme = theme

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        escaped_title = title.replace("&", "&&")
        self.btn_header = QPushButton(f">  {escaped_title}")
        self.btn_header.setCursor(Qt.PointingHandCursor)
        self.btn_header.clicked.connect(self.toggle)
        layout.addWidget(self.btn_header)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 4, 0, 4)
        self.content_layout.setSpacing(8)
        self.content_widget.setVisible(False)
        layout.addWidget(self.content_widget)
        self._apply_style()

    def set_theme(self, theme: str):
        self._theme = theme
        self._apply_style()

    def _apply_style(self):
        color = "#007AFF" if self._theme == "light" else "#0A84FF"
        hover = "#0051A8" if self._theme == "light" else "#409CFF"
        self.btn_header.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {color};
                font-size: 12px;
                font-weight: 600;
                text-align: left;
                border: none;
                padding: 4px 0px;
            }}
            QPushButton:hover {{
                color: {hover};
            }}
        """)

    def toggle(self):
        self._is_expanded = not self._is_expanded
        self.content_widget.setVisible(self._is_expanded)
        icon = "v" if self._is_expanded else ">"
        escaped_title = self._raw_title.replace("&", "&&")
        self.btn_header.setText(f"{icon}  {escaped_title}")

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)


class AppleRecordButton(QPushButton):
    """
    Apple Camera shutter trigger:
    Outer circular ring with clean spacing, center red disc that transforms
    into a rounded square when active.
    """
    def __init__(self, parent=None, theme="dark"):
        super().__init__(parent)
        self.setFixedSize(48, 48)
        self.setCursor(Qt.PointingHandCursor)
        self.setCheckable(True)
        self.setStyleSheet("background: transparent; border: none; padding: 0px; margin: 0px;")
        self._is_recording = False
        self._theme = theme

    def set_theme(self, theme: str):
        self._theme = theme
        self.update()

    def set_recording(self, recording: bool):
        self._is_recording = recording
        self.setChecked(recording)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = float(self.width())
        h = float(self.height())
        cx, cy = w / 2.0, h / 2.0

        # Outer Ring: Keep pen_w margin so stroke is strictly inside widget bounds
        pen_w = 2.5
        ring_color = QColor(255, 255, 255, 230) if self._theme == "dark" else QColor(100, 100, 105, 220)
        painter.setPen(QPen(ring_color, pen_w))
        painter.setBrush(Qt.NoBrush)
        ring_rect = QRectF(pen_w, pen_w, w - 2 * pen_w, h - 2 * pen_w)
        painter.drawEllipse(ring_rect)

        # Inner Recording Indicator (System Red)
        if not self._is_recording:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 59, 48)))  # iOS System Red
            inner_radius = 16.0
            painter.drawEllipse(QRectF(cx - inner_radius, cy - inner_radius, inner_radius * 2, inner_radius * 2))
        else:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 69, 58)))
            sq_size = 16.0
            painter.drawRoundedRect(
                QRectF(cx - sq_size / 2.0, cy - sq_size / 2.0, sq_size, sq_size),
                4.0, 4.0
            )



class ApplePillButton(QPushButton):
    """
    Clean frosted pill action button for both Light and Dark themes.
    Strictly NO emojis.
    """
    def __init__(self, text, is_accent=False, parent=None, theme="dark"):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(34)
        self._is_accent = is_accent
        self._theme = theme
        self._is_active = False
        self._update_style()

    def set_theme(self, theme: str):
        self._theme = theme
        self._update_style()

    def set_active_state(self, active: bool):
        self._is_active = active
        self._update_style()

    def _update_style(self):
        if self._is_active:
            # Active Apple Blue or Amber state
            self.setStyleSheet("""
                QPushButton {
                    background-color: #007AFF;
                    color: #FFFFFF;
                    font-size: 12px;
                    font-weight: 600;
                    border-radius: 17px;
                    padding: 0 16px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #0062CC;
                }
            """)
        elif self._is_accent:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #007AFF;
                    color: #FFFFFF;
                    font-size: 12px;
                    font-weight: 600;
                    border-radius: 17px;
                    padding: 0 16px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #0062CC;
                }
            """)
        elif self._theme == "light":
            self.setStyleSheet("""
                QPushButton {
                    background-color: #F2F2F7;
                    color: #1D1D1F;
                    font-size: 12px;
                    font-weight: 500;
                    border-radius: 17px;
                    padding: 0 14px;
                    border: 1px solid #D1D1D6;
                }
                QPushButton:hover {
                    background-color: #E5E5EA;
                    color: #000000;
                }
                QPushButton:pressed {
                    background-color: #D8D8DC;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #2C2C2E;
                    color: #F2F2F7;
                    font-size: 12px;
                    font-weight: 500;
                    border-radius: 17px;
                    padding: 0 14px;
                    border: 1px solid #3A3A3C;
                }
                QPushButton:hover {
                    background-color: #3A3A3C;
                    color: #FFFFFF;
                    border-color: #48484A;
                }
                QPushButton:pressed {
                    background-color: #1C1C1E;
                }
            """)
