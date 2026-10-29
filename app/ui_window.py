import os
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QSettings, QRectF, QPointF
from PySide6.QtGui import (QImage, QFont, QColor, QPainter, QPen, QBrush, QShortcut, QKeySequence,
                           QPainterPath)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QFileDialog, QFrame,
    QSizePolicy, QScrollArea, QStackedWidget, QColorDialog, QMessageBox,
)

from app.camera.capture import CameraCaptureThread, list_camera_devices
from app.camera.virtual_cam import VirtualCameraOutput
from app.tracking.face_tracker import FaceTracker
from app.tracking.hand_tracker import HandTracker
from app.tracking.voice_meter import VoiceMeter
from app.effects.scribble_mask import ScribbleMaskEngine
from app.effects.background_engine import BackgroundEngine
from app.recording.video_recorder import VideoRecorder, list_audio_devices
from app.pipeline import PipelineWorker
from app.apple_widgets import (
    AppleSwitch, AppleSegmentedControl, AppleCard, AppleRow, AppleSliderRow, AppleDisclosureSection,
    AppleRecordButton, ApplePillButton,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SHAPES = ["head", "cube", "star", "diamond", "pill"]
STYLES = ["ink", "chalk", "neon", "sunny"]
EYES = ["knot", "cartoon", "dot"]
BG_MODES = [BackgroundEngine.MODE_ORIGINAL, BackgroundEngine.MODE_BLUR, BackgroundEngine.MODE_STUDIO,
            BackgroundEngine.MODE_SOLID, BackgroundEngine.MODE_IMAGE]
TONGUE_SENS = ["medium", "low", "high", "off"]
PRESETS = ["Quality", "Balanced", "Low-End"]


# =============================================================================
# Video canvas
# =============================================================================
class VideoCanvas(QWidget):
    """
    Preview surface. Draws the newest frame straight from the numpy buffer (QImage BGR888,
    no colour conversion or pixmap copies) and paints the HUD with QPainter on top, so the
    HUD never touches the recorded / broadcast frames.

    Mouse: click a person to hide/show them (Person Isolation); in keep-zone mode, drag a
    rectangle; right-click a keep zone to delete it.
    """
    clicked = Signal(float, float, bool)
    zone_drawn = Signal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(480, 300)
        self.setMouseTracking(True)
        self._frame = None
        self._img = None
        self.metrics = {}
        self.show_hud = True
        self.people = []
        self.keep_zones = []
        self.zone_mode = False
        self._drag0 = None
        self._drag1 = None
        self.toast = None
        self.toast_until = 0.0
        self.placeholder = "Connecting camera..."

    def set_frame(self, frame, metrics):
        self._frame = frame
        h, w = frame.shape[:2]
        self._img = QImage(frame.data, w, h, w * 3, QImage.Format_BGR888)
        self.metrics = metrics
        self.update()

    def show_toast(self, text, seconds=4.0):
        self.toast = text
        self.toast_until = time.monotonic() + seconds
        self.update()

    def _target(self):
        if self._img is None:
            return QRectF(0, 0, self.width(), self.height())
        iw, ih = self._img.width(), self._img.height()
        s = min(self.width() / iw, self.height() / ih)
        tw, th = iw * s, ih * s
        return QRectF((self.width() - tw) / 2, (self.height() - th) / 2, tw, th)

    def _to_norm(self, pos):
        r = self._target()
        if r.width() <= 0 or not r.contains(pos):
            return None
        return (pos.x() - r.x()) / r.width(), (pos.y() - r.y()) / r.height()

    # ---- painting --------------------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 16, 16)
        p.setClipPath(path)
        p.fillRect(self.rect(), QColor(0, 0, 0))
        if self._img is None:
            p.setPen(QColor(140, 140, 145))
            p.setFont(QFont("Segoe UI", 12))
            p.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
            return
        r = self._target()
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.drawImage(r, self._img)
        if self.people:
            self._paint_people(p, r)
        if self.keep_zones or self._drag0:
            self._paint_zones(p, r)
        if self.show_hud:
            self._paint_hud(p, r)
        if self.toast and time.monotonic() < self.toast_until:
            self._pill(p, r.center().x(), r.bottom() - 40, self.toast, QColor(20, 20, 24, 225), QColor(245, 245, 247), center=True)

    def _pill(self, p, x, y, text, bg, fg, dot=None, center=False, size=10):
        f = QFont("Segoe UI", size, QFont.DemiBold)
        p.setFont(f)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text)
        w = tw + (34 if dot else 22)
        h = fm.height() + 10
        if center:
            x -= w / 2
        rect = QRectF(x, y, w, h)
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(rect, h / 2, h / 2)
        tx = x + 11
        if dot:
            p.setBrush(dot)
            p.drawEllipse(QPointF(x + 14, y + h / 2), 4, 4)
            tx = x + 24
        p.setPen(fg)
        p.drawText(QRectF(tx, y, tw + 2, h), Qt.AlignVCenter | Qt.AlignLeft, text)
        return w, h

    def _paint_hud(self, p, r):
        m = self.metrics
        bg = QColor(18, 18, 22, 200)
        fg = QColor(236, 236, 240)
        fps = m.get('fps', 0.0)
        dot = QColor(52, 199, 89) if fps >= 25 else (QColor(255, 204, 0) if fps >= 18 else QColor(255, 69, 58))
        x, y = r.x() + 14, r.y() + 14
        _, h = self._pill(p, x, y, f"{fps:4.1f} FPS   {m.get('latency_ms', 0):.0f} ms latency", bg, fg, dot)
        mode = m.get('privacy', 'clear')
        labels = {
            'helmet': ("HEAD LOCKED", QColor(10, 132, 255)),
            'holding': ("HOLDING HELMET (face hidden)", QColor(255, 159, 10)),
            'mosaic': ("PRIVACY MOSAIC (looking for face)", QColor(255, 159, 10)),
            'curtain': ("BE RIGHT BACK", QColor(191, 90, 242)),
            'clear': ("NO PERSON", QColor(142, 142, 147)),
        }
        text, col = labels.get(mode, labels['clear'])
        if mode == 'helmet':
            text += f"   Y {m.get('yaw', 0):+.0f}   P {m.get('pitch', 0):+.0f}"
        self._pill(p, x, y + h + 8, text, bg, fg, col)
        detail = f"track {m.get('track_ms', 0):.0f} ms  |  matte {m.get('bg_ms', 0):.0f} ms  |  draw {m.get('scribble_ms', 0):.0f} ms"
        self._pill(p, x, y + 2 * (h + 8), detail, QColor(18, 18, 22, 160), QColor(170, 170, 178), size=9)
        rx = r.right() - 14
        if m.get('is_recording'):
            w, _ = self._pill(p, 0, -100, "REC", bg, fg, QColor(255, 59, 48))
            self._pill(p, rx - w, y, "REC", QColor(60, 16, 16, 220), fg, QColor(255, 59, 48))
            rx -= w + 8
        if m.get('is_vcam'):
            w, _ = self._pill(p, 0, -100, "LIVE V-CAM", bg, fg, QColor(10, 132, 255))
            self._pill(p, rx - w, y, "LIVE V-CAM", QColor(14, 30, 58, 220), fg, QColor(10, 132, 255))

    def _norm_rect(self, r, b):
        x0, y0, x1, y1 = b
        return QRectF(r.x() + x0 * r.width(), r.y() + y0 * r.height(), (x1 - x0) * r.width(), (y1 - y0) * r.height())

    def _paint_people(self, p, r):
        for tid, box, kept, is_me in self.people:
            rect = self._norm_rect(r, box)
            col = QColor(10, 132, 255) if is_me else (QColor(52, 199, 89) if kept else QColor(255, 69, 58))
            pen = QPen(col, 2, Qt.SolidLine if kept else Qt.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(rect, 8, 8)
            label = "You" if is_me else (f"Person {tid}  (shown)" if kept else f"Person {tid}  (hidden)")
            self._pill(p, rect.x() + 6, rect.y() + 6, label, QColor(col.red(), col.green(), col.blue(), 210), QColor(255, 255, 255), size=9)

    def _paint_zones(self, p, r):
        pen = QPen(QColor(52, 199, 89), 2, Qt.DashLine)
        p.setPen(pen)
        p.setBrush(QColor(52, 199, 89, 30))
        for z in self.keep_zones:
            p.drawRect(self._norm_rect(r, z))
        if self._drag0 and self._drag1:
            (x0, y0), (x1, y1) = self._drag0, self._drag1
            p.drawRect(self._norm_rect(r, (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))))

    # ---- mouse -----------------------------------------------------------------------
    def mousePressEvent(self, e):
        pos = e.position()
        n = self._to_norm(pos)
        if n is None:
            return
        if self.zone_mode and e.button() == Qt.LeftButton:
            self._drag0 = n
            self._drag1 = n
            return
        self.clicked.emit(n[0], n[1], e.button() == Qt.RightButton)

    def mouseMoveEvent(self, e):
        if self._drag0 is not None:
            n = self._to_norm(e.position())
            if n is not None:
                self._drag1 = n
                self.update()

    def mouseReleaseEvent(self, e):
        if self._drag0 is not None and self._drag1 is not None:
            (x0, y0), (x1, y1) = self._drag0, self._drag1
            self._drag0 = self._drag1 = None
            self.zone_mode = False
            self.setCursor(Qt.ArrowCursor)
            if abs(x1 - x0) > 0.01 and abs(y1 - y0) > 0.01:
                self.zone_drawn.emit(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            self.update()


# =============================================================================
# Main window
# =============================================================================
class ScribbleCameraWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scribble Studio")
        self.resize(1380, 880)
        self.setMinimumSize(1060, 700)
        ini = os.environ.get("SCRIBBLE_SETTINGS_INI")
        self.settings = QSettings(ini, QSettings.IniFormat) if ini else QSettings("ScribbleStudio", "ScribbleCamera")
        self.current_theme = self.settings.value("ui/theme", "dark")
        self.all_apple_widgets = []
        self._appliers = []
        self._loading = True

        self.models_dir = os.path.join(BASE_DIR, "app", "models")
        m = self.models_dir
        self.test_video_path = r"D:\Ehtishaam\Files\Mine\New Branding\Scribble face vid test\VID_20260430_222927.mp4"

        # Engines (live across camera/pipeline restarts). Landmark workers spawn now and load
        # their models in parallel while the UI is built.
        self.tracker = FaceTracker(os.path.join(m, "face_landmarker.task"),
                                   pose_model_path=os.path.join(m, "pose_landmarker_lite.task"), use_worker=True)
        self.hands = HandTracker(os.path.join(m, "hand_landmarker.task"), use_worker=True)
        self.scribble_engine = ScribbleMaskEngine()
        self.bg_engine = BackgroundEngine(os.path.join(m, "selfie_segmenter.tflite"), models_dir=m)
        self.recorder = VideoRecorder(output_dir=os.path.join(BASE_DIR, "recordings"))
        self.virtual_cam = VirtualCameraOutput()
        self.voice = VoiceMeter()

        self.capture_thread = None
        self.pipeline_worker = None
        self.current_source = None
        self.record_start = 0.0
        self.pipe = dict(enable_scribble=True, enable_expressions=True, enable_boil=True, sketch_fps=12.0,
                         sketch_noise=0.85, scribble_scale=1.6, privacy_guard=True, curtain=False)

        self.rec_timer = QTimer(self)
        self.rec_timer.setInterval(500)
        self.rec_timer.timeout.connect(self._update_record_timer)

        self._setup_ui()
        self._setup_shortcuts()
        self.set_theme(self.current_theme)
        for apply in self._appliers:
            apply()
        self._loading = False
        self._start_pipeline(self._initial_source())

    # ------------------------------------------------------------------ settings helpers
    def _load(self, key, default, typ):
        v = self.settings.value(key, default)
        try:
            if typ is bool:
                return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
            return typ(v)
        except Exception:
            return default

    def _save(self, key, value):
        if not self._loading:
            self.settings.setValue(key, value)

    def _reg(self, w):
        self.all_apple_widgets.append(w)
        return w

    def _slider(self, key, label, lo, hi, default, apply, unit="", fmt=None):
        v = int(np.clip(self._load(key, default, int), lo, hi))
        w = self._reg(AppleSliderRow(label, lo, hi, v, unit=unit, formatter=fmt, theme=self.current_theme))

        def on(val):
            self._save(key, val)
            apply(val)
        w.valueChanged.connect(on)
        self._appliers.append(lambda: apply(w.value()))
        return w

    def _switch(self, key, label, subtitle, default, apply):
        v = self._load(key, default, bool)
        sw = self._reg(AppleSwitch(checked=v, theme=self.current_theme))

        def on(val):
            self._save(key, bool(val))
            apply(bool(val))
        sw.toggled.connect(on)
        self._appliers.append(lambda: apply(sw.isChecked()))
        row = self._reg(AppleRow(label, sw, subtitle, theme=self.current_theme))
        row.switch = sw
        return row

    def _segmented(self, key, items, default_idx, apply, label=None, subtitle=None):
        idx = int(np.clip(self._load(key, default_idx, int), 0, len(items) - 1))
        seg = self._reg(AppleSegmentedControl(items, theme=self.current_theme))
        seg.set_current_index(idx)

        def on(i):
            self._save(key, i)
            apply(i)
        seg.currentIndexChanged.connect(on)
        self._appliers.append(lambda: apply(seg.current_index()))
        if label is None:
            seg.control = seg
            return seg
        row = self._reg(AppleRow(label, seg, subtitle, theme=self.current_theme))
        row.control = seg
        return row

    def _set_pipe(self, key, value):
        self.pipe[key] = value
        if self.pipeline_worker is not None:
            setattr(self.pipeline_worker, key, value)

    def _note(self, text):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("font-size: 11px;")
        return lbl

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(14, 14, 14, 14)
        main.setSpacing(14)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(10)
        self.video = VideoCanvas()
        self.video.clicked.connect(self._on_video_clicked)
        self.video.zone_drawn.connect(self._on_zone_drawn)
        lv.addWidget(self.video, stretch=1)

        self.dock_frame = QFrame()
        self.dock_frame.setObjectName("dockFrame")
        self.dock_frame.setFixedHeight(68)
        dock = QHBoxLayout(self.dock_frame)
        dock.setContentsMargins(18, 6, 18, 6)
        dock.setSpacing(12)
        self.btn_vcam = self._reg(ApplePillButton("Virtual Cam", theme=self.current_theme))
        self.btn_vcam.clicked.connect(self._toggle_virtual_cam)
        dock.addWidget(self.btn_vcam)
        self.btn_curtain = self._reg(ApplePillButton("Be Right Back", theme=self.current_theme))
        self.btn_curtain.clicked.connect(self._toggle_curtain)
        dock.addWidget(self.btn_curtain)
        self.btn_hud = self._reg(ApplePillButton("HUD", theme=self.current_theme))
        self.btn_hud.set_active_state(True)
        self.btn_hud.clicked.connect(self._toggle_hud)
        dock.addWidget(self.btn_hud)
        dock.addStretch(1)
        shutter = QWidget()
        shutter.setStyleSheet("background: transparent; border: none;")
        sl = QHBoxLayout(shutter)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(10)
        self.btn_record = self._reg(AppleRecordButton(theme=self.current_theme))
        self.btn_record.clicked.connect(self._toggle_record)
        sl.addWidget(self.btn_record)
        self.lbl_record_timer = QLabel("Ready")
        self.lbl_record_timer.setFixedWidth(70)
        sl.addWidget(self.lbl_record_timer)
        dock.addWidget(shutter)
        dock.addStretch(1)
        self.btn_calib = self._reg(ApplePillButton("Calibrate Face", theme=self.current_theme))
        self.btn_calib.clicked.connect(self._calibrate_face)
        dock.addWidget(self.btn_calib)
        self.btn_snap = self._reg(ApplePillButton("Snapshot", theme=self.current_theme))
        self.btn_snap.clicked.connect(self._snapshot)
        dock.addWidget(self.btn_snap)
        self.btn_open_rec = self._reg(ApplePillButton("Recordings", theme=self.current_theme))
        self.btn_open_rec.clicked.connect(self._open_recordings_folder)
        dock.addWidget(self.btn_open_rec)
        lv.addWidget(self.dock_frame)

        self.status_bar_label = QLabel("Starting...")
        lv.addWidget(self.status_bar_label)
        main.addWidget(left, stretch=3)

        # ---- sidebar
        self.sidebar_frame = QFrame()
        self.sidebar_frame.setObjectName("sidebarFrame")
        self.sidebar_frame.setFixedWidth(370)
        sb = QVBoxLayout(self.sidebar_frame)
        sb.setContentsMargins(16, 16, 16, 16)
        sb.setSpacing(12)
        header = QWidget()
        header.setStyleSheet("background: transparent; border: none;")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(0, 0, 0, 0)
        titles = QWidget()
        tl = QVBoxLayout(titles)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(1)
        self.lbl_studio_title = QLabel("Scribble Studio")
        self.lbl_studio_title.setFont(QFont("Segoe UI", 15, QFont.Bold))
        tl.addWidget(self.lbl_studio_title)
        self.lbl_studio_sub = QLabel("Private 3D scribble head + live background")
        tl.addWidget(self.lbl_studio_sub)
        hl.addWidget(titles, stretch=1)
        self.seg_theme = self._reg(AppleSegmentedControl(["Light", "Dark"], theme=self.current_theme))
        self.seg_theme.setFixedWidth(110)
        self.seg_theme.set_current_index(0 if self.current_theme == "light" else 1)
        self.seg_theme.currentIndexChanged.connect(lambda i: self.set_theme("light" if i == 0 else "dark"))
        hl.addWidget(self.seg_theme)
        sb.addWidget(header)

        self.tab_nav = self._reg(AppleSegmentedControl(["Helmet", "Face", "Stage", "Studio"], theme=self.current_theme))
        sb.addWidget(self.tab_nav)
        self.tabs_stack = QStackedWidget()
        self.tab_nav.currentIndexChanged.connect(self.tabs_stack.setCurrentIndex)
        sb.addWidget(self.tabs_stack, stretch=1)
        main.addWidget(self.sidebar_frame)

        self.tabs_stack.addWidget(self._build_helmet_tab())
        self.tabs_stack.addWidget(self._build_face_tab())
        self.tabs_stack.addWidget(self._build_stage_tab())
        self.tabs_stack.addWidget(self._build_studio_tab())

    @staticmethod
    def _compact(combo):
        # Long device names must not widen the sidebar.
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(13)
        return combo

    def _page(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; } QScrollArea > QWidget > QWidget { background: transparent; }")
        inner = QWidget()
        scroll.setWidget(inner)
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(12)
        return scroll, lay

    # ---- Helmet tab
    def _build_helmet_tab(self):
        page, lay = self._page()
        sc = self.scribble_engine
        sph = sc.sphere_3d
        card = self._reg(AppleCard("3D Scribble Helmet", theme=self.current_theme))
        card.add_widget(self._switch("helmet/enabled", "Scribble Helmet", "Solid 3D head that hides your face",
                                     True, lambda v: self._set_pipe('enable_scribble', v)))
        card.add_widget(self._segmented("helmet/shape", ["Head", "Cube", "Star", "Diamond", "Pill"], 0,
                                        lambda i: sc.set_shape(SHAPES[i])))
        card.add_widget(self._segmented("helmet/style", ["Ink", "Chalk", "Neon", "Sunny"], 0,
                                        lambda i: sc.set_style(STYLES[i]), label="Style"))
        card.add_widget(self._slider("helmet/scale", "Helmet Size", 100, 230, 160,
                                     lambda v: self._set_pipe('scribble_scale', v / 100.0), unit="%"))
        card.add_widget(self._slider("helmet/shading", "3D Shading", 0, 100, 75,
                                     lambda v: setattr(sph, 'shading', v / 100.0), unit="%"))
        card.add_widget(self._slider("helmet/density", "Scribbles", 10, 84, 84, sc.set_density))
        card.add_widget(self._slider("helmet/line", "Line Width", 1, 6, 3, sc.set_line_width, unit=" px"))
        card.add_widget(self._switch("helmet/outline", "Hand-drawn Outline", None, True,
                                     lambda v: setattr(sph, 'show_outline', v)))
        lay.addWidget(card)

        motion = self._reg(AppleCard("Sketch Motion", theme=self.current_theme))
        motion.add_widget(self._switch("helmet/boil", "Line Boil", "Stop-motion redraw of every stroke", True,
                                       lambda v: self._set_pipe('enable_boil', v)))
        motion.add_widget(self._slider("helmet/boil_fps", "Boil Speed", 4, 24, 12,
                                       lambda v: self._set_pipe('sketch_fps', float(v)), unit=" fps"))
        motion.add_widget(self._slider("helmet/noise", "Wobble", 0, 150, 85,
                                       lambda v: self._set_pipe('sketch_noise', v / 100.0), unit="%"))
        lay.addWidget(motion)

        tune = self._reg(AppleDisclosureSection("Fit & Offset Calibration", theme=self.current_theme))
        fmt = lambda v: f"{v / 10:+.1f} cm"
        tune.add_widget(self._slider("helmet/off_y", "Vertical", -100, 100, 15, lambda v: sc.set_offsets(offset_y=v / 10.0), fmt=fmt))
        tune.add_widget(self._slider("helmet/off_x", "Horizontal", -100, 100, 0, lambda v: sc.set_offsets(offset_x=v / 10.0), fmt=fmt))
        tune.add_widget(self._slider("helmet/off_z", "Depth", -100, 100, -5, lambda v: sc.set_offsets(offset_z=v / 10.0), fmt=fmt))
        tune.add_widget(self._slider("helmet/opacity", "Body Opacity", 60, 100, 100,
                                     lambda v: setattr(sph, 'fill_opacity', v / 100.0), unit="%"))
        tune.add_widget(self._note("Keep Body Opacity at 100% for full privacy — lower values let your real face show through."))
        lay.addWidget(tune)
        lay.addStretch(1)
        return page

    # ---- Face tab
    def _build_face_tab(self):
        page, lay = self._page()
        ff = self.scribble_engine.features
        card = self._reg(AppleCard("Expression Mimicry", theme=self.current_theme))
        card.add_widget(self._switch("face/enabled", "Mimic Expressions", "Lips, eyelids, gaze and brows from your face",
                                     True, lambda v: self._set_pipe('enable_expressions', v)))
        card.add_widget(self._slider("face/gain", "Exaggeration", 50, 300, 160,
                                     lambda v: setattr(ff, 'gain', v / 100.0), fmt=lambda v: f"{v / 100:.1f}x"))
        card.add_widget(self._slider("face/talk", "Talk Sensitivity", 50, 250, 120,
                                     lambda v: setattr(ff, 'talk_sensitivity', v / 100.0), unit="%"))
        card.add_widget(self._switch("face/voice", "Voice Lip-Sync", "Your mic opens the mouth when you speak",
                                     False, self._apply_voice))
        card.add_widget(self._segmented("face/eyes", ["Scribble", "Cartoon", "Dots"], 0,
                                        lambda i: setattr(ff, 'eye_style', EYES[i]), label="Eyes"))
        card.add_widget(self._switch("face/brows", "Eyebrows", None, True, lambda v: setattr(ff, 'show_brows', v)))
        card.add_widget(self._slider("face/size", "Face Size", 60, 150, 100,
                                     lambda v: setattr(ff, 'feature_scale', v / 100.0), unit="%"))
        card.add_widget(self._slider("face/height", "Face Height", -60, 60, 0,
                                     lambda v: setattr(ff, 'feature_height', v / 10.0), fmt=lambda v: f"{v / 10:+.1f} cm"))
        btn = self._reg(ApplePillButton("Calibrate Neutral Face  (C)", is_accent=True, theme=self.current_theme))
        btn.clicked.connect(self._calibrate_face)
        card.add_widget(btn)
        card.add_widget(self._note("Relax your face and look at the camera, then click. The drawn face treats this as "
                                   "\"neutral\" and exaggerates every movement away from it. It also self-calibrates "
                                   "slowly whenever your face is relaxed. Talk Sensitivity sets how easily the mouth "
                                   "opens while you speak; Voice Lip-Sync uses the Recording microphone (or your "
                                   "default mic) — turn it off if other people or music are audible."))
        lay.addWidget(card)

        fun = self._reg(AppleCard("Extras", theme=self.current_theme))
        fun.add_widget(self._switch("face/emotes", "Emote Stickers", "Blush, anger vein, sweat drop, ?, Zzz, hearts, !!",
                                    False, lambda v: setattr(ff, 'emotes_enabled', v)))
        fun.add_widget(self._segmented("face/tongue_style", ["Sprite", "Organic"], 0,
                                       lambda i: ff.set_tongue_style(["sprite", "organic"][i]), label="Tongue"))
        fun.add_widget(self._segmented("face/tongue_sens", ["Medium", "Low", "High", "Off"], 0,
                                       lambda i: ff.set_tongue_sensitivity(TONGUE_SENS[i]), label="Tongue Detect"))
        lay.addWidget(fun)
        lay.addStretch(1)
        return page

    # ---- Stage tab
    def _build_stage_tab(self):
        page, lay = self._page()
        bg = self.bg_engine
        card = self._reg(AppleCard("Background", theme=self.current_theme))
        self.seg_bg = self._segmented("bg/mode", ["Original", "Blur", "Studio", "Solid", "Image"], 0, self._apply_bg_mode)
        card.add_widget(self.seg_bg)
        self.slider_blur = self._slider("bg/blur", "Blur Amount", 15, 95, 55, lambda v: setattr(bg, 'blur_strength', v), unit="%")
        card.add_widget(self.slider_blur)
        self.btn_color = self._reg(ApplePillButton("Pick Solid Color...", theme=self.current_theme))
        self.btn_color.clicked.connect(self._pick_color)
        card.add_widget(self.btn_color)
        self.btn_pick_img = self._reg(ApplePillButton("Choose Background Image...", is_accent=True, theme=self.current_theme))
        self.btn_pick_img.clicked.connect(self._pick_image)
        card.add_widget(self.btn_pick_img)
        lay.addWidget(card)
        col = self.settings.value("bg/solid", "#00ff00")
        qc = QColor(col)
        bg.set_solid_color((qc.blue(), qc.green(), qc.red()))
        img = self.settings.value("bg/image", "")
        if img:
            bg.set_custom_image(img)

        iso = self._reg(AppleCard("People in Frame", theme=self.current_theme))
        iso.add_widget(self._switch("iso/enabled", "Person Control", "Pick who appears in the video",
                                    False, self._apply_isolation))
        iso.add_widget(self._segmented("iso/mode", ["Everyone", "Only Me"], 1,
                                       lambda i: bg.isolator.set_mode(["everyone", "only_me"][i]), label="Default"))
        iso.add_widget(self._note("Click anyone in the preview to hide or show them — the choice follows them as they "
                                  "move. \"Only Me\" hides anybody who walks in behind you automatically. Hidden people "
                                  "are replaced with the empty background the app learns while you use it."))
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        b1 = self._reg(ApplePillButton("Add Keep Zone", theme=self.current_theme))
        b1.clicked.connect(self._start_zone)
        b2 = self._reg(ApplePillButton("Clear", theme=self.current_theme))
        b2.clicked.connect(self._clear_isolation)
        rl.addWidget(b1)
        rl.addWidget(b2)
        iso.add_widget(row)
        iso.add_widget(self._note("Keep Zone: drag a box over something the matte misses (a friend, a guitar, a pet) to "
                                  "always show it. Right-click a zone to delete it."))
        lay.addWidget(iso)

        matte = self._reg(AppleDisclosureSection("Matte Quality & Edges", theme=self.current_theme))
        self.matting_engine_combo = self._compact(QComboBox())
        engines = []
        if bg.has_cuda:
            engines += [("Neural RVM ResNet50 (best edges)", BackgroundEngine.ENGINE_RVM_RESNET50),
                        ("Neural RVM MobileNet (fast)", BackgroundEngine.ENGINE_RVM_MOBILENET)]
        engines += [("MediaPipe (lightest, CPU)", BackgroundEngine.ENGINE_MEDIAPIPE)]
        for name, key in engines:
            self.matting_engine_combo.addItem(name, key)
        saved = self.settings.value("bg/engine", BackgroundEngine.ENGINE_RVM_MOBILENET if bg.has_cuda else BackgroundEngine.ENGINE_MEDIAPIPE)
        i = max(0, self.matting_engine_combo.findData(saved))
        self.matting_engine_combo.setCurrentIndex(i)
        self.matting_engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        self._appliers.append(lambda: bg.set_engine_type(self.matting_engine_combo.currentData()))
        matte.add_widget(self._reg(AppleRow("Engine", self.matting_engine_combo, theme=self.current_theme)))
        self.slider_detail = self._slider("bg/detail", "Detail", 20, 60, 40, lambda v: setattr(bg, 'downsample_ratio', v / 100.0), unit="%")
        matte.add_widget(self.slider_detail)
        matte.add_widget(self._slider("bg/hardness", "Edge Hardness", 0, 100, 50, lambda v: setattr(bg, 'hardness', v / 100.0), unit="%"))
        matte.add_widget(self._slider("bg/choke", "Edge Choke", -4, 3, 0, lambda v: setattr(bg, 'edge_choke', v), fmt=lambda v: f"{v:+d} px"))
        matte.add_widget(self._switch("bg/decontam", "Clean Edge Colors", "Removes background color spill on edges",
                                      True, lambda v: setattr(bg, 'enable_decontamination', v)))
        self.row_solid_hands = self._switch("bg/handprior", "Solid Hands", "Keeps fast-moving hands from going see-through",
                                            True, lambda v: setattr(bg, 'hand_prior', v))
        matte.add_widget(self.row_solid_hands)
        matte.add_widget(self._switch("bg/guided", "Edge Refine (CPU engine)", None, True,
                                      lambda v: setattr(bg, 'use_guided_filter', v)))
        matte.add_widget(self._switch("bg/relight", "Light Wrap", "Soft rim light from the new background", False,
                                      lambda v: bg.set_relighting_params(enabled=v)))
        btn_plate = self._reg(ApplePillButton("Re-learn Empty Background", theme=self.current_theme))
        btn_plate.clicked.connect(lambda: (bg.reset_clean_plate(), self.video.show_toast("Background plate reset — it re-learns as you move")))
        matte.add_widget(btn_plate)
        lay.addWidget(matte)
        lay.addStretch(1)
        return page

    # ---- Studio tab
    def _build_studio_tab(self):
        page, lay = self._page()
        card = self._reg(AppleCard("Camera", theme=self.current_theme))
        self.camera_combo = self._compact(QComboBox())
        self._populate_cameras()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        card.add_widget(self._reg(AppleRow("Device", self.camera_combo, theme=self.current_theme)))
        self.res_combo = self._compact(QComboBox())
        self.res_combo.addItem("720p (recommended)", (1280, 720))
        self.res_combo.addItem("1080p", (1920, 1080))
        self.res_combo.setCurrentIndex(int(self._load("cam/res", 0, int)))
        self.res_combo.currentIndexChanged.connect(self._on_resolution_changed)
        card.add_widget(self._reg(AppleRow("Resolution", self.res_combo, theme=self.current_theme)))
        card.add_widget(self._switch("cam/mirror", "Mirror", "Flip left/right (like a selfie camera)", False, self._apply_mirror))
        b = self._reg(ApplePillButton("Refresh Cameras", theme=self.current_theme))
        b.clicked.connect(self._refresh_cameras)
        card.add_widget(b)
        lay.addWidget(card)

        perf = self._reg(AppleCard("Performance", theme=self.current_theme))
        self.seg_preset = self._reg(AppleSegmentedControl(PRESETS, theme=self.current_theme))
        self.seg_preset.set_current_index(int(self._load("perf/preset", 1, int)))
        self.seg_preset.currentIndexChanged.connect(self._apply_preset)
        perf.add_widget(self.seg_preset)
        self.lbl_preset = self._note("")
        perf.add_widget(self.lbl_preset)
        perf.add_widget(self._note("Hand tracking — Smart runs every frame only while a hand is near your face:"))
        perf.add_widget(self._segmented("perf/hands", ["Smart", "Every Frame", "Off"], 0, self._apply_hand_mode))
        lay.addWidget(perf)

        priv = self._reg(AppleCard("Privacy", theme=self.current_theme))
        priv.add_widget(self._switch("priv/guard", "Privacy Guard", "Keeps the helmet on through lost tracking; mosaics unknown heads",
                                     True, self._apply_guard))
        priv.add_widget(self._switch("priv/hands_over", "Hands in Front of Helmet", "Your hands can cover the scribble face",
                                     True, lambda v: setattr(self.scribble_engine, 'hands_over_helmet', v)))
        priv.add_widget(self._note("Hotkey B toggles the \"Be Right Back\" screen instantly (recording and virtual camera)."))
        lay.addWidget(priv)

        rec = self._reg(AppleCard("Recording", theme=self.current_theme))
        self.audio_combo = self._compact(QComboBox())
        self.audio_combo.addItem("No audio", "")
        for dev in list_audio_devices(self.recorder.ffmpeg_path):
            self.audio_combo.addItem(dev, dev)
        saved_mic = self.settings.value("rec/audio", "")
        j = self.audio_combo.findData(saved_mic)
        self.audio_combo.setCurrentIndex(j if j >= 0 else 0)
        self.audio_combo.currentIndexChanged.connect(self._on_mic_changed)
        rec.add_widget(self._reg(AppleRow("Microphone", self.audio_combo, theme=self.current_theme)))
        rec.add_widget(self._slider("rec/av_offset", "Audio Delay", -300, 500, 120,
                                    lambda v: setattr(self.recorder, 'audio_offset_ms', v), unit=" ms"))
        rec.add_widget(self._switch("rec/matte", "Export Matte Track", "Extra *_matte.mp4 (white = you) for AE / Premiere",
                                    False, lambda v: setattr(self.recorder, 'record_matte', v)))
        enc = "hardware (NVENC/AMF/QSV) if available" if self.recorder.ffmpeg_path else "OpenCV (install FFmpeg for audio + GPU encoding)"
        rec.add_widget(self._note(f"Recordings keep real-time speed even if FPS dips. Encoder: {enc}. "
                                  f"Audio Delay lines your voice up with the camera's latency (DroidCam over Wi-Fi ≈ 100-200 ms)."))
        rec.add_widget(self._switch("ui/hud", "Preview HUD", "Stats overlay (never recorded)", True, self._apply_hud))
        lay.addWidget(rec)
        lay.addStretch(1)
        self._update_preset_label()
        self._appliers.append(lambda: self.tracker.set_proxy_width(self._load("perf/proxy", 512, int)))
        return page

    # ------------------------------------------------------------------ appliers
    def _apply_bg_mode(self, i):
        self.bg_engine.mode = BG_MODES[i]
        self.slider_blur.setVisible(i == 1)
        self.btn_color.setVisible(i == 3)
        self.btn_pick_img.setVisible(i == 4)
        if i == 4 and self.bg_engine.custom_bg_img is None and not self._loading:
            self._pick_image()

    def _apply_isolation(self, v):
        self.bg_engine.set_person_isolation(v)
        if not v:
            self.video.people = []
            self.video.keep_zones = []

    def _apply_mirror(self, v):
        if self.capture_thread is not None:
            self.capture_thread.mirror = bool(v)

    def _apply_guard(self, v):
        self._set_pipe('privacy_guard', v)
        self.tracker.set_privacy_hold(v)

    def _apply_voice(self, v):
        if not v:
            self.voice.stop()
            return
        mic = self.audio_combo.currentData() if hasattr(self, 'audio_combo') else ""
        if not self.voice.start(mic):
            self.video.show_toast(f"Voice Lip-Sync: {self.voice.error}")

    def _on_mic_changed(self, _):
        self._save("rec/audio", self.audio_combo.currentData())
        if self.voice.running:
            self._apply_voice(True)

    def _apply_hud(self, v):
        self.video.show_hud = bool(v)
        self.btn_hud.set_active_state(bool(v))

    def _apply_hand_mode(self, i):
        # Smart: every frame while a hand is near your face, otherwise every N frames
        # (N from the performance preset). Every Frame: always. Off: no hand tracking.
        self.hands.enabled = i != 2
        base_n = {0: 1, 1: 2, 2: 3}.get(self.seg_preset.current_index() if hasattr(self, 'seg_preset') else 1, 2)
        self.hands.every_n = 1 if i == 1 else base_n

    def _apply_preset(self, i):
        """
        One-tap bundles. Settings chosen here are ordinary settings you can still tweak.
        The camera resolution is deliberately NOT part of a preset: Quality used to switch
        to 1080p, and a phone camera at 1080p (DroidCam over Wi-Fi) plus 2.25x the pixels in
        every stage is what made it crawl at 1-15 FPS. Presets now differ where it counts at
        the resolution you picked: matte network, matte detail, face-tracking detail and
        how often hands are tracked.
        """
        self._save("perf/preset", i)
        cuda = self.bg_engine.has_cuda
        if i == 0:     # Quality
            engine = BackgroundEngine.ENGINE_RVM_RESNET50 if cuda else BackgroundEngine.ENGINE_MEDIAPIPE
            proxy, detail = 640, 50
        elif i == 1:   # Balanced
            engine = BackgroundEngine.ENGINE_RVM_MOBILENET if cuda else BackgroundEngine.ENGINE_MEDIAPIPE
            proxy, detail = 512, 40
        else:          # Low-End
            engine = BackgroundEngine.ENGINE_RVM_MOBILENET if cuda else BackgroundEngine.ENGINE_MEDIAPIPE
            proxy, detail = 384, 25
        self.tracker.set_proxy_width(proxy)
        self.settings.setValue("perf/proxy", proxy)
        k = self.matting_engine_combo.findData(engine)
        if k >= 0:
            self.matting_engine_combo.setCurrentIndex(k)
        self.slider_detail.setValue(detail)
        # Fast hands vanishing is what Solid Hands prevents — every preset turns it on.
        self.row_solid_hands.switch.setChecked(True)
        self._apply_hand_mode(int(self._load("perf/hands", 0, int)))
        self._update_preset_label()
        if not self._loading:
            self.video.show_toast(f"{PRESETS[i]} preset applied")

    def _update_preset_label(self):
        i = self.seg_preset.current_index()
        txt = {
            0: "Quality: ResNet50 matte at high detail (sharpest hair/finger edges), hi-res face tracking, hands every "
               "frame. Keeps your camera resolution — 720p is recommended; 1080p from a phone is much slower.",
            1: "Balanced: fast neural matte, hands tracked every other frame (every frame near your face).",
            2: "Low-End: lightest matte and tracking — leaves the most CPU/GPU for OBS, Premiere or After Effects.",
        }[i]
        self.lbl_preset.setText(txt)

    # ------------------------------------------------------------------ theme
    def set_theme(self, theme):
        self.current_theme = theme
        self.settings.setValue("ui/theme", theme)
        for w in self.all_apple_widgets:
            if hasattr(w, "set_theme"):
                w.set_theme(theme)
        light = theme == "light"
        bg, fg, sub = ("#F2F2F7", "#1D1D1F", "#6E6E73") if light else ("#121214", "#F2F2F7", "#8E8E93")
        panel, border = ("#FFFFFF", "#E5E5EA") if light else ("#161618", "#2C2C2E")
        combo_bg, combo_bd = ("#FFFFFF", "#D1D1D6") if light else ("#242426", "#38383A")
        accent = "#007AFF" if light else "#0A84FF"
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {bg}; }}
            QWidget {{ color: {fg}; font-family: "Segoe UI Variable Text", "Segoe UI", sans-serif; }}
            QLabel {{ color: {sub}; }}
            QComboBox {{ background-color: {combo_bg}; color: {fg}; border: 1px solid {combo_bd}; border-radius: 8px;
                         padding: 4px 10px; font-size: 12px; min-height: 20px; max-width: 210px; }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox QAbstractItemView {{ background-color: {combo_bg}; color: {fg}; selection-background-color: {accent};
                                          selection-color: #FFFFFF; border: 1px solid {combo_bd}; }}
            QScrollBar:vertical {{ width: 6px; background: transparent; }}
            QScrollBar::handle:vertical {{ background: {combo_bd}; border-radius: 3px; min-height: 24px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        """)
        self.dock_frame.setStyleSheet(f"QFrame#dockFrame {{ background-color: {panel}; border-radius: 20px; border: 1px solid {border}; }}")
        self.sidebar_frame.setStyleSheet(f"QFrame#sidebarFrame {{ background-color: {panel}; border-radius: 16px; border: 1px solid {border}; }}")
        self.lbl_studio_title.setStyleSheet(f"color: {fg};")
        self.lbl_studio_sub.setStyleSheet(f"color: {sub}; font-size: 11px;")
        self.status_bar_label.setStyleSheet(f"color: {sub}; font-size: 11px; padding-left: 6px;")
        if not self.recorder.is_recording:
            self.lbl_record_timer.setStyleSheet(f"color: {sub}; font-size: 13px; font-weight: 600;")

    # ------------------------------------------------------------------ shortcuts
    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._toggle_record)
        QShortcut(QKeySequence("V"), self, activated=self._toggle_virtual_cam)
        QShortcut(QKeySequence("H"), self, activated=self._toggle_hud)
        QShortcut(QKeySequence("B"), self, activated=self._toggle_curtain)
        QShortcut(QKeySequence("C"), self, activated=self._calibrate_face)
        QShortcut(QKeySequence("S"), self, activated=self._snapshot)

    # ------------------------------------------------------------------ camera / pipeline
    def _populate_cameras(self):
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for idx, dev in enumerate(list_camera_devices()):
            self.camera_combo.addItem(dev, idx)
        if os.path.exists(self.test_video_path):
            self.camera_combo.addItem("Demo Video File", self.test_video_path)
        self.camera_combo.addItem("Video File...", "__file__")
        self.camera_combo.blockSignals(False)

    def _initial_source(self):
        saved = self.settings.value("cam/device", "")
        names = [self.camera_combo.itemText(i) for i in range(self.camera_combo.count())]
        pick = None
        if saved and saved in names:
            pick = names.index(saved)
        else:
            for i, n in enumerate(names):
                if "droidcam" in n.lower():
                    pick = i
                    break
        if pick is None:
            pick = 0
        self.camera_combo.blockSignals(True)
        self.camera_combo.setCurrentIndex(pick)
        self.camera_combo.blockSignals(False)
        data = self.camera_combo.itemData(pick)
        return data if data != "__file__" else 0

    def _refresh_cameras(self):
        cur = self.camera_combo.currentText()
        self._populate_cameras()
        k = self.camera_combo.findText(cur)
        self.camera_combo.blockSignals(True)
        self.camera_combo.setCurrentIndex(max(0, k))
        self.camera_combo.blockSignals(False)

    def _on_camera_changed(self, index):
        if index < 0:
            return
        data = self.camera_combo.itemData(index)
        if data == "__file__":
            fname, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.mkv *.avi *.mov)")
            if not fname:
                return
            data = fname
        else:
            self.settings.setValue("cam/device", self.camera_combo.itemText(index))
        self._start_pipeline(data)

    def _on_resolution_changed(self, idx):
        self._save("cam/res", idx)
        if self.current_source is not None and not self._loading:
            self._start_pipeline(self.current_source)

    def _on_engine_changed(self, _):
        key = self.matting_engine_combo.currentData()
        self._save("bg/engine", key)
        self.bg_engine.set_engine_type(key)

    def _start_pipeline(self, source):
        if self.pipeline_worker:
            try:
                self.pipeline_worker.frame_ready.disconnect()
            except Exception:
                pass
            self.pipeline_worker.stop()
            self.pipeline_worker = None
        if self.capture_thread:
            self.capture_thread.stop()
        self.tracker.reset()
        self.bg_engine.reset_states()
        self.current_source = source
        self.video.placeholder = f"Connecting to {self.camera_combo.currentText()}..."
        res = self.res_combo.currentData() or (1280, 720)
        self.capture_thread = CameraCaptureThread(source=source, target_fps=30, width=res[0], height=res[1],
                                                  mirror=self._load("cam/mirror", False, bool))
        self.capture_thread.start()
        w = PipelineWorker(self.capture_thread, self.tracker, self.hands, self.scribble_engine, self.bg_engine,
                           self.recorder, self.virtual_cam)
        for k, v in self.pipe.items():
            setattr(w, k, v)
        w.voice = self.voice
        w.frame_ready.connect(self._on_frame_ready)
        w.status_updated.connect(self.status_bar_label.setText)
        self.pipeline_worker = w
        w.start()

    def _on_frame_ready(self):
        if self.pipeline_worker is None:
            return
        item = self.pipeline_worker.take_latest()
        if item is None:
            return
        frame, m = item
        self.video.people = m.get('people', []) if self.bg_engine.isolator.enabled else []
        self.video.keep_zones = list(self.bg_engine.isolator.keep_zones) if self.bg_engine.isolator.enabled else []
        self.video.set_frame(frame, m)
        h, w = frame.shape[:2]
        cap = self.capture_thread
        cam_fps = cap.actual_fps if cap else 0
        state = {'helmet': "Head locked", 'holding': "Holding helmet (tracking recovering)",
                 'mosaic': "Mosaic until face found", 'curtain': "Be Right Back screen", 'clear': "No person"}
        self.status_bar_label.setText(
            f"{w}x{h}  |  camera {cam_fps:.0f} fps  |  {state.get(m.get('privacy'), '')}  |  matte: {m.get('engine') or 'off'}"
            + ("  |  REC" if m.get('is_recording') else "") + ("  |  V-CAM" if m.get('is_vcam') else "")
            + "      Space rec · V v-cam · B be-right-back · C calibrate · S snapshot · H HUD")

    # ------------------------------------------------------------------ interactions
    def _on_video_clicked(self, xn, yn, right):
        iso = self.bg_engine.isolator
        if not iso.enabled:
            return
        if right and iso.remove_zone_at(xn, yn):
            self.video.show_toast("Keep zone removed")
            return
        r = iso.click(xn, yn)
        if r is not None:
            tid, kept = r
            if tid == iso.me_tid:
                self.video.show_toast("That's you — you're always kept")
            else:
                self.video.show_toast(f"Person {tid} will be {'shown' if kept else 'hidden'}")

    def _start_zone(self):
        if not self.bg_engine.isolator.enabled:
            self.video.show_toast("Turn on Person Control first")
            return
        self.video.zone_mode = True
        self.video.setCursor(Qt.CrossCursor)
        self.video.show_toast("Drag a box over what you want to always keep")

    def _on_zone_drawn(self, x0, y0, x1, y1):
        self.bg_engine.isolator.add_keep_zone(x0, y0, x1, y1)
        self.video.show_toast("Keep zone added")

    def _clear_isolation(self):
        self.bg_engine.isolator.clear_selections()
        self.video.show_toast("Person selections and keep zones cleared")

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(self.settings.value("bg/solid", "#00ff00")), self, "Solid Background Color")
        if c.isValid():
            self.settings.setValue("bg/solid", c.name())
            self.bg_engine.set_solid_color((c.blue(), c.green(), c.red()))

    def _pick_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Background Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if fname and self.bg_engine.set_custom_image(fname):
            self.settings.setValue("bg/image", fname)
            self.video.show_toast(f"Background: {os.path.basename(fname)}")

    def _calibrate_face(self):
        self.scribble_engine.features.calibrate_neutral()
        self.video.show_toast("Hold a relaxed face... calibrating neutral expression")

    def _toggle_hud(self):
        v = not self.video.show_hud
        self._apply_hud(v)
        self.settings.setValue("ui/hud", v)

    def _toggle_curtain(self):
        v = not self.pipe.get('curtain', False)
        self._set_pipe('curtain', v)
        self.btn_curtain.set_active_state(v)
        self.video.show_toast("Be Right Back screen ON" if v else "Back live")

    def _toggle_record(self):
        if not self.recorder.is_recording:
            _, frame = self.capture_thread.get_latest_frame() if self.capture_thread else (0, None)
            if frame is None:
                self.video.show_toast("No camera frames yet")
                return
            h, w = frame.shape[:2]
            self.recorder.audio_device = self.audio_combo.currentData() or None
            fps = int(round(self.capture_thread.target_fps)) if self.capture_thread.is_video_file else 30
            self.recorder.start(width=w, height=h, fps=fps)
            self.btn_record.set_recording(True)
            self.record_start = time.time()
            self.rec_timer.start()
            self.lbl_record_timer.setText("00:00")
            self.lbl_record_timer.setStyleSheet("color: #FF3B30; font-weight: 700; font-size: 13px;")
        else:
            self.rec_timer.stop()
            stats = self.recorder.stop()
            self.btn_record.set_recording(False)
            self.lbl_record_timer.setText("Ready")
            sub = "#6E6E73" if self.current_theme == "light" else "#8E8E93"
            self.lbl_record_timer.setStyleSheet(f"color: {sub}; font-size: 13px; font-weight: 600;")
            msg = f"Saved {os.path.basename(stats.get('file_path', ''))}  ({stats.get('duration_sec', 0):.0f}s, {stats.get('file_size_mb', 0):.1f} MB"
            msg += ", + matte" if stats.get('matte_path') else ""
            msg += ")"
            self.video.show_toast(msg, 6.0)
            self.status_bar_label.setText(msg)

    def _snapshot(self):
        """Saves the current output frame (no HUD) as a PNG — great for thumbnails."""
        frame = self.video._frame
        if frame is None:
            return
        d = os.path.join(os.path.abspath(self.recorder.output_dir), "snapshots")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, time.strftime("snapshot_%Y%m%d_%H%M%S.png"))
        ok, buf = cv2.imencode(".png", frame)
        if ok:
            buf.tofile(path)
            self.video.show_toast(f"Snapshot saved: snapshots/{os.path.basename(path)}")

    def _update_record_timer(self):
        if self.recorder.is_recording:
            e = int(time.time() - self.record_start)
            self.lbl_record_timer.setText(f"{e // 60:02d}:{e % 60:02d}")

    def _open_recordings_folder(self):
        d = os.path.abspath(self.recorder.output_dir)
        os.makedirs(d, exist_ok=True)
        os.startfile(d)

    def _toggle_virtual_cam(self):
        if not self.virtual_cam.active:
            _, frame = self.capture_thread.get_latest_frame() if self.capture_thread else (0, None)
            if frame is None:
                self.video.show_toast("No camera frames yet")
                return
            h, w = frame.shape[:2]
            if self.virtual_cam.start(width=w, height=h, fps=30):
                self.btn_vcam.setText("Live V-Cam")
                self.btn_vcam.set_active_state(True)
                self.video.show_toast(f"Virtual camera live: {self.virtual_cam.device}")
            else:
                QMessageBox.warning(self, "Virtual Camera", f"Could not open the virtual camera:\n{self.virtual_cam.last_error}\n\n"
                                    "Install OBS Studio (it provides 'OBS Virtual Camera').")
        else:
            self.virtual_cam.stop()
            self.btn_vcam.setText("Virtual Cam")
            self.btn_vcam.set_active_state(False)

    def closeEvent(self, event):
        if self.recorder.is_recording:
            self.recorder.stop()
        if self.virtual_cam.active:
            self.virtual_cam.stop()
        if self.pipeline_worker:
            self.pipeline_worker.stop()
        if self.capture_thread:
            self.capture_thread.stop()
        self.tracker.close()
        self.hands.close()
        self.bg_engine.close()
        self.voice.close()
        self.settings.sync()
        event.accept()
