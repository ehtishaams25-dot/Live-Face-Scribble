import os
import sys
import time
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QFont, QIcon, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QCheckBox, QGroupBox,
    QFileDialog, QMessageBox, QFrame, QSizePolicy, QScrollArea, QStackedWidget
)

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.camera.capture import CameraCaptureThread, list_camera_devices
from app.camera.virtual_cam import VirtualCameraOutput
from app.tracking.face_tracker import FaceTracker
from app.effects.scribble_mask import ScribbleMaskEngine
from app.effects.background_engine import BackgroundEngine
from app.recording.video_recorder import VideoRecorder
from app.apple_widgets import (
    AppleSwitch, AppleSegmentedControl, AppleCard, AppleRow,
    AppleSliderRow, AppleDisclosureSection, AppleRecordButton, ApplePillButton
)


class PipelineWorker(QThread):
    """
    Background worker thread running 3D tracking, effects, compositing,
    environment relighting, hand foreground occlusion, recording, and virtual camera.
    """
    frame_processed = Signal(np.ndarray, dict)
    status_updated = Signal(str)

    def __init__(self, capture_thread: CameraCaptureThread,
                 tracker: FaceTracker,
                 scribble_engine: ScribbleMaskEngine,
                 bg_engine: BackgroundEngine,
                 recorder: VideoRecorder,
                 virtual_cam: VirtualCameraOutput):
        super().__init__()
        self.capture = capture_thread
        self.tracker = tracker
        self.scribble_engine = scribble_engine
        self.bg_engine = bg_engine
        self.recorder = recorder
        self.virtual_cam = virtual_cam

        self.running = True
        
        # 3D Head Space Scribble Helmet settings
        self.enable_scribble = True
        self.enable_expressions = True
        self.scribble_shape = "head"
        self.scribble_density = 90
        self.scribble_thickness = 3
        self.offset_x = 0.0
        self.offset_y = 1.0
        self.offset_z = -0.5
        self.scribble_scale = 1.45
        self.parallax_gain = 1.6
        self.enable_boil = True
        self.boil_fps = 12.0
        self.sketch_fps = 12.0
        self.sketch_noise = 0.85
        self.obscuration_strength = 0.95
        self.scribble_style = "white"
        self.cavity_darkness = 0.85
        self.tongue_style = "sprite"
        self.tongue_sensitivity = "medium"

        # Background & Relighting settings (Default: MediaPipe for zero-ghosting on hands)
        self.bg_mode = BackgroundEngine.MODE_ORIGINAL
        self.blur_strength = 55
        self.enable_relighting = False
        self.rim_intensity = 0.20
        self.ambient_intensity = 0.0
        self.use_guided_filter = False
        self.custom_bg_path = None
        self.matting_engine = BackgroundEngine.ENGINE_RVM_MOBILENET if getattr(bg_engine, 'has_cuda', False) else BackgroundEngine.ENGINE_MEDIAPIPE
        self.downsample_ratio = 0.45
        self.edge_choke = 0
        self.enable_decontamination = True

        self.last_frame_id = -1
        self.metrics = {
            'fps': 0.0,
            'total_ms': 0.0,
            'track_ms': 0.0,
            'bg_ms': 0.0,
            'scribble_ms': 0.0,
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0,
            'skull_trans': np.zeros(3),
            'has_face': False,
            'is_coasting': False,
            'confidence': 1.0,
            'tracking_mode': 'lost',
            'privacy_shield_active': True,
            'relight_enabled': True,
            'is_recording': False,
            'is_vcam': False
        }

    def run(self):
        fps_count = 0
        fps_timer = time.time()
        start_time = time.time()
        
        while self.running:
            t0 = time.time()
            elapsed_sec = t0 - start_time
            frame_id, frame = self.capture.get_latest_frame()

            if frame is None or frame_id == self.last_frame_id:
                time.sleep(0.005)
                continue

            self.last_frame_id = frame_id
            h, w = frame.shape[:2]
            ts_ms = int(time.time() * 1000)

            try:
                # 1. 3D Head Space Tracking
                t_track0 = time.time()
                track_res = self.tracker.process_frame(frame, timestamp_ms=ts_ms)
                track_ms = (time.time() - t_track0) * 1000.0

                # 2. Background Segmentation & Relighting
                t_bg0 = time.time()
                self.bg_engine.use_guided_filter = self.use_guided_filter
                self.bg_engine.set_relighting_params(
                    enabled=self.enable_relighting,
                    rim_intensity=self.rim_intensity,
                    ambient_intensity=self.ambient_intensity
                )
                composited = self.bg_engine.process(
                    frame_bgr=frame,
                    mode=self.bg_mode,
                    blur_strength=self.blur_strength,
                    timestamp_ms=ts_ms,
                    engine_type=self.matting_engine,
                    downsample_ratio=self.downsample_ratio,
                    edge_choke=self.edge_choke,
                    enable_decontamination=self.enable_decontamination
                )
                bg_ms = (time.time() - t_bg0) * 1000.0
                hand_mask = self.bg_engine.get_latest_hand_mask()

                # 3. 3D Anatomical Scribble Helmet with Hand Occlusion Awareness
                t_sc0 = time.time()
                if track_res.get('has_face', False):
                    if self.enable_scribble:
                        self.scribble_engine.set_shape(self.scribble_shape)
                        self.scribble_engine.set_density(self.scribble_density)
                        self.scribble_engine.set_line_width(self.scribble_thickness)
                        self.scribble_engine.set_offsets(self.offset_x, self.offset_y, self.offset_z)
                        self.scribble_engine.set_tongue_style(self.tongue_style)
                        self.scribble_engine.set_tongue_sensitivity(self.tongue_sensitivity)

                        composited = self.scribble_engine.apply(
                            frame_bgr=composited,
                            tracking_data=track_res,
                            size_scale=self.scribble_scale,
                            parallax_gain=self.parallax_gain,
                            obscuration_strength=self.obscuration_strength,
                            enable_boil=self.enable_boil,
                            style=self.scribble_style,
                            cavity_darkness=self.cavity_darkness,
                            sketch_fps=self.sketch_fps,
                            sketch_noise=self.sketch_noise,
                            t_sec=elapsed_sec,
                            enable_expressions=self.enable_expressions,
                            clean_camera_frame=composited,
                            timestamp_ms=ts_ms,
                            hand_mask=hand_mask
                        )

                scribble_ms = (time.time() - t_sc0) * 1000.0

                # 4. Asynchronous Recording
                if self.recorder.is_recording:
                    self.recorder.write_frame(composited)

                # 5. Virtual Camera Streaming
                if self.virtual_cam.active:
                    self.virtual_cam.send_frame(composited)

                total_ms = (time.time() - t0) * 1000.0

                # FPS calculation
                fps_count += 1
                if time.time() - fps_timer >= 0.5:
                    self.metrics['fps'] = fps_count / (time.time() - fps_timer)
                    fps_count = 0
                    fps_timer = time.time()

                self.metrics['total_ms'] = total_ms
                self.metrics['track_ms'] = track_ms
                self.metrics['bg_ms'] = bg_ms
                self.metrics['scribble_ms'] = scribble_ms
                self.metrics['has_face'] = track_res.get('has_face', False)
                self.metrics['is_coasting'] = track_res.get('is_coasting', False)
                self.metrics['confidence'] = track_res.get('confidence', 0.0)
                self.metrics['tracking_mode'] = track_res.get('tracking_mode', 'lost')
                self.metrics['relight_enabled'] = self.enable_relighting
                if self.metrics['has_face']:
                    self.metrics['yaw'] = track_res['yaw']
                    self.metrics['pitch'] = track_res['pitch']
                    self.metrics['roll'] = track_res['roll']
                    self.metrics['skull_trans'] = track_res.get('translation_skull', np.zeros(3))
                self.metrics['is_recording'] = self.recorder.is_recording
                self.metrics['is_vcam'] = self.virtual_cam.active

                # Send to UI
                self.frame_processed.emit(composited, self.metrics.copy())
            except Exception as e:
                print(f"[PipelineWorker] Error processing frame: {e}")

    def stop(self):
        self.running = False
        self.wait(1000)


class ScribbleCameraWindow(QMainWindow):
    """
    Scribble Studio — Apple-inspired Minimalist Creative Video Interface.
    Features:
    - Apple Light and Dark theme switching with Apple System Blue (#007AFF) accents
    - True 3D anatomical scribble helmet wrapping around skull front, crown, sides, and back
    - Real-time foreground hand occlusion (hands in front of helmet)
    - Zero emojis anywhere in the interface
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scribble Studio")
        self.resize(1340, 860)
        self.setMinimumSize(1040, 680)

        # Paths
        self.models_dir = os.path.join(BASE_DIR, "app", "models")
        self.lmk_model = os.path.join(self.models_dir, "face_landmarker.task")
        self.pose_model = os.path.join(self.models_dir, "pose_landmarker_full.task")
        self.seg_model = os.path.join(self.models_dir, "selfie_segmenter.tflite")
        self.scribble_tex = os.path.join(self.models_dir, "assets", "scribble_texture.png")
        self.test_video_path = r"D:\Ehtishaam\Files\Mine\New Branding\Scribble face vid test\VID_20260430_222927.mp4"

        # Theme state: "dark" or "light"
        self.current_theme = "dark"
        self.all_apple_widgets = []

        # Core engines
        self.tracker = FaceTracker(self.lmk_model, pose_model_path=self.pose_model)
        self.scribble_engine = ScribbleMaskEngine(self.scribble_tex, boil_fps=12.0)
        self.bg_engine = BackgroundEngine(self.seg_model, models_dir=self.models_dir)
        self.recorder = VideoRecorder(output_dir=os.path.join(BASE_DIR, "recordings"))
        self.virtual_cam = VirtualCameraOutput()

        # Capture & pipeline threads
        self.capture_thread = None
        self.pipeline_worker = None

        self.current_source = 0
        self.show_hud = True
        self.record_start_sec = 0.0

        # Recording duration timer
        self.rec_timer = QTimer(self)
        self.rec_timer.setInterval(1000)
        self.rec_timer.timeout.connect(self._update_record_timer)

        self._setup_ui()
        self.set_theme("dark")
        self._start_pipeline(source=0)

    def _setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(14)

        # =========================================================================
        # LEFT: CONTENT-FIRST VIDEO VIEWPORT + FLOATING ACTION DOCK
        # =========================================================================
        viewport_container = QWidget()
        viewport_layout = QVBoxLayout(viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(10)

        # Video canvas
        self.video_label = QLabel()
        self.video_label.setObjectName("videoCanvas")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        viewport_layout.addWidget(self.video_label, stretch=1)

        # Floating Action Dock
        self.dock_frame = QFrame()
        self.dock_frame.setObjectName("dockFrame")
        self.dock_frame.setFixedHeight(68)
        dock_layout = QHBoxLayout(self.dock_frame)
        dock_layout.setContentsMargins(18, 6, 18, 6)
        dock_layout.setSpacing(12)
        dock_layout.setAlignment(Qt.AlignVCenter)

        # Left action group (Broadcast & HUD)
        self.btn_vcam = ApplePillButton("Virtual Cam", theme=self.current_theme)
        self.btn_vcam.clicked.connect(self._toggle_virtual_cam)
        self.all_apple_widgets.append(self.btn_vcam)
        dock_layout.addWidget(self.btn_vcam)

        self.btn_hud = ApplePillButton("HUD", theme=self.current_theme)
        self.btn_hud.set_active_state(True)
        self.btn_hud.clicked.connect(self._toggle_hud_dock)
        self.all_apple_widgets.append(self.btn_hud)
        dock_layout.addWidget(self.btn_hud)

        dock_layout.addStretch(1)

        # Center action group: Camera Shutter & Timer
        shutter_container = QWidget()
        shutter_container.setStyleSheet("background: transparent; border: none;")
        shutter_layout = QHBoxLayout(shutter_container)
        shutter_layout.setContentsMargins(0, 0, 0, 0)
        shutter_layout.setSpacing(10)
        shutter_layout.setAlignment(Qt.AlignCenter)

        self.btn_record = AppleRecordButton(theme=self.current_theme)
        self.btn_record.clicked.connect(self._toggle_record)
        self.all_apple_widgets.append(self.btn_record)
        shutter_layout.addWidget(self.btn_record, alignment=Qt.AlignVCenter)

        self.lbl_record_timer = QLabel("Ready")
        self.lbl_record_timer.setFixedWidth(70)
        self.lbl_record_timer.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.lbl_record_timer.setStyleSheet("font-size: 13px; font-weight: 600; background: transparent; border: none;")
        shutter_layout.addWidget(self.lbl_record_timer, alignment=Qt.AlignVCenter)
        dock_layout.addWidget(shutter_container)

        dock_layout.addStretch(1)

        # Right action group: Open Recordings Folder
        self.btn_open_rec = ApplePillButton("Recordings", theme=self.current_theme)
        self.btn_open_rec.clicked.connect(self._open_recordings_folder)
        self.all_apple_widgets.append(self.btn_open_rec)
        dock_layout.addWidget(self.btn_open_rec)

        viewport_layout.addWidget(self.dock_frame, stretch=0)

        # Status note
        self.status_bar_label = QLabel("Connecting camera...")
        self.status_bar_label.setStyleSheet("font-size: 11px; padding-left: 6px;")
        viewport_layout.addWidget(self.status_bar_label, stretch=0)

        main_layout.addWidget(viewport_container, stretch=3)

        # =========================================================================
        # RIGHT: APPLE INSPECTOR SIDEBAR WITH SEGMENTED TABS
        # =========================================================================
        self.sidebar_frame = QFrame()
        self.sidebar_frame.setObjectName("sidebarFrame")
        self.sidebar_frame.setFixedWidth(355)
        sidebar_layout = QVBoxLayout(self.sidebar_frame)
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(12)

        # Sidebar Header with Theme Switcher (Fixed height, no vertical gap)
        header_widget = QWidget()
        header_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        header_widget.setStyleSheet("background: transparent; border: none;")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        titles_box = QWidget()
        titles_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        titles_box.setStyleSheet("background: transparent; border: none;")
        titles_layout = QVBoxLayout(titles_box)
        titles_layout.setAlignment(Qt.AlignTop)
        titles_layout.setContentsMargins(0, 0, 0, 0)
        titles_layout.setSpacing(1)

        self.lbl_studio_title = QLabel("Scribble Studio")
        self.lbl_studio_title.setFont(QFont("Segoe UI", 15, QFont.Bold))
        titles_layout.addWidget(self.lbl_studio_title)

        self.lbl_studio_sub = QLabel("Real-Time 3D & Spatial Relighting")
        self.lbl_studio_sub.setStyleSheet("font-size: 11px;")
        titles_layout.addWidget(self.lbl_studio_sub)
        header_layout.addWidget(titles_box, stretch=1)

        # Theme Switcher Segmented Control
        self.seg_theme = AppleSegmentedControl(["Light", "Dark"], theme=self.current_theme)
        self.seg_theme.setFixedWidth(110)
        self.seg_theme.set_current_index(1)  # Dark default
        self.seg_theme.currentIndexChanged.connect(self._on_theme_changed)
        self.all_apple_widgets.append(self.seg_theme)
        header_layout.addWidget(self.seg_theme, stretch=0)

        sidebar_layout.addWidget(header_widget, stretch=0)

        # macOS Segmented Navigation Bar (NO EMOJIS)
        self.tab_nav = AppleSegmentedControl(["Helmet", "Stage", "Studio"], theme=self.current_theme)
        self.all_apple_widgets.append(self.tab_nav)
        sidebar_layout.addWidget(self.tab_nav, stretch=0)

        # Stacked Pages
        self.tabs_stack = QStackedWidget()
        self.tab_nav.currentIndexChanged.connect(self.tabs_stack.setCurrentIndex)
        sidebar_layout.addWidget(self.tabs_stack, stretch=1)

        # -------------------------------------------------------------------------
        # TAB 0: HELMET & MIMICRY
        # -------------------------------------------------------------------------
        page_helmet_scroll = self._create_scrollable_container()
        page_helmet_layout = QVBoxLayout(page_helmet_scroll.widget())
        page_helmet_layout.setContentsMargins(0, 4, 0, 4)
        page_helmet_layout.setSpacing(12)

        # Card: 3D Scribble Helmet
        card_helmet = AppleCard("3D Helmet", theme=self.current_theme)
        self.all_apple_widgets.append(card_helmet)

        self.chk_scribble = AppleSwitch(checked=True, theme=self.current_theme)
        self.chk_scribble.toggled.connect(self._on_scribble_toggled)
        self.all_apple_widgets.append(self.chk_scribble)
        row_scribble = AppleRow("3D Scribble Helmet", self.chk_scribble, "Render responsive spatial head rig", theme=self.current_theme)
        self.all_apple_widgets.append(row_scribble)
        card_helmet.add_widget(row_scribble)

        # 5 Predetermined Helmet Shapes: Head, Cube, Star, Diamond, Pill
        self.seg_helmet_shape = AppleSegmentedControl(["Head", "Cube", "Star", "Diamond", "Pill"], theme=self.current_theme)
        self.seg_helmet_shape.currentIndexChanged.connect(self._on_shape_changed)
        self.all_apple_widgets.append(self.seg_helmet_shape)
        row_shape = AppleRow("Shape", self.seg_helmet_shape, theme=self.current_theme)
        self.all_apple_widgets.append(row_shape)
        card_helmet.add_widget(row_shape)

        self.seg_helmet_style = AppleSegmentedControl(["Chalk", "Charcoal", "Cyber"], theme=self.current_theme)
        self.seg_helmet_style.currentIndexChanged.connect(self._on_scribble_style_changed)
        self.all_apple_widgets.append(self.seg_helmet_style)
        row_style = AppleRow("Style", self.seg_helmet_style, theme=self.current_theme)
        self.all_apple_widgets.append(row_style)
        card_helmet.add_widget(row_style)

        self.slider_density = AppleSliderRow("Coil Density", 30, 150, 90, unit=" coils", theme=self.current_theme)
        self.slider_density.valueChanged.connect(self._on_density_changed)
        self.all_apple_widgets.append(self.slider_density)
        card_helmet.add_widget(self.slider_density)

        self.slider_size = AppleSliderRow("Helmet Scale", 70, 200, 145, unit="%", theme=self.current_theme)
        self.slider_size.valueChanged.connect(self._on_size_changed)
        self.all_apple_widgets.append(self.slider_size)
        card_helmet.add_widget(self.slider_size)

        self.slider_thickness = AppleSliderRow("Line Width", 1, 6, 3, unit=" px", theme=self.current_theme)
        self.slider_thickness.valueChanged.connect(self._on_thickness_changed)
        self.all_apple_widgets.append(self.slider_thickness)
        card_helmet.add_widget(self.slider_thickness)
        page_helmet_layout.addWidget(card_helmet)

        # Card: Expressions & Motion
        card_expressions = AppleCard("Expressions & Motion", theme=self.current_theme)
        self.all_apple_widgets.append(card_expressions)

        self.chk_expressions = AppleSwitch(checked=True, theme=self.current_theme)
        self.chk_expressions.toggled.connect(self._on_expressions_toggled)
        self.all_apple_widgets.append(self.chk_expressions)
        row_exp = AppleRow("Mimic Expressions", self.chk_expressions, "Hand-drawn sketch sprite mimicry", theme=self.current_theme)
        self.all_apple_widgets.append(row_exp)
        card_expressions.add_widget(row_exp)

        self.chk_boil = AppleSwitch(checked=True, theme=self.current_theme)
        self.chk_boil.toggled.connect(self._on_boil_toggled)
        self.all_apple_widgets.append(self.chk_boil)
        row_boil = AppleRow("12 FPS Sketch Boil", self.chk_boil, "Stop-motion line vibration", theme=self.current_theme)
        self.all_apple_widgets.append(row_boil)
        card_expressions.add_widget(row_boil)

        self.seg_tongue_style = AppleSegmentedControl(["Reference", "Chalk Red"], theme=self.current_theme)
        self.seg_tongue_style.currentIndexChanged.connect(self._on_tongue_style_changed)
        self.all_apple_widgets.append(self.seg_tongue_style)
        row_tongue = AppleRow("Tongue Style", self.seg_tongue_style, theme=self.current_theme)
        self.all_apple_widgets.append(row_tongue)
        card_expressions.add_widget(row_tongue)

        self.seg_tongue_sens = AppleSegmentedControl(["Medium", "Low", "High", "Off"], theme=self.current_theme)
        self.seg_tongue_sens.currentIndexChanged.connect(self._on_tongue_sens_changed)
        self.all_apple_widgets.append(self.seg_tongue_sens)
        row_sens = AppleRow("Sensitivity", self.seg_tongue_sens, theme=self.current_theme)
        self.all_apple_widgets.append(row_sens)
        card_expressions.add_widget(row_sens)
        page_helmet_layout.addWidget(card_expressions)

        # Progressive Disclosure: Fine-Tuning & Offset Calibration
        disclosure_tuning = AppleDisclosureSection("Fine-Tuning & Offset Calibration", theme=self.current_theme)
        self.all_apple_widgets.append(disclosure_tuning)

        self.slider_offset_y = AppleSliderRow("Vertical (Y)", -100, 100, 15, formatter=lambda v: f"{v/10:+.1f} cm", theme=self.current_theme)
        self.slider_offset_y.valueChanged.connect(self._on_offset_y_changed)
        self.all_apple_widgets.append(self.slider_offset_y)
        disclosure_tuning.add_widget(self.slider_offset_y)

        self.slider_offset_x = AppleSliderRow("Horizontal (X)", -100, 100, 0, formatter=lambda v: f"{v/10:+.1f} cm", theme=self.current_theme)
        self.slider_offset_x.valueChanged.connect(self._on_offset_x_changed)
        self.all_apple_widgets.append(self.slider_offset_x)
        disclosure_tuning.add_widget(self.slider_offset_x)

        self.slider_offset_z = AppleSliderRow("Depth (Z)", -100, 100, -5, formatter=lambda v: f"{v/10:+.1f} cm", theme=self.current_theme)
        self.slider_offset_z.valueChanged.connect(self._on_offset_z_changed)
        self.all_apple_widgets.append(self.slider_offset_z)
        disclosure_tuning.add_widget(self.slider_offset_z)

        self.slider_noise = AppleSliderRow("Tooth Noise", 0, 150, 85, unit="%", theme=self.current_theme)
        self.slider_noise.valueChanged.connect(self._on_noise_changed)
        self.all_apple_widgets.append(self.slider_noise)
        disclosure_tuning.add_widget(self.slider_noise)
        page_helmet_layout.addWidget(disclosure_tuning)

        page_helmet_layout.addStretch(1)
        self.tabs_stack.addWidget(page_helmet_scroll)

        # -------------------------------------------------------------------------
        # TAB 1: STAGE & COMPOSITING
        # -------------------------------------------------------------------------
        page_stage_scroll = self._create_scrollable_container()
        page_stage_layout = QVBoxLayout(page_stage_scroll.widget())
        page_stage_layout.setContentsMargins(0, 4, 0, 4)
        page_stage_layout.setSpacing(12)

        # Card: Background Environment
        card_bg = AppleCard("Environment Stage", theme=self.current_theme)
        self.all_apple_widgets.append(card_bg)

        self.seg_bg_mode = AppleSegmentedControl(["Original", "Blur", "Studio", "Solid", "Custom"], theme=self.current_theme)
        self.seg_bg_mode.currentIndexChanged.connect(self._on_bg_mode_changed)
        self.all_apple_widgets.append(self.seg_bg_mode)
        card_bg.add_widget(self.seg_bg_mode)

        self.slider_blur = AppleSliderRow("Blur Radius", 15, 95, 55, unit="%", theme=self.current_theme)
        self.slider_blur.valueChanged.connect(self._on_blur_changed)
        self.slider_blur.setVisible(False)
        self.all_apple_widgets.append(self.slider_blur)
        card_bg.add_widget(self.slider_blur)

        self.btn_pick_img = ApplePillButton("Choose Custom Image...", is_accent=True, theme=self.current_theme)
        self.btn_pick_img.clicked.connect(self._on_pick_custom_bg)
        self.btn_pick_img.setVisible(False)
        self.all_apple_widgets.append(self.btn_pick_img)
        card_bg.add_widget(self.btn_pick_img)
        page_stage_layout.addWidget(card_bg)

        # Card: Spatial Relighting
        card_relight = AppleCard("Spatial Relighting & Light Wrap", theme=self.current_theme)
        self.all_apple_widgets.append(card_relight)

        self.chk_relight = AppleSwitch(checked=False, theme=self.current_theme)
        self.chk_relight.toggled.connect(self._on_relight_toggled)
        self.all_apple_widgets.append(self.chk_relight)
        row_relight = AppleRow("Environment Relight", self.chk_relight, "Subtle edge light wrap integration", theme=self.current_theme)
        self.all_apple_widgets.append(row_relight)
        card_relight.add_widget(row_relight)

        self.slider_rim = AppleSliderRow("Rim Glow", 0, 150, 20, unit="%", theme=self.current_theme)
        self.slider_rim.valueChanged.connect(self._on_rim_changed)
        self.all_apple_widgets.append(self.slider_rim)
        card_relight.add_widget(self.slider_rim)

        self.slider_ambient = AppleSliderRow("Ambient Tint", 0, 40, 0, unit="%", theme=self.current_theme)
        self.slider_ambient.valueChanged.connect(self._on_ambient_changed)
        self.all_apple_widgets.append(self.slider_ambient)
        card_relight.add_widget(self.slider_ambient)
        page_stage_layout.addWidget(card_relight)

        # Progressive Disclosure: Matte & Edge Refinement
        disclosure_matte = AppleDisclosureSection("Matte Precision & Refinement", theme=self.current_theme)
        self.all_apple_widgets.append(disclosure_matte)

        self.matting_engine_combo = QComboBox()
        self.matting_engine_combo.addItem("Neural RVM MobileNet (Speed)", BackgroundEngine.ENGINE_RVM_MOBILENET)
        self.matting_engine_combo.addItem("Neural RVM ResNet50 (Studio)", BackgroundEngine.ENGINE_RVM_RESNET50)
        self.matting_engine_combo.addItem("MediaPipe Segmenter (CPU)", BackgroundEngine.ENGINE_MEDIAPIPE)
        if not getattr(self.bg_engine, 'has_cuda', False):
            self.matting_engine_combo.setCurrentIndex(2)
        else:
            self.matting_engine_combo.setCurrentIndex(0)
        self.matting_engine_combo.currentIndexChanged.connect(self._on_matting_engine_changed)
        row_matting = AppleRow("Matte Engine", self.matting_engine_combo, theme=self.current_theme)
        self.all_apple_widgets.append(row_matting)
        disclosure_matte.add_widget(row_matting)

        self.chk_decontaminate = AppleSwitch(checked=True, theme=self.current_theme)
        self.chk_decontaminate.toggled.connect(self._on_decontaminate_toggled)
        self.all_apple_widgets.append(self.chk_decontaminate)
        row_decontam = AppleRow("Decontaminate Edges", self.chk_decontaminate, "Suppresses boundary halo artifacts", theme=self.current_theme)
        self.all_apple_widgets.append(row_decontam)
        disclosure_matte.add_widget(row_decontam)

        self.chk_guided = AppleSwitch(checked=False, theme=self.current_theme)
        self.chk_guided.toggled.connect(self._on_guided_toggled)
        self.all_apple_widgets.append(self.chk_guided)
        row_guided = AppleRow("Guided Filter (Hair)", self.chk_guided, "Fine hair strand alpha edge refinement", theme=self.current_theme)
        self.all_apple_widgets.append(row_guided)
        disclosure_matte.add_widget(row_guided)

        self.slider_choke = AppleSliderRow("Edge Choke", -4, 2, 0, formatter=lambda v: f"{v:+d} px", theme=self.current_theme)
        self.slider_choke.valueChanged.connect(self._on_choke_changed)
        self.all_apple_widgets.append(self.slider_choke)
        disclosure_matte.add_widget(self.slider_choke)

        self.slider_precision = AppleSliderRow("Matte Detail", 20, 50, 45, unit="%", theme=self.current_theme)
        self.slider_precision.valueChanged.connect(self._on_precision_changed)
        self.all_apple_widgets.append(self.slider_precision)
        disclosure_matte.add_widget(self.slider_precision)
        page_stage_layout.addWidget(disclosure_matte)

        page_stage_layout.addStretch(1)
        self.tabs_stack.addWidget(page_stage_scroll)

        # -------------------------------------------------------------------------
        # TAB 2: STUDIO & CAMERA
        # -------------------------------------------------------------------------
        page_studio_scroll = self._create_scrollable_container()
        page_studio_layout = QVBoxLayout(page_studio_scroll.widget())
        page_studio_layout.setContentsMargins(0, 4, 0, 4)
        page_studio_layout.setSpacing(12)

        # Card: Video Input
        card_input = AppleCard("Video Input", theme=self.current_theme)
        self.all_apple_widgets.append(card_input)

        self.camera_combo = QComboBox()
        self._populate_cameras()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        row_cam = AppleRow("Camera Device", self.camera_combo, theme=self.current_theme)
        self.all_apple_widgets.append(row_cam)
        card_input.add_widget(row_cam)

        self.res_combo = QComboBox()
        self.res_combo.addItem("1080p Full HD (1920x1080)", (1920, 1080))
        self.res_combo.addItem("720p HD (1280x720)", (1280, 720))
        self.res_combo.currentIndexChanged.connect(self._on_resolution_changed)
        row_res = AppleRow("Resolution", self.res_combo, theme=self.current_theme)
        self.all_apple_widgets.append(row_res)
        card_input.add_widget(row_res)

        self.btn_refresh_cam = ApplePillButton("Refresh Connected Cameras", theme=self.current_theme)
        self.btn_refresh_cam.clicked.connect(self._refresh_cameras)
        self.all_apple_widgets.append(self.btn_refresh_cam)
        card_input.add_widget(self.btn_refresh_cam)
        page_studio_layout.addWidget(card_input)

        # Card: Display & Telemetry
        card_display = AppleCard("Display & Telemetry", theme=self.current_theme)
        self.all_apple_widgets.append(card_display)

        self.chk_hud = AppleSwitch(checked=True, theme=self.current_theme)
        self.chk_hud.toggled.connect(self._on_hud_toggled)
        self.all_apple_widgets.append(self.chk_hud)
        row_hud = AppleRow("Overlay HUD", self.chk_hud, "Floating visionOS-style metrics", theme=self.current_theme)
        self.all_apple_widgets.append(row_hud)
        card_display.add_widget(row_hud)

        self.lbl_privacy_info = QLabel("Real-time HUD telemetry displays live FPS, processing latency, head pose angles, and recording status.")
        self.lbl_privacy_info.setWordWrap(True)
        self.lbl_privacy_info.setStyleSheet("font-size: 11px; line-height: 1.4;")
        card_display.add_widget(self.lbl_privacy_info)
        page_studio_layout.addWidget(card_display)

        # Card: Recordings & Storage
        card_storage = AppleCard("Recordings & Storage", theme=self.current_theme)
        self.all_apple_widgets.append(card_storage)

        rec_dir = os.path.abspath(self.recorder.output_dir)
        self.lbl_storage_info = QLabel(f"Location:\n{rec_dir}")
        self.lbl_storage_info.setWordWrap(True)
        self.lbl_storage_info.setStyleSheet("font-size: 11px;")
        card_storage.add_widget(self.lbl_storage_info)

        self.btn_open_rec_sidebar = ApplePillButton("Open Recordings Folder", theme=self.current_theme)
        self.btn_open_rec_sidebar.clicked.connect(self._open_recordings_folder)
        self.all_apple_widgets.append(self.btn_open_rec_sidebar)
        card_storage.add_widget(self.btn_open_rec_sidebar)
        page_studio_layout.addWidget(card_storage)

        page_studio_layout.addStretch(1)
        self.tabs_stack.addWidget(page_studio_scroll)

        sidebar_layout.addWidget(self.tabs_stack)
        main_layout.addWidget(self.sidebar_frame, stretch=0)

    def _on_theme_changed(self, idx):
        theme = "light" if idx == 0 else "dark"
        self.set_theme(theme)

    def set_theme(self, theme: str):
        self.current_theme = theme

        # Update all custom Apple widgets
        for widget in self.all_apple_widgets:
            if hasattr(widget, "set_theme"):
                widget.set_theme(theme)

        if theme == "light":
            # Apple Light Theme: Pure White, Soft Light Gray, and Apple Blue
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #F2F2F7;
                }
                QWidget {
                    color: #1D1D1F;
                    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI Variable Text", "Segoe UI", sans-serif;
                }
                QLabel {
                    color: #6E6E73;
                }
                QComboBox {
                    background-color: #FFFFFF;
                    color: #1D1D1F;
                    border: 1px solid #D1D1D6;
                    border-radius: 8px;
                    padding: 4px 10px;
                    font-size: 12px;
                    font-weight: 500;
                    min-height: 20px;
                }
                QComboBox::drop-down {
                    border: none;
                    width: 20px;
                }
                QComboBox QAbstractItemView {
                    background-color: #FFFFFF;
                    color: #1D1D1F;
                    selection-background-color: #007AFF;
                    selection-color: #FFFFFF;
                    border: 1px solid #D1D1D6;
                    border-radius: 8px;
                    padding: 4px;
                }
                QScrollBar:vertical {
                    width: 6px;
                    background: transparent;
                    margin: 0px;
                }
                QScrollBar::handle:vertical {
                    background: #C7C7CC;
                    border-radius: 3px;
                    min-height: 24px;
                }
                QScrollBar::handle:vertical:hover {
                    background: #AEAEB2;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: none;
                }
            """)
            self.dock_frame.setStyleSheet("""
                QFrame#dockFrame {
                    background-color: #FFFFFF;
                    border-radius: 20px;
                    border: 1px solid #D1D1D6;
                }
            """)
            self.sidebar_frame.setStyleSheet("""
                QFrame#sidebarFrame {
                    background-color: #FFFFFF;
                    border-radius: 16px;
                    border: 1px solid #E5E5EA;
                }
                QScrollArea {
                    border: none;
                    background-color: transparent;
                }
                QScrollArea > QWidget > QWidget {
                    background-color: transparent;
                }
            """)
            self.video_label.setStyleSheet("""
                QLabel#videoCanvas {
                    background-color: #000000;
                    border-radius: 16px;
                    border: 1px solid #D1D1D6;
                }
            """)
            self.lbl_studio_title.setStyleSheet("color: #000000; letter-spacing: -0.2px;")
            self.lbl_studio_sub.setStyleSheet("color: #6E6E73; font-size: 11px;")
            self.status_bar_label.setStyleSheet("color: #6E6E73; font-size: 11px; padding-left: 6px;")
            self.lbl_record_timer.setStyleSheet("color: #1D1D1F; font-size: 13px; font-weight: 600; background: transparent; border: none;")
        else:
            # Apple Dark Theme: Deep Space Black, Silver Text, and Apple Blue
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #121214;
                }
                QWidget {
                    color: #F2F2F7;
                    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI Variable Text", "Segoe UI", sans-serif;
                }
                QLabel {
                    color: #8E8E93;
                }
                QComboBox {
                    background-color: #242426;
                    color: #F2F2F7;
                    border: 1px solid #38383A;
                    border-radius: 8px;
                    padding: 4px 10px;
                    font-size: 12px;
                    font-weight: 500;
                    min-height: 20px;
                }
                QComboBox::drop-down {
                    border: none;
                    width: 20px;
                }
                QComboBox QAbstractItemView {
                    background-color: #242426;
                    color: #F2F2F7;
                    selection-background-color: #0A84FF;
                    selection-color: #FFFFFF;
                    border: 1px solid #38383A;
                    border-radius: 8px;
                    padding: 4px;
                }
                QScrollBar:vertical {
                    width: 6px;
                    background: transparent;
                    margin: 0px;
                }
                QScrollBar::handle:vertical {
                    background: #3A3A3C;
                    border-radius: 3px;
                    min-height: 24px;
                }
                QScrollBar::handle:vertical:hover {
                    background: #505054;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: none;
                }
            """)
            self.dock_frame.setStyleSheet("""
                QFrame#dockFrame {
                    background-color: #1C1C1E;
                    border-radius: 20px;
                    border: 1px solid #2C2C2E;
                }
            """)
            self.sidebar_frame.setStyleSheet("""
                QFrame#sidebarFrame {
                    background-color: #161618;
                    border-radius: 16px;
                    border: 1px solid #2C2C2E;
                }
                QScrollArea {
                    border: none;
                    background-color: transparent;
                }
                QScrollArea > QWidget > QWidget {
                    background-color: transparent;
                }
            """)
            self.video_label.setStyleSheet("""
                QLabel#videoCanvas {
                    background-color: #000000;
                    border-radius: 16px;
                    border: 1px solid #2C2C2E;
                }
            """)
            self.lbl_studio_title.setStyleSheet("color: #F2F2F7; letter-spacing: -0.2px;")
            self.lbl_studio_sub.setStyleSheet("color: #8E8E93; font-size: 11px;")
            self.status_bar_label.setStyleSheet("color: #636366; font-size: 11px; padding-left: 6px;")
            self.lbl_record_timer.setStyleSheet("color: #8E8E93; font-size: 13px; font-weight: 600; background: transparent; border: none;")

    def _create_scrollable_container(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
        """)
        container = QWidget()
        scroll.setWidget(container)
        return scroll

    def _populate_cameras(self):
        self.camera_combo.clear()
        devices = list_camera_devices()
        for idx, dev in enumerate(devices):
            self.camera_combo.addItem(f"{dev}", idx)
        
        if os.path.exists(self.test_video_path):
            self.camera_combo.addItem("Demo Video File", self.test_video_path)
            
        self.camera_combo.addItem("Custom Video File...", "__file__")

    def _refresh_cameras(self):
        cur_idx = self.camera_combo.currentIndex()
        self._populate_cameras()
        if cur_idx < self.camera_combo.count():
            self.camera_combo.setCurrentIndex(cur_idx)

    def _on_camera_changed(self, index):
        if index < 0: return
        data = self.camera_combo.itemData(index)
        if data == "__file__":
            fname, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.mkv *.avi *.mov)")
            if fname:
                self._start_pipeline(source=fname)
            else:
                self.camera_combo.setCurrentIndex(0)
        else:
            self._start_pipeline(source=data)

    def _start_pipeline(self, source):
        # Stop existing
        if self.pipeline_worker:
            try:
                self.pipeline_worker.frame_processed.disconnect()
            except Exception:
                pass
            self.pipeline_worker.stop()
        if self.capture_thread:
            self.capture_thread.stop()

        self.current_source = source
        self.status_bar_label.setText(f"Connecting to source: {source}...")

        # Initialize capture thread
        res = self.res_combo.currentData() if hasattr(self, 'res_combo') and self.res_combo.currentData() else (1920, 1080)
        self.capture_thread = CameraCaptureThread(source=source, target_fps=30, width=res[0], height=res[1])
        self.capture_thread.start()

        # Initialize pipeline worker
        self.pipeline_worker = PipelineWorker(
            capture_thread=self.capture_thread,
            tracker=self.tracker,
            scribble_engine=self.scribble_engine,
            bg_engine=self.bg_engine,
            recorder=self.recorder,
            virtual_cam=self.virtual_cam
        )

        # Synchronize UI values
        self.pipeline_worker.enable_scribble = self.chk_scribble.isChecked()
        self.pipeline_worker.enable_expressions = self.chk_expressions.isChecked()
        style_keys = ["white", "dark", "cyber"]
        self.pipeline_worker.scribble_style = style_keys[self.seg_helmet_style.current_index()]
        self.pipeline_worker.scribble_density = self.slider_density.value()
        self.pipeline_worker.scribble_thickness = self.slider_thickness.value()
        self.pipeline_worker.offset_y = self.slider_offset_y.value() / 10.0
        self.pipeline_worker.offset_x = self.slider_offset_x.value() / 10.0
        self.pipeline_worker.offset_z = self.slider_offset_z.value() / 10.0
        self.pipeline_worker.scribble_scale = self.slider_size.value() / 100.0
        tongue_styles = ["sprite", "organic"]
        self.pipeline_worker.tongue_style = tongue_styles[self.seg_tongue_style.current_index()]
        tongue_sens = ["medium", "low", "high", "off"]
        self.pipeline_worker.tongue_sensitivity = tongue_sens[self.seg_tongue_sens.current_index()]
        self.pipeline_worker.enable_boil = self.chk_boil.isChecked()
        self.pipeline_worker.sketch_noise = self.slider_noise.value() / 100.0
        self.pipeline_worker.blur_strength = self.slider_blur.value()
        self.pipeline_worker.enable_relighting = self.chk_relight.isChecked()
        self.pipeline_worker.rim_intensity = self.slider_rim.value() / 100.0
        self.pipeline_worker.ambient_intensity = self.slider_ambient.value() / 100.0
        self.pipeline_worker.use_guided_filter = self.chk_guided.isChecked()
        self.pipeline_worker.matting_engine = self.matting_engine_combo.currentData()
        self.pipeline_worker.edge_choke = self.slider_choke.value()
        self.pipeline_worker.enable_decontamination = self.chk_decontaminate.isChecked()
        self.pipeline_worker.downsample_ratio = self.slider_precision.value() / 100.0
        self._update_bg_mode_on_worker()

        self.pipeline_worker.frame_processed.connect(self._on_frame_processed)
        self.pipeline_worker.start()

    @Slot(np.ndarray, dict)
    def _on_frame_processed(self, frame_bgr: np.ndarray, metrics: dict):
        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w

        if self.show_hud:
            frame_display = self._draw_apple_hud(frame_bgr, metrics, w, h)
            rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        lbl_size = self.video_label.size()
        pixmap = QPixmap.fromImage(qimg).scaled(
            lbl_size, Qt.KeepAspectRatio, Qt.FastTransformation
        )
        self.video_label.setPixmap(pixmap)

        # Update status bar (NO EMOJIS)
        if metrics.get('has_face', False):
            if metrics.get('is_coasting', False):
                face_str = f"Head: Coasting ({int(metrics.get('confidence', 0)*100)}%)"
            else:
                face_str = "Head: 3D Locked"
        else:
            face_str = "Head: Searching for face"

        rec_str = "  |  REC" if metrics.get('is_recording', False) else ""
        vcam_str = "  |  V-CAM" if metrics.get('is_vcam', False) else ""
        self.status_bar_label.setText(f"{w}x{h}  |  {face_str}{rec_str}{vcam_str}")

    def _draw_apple_hud(self, frame: np.ndarray, metrics: dict, w: int, h: int) -> np.ndarray:
        """
        Renders sleek visionOS/macOS floating translucent pill badges (NO EMOJIS).
        """
        out = frame.copy()

        fps = metrics.get('fps', 0.0)
        tot_ms = metrics.get('total_ms', 0.0)
        has_face = metrics.get('has_face', False)
        is_coasting = metrics.get('is_coasting', False)

        # 1. Top-Left Pill: FPS & Processing Latency
        pw, ph = 240, 32
        px1, py1 = 16, 16
        px2, py2 = px1 + pw, py1 + ph

        if px2 < w and py2 < h:
            roi1 = out[py1:py2, px1:px2]
            pill1 = np.full_like(roi1, (20, 20, 22))
            cv2.addWeighted(pill1, 0.75, roi1, 0.25, 0, roi1)
            cv2.rectangle(out, (px1, py1), (px2, py2), (55, 55, 60), 1)

            dot_col = (0, 122, 255) if fps >= 25 else (50, 180, 255)  # Apple Blue
            cv2.circle(out, (px1 + 16, py1 + 16), 4, dot_col, -1)
            cv2.putText(out, f"{fps:.1f} FPS   {tot_ms:.1f} ms   {w}x{h}", (px1 + 28, py1 + 21),
                        cv2.FONT_HERSHEY_DUPLEX, 0.44, (240, 240, 245), 1, cv2.LINE_AA)

        # 2. Status Pill (Below FPS pill)
        py1_s = py2 + 8
        py2_s = py1_s + 28
        pw_s = 280
        px2_s = px1 + pw_s

        if px2_s < w and py2_s < h:
            roi2 = out[py1_s:py2_s, px1:px2_s]
            pill2 = np.full_like(roi2, (20, 20, 22))
            cv2.addWeighted(pill2, 0.75, roi2, 0.25, 0, roi2)
            cv2.rectangle(out, (px1, py1_s), (px2_s, py2_s), (55, 55, 60), 1)

            if has_face:
                yaw = metrics.get('yaw', 0.0)
                pitch = metrics.get('pitch', 0.0)
                if is_coasting:
                    cv2.circle(out, (px1 + 14, py1_s + 14), 4, (50, 180, 255), -1)
                    cv2.putText(out, "COASTING (PREDICTIVE)", (px1 + 26, py1_s + 18),
                                cv2.FONT_HERSHEY_DUPLEX, 0.38, (200, 200, 210), 1, cv2.LINE_AA)
                else:
                    cv2.circle(out, (px1 + 14, py1_s + 14), 4, (0, 122, 255), -1)  # Apple Blue
                    cv2.putText(out, f"3D HEAD LOCKED  |  Y: {yaw:+.1f}  P: {pitch:+.1f}", (px1 + 26, py1_s + 18),
                                cv2.FONT_HERSHEY_DUPLEX, 0.38, (220, 225, 230), 1, cv2.LINE_AA)
            else:
                cv2.circle(out, (px1 + 14, py1_s + 14), 4, (120, 120, 125), -1)
                cv2.putText(out, "SEARCHING FOR FACE", (px1 + 26, py1_s + 18),
                            cv2.FONT_HERSHEY_DUPLEX, 0.38, (200, 200, 205), 1, cv2.LINE_AA)

        # 3. Top-Right Badges: REC & V-CAM
        rx = w - 16
        if metrics.get('is_recording', False):
            rw, rh = 84, 30
            rx1 = rx - rw
            ry1 = 16
            rx2 = rx
            ry2 = ry1 + rh
            if rx1 >= 0 and ry2 < h:
                roi_rec = out[ry1:ry2, rx1:rx2]
                pill_rec = np.full_like(roi_rec, (30, 20, 20))
                cv2.addWeighted(pill_rec, 0.85, roi_rec, 0.15, 0, roi_rec)
                cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (60, 30, 30), 1)
                cv2.circle(out, (rx1 + 16, ry1 + 15), 5, (50, 50, 255), -1)
                cv2.putText(out, "REC", (rx1 + 28, ry1 + 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
                rx = rx1 - 10

        if metrics.get('is_vcam', False):
            rw, rh = 80, 30
            rx1 = rx - rw
            ry1 = 16
            rx2 = rx
            ry2 = ry1 + rh
            if rx1 >= 0 and ry2 < h:
                roi_vc = out[ry1:ry2, rx1:rx2]
                pill_vc = np.full_like(roi_vc, (20, 26, 34))
                cv2.addWeighted(pill_vc, 0.85, roi_vc, 0.15, 0, roi_vc)
                cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (40, 60, 80), 1)
                cv2.putText(out, "LIVE", (rx1 + 20, ry1 + 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 180, 50), 1, cv2.LINE_AA)

        return out

    # =========================================================================
    # CONTROL SLOTS & EVENT HANDLERS
    # =========================================================================
    def _on_scribble_toggled(self, checked):
        if self.pipeline_worker:
            self.pipeline_worker.enable_scribble = checked

    def _on_expressions_toggled(self, checked):
        if self.pipeline_worker:
            self.pipeline_worker.enable_expressions = checked

    def _on_shape_changed(self, idx):
        shapes = ["head", "cube", "star", "diamond", "pill"]
        if 0 <= idx < len(shapes) and self.pipeline_worker:
            self.pipeline_worker.scribble_shape = shapes[idx]

    def _on_scribble_style_changed(self, idx):
        styles = ["white", "dark", "cyber"]
        if 0 <= idx < len(styles) and self.pipeline_worker:
            self.pipeline_worker.scribble_style = styles[idx]

    def _on_size_changed(self, val):
        if self.pipeline_worker:
            self.pipeline_worker.scribble_scale = val / 100.0

    def _on_density_changed(self, val):
        if self.pipeline_worker:
            self.pipeline_worker.scribble_density = val

    def _on_thickness_changed(self, val):
        if self.pipeline_worker:
            self.pipeline_worker.scribble_thickness = val

    def _on_offset_y_changed(self, val):
        cm = val / 10.0
        if self.pipeline_worker:
            self.pipeline_worker.offset_y = cm

    def _on_offset_x_changed(self, val):
        cm = val / 10.0
        if self.pipeline_worker:
            self.pipeline_worker.offset_x = cm

    def _on_offset_z_changed(self, val):
        cm = val / 10.0
        if self.pipeline_worker:
            self.pipeline_worker.offset_z = cm

    def _on_tongue_style_changed(self, idx):
        styles = ["sprite", "organic"]
        if 0 <= idx < len(styles) and self.pipeline_worker:
            self.pipeline_worker.tongue_style = styles[idx]

    def _on_tongue_sens_changed(self, idx):
        sens = ["medium", "low", "high", "off"]
        if 0 <= idx < len(sens) and self.pipeline_worker:
            self.pipeline_worker.tongue_sensitivity = sens[idx]

    def _on_boil_toggled(self, checked):
        if self.pipeline_worker:
            self.pipeline_worker.enable_boil = checked

    def _on_noise_changed(self, val):
        if self.pipeline_worker:
            self.pipeline_worker.sketch_noise = val / 100.0

    def _on_resolution_changed(self, idx):
        if self.current_source is not None:
            self._start_pipeline(source=self.current_source)

    def _on_bg_mode_changed(self, idx):
        self.btn_pick_img.setVisible(idx == 4)
        self.slider_blur.setVisible(idx == 1)
        self._update_bg_mode_on_worker()

    def _update_bg_mode_on_worker(self):
        if not self.pipeline_worker: return
        idx = self.seg_bg_mode.current_index()
        modes = [
            BackgroundEngine.MODE_ORIGINAL,
            BackgroundEngine.MODE_BLUR,
            BackgroundEngine.MODE_STUDIO,
            BackgroundEngine.MODE_SOLID,
            BackgroundEngine.MODE_IMAGE
        ]
        if 0 <= idx < len(modes):
            self.pipeline_worker.bg_mode = modes[idx]

    def _on_blur_changed(self, val):
        if self.pipeline_worker:
            self.pipeline_worker.blur_strength = val

    def _on_relight_toggled(self, checked):
        if self.pipeline_worker:
            self.pipeline_worker.enable_relighting = checked

    def _on_rim_changed(self, val):
        if self.pipeline_worker:
            self.pipeline_worker.rim_intensity = val / 100.0

    def _on_ambient_changed(self, val):
        if self.pipeline_worker:
            self.pipeline_worker.ambient_intensity = val / 100.0

    def _on_guided_toggled(self, checked):
        if self.pipeline_worker:
            self.pipeline_worker.use_guided_filter = checked

    def _on_matting_engine_changed(self, idx):
        engine = self.matting_engine_combo.currentData()
        if self.pipeline_worker:
            self.pipeline_worker.matting_engine = engine
        self.bg_engine.set_engine_type(engine)

    def _on_choke_changed(self, val):
        if self.pipeline_worker:
            self.pipeline_worker.edge_choke = val

    def _on_decontaminate_toggled(self, checked):
        if self.pipeline_worker:
            self.pipeline_worker.enable_decontamination = checked

    def _on_precision_changed(self, val):
        if self.pipeline_worker:
            self.pipeline_worker.downsample_ratio = val / 100.0

    def _on_pick_custom_bg(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Background Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if fname and self.bg_engine.set_custom_image(fname):
            QMessageBox.information(self, "Background Set", f"Virtual background loaded:\n{os.path.basename(fname)}")

    def _on_hud_toggled(self, checked):
        self.show_hud = checked
        self.btn_hud.set_active_state(checked)

    def _toggle_hud_dock(self):
        self.show_hud = not self.show_hud
        self.chk_hud.setChecked(self.show_hud)
        self.btn_hud.set_active_state(self.show_hud)

    def _toggle_record(self):
        if not self.recorder.is_recording:
            # Start recording
            _, frame = self.capture_thread.get_latest_frame()
            if frame is not None:
                h, w = frame.shape[:2]
                rec_file = self.recorder.start(width=w, height=h, fps=30)
                self.btn_record.set_recording(True)
                self.record_start_sec = time.time()
                self.rec_timer.start()
                self.lbl_record_timer.setText("00:00")
                self.lbl_record_timer.setStyleSheet("color: #FF3B30; font-weight: 700; font-size: 13px; background: transparent; border: none;")
        else:
            # Stop recording
            self.rec_timer.stop()
            stats = self.recorder.stop()
            self.btn_record.set_recording(False)
            self.lbl_record_timer.setText("Ready")
            timer_color = "#1D1D1F" if self.current_theme == "light" else "#8E8E93"
            self.lbl_record_timer.setStyleSheet(f"color: {timer_color}; font-weight: 500; font-size: 13px; background: transparent; border: none;")
            QMessageBox.information(
                self, "Recording Saved",
                f"Video saved successfully:\n\n{stats.get('file_path')}\n\n"
                f"Duration: {stats.get('duration_sec', 0):.1f}s\n"
                f"Frames: {stats.get('frame_count', 0)}\n"
                f"Size: {stats.get('file_size_mb', 0):.2f} MB"
            )

    def _update_record_timer(self):
        if self.recorder.is_recording:
            elapsed = int(time.time() - self.record_start_sec)
            mins = elapsed // 60
            secs = elapsed % 60
            self.lbl_record_timer.setText(f"{mins:02d}:{secs:02d}")

    def _open_recordings_folder(self):
        rec_dir = os.path.abspath(self.recorder.output_dir)
        os.makedirs(rec_dir, exist_ok=True)
        os.startfile(rec_dir)

    def _toggle_virtual_cam(self):
        if not self.virtual_cam.active:
            _, frame = self.capture_thread.get_latest_frame()
            if frame is not None:
                h, w = frame.shape[:2]
                success = self.virtual_cam.start(width=w, height=h, fps=30)
                if success:
                    self.btn_vcam.setText("Live V-Cam")
                    self.btn_vcam.set_active_state(True)
                else:
                    err = self.virtual_cam.last_error or "Ensure OBS or virtual camera driver is available."
                    QMessageBox.warning(self, "Virtual Camera Notice", f"Could not open virtual camera:\n{err}")
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
        self.bg_engine.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ScribbleCameraWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
