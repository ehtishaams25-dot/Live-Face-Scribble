import cv2
import numpy as np

from app.effects.scribble_sphere_3d import ScribbleSphere3DEngine
from app.effects.hand_drawn_expressions import HandDrawnExpressionsRenderer


class ScribbleMaskEngine:
    """
    Surreal 3D Multi-Shape Scribble Helmet Engine with Real-Time Expression Mimicry
    and Foreground Hand Occlusion Awareness:
    - 5 Predetermined Shapes: Head, Cube, Star, Diamond, Pill.
    - Reuses the single shared hand occlusion mask computed once per frame by
      BackgroundEngine (avoids loading/running a second redundant HandLandmarker).
    - Strict Layered Compositing: Camera Hand -> Foreground Overlay -> Zero Scribble Leak.
    - Responsive slider control over density, thickness, 3D scale, and X/Y/Z offsets.
    """

    STYLE_WHITE = ScribbleSphere3DEngine.STYLE_WHITE
    STYLE_DARK = ScribbleSphere3DEngine.STYLE_DARK
    STYLE_CYBER = ScribbleSphere3DEngine.STYLE_CYBER

    SHAPE_HEAD = ScribbleSphere3DEngine.SHAPE_HEAD
    SHAPE_CUBE = ScribbleSphere3DEngine.SHAPE_CUBE
    SHAPE_STAR = ScribbleSphere3DEngine.SHAPE_STAR
    SHAPE_DIAMOND = ScribbleSphere3DEngine.SHAPE_DIAMOND
    SHAPE_PILL = ScribbleSphere3DEngine.SHAPE_PILL

    def __init__(self, texture_path: str = None, boil_fps: float = 12.0, radius_cm: float = 14.5):
        self.boil_fps = float(boil_fps)
        self.base_rx = 14.5
        self.base_ry = 18.5
        self.base_rz = 15.5
        self.style = self.STYLE_WHITE
        self.cavity_darkness = 0.85
        self.line_width = 3
        self.density = 90
        self.offset_x = 0.0
        self.offset_y = 1.0
        self.offset_z = -0.5
        self.sketch_noise = 0.85

        # Core 3D Multi-Shape Scribble Helmet Engine
        self.sphere_3d = ScribbleSphere3DEngine(
            semi_width_x=self.base_rx,
            semi_height_y=self.base_ry,
            semi_depth_z=self.base_rz,
            center_offset_x=self.offset_x,
            center_offset_y=self.offset_y,
            center_offset_z=self.offset_z,
            line_thickness=self.line_width,
            density=self.density,
            cavity_darkness=self.cavity_darkness,
            shape=self.SHAPE_HEAD
        )

        # Real-Time Hand-Drawn Expression Mimicry Engine
        self.expressions_engine = HandDrawnExpressionsRenderer(boil_fps=self.boil_fps)

    def set_shape(self, shape_name: str):
        self.sphere_3d.set_shape(shape_name)

    def set_style(self, style: str):
        if style in (self.STYLE_WHITE, self.STYLE_DARK, self.STYLE_CYBER):
            self.style = style

    def set_cavity_darkness(self, darkness: float):
        self.cavity_darkness = float(np.clip(darkness, 0.0, 1.0))
        self.sphere_3d.set_cavity_darkness(self.cavity_darkness)

    def set_density(self, density: int):
        self.density = int(density)
        self.sphere_3d.set_density(self.density)

    def set_line_width(self, width: int):
        self.line_width = max(1, min(8, int(width)))
        self.sphere_3d.set_line_thickness(self.line_width)

    def set_scale(self, scale_ratio: float):
        self.sphere_3d.set_scale(scale_ratio)

    def set_offsets(self, offset_x: float = None, offset_y: float = None, offset_z: float = None):
        if offset_x is not None:
            self.offset_x = float(offset_x)
        if offset_y is not None:
            self.offset_y = float(offset_y)
        if offset_z is not None:
            self.offset_z = float(offset_z)
        self.sphere_3d.set_offsets(self.offset_x, self.offset_y, self.offset_z)

    def set_tongue_style(self, style_name: str):
        self.expressions_engine.set_tongue_style(style_name)

    def set_tongue_sensitivity(self, sensitivity: str):
        self.expressions_engine.set_tongue_sensitivity(sensitivity)

    def apply(self, frame_bgr: np.ndarray, tracking_data: dict,
              size_scale: float = 1.45,
              parallax_gain: float = 1.6,
              obscuration_strength: float = 0.0,
              enable_boil: bool = True,
              style: str = None,
              cavity_darkness: float = None,
              t_sec: float = None,
              sketch_fps: float = None,
              sketch_noise: float = None,
              enable_expressions: bool = True,
              clean_camera_frame: np.ndarray = None,
              timestamp_ms: int = None,
              hand_mask: np.ndarray = None) -> np.ndarray:
        if not tracking_data.get('has_face', False):
            return frame_bgr

        active_style = style if style is not None else self.style
        active_cavity = cavity_darkness if cavity_darkness is not None else self.cavity_darkness
        active_fps = sketch_fps if sketch_fps is not None else self.boil_fps
        active_noise = sketch_noise if sketch_noise is not None else self.sketch_noise

        # Dynamic scale adjustment
        self.sphere_3d.set_scale(size_scale / 1.45 if size_scale > 0.1 else 1.0)

        # Clean copy of the current source/composited frame before the scribble helmet is added.
        clean_composite = clean_camera_frame if clean_camera_frame is not None else frame_bgr.copy()

        # 1. Hand Foreground Mask for Occlusion — supplied by BackgroundEngine
        #    (the single shared HandLandmarker instance for the whole pipeline).

        # 2. Render 3D Scribble Helmet
        out_frame = self.sphere_3d.apply(
            frame_bgr=frame_bgr,
            tracking_data=tracking_data,
            t_sec=t_sec,
            style=active_style,
            cavity_darkness=active_cavity,
            line_width=self.line_width,
            enable_boil=enable_boil,
            sketch_fps=active_fps,
            sketch_noise=active_noise,
            hand_mask=hand_mask
        )

        # 3. Render Hand-Drawn Animated Facial Expression Mimicry
        if enable_expressions:
            out_frame = self.expressions_engine.render(
                frame_bgr=out_frame,
                tracking_data=tracking_data,
                style=active_style,
                t_sec=t_sec,
                line_thickness=max(2, self.line_width)
            )

        # 4. Strict Foreground Hand Compositing
        # Guarantees that hands placed in front of the face/mouth ALWAYS stay on top
        # of BOTH the scribble helmet and the facial expressions, WITHOUT leaking real room background!
        if hand_mask is not None and np.any(hand_mask > 20):
            alpha_hand = (hand_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
            out_frame[:] = (clean_composite * alpha_hand + out_frame * (1.0 - alpha_hand)).astype(np.uint8)

        return out_frame
