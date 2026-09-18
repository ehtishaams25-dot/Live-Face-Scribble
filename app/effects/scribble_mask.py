import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app.effects.scribble_sphere_3d import ScribbleSphere3DEngine
from app.effects.hand_drawn_expressions import HandDrawnExpressionsRenderer


class ScribbleMaskEngine:
    """
    Surreal 3D Multi-Shape Scribble Helmet Engine with Real-Time Expression Mimicry
    and Foreground Hand Occlusion Awareness:
    - 5 Predetermined Shapes: Head, Cube, Star, Diamond, Pill.
    - HandLandmarker integration with high-speed proxy and low confidence threshold (0.25)
      to detect hands even when flat over the mouth/face.
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

        # Real-Time Hand Landmarker for Foreground Occlusion
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        hand_model_path = os.path.join(models_dir, "hand_landmarker.task")
        self.hand_landmarker = None
        if os.path.exists(hand_model_path):
            try:
                base_options = python.BaseOptions(model_asset_path=hand_model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.25,
                    min_hand_presence_confidence=0.25
                )
                self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
            except Exception as e:
                print(f"[ScribbleMaskEngine] HandLandmarker init warning: {e}")

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

    def _detect_hand_occlusion_mask(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """
        Detects hands using a fast 480p proxy (4ms) and creates an anti-aliased
        foreground occlusion mask with monotonic timestamp protection.
        """
        if self.hand_landmarker is None:
            return None

        h, w = frame_bgr.shape[:2]
        if not hasattr(self, '_last_hand_ts'):
            self._last_hand_ts = -1
        if timestamp_ms is None or timestamp_ms <= self._last_hand_ts:
            self._last_hand_ts = self._last_hand_ts + 1 if self._last_hand_ts >= 0 else 0
            ts = self._last_hand_ts
        else:
            self._last_hand_ts = timestamp_ms
            ts = timestamp_ms

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if w > 480:
                pw = 480
                ph = int(h * 480 / w)
                rgb_proxy = cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_LINEAR)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_proxy)
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            res = self.hand_landmarker.detect_for_video(mp_image, ts)

            if not res.hand_landmarks or len(res.hand_landmarks) == 0:
                return None

            mask = np.zeros((h, w), dtype=np.uint8)

            for hand_lmks in res.hand_landmarks:
                pts_2d = []
                for lm in hand_lmks:
                    px = int(np.clip(lm.x * w, 0, w - 1))
                    py = int(np.clip(lm.y * h, 0, h - 1))
                    pts_2d.append([px, py])

                pts_np = np.array(pts_2d, dtype=np.int32)
                # Compute convex hull of hand landmarks
                hull = cv2.convexHull(pts_np)
                cv2.fillConvexPoly(mask, hull, 255)

                # Connect finger bones with thick lines
                bones = [
                    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
                    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
                    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
                    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
                ]
                for p1, p2 in bones:
                    cv2.line(mask, tuple(pts_2d[p1]), tuple(pts_2d[p2]), 255, thickness=32)

            # Dilate slightly to encompass fleshy fingers
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
            mask = cv2.dilate(mask, kernel)
            # Gaussian blur for soft anti-aliased edge blending
            mask = cv2.GaussianBlur(mask, (9, 9), 0)
            return mask
        except Exception:
            return None

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
        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)

        # Dynamic scale adjustment
        self.sphere_3d.set_scale(size_scale / 1.45 if size_scale > 0.1 else 1.0)

        # Clean copy of the current source/composited frame before the scribble helmet is added.
        clean_composite = clean_camera_frame if clean_camera_frame is not None else frame_bgr.copy()

        # 1. Obtain Hand Foreground Mask for Occlusion (reuse background engine mask or detect)
        if hand_mask is None:
            hand_mask = self._detect_hand_occlusion_mask(frame_bgr, ts)

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
