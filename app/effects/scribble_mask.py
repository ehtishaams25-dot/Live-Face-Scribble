import cv2
import numpy as np

from app.effects.scribble_sphere_3d import ScribbleSphere3DEngine, resolve_style
from app.effects.face_features import FaceFeatureRenderer
from app.tracking.hand_tracker import HandTracker


class ScribbleMaskEngine:
    """
    Composes the full scribble head for one frame:
        privacy core -> solid shaded 3D helmet -> scribble strokes -> outline
        -> landmark-driven face -> hands raised in front of the face on top.

    Privacy rules enforced here:
      * The helmet body is opaque by default and its silhouette is the exact projected
        solid (see ScribbleSphere3DEngine), unioned with the *current* face-landmark hull
        grown by `core_margin`, so a pose that lags a fast move by a frame still covers
        the real face.
      * Hands are drawn over the helmet only from this frame's hand landmarks, using
        masks shrunk inside the fingers — never from stale or loose masks.
    """

    SHAPE_HEAD = ScribbleSphere3DEngine.SHAPE_HEAD
    SHAPE_CUBE = ScribbleSphere3DEngine.SHAPE_CUBE
    SHAPE_STAR = ScribbleSphere3DEngine.SHAPE_STAR
    SHAPE_DIAMOND = ScribbleSphere3DEngine.SHAPE_DIAMOND
    SHAPE_PILL = ScribbleSphere3DEngine.SHAPE_PILL

    def __init__(self, texture_path: str = None, boil_fps: float = 12.0, radius_cm: float = 14.5):
        self.boil_fps = float(boil_fps)
        self.style = "ink"
        self.line_width = 3
        self.density = 90
        self.offset_x, self.offset_y, self.offset_z = 0.0, 1.5, -0.5
        self.privacy_core = True
        self.core_margin = 0.18
        self.hands_over_helmet = True
        self.uncertain_growth = 1.12
        self._boost = 1.0
        self.sphere_3d = ScribbleSphere3DEngine(
            semi_width_x=14.5, semi_height_y=18.5, semi_depth_z=15.5,
            center_offset_x=self.offset_x, center_offset_y=self.offset_y, center_offset_z=self.offset_z,
            line_thickness=self.line_width, density=self.density)
        self.expressions_engine = FaceFeatureRenderer(boil_fps=self.boil_fps)
        self.features = self.expressions_engine

    # -- settings passthroughs --------------------------------------------------------
    def set_shape(self, shape_name: str):
        self.sphere_3d.set_shape(shape_name)

    def set_style(self, style: str):
        self.style = resolve_style(style)

    def set_density(self, density: int):
        self.density = int(density)
        self.sphere_3d.set_density(self.density)

    def set_line_width(self, width: int):
        self.line_width = max(1, min(8, int(width)))
        self.sphere_3d.set_line_thickness(self.line_width)

    def set_scale(self, scale_ratio: float):
        self.sphere_3d.set_scale(scale_ratio)

    def set_offsets(self, offset_x=None, offset_y=None, offset_z=None):
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

    # -- render ----------------------------------------------------------------------
    def _privacy_hull(self, tracking_data):
        lm = tracking_data.get('landmarks_2d')
        if not self.privacy_core or lm is None or len(lm) < 10:
            return None
        hull = cv2.convexHull(lm.astype(np.float32)).reshape(-1, 2)
        c = hull.mean(axis=0)
        return (hull - c) * (1.0 + self.core_margin) + c

    def apply(self, frame_bgr: np.ndarray, tracking_data: dict, size_scale: float = 1.45,
              parallax_gain: float = 1.6, obscuration_strength: float = 1.0, enable_boil: bool = True,
              style: str = None, cavity_darkness: float = None, t_sec: float = None, sketch_fps: float = None,
              sketch_noise: float = None, enable_expressions: bool = True, clean_camera_frame: np.ndarray = None,
              timestamp_ms: int = None, hand_mask: np.ndarray = None, hands=None, hands_fresh: bool = False) -> np.ndarray:
        if not tracking_data.get('has_face', False):
            return frame_bgr
        active_style = resolve_style(style if style is not None else self.style)
        fps = sketch_fps if sketch_fps is not None else self.boil_fps
        noise = 0.85 if sketch_noise is None else sketch_noise
        # Uncertain pose (holding through a tracking dropout) -> grow the helmet a little
        # as a safety margin; eased so the size never visibly pops.
        target = 1.0 if tracking_data.get('face_fresh', True) else self.uncertain_growth
        self._boost += (target - self._boost) * 0.3
        self.sphere_3d.set_scale((size_scale / 1.45 if size_scale > 0.1 else 1.0) * self._boost)
        self.sphere_3d.core_scale = self._boost

        # Hands in front of the face: remember the clean pixels before we paint.
        occluders = hands if (self.hands_over_helmet and hands_fresh and hands) else None
        clean = None
        if occluders is not None:
            clean = clean_camera_frame if clean_camera_frame is not None and clean_camera_frame is not frame_bgr else frame_bgr.copy()

        self.sphere_3d.apply(
            frame_bgr=frame_bgr, tracking_data=tracking_data, t_sec=t_sec, style=active_style,
            line_width=self.line_width, enable_boil=enable_boil, sketch_fps=fps, sketch_noise=noise,
            privacy_hull=self._privacy_hull(tracking_data))

        if enable_expressions and self.sphere_3d.last_sil_bbox is not None:
            H = frame_bgr.shape[0]
            px_scale = max(1.0, min(frame_bgr.shape[:2]) / 720.0)
            self.expressions_engine.render(frame_bgr, tracking_data, self.sphere_3d, style=active_style,
                                           t_sec=t_sec, line_px=max(1, int(round(self.line_width * px_scale))))

        if occluders is not None and self.sphere_3d.last_sil_bbox is not None:
            self._composite_hands(frame_bgr, clean, occluders)
        return frame_bgr

    def _composite_hands(self, frame, clean, hands):
        x0, y0, x1, y1 = self.sphere_3d.last_sil_bbox
        hb = HandTracker.bbox(hands, pad=0.3)
        if hb is None:
            return
        # Only the overlap between hands and helmet needs work.
        ox0, oy0 = max(x0, hb[0]), max(y0, hb[1])
        ox1, oy1 = min(x1, hb[2]), min(y1, hb[3])
        if ox1 - ox0 < 2 or oy1 - oy0 < 2:
            return
        m = HandTracker.draw_mask((oy1 - oy0, ox1 - ox0), hands, offset=(ox0, oy0), thickness=0.85)
        if not m.any():
            return
        m = cv2.erode(m, np.ones((3, 3), np.uint8))
        a = cv2.GaussianBlur(m, (3, 3), 0).astype(np.float32) / 255.0
        roi = frame[oy0:oy1, ox0:ox1]
        roi[:] = cv2.blendLinear(clean[oy0:oy1, ox0:ox1], roi, a, 1.0 - a)
