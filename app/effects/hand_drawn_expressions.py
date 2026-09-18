import time
import cv2
import numpy as np

from app.tracking.temporal_smoother import OneEuroFilter


class HandDrawnExpressionsRenderer:
    """
    Hand-Drawn Pencil & Chalk Expression Mimicry Engine based on the 30-Expression Scribble Sprite Sheet.
    Features:
    - CIRCLE EYES: Round scribble circle knot eyes that continuously track gaze and morph dynamically
      (crescents for smile/proud, chevrons for laugh/disgust, wink chevron for winking, oval for surprised).
    - ZERO NOSE: Clean, minimalist hand-drawn face aesthetic matching the user's reference sheet.
    - CONDITIONAL EYEBROWS: Eyebrows are strictly absent on Neutral, Happy, Laughing, Surprised, Sleepy,
      Wink, and Calm. Eyebrows appear ONLY when Angry, Worried/Scared, or Confused/Skeptical.
    - NEUTRAL DEFAULT: A clean horizontal straight-line hand-sketched mouth and two round scribble circle eyes.
    - CONTINUOUS SPEECH VISEME MORPHING: Mouth aperture expands/contracts fluidly with speech syllables without tongue.
    - 2 RED TONGUE STYLES:
      * Style 1: Hand-Drawn U-Tongue with vertical center line & horizontal sketch hatchings (matching user sheet & crop).
      * Style 2: Organic Hand-Drawn Red Tongue with smooth curved cleft.
      * Strict Speech Separation: Talking or opening the mouth NEVER triggers the tongue.
    - CONTRAST INVERSION & SCLERA BACKING: High-contrast paper backing ensures expressions pop on any helmet/background.
    """

    COLOR_BLACK = (18, 18, 24)
    COLOR_WHITE = (255, 255, 255)
    COLOR_RED_SPRITE = (30, 30, 220)   # Vibrant red chalk (BGR)
    COLOR_RED_ORGANIC = (15, 15, 225)  # Crimson organic red (BGR)

    TONGUE_STYLE_SPRITE = "sprite"
    TONGUE_STYLE_ORGANIC = "organic"

    TONGUE_SENSITIVITY_OFF = "off"
    TONGUE_SENSITIVITY_LOW = "low"
    TONGUE_SENSITIVITY_MED = "medium"
    TONGUE_SENSITIVITY_HIGH = "high"

    def __init__(self, boil_fps: float = 12.0):
        self.boil_fps = float(boil_fps)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Adaptive per-channel 1-Euro smoothing for blendshapes — consistent with the
        # pose smoothing strategy in temporal_smoother.py, and unlike a flat exponential
        # filter it stays snappy on fast expression changes while remaining stable at rest.
        self._bs_filters = {}
        self.smooth_bs = {}

        # Hysteresis state for discrete expression classification (shocked/angry/etc.):
        # entering a state requires a higher threshold than staying in it, which kills
        # the near-boundary flicker a single hard threshold would otherwise cause.
        self._state_active = {}

        # Facial surface depth in 3D Head Space (on the cranial surface, NOT floating in front)
        self.z_surf = -7.6

        # Tongue configuration
        self.tongue_style = self.TONGUE_STYLE_SPRITE
        self.tongue_sensitivity = self.TONGUE_SENSITIVITY_MED

    def reset(self):
        self.smooth_bs.clear()
        self._bs_filters.clear()
        self._state_active.clear()

    def set_tongue_style(self, style: str):
        if style in (self.TONGUE_STYLE_SPRITE, self.TONGUE_STYLE_ORGANIC, "red_chalk"):
            self.tongue_style = style

    def set_tongue_sensitivity(self, sensitivity: str):
        if sensitivity in (self.TONGUE_SENSITIVITY_OFF, self.TONGUE_SENSITIVITY_LOW,
                           self.TONGUE_SENSITIVITY_MED, self.TONGUE_SENSITIVITY_HIGH):
            self.tongue_sensitivity = sensitivity

    def _get_smoothed_blendshapes(self, raw_bs: dict, t: float) -> dict:
        for k, v in raw_bs.items():
            f = self._bs_filters.get(k)
            if f is None:
                # min_cutoff=2.0 keeps low-speed jitter down without adding perceptible
                # lag; beta=0.4 lets the filter open up fast for quick speech/blink
                # transients instead of smearing them like a flat-alpha filter would.
                f = OneEuroFilter(min_cutoff=2.0, beta=0.4, d_cutoff=1.0)
                self._bs_filters[k] = f
            self.smooth_bs[k] = float(f.filter(v, t))

        return self.smooth_bs

    def _hyst(self, key: str, value: float, enter_thresh: float, exit_thresh: float) -> bool:
        """Schmitt-trigger style hysteresis: a higher bar to enter a state than to stay in it."""
        was_active = self._state_active.get(key, False)
        thresh = exit_thresh if was_active else enter_thresh
        is_active = value > thresh
        self._state_active[key] = is_active
        return is_active

    def _make_scribble_knot_eye(self, ex: float, ey: float, z_surf: float,
                                radius: float = 1.35, gaze_x: float = 0.0, gaze_y: float = 0.0,
                                blink: float = 0.0, squint: float = 0.0, smile: float = 0.0,
                                is_shocked: bool = False, is_angry: bool = False, is_laughing: bool = False,
                                is_sleepy: bool = False, sign: float = 1.0, rng: np.random.RandomState = None) -> list:
        """
        Renders an authentic hand-drawn scribble circle knot eye that morphs according to expression.
        - Neutral / Default: Dense round scribble knot of looping pencil lines.
        - Smile / Proud: Joyful upward sketch crescent (^^).
        - Laugh / Disgust: Squinted angled chevrons (><).
        - Wink / Closed: Hand-sketched wink chevron (>).
        - Surprised / Shocked: Enlarged open circular scribble loop.
        - Sleepy / Tired: Flattened horizontal oval scribble ellipse.
        - Gaze tracking: Eye knot shifts dynamically with user gaze.
        """
        curves = []

        # 1. Closed Eye / Winking
        if blink > 0.45:
            w = radius * 1.25
            chevron = [
                [ex - w * 0.9, ey + 0.30, z_surf],
                [ex, ey - 0.20, z_surf],
                [ex + w * 0.9, ey + 0.30, z_surf]
            ]
            curves.append(np.array(chevron, dtype=np.float64))
            # Double stroke for sketchbook depth
            curves.append(np.array([[p[0], p[1] + 0.15, p[2]] for p in chevron], dtype=np.float64))
            return curves

        # 2. Laughing / Disgusted Squint (><)
        if is_laughing or (squint > 0.65 and smile > 0.45):
            w = radius * 1.15
            h = radius * 0.75
            p_apex = [ex + sign * w * 0.8, ey, z_surf]
            p_top = [ex - sign * w * 0.8, ey - h, z_surf]
            p_bot = [ex - sign * w * 0.8, ey + h, z_surf]
            curves.append(np.array([p_top, p_apex, p_bot], dtype=np.float64))
            curves.append(np.array([[p_top[0] + 0.1, p_top[1], z_surf],
                                    [p_apex[0] + 0.1, p_apex[1], z_surf],
                                    [p_bot[0] + 0.1, p_bot[1], z_surf]], dtype=np.float64))
            return curves

        # 3. Happy / Proud Smiling Crescent (^^)
        # In sprite sheet, happy eyes are SOLID ROUND SCRIBBLE KNOTS!
        # Only extreme squeeze squint (proud / anime squeeze) uses (^^)
        if squint > 0.78 and smile > 0.75 and blink > 0.45:
            w = radius * 1.20
            h = 0.65 * (1.0 + smile * 0.4)
            arc1, arc2 = [], []
            for t in np.linspace(-1.0, 1.0, 12):
                x = ex + t * w
                y = ey - h * (1.0 - t**2)
                arc1.append([x + (rng.uniform(-0.04, 0.04) if rng else 0.0), y, z_surf])
                arc2.append([x, y + 0.22, z_surf])
            curves.append(np.array(arc1, dtype=np.float64))
            curves.append(np.array(arc2, dtype=np.float64))
            return curves

        # 4. Sleepy / Tired Flattened Horizontal Scribble
        if is_sleepy:
            w = radius * 1.25
            h = radius * 0.45
            for loop in range(3):
                oval = []
                phase = loop * 1.8 + (rng.uniform(0, 2 * np.pi) if rng else 0.0)
                for th in np.linspace(0, 2 * np.pi, 14):
                    dr = rng.uniform(-0.06, 0.06) if rng else 0.0
                    x = ex + gaze_x * 0.35 + (w + dr) * np.cos(th + phase)
                    y = ey + gaze_y * 0.25 + (h + dr) * np.sin(th + phase)
                    oval.append([x, y, z_surf])
                curves.append(np.array(oval, dtype=np.float64))
            return curves

        # 5. Solid Round Scribble Knot Ball Eye (Neutral / Happy / Sad / Excited / Angry / Default)
        # Matches the 30-expression scribble sprite sheet: dense tangled ball of yarn doodle!
        r = radius * (1.35 if is_shocked else 0.88 if is_angry else 1.0)
        cx = ex + gaze_x * 0.45
        cy = ey + gaze_y * 0.35

        # 18 dense looping circular scribble passes creating the solid yarn-ball doodle
        num_loops = 18
        for loop in range(num_loops):
            pts = []
            phase = loop * 1.37 + (rng.uniform(0, 2 * np.pi) if rng else 0.0)
            r_scale = 0.22 + 0.78 * np.sin((loop / float(num_loops - 1)) * (np.pi / 2.0))
            wobble_freq = rng.choice([3, 4, 5]) if rng else 3
            wobble_amp = (rng.uniform(0.08, 0.18) if rng else 0.12) * r

            steps = 18
            for th in np.linspace(0, 2 * np.pi, steps, endpoint=False):
                dr = np.sin(wobble_freq * th + phase) * wobble_amp + (rng.uniform(-0.05, 0.05) if rng else 0.0) * r
                x = cx + (r * r_scale + dr) * np.cos(th + phase * 0.25)
                y = cy + (r * r_scale + dr) * np.sin(th + phase * 0.25)
                pts.append([x, y, z_surf])
            curves.append(np.array(pts, dtype=np.float64))

        # Dense graphite / chalk cross-hatching passes inside the knot core
        core_r = r * 0.75
        for ang in [0.0, np.pi * 0.25, np.pi * 0.50, np.pi * 0.75]:
            dr1 = (rng.uniform(-0.08, 0.08) if rng else 0.0) * r
            dr2 = (rng.uniform(-0.08, 0.08) if rng else 0.0) * r
            curves.append(np.array([
                [cx - (core_r + dr1) * np.cos(ang), cy - (core_r + dr1) * np.sin(ang), z_surf],
                [cx + (core_r + dr2) * np.cos(ang), cy + (core_r + dr2) * np.sin(ang), z_surf]
            ], dtype=np.float64))

        return curves

    def _make_eyebrows(self, brow_up: float, brow_down_l: float, brow_down_r: float,
                       brow_outer_up_l: float, brow_outer_up_r: float,
                       z_surf: float, rng: np.random.RandomState = None) -> list:
        """
        Continuous eyebrow rendering driven directly by blendshape magnitude, always
        present rather than gated to a handful of discrete extreme emotion states —
        a real face's brows move constantly during ordinary speech. Height, tilt, and
        stroke weight are continuous functions of brow_up/brow_down/outer_up per side,
        so motion (including natural left/right asymmetry, e.g. a skeptical raised
        brow) crossfades smoothly instead of snapping between fixed presets.
        """
        curves = []
        for sign, brow_down, outer_up in [(1.0, brow_down_l, brow_outer_up_l), (-1.0, brow_down_r, brow_outer_up_r)]:
            bx_center = sign * 3.3
            by = -5.4
            brow_w = 1.9

            lift = brow_up * 1.3 + outer_up * 0.9   # raises the brow (surprise / worry / confusion)
            drop = brow_down * 1.1                   # lowers & furrows the brow (anger)
            base_y = by - lift + drop
            # Furrowing angles the inner end down (\ /); raising angles it up (/ \).
            tilt = (drop - lift) * 0.55

            activity = float(np.clip(lift + drop, 0.0, 1.5))
            n_strokes = 2 if activity > 0.12 else 1

            guide = []
            for t in np.linspace(-1.0, 1.0, 10):
                x = bx_center + t * brow_w
                y = base_y - 0.20 * (1.0 - t**2) - (tilt * t * sign)
                guide.append([x, y, z_surf])
            guide_arr = np.array(guide, dtype=np.float64)
            for s in range(n_strokes):
                offset_y = (s - (n_strokes - 1) / 2.0) * 0.22
                curves.append(np.array([[p[0], p[1] + offset_y, p[2]] for p in guide_arr], dtype=np.float64))

        return curves

    def _make_mouth(self, my: float, mw: float, z_surf: float,
                    smile: float = 0.0, jaw: float = 0.0, frown: float = 0.0,
                    pucker: float = 0.0, funnel: float = 0.0, stretch: float = 0.0,
                    is_shocked: bool = False, is_angry: bool = False, is_laughing: bool = False,
                    speech_viseme: float = 0.0, show_tongue: bool = False,
                    tongue_style: str = "sprite",
                    rng: np.random.RandomState = None) -> tuple:
        """
        Renders the hand-drawn mouth matching the 30-expression scribble sprite sheet:
        - Neutral: Single straight horizontal hand-sketched line (-).
        - Happy Smile: Curved upward crescent (‿).
        - Sad Frown: Curved downward crescent (⁀).
        - Surprised / Shocked: Round circular scribble ring (O).
        - Laughing / Excited: Wide open grinning mouth with dark shaded cavity.
        - Speech / Talking: Continuous fluid aperture morphing without tongue.
        - Tongue Out: Smiling or parted mouth seam with U-tongue geometry.
        """
        curves = []
        tongue_curves = []

        # --------------------------------------------------------------
        # 1. TONGUE OUT MOUTH BASE (Only when tongue is genuinely protruded, NEVER during speech!)
        # --------------------------------------------------------------
        if show_tongue:
            # Curved smiling / parted mouth seam that the tongue hangs from
            top_lip = []
            for t in np.linspace(-1.0, 1.0, 12):
                x = t * mw
                y = my - 0.20 * (1.0 - t**2)
                top_lip.append([x + (rng.uniform(-0.04, 0.04) if rng else 0.0), y, z_surf])
            curves.append(np.array(top_lip, dtype=np.float64))
            curves.append(np.array([[p[0], p[1] + 0.18, p[2]] for p in top_lip], dtype=np.float64))

            # Tongue Geometry (Matching row 4 col 2 of sprite sheet)
            tw = mw * 0.44
            t_drop = 2.10
            y_tip = my + t_drop

            # Smooth U-shape outline of the tongue
            u_tongue = []
            for t in np.linspace(-1.0, 1.0, 18):
                tx = t * tw
                ty_curve = my + 0.20 + (t_drop * (1.0 - abs(t)**2.4))
                u_tongue.append([tx + (rng.uniform(-0.03, 0.03) if rng else 0.0), ty_curve, z_surf - 0.2])
            tongue_curves.append(np.array(u_tongue, dtype=np.float64))

            # Vertical center line down the tongue (|) matching sprite sheet
            cleft = []
            for ty in np.linspace(my + 0.35, y_tip - 0.30, 8):
                cleft.append([rng.uniform(-0.02, 0.02) if rng else 0.0, ty, z_surf - 0.2])
            tongue_curves.append(np.array(cleft, dtype=np.float64))

            # Version 2 Organic Chalk: sketch hatchings and double outline
            if tongue_style in (self.TONGUE_STYLE_ORGANIC, "red_chalk"):
                u_tongue_inner = []
                for pt in u_tongue:
                    u_tongue_inner.append([pt[0] * 0.92, pt[1] - 0.10, pt[2]])
                tongue_curves.append(np.array(u_tongue_inner, dtype=np.float64))
                for hy_frac in [0.45, 0.72]:
                    hy = my + t_drop * hy_frac
                    w_h = tw * (1.0 - (hy_frac - 0.3)**2) * 0.70
                    tongue_curves.append(np.array([
                        [-w_h + (rng.uniform(-0.03, 0.03) if rng else 0.0), hy, z_surf - 0.2],
                        [w_h + (rng.uniform(-0.03, 0.03) if rng else 0.0), hy, z_surf - 0.2]
                    ], dtype=np.float64))

            # Corner smile ticks
            for sign in [1.0, -1.0]:
                curves.append(np.array([[sign * mw, my, z_surf], [sign * (mw + 0.35), my - 0.40, z_surf]], dtype=np.float64))
            return curves, tongue_curves

        # --------------------------------------------------------------
        # 2. SURPRISED / SHOCKED (Round open circular scribble ring O)
        # --------------------------------------------------------------
        if is_shocked or jaw > 0.42:
            rad_x = 1.35 * (1.0 + 0.15 * stretch)
            rad_y = 1.85 * (1.0 + 0.45 * jaw)
            for loop in range(3):
                ring = []
                phase = loop * 1.8 + (rng.uniform(0, 2 * np.pi) if rng else 0.0)
                for th in np.linspace(0, 2 * np.pi, 16):
                    dr = rng.uniform(-0.08, 0.08) if rng else 0.0
                    x = (rad_x + dr) * np.cos(th + phase)
                    y = my + (rad_y + dr) * np.sin(th + phase)
                    ring.append([x, y, z_surf])
                curves.append(np.array(ring, dtype=np.float64))
            return curves, tongue_curves

        # --------------------------------------------------------------
        # 3. LAUGHING / EXCITED (Wide open grinning mouth)
        # --------------------------------------------------------------
        if is_laughing or (smile > 0.42 and jaw > 0.18):
            open_gap = max(0.9, jaw * 2.8 + 0.4)
            top_pts, bot_pts = [], []
            for t in np.linspace(-1.0, 1.0, 12):
                x = t * mw
                lift = -smile * 0.9 * (1.0 - t**2)
                top_pts.append([x, my + lift - 0.25 * (1.0 - t**2), z_surf])
                bot_pts.append([x, my + lift + open_gap * (1.0 - 0.4 * t**2), z_surf])
            curves.append(np.array(top_pts, dtype=np.float64))
            curves.append(np.array(bot_pts, dtype=np.float64))

            # Laugh ticks on side
            for sign in [1.0, -1.0]:
                curves.append(np.array([[sign * (mw + 0.4), my - 0.5, z_surf],
                                        [sign * (mw + 0.7), my, z_surf],
                                        [sign * (mw + 0.4), my + 0.5, z_surf]], dtype=np.float64))
            return curves, tongue_curves

        # --------------------------------------------------------------
        # 4. SPEECH / TALKING (Fluid continuous mouth aperture)
        # --------------------------------------------------------------
        if speech_viseme > 0.18 or jaw > 0.06:
            open_gap = max(0.35, jaw * 2.6 + funnel * 1.4 + speech_viseme * 0.9)
            top_lip, bot_lip = [], []
            for t in np.linspace(-1.0, 1.0, 10):
                x = t * mw
                top_lip.append([x, my - open_gap * 0.45 * (1.0 - t**2), z_surf])
                bot_lip.append([x, my + open_gap * 0.55 * (1.0 - t**2), z_surf])
            curves.append(np.array(top_lip, dtype=np.float64))
            curves.append(np.array(bot_lip, dtype=np.float64))
            return curves, tongue_curves

        # --------------------------------------------------------------
        # 5. HAPPY SMILE (Upward curved crescent ‿)
        # --------------------------------------------------------------
        if smile > 0.22:
            smile_arch = smile * 1.35
            line1, line2 = [], []
            for t in np.linspace(-1.0, 1.0, 12):
                x = t * mw
                y = my + smile_arch * (1.0 - t**2)
                line1.append([x + (rng.uniform(-0.04, 0.04) if rng else 0.0), y, z_surf])
                line2.append([x, y + 0.20, z_surf])
            curves.append(np.array(line1, dtype=np.float64))
            curves.append(np.array(line2, dtype=np.float64))

            # Smile corner ticks flicking upward
            for sign in [1.0, -1.0]:
                curves.append(np.array([[sign * mw, my, z_surf],
                                        [sign * (mw + 0.35), my - 0.40, z_surf]], dtype=np.float64))
            return curves, tongue_curves

        # --------------------------------------------------------------
        # 6. SAD FROWN (Downward curved crescent ⁀)
        # --------------------------------------------------------------
        if frown > 0.25:
            frown_arch = frown * 1.15
            line1, line2 = [], []
            for t in np.linspace(-1.0, 1.0, 12):
                x = t * mw
                y = my - frown_arch * (1.0 - t**2)
                line1.append([x + (rng.uniform(-0.04, 0.04) if rng else 0.0), y, z_surf])
                line2.append([x, y + 0.20, z_surf])
            curves.append(np.array(line1, dtype=np.float64))
            curves.append(np.array(line2, dtype=np.float64))

            # Frown corner ticks drooping downward
            for sign in [1.0, -1.0]:
                curves.append(np.array([[sign * mw, my, z_surf],
                                        [sign * (mw + 0.35), my + 0.40, z_surf]], dtype=np.float64))
            return curves, tongue_curves

        # --------------------------------------------------------------
        # 7. NEUTRAL / CALM (DEFAULT: Straight horizontal scribble line -)
        # --------------------------------------------------------------
        for s in range(2):
            offset_y = (s - 0.5) * 0.25
            line = []
            for t in np.linspace(-1.0, 1.0, 10):
                x = t * mw
                y = my + offset_y + (rng.uniform(-0.04, 0.04) if rng else 0.0)
                line.append([x, y, z_surf])
            curves.append(np.array(line, dtype=np.float64))

        return curves, tongue_curves

    def _draw_curves_batched(self, curves, rvec, tvec, cam_mat, color, thickness, target_img):
        """Projects and draws a list of variable-length 3D curves with one
        cv2.projectPoints + one cv2.polylines call instead of per-curve calls."""
        valid = [c for c in curves if len(c) >= 2]
        if not valid:
            return
        lens = [len(c) for c in valid]
        splits = np.cumsum(lens)[:-1]
        all_pts = np.vstack(valid)
        proj, _ = cv2.projectPoints(all_pts, rvec, tvec, cam_mat, self.dist_coeffs)
        pts_2d = proj.reshape(-1, 2)
        polylines = [p.astype(np.int32).reshape(-1, 1, 2) for p in np.split(pts_2d, splits)]
        cv2.polylines(target_img, polylines, False, color, thickness, cv2.LINE_AA)

    def render(self, frame_bgr: np.ndarray, tracking_data: dict,
               style: str = "white", t_sec: float = None,
               line_thickness: int = 3) -> np.ndarray:
        """
        Renders the hand-drawn sketch expressions onto frame_bgr matching the 30-expression sprite sheet.
        """
        if not tracking_data.get('has_face', False):
            return frame_bgr

        rvec = tracking_data.get('rvec')
        tvec = tracking_data.get('tvec')
        cam_mat = tracking_data.get('cam_matrix')
        raw_bs = tracking_data.get('blendshapes', {})

        if rvec is None or tvec is None or cam_mat is None:
            return frame_bgr

        cur_t = t_sec if t_sec is not None else time.time()
        step_id = int(cur_t * self.boil_fps)
        rng = np.random.RandomState(step_id & 0x7FFFFFFF)

        bs = self._get_smoothed_blendshapes(raw_bs, cur_t)

        # Facing direction check (yaw fade)
        R_head, _ = cv2.Rodrigues(rvec)
        forward_cam = R_head @ np.array([0.0, 0.0, -1.0])
        dot = -forward_cam[2]
        if dot < 0.25:
            return frame_bgr

        alpha = 1.0 if dot > 0.55 else max(0.0, (dot - 0.25) / 0.30)
        z_surf = self.z_surf
        tz = float(tvec.flatten()[2]) if tvec is not None and len(tvec.flatten()) > 2 else 60.0


        # Blendshape signals
        smile_l = bs.get('mouthSmileLeft', 0.0)
        smile_r = bs.get('mouthSmileRight', 0.0)
        smile = max(smile_l, smile_r)
        # Direct real-time lip tracking from 2D facial landmarks (Instant speech synchronization)
        lmks_2d = tracking_data.get('landmarks_2d')
        if lmks_2d is not None and len(lmks_2d) > 291:
            lip_open_px = float(np.linalg.norm(lmks_2d[14] - lmks_2d[13]))
            lip_width_px = float(np.linalg.norm(lmks_2d[291] - lmks_2d[61]))
            open_ratio = lip_open_px / max(10.0, lip_width_px)
            raw_jaw = bs.get('jawOpen', 0.0)
            # High-fidelity speech viseme drives mouth opening with zero lag
            speech_viseme = max(bs.get('speechViseme', 0.0), float(np.clip((open_ratio - 0.03) * 5.0, 0.0, 1.0)))
            jaw = max(raw_jaw, float(np.clip((open_ratio - 0.04) * 3.5, 0.0, 1.0)))
        else:
            jaw = bs.get('jawOpen', 0.0)
            speech_viseme = bs.get('speechViseme', 0.0)

        frown = max(bs.get('mouthFrownLeft', 0.0), bs.get('mouthFrownRight', 0.0))
        pucker = bs.get('mouthPucker', 0.0)
        funnel = bs.get('mouthFunnel', 0.0)
        stretch = max(bs.get('mouthStretchLeft', 0.0), bs.get('mouthStretchRight', 0.0))
        tongue_out = bs.get('tongueOut', 0.0)

        blink_l = bs.get('eyeBlinkLeft', 0.0)
        blink_r = bs.get('eyeBlinkRight', 0.0)
        squint_l = bs.get('eyeSquintLeft', 0.0)
        squint_r = bs.get('eyeSquintRight', 0.0)
        brow_up = bs.get('browInnerUp', 0.0)
        brow_down_l = bs.get('browDownLeft', 0.0)
        brow_down_r = bs.get('browDownRight', 0.0)
        brow_down = max(brow_down_l, brow_down_r)
        brow_outer_up_l = bs.get('browOuterUpLeft', 0.0)
        brow_outer_up_r = bs.get('browOuterUpRight', 0.0)

        # Eye gaze tracking
        look_x = (bs.get('eyeLookOutRight', 0.0) - bs.get('eyeLookInRight', 0.0) +
                  bs.get('eyeLookInLeft', 0.0) - bs.get('eyeLookOutLeft', 0.0)) * 0.40
        look_y = (bs.get('eyeLookDownLeft', 0.0) - bs.get('eyeLookUpLeft', 0.0)) * 0.30

        # Expression classification. Each state's primary driving signal is gated
        # through hysteresis (a higher bar to enter than to stay), which prevents
        # the visible discrete "pose snapping" that a single hard threshold causes
        # when a blendshape hovers right at the boundary.
        jaw_shock = self._hyst('jaw_shock', jaw, 0.38, 0.30)
        is_shocked = (jaw_shock and brow_up > 0.30 and smile < 0.25)

        brow_down_angry = self._hyst('brow_down_angry', brow_down, 0.38, 0.30)
        is_angry = (brow_down_angry and smile < 0.20 and not is_shocked)

        frown_worried = self._hyst('frown_worried', frown, 0.20, 0.14)
        is_worried = (brow_up > 0.35 and frown_worried and not is_shocked and not is_angry)

        jaw_scared = self._hyst('jaw_scared', jaw, 0.25, 0.18)
        is_scared = (brow_up > 0.40 and jaw_scared and not is_shocked)

        smile_laugh = self._hyst('smile_laugh', smile, 0.35, 0.28)
        is_laughing = (smile_laugh and jaw > 0.18)

        brow_asym = abs(brow_outer_up_l - brow_outer_up_r)
        is_confused = self._hyst('confused', brow_asym, 0.25, 0.18)
        is_skeptical = is_confused

        blink_sleepy = self._hyst('blink_sleepy', max(blink_l, blink_r), 0.35, 0.27)
        is_sleepy = (blink_sleepy and jaw < 0.15 and smile < 0.15)

        # STRICT TONGUE ACTIVATION GATE (IMMUNITY DURING SPEECH)
        show_tongue = False
        if self.tongue_sensitivity != self.TONGUE_SENSITIVITY_OFF:
            thresh = {
                self.TONGUE_SENSITIVITY_LOW: 0.65,
                self.TONGUE_SENSITIVITY_MED: 0.45,
                self.TONGUE_SENSITIVITY_HIGH: 0.28
            }.get(self.tongue_sensitivity, 0.45)

            # ABSOLUTE SPEECH IMMUNITY:
            # If the user is speaking, moving jaw, laughing, smiling with open jaw,
            # or exhibiting speech visemes, tongue is 100% locked out!
            is_speaking = (speech_viseme > 0.05 or jaw > 0.22 or is_laughing or is_shocked or
                           funnel > 0.10 or pucker > 0.15 or (jaw > 0.06 and smile > 0.20))

            if not is_speaking and tongue_out > thresh:
                show_tongue = True

        my = 4.2
        mw = 2.4 * (1.0 + 0.35 * stretch - 0.30 * pucker + 0.35 * smile)

        # ----------------------------------------------------------------------
        # CONTRAST BACKING PASSES
        # ----------------------------------------------------------------------
        eye_bg_col = (14, 12, 10) if style != "white" else (248, 248, 252)

        # A. Eye Backing Discs (Subtle cavity shading without vector border ring!)
        for sign, blk in [(1.0, blink_l), (-1.0, blink_r)]:
            if blk < 0.50 and not is_sleepy:
                ex = sign * 3.3 + look_x * 0.45
                ey = -2.8 + look_y * 0.35
                pt3d = np.array([[ex, ey, z_surf]], dtype=np.float64)
                proj, _ = cv2.projectPoints(pt3d, rvec, tvec, cam_mat, self.dist_coeffs)
                cx, cy = int(proj[0, 0, 0]), int(proj[0, 0, 1])
                r_scale = 1.35 if is_shocked else 1.0
                rx_px = int(cam_mat[0, 0] * (1.35 * r_scale) / tz)
                ry_px = int(cam_mat[1, 1] * (1.35 * r_scale) / tz)
                if rx_px > 3 and ry_px > 3:
                    # Clean fill only — NO vector circle border line!
                    cv2.ellipse(frame_bgr, (cx, cy), (rx_px, ry_px), 0, 0, 360, eye_bg_col, -1, cv2.LINE_AA)

        # B. Shaded Oral Cavity Backing (Open mouth / Laugh / Shock)
        if jaw > 0.12 or is_shocked or is_laughing or speech_viseme > 0.22:
            open_gap = max(0.40, jaw * 2.8 + funnel * 1.5 + speech_viseme * 1.0)
            cavity_pt = np.array([[0.0, my + open_gap * 0.30, z_surf]], dtype=np.float64)
            proj_cav, _ = cv2.projectPoints(cavity_pt, rvec, tvec, cam_mat, self.dist_coeffs)
            mcx, mcy = int(proj_cav[0, 0, 0]), int(proj_cav[0, 0, 1])
            m_rx = int(cam_mat[0, 0] * (mw * 0.70) / tz)
            m_ry = int(cam_mat[1, 1] * (max(0.5, open_gap * 0.50)) / tz)
            if m_rx > 3 and m_ry > 2:
                cav_col = (18, 15, 14) if style == "white" else (240, 240, 245)
                cv2.ellipse(frame_bgr, (mcx, mcy), (m_rx, m_ry), 0, 0, 360, cav_col, -1, cv2.LINE_AA)

        # C. Hand-Drawn Red Tongue Backing Fill
        if show_tongue:
            tw = mw * 0.44
            t_drop = 1.95
            tip_pt = np.array([[0.0, my + t_drop * 0.55, z_surf]], dtype=np.float64)
            proj_tip, _ = cv2.projectPoints(tip_pt, rvec, tvec, cam_mat, self.dist_coeffs)
            px_tip, py_tip = int(proj_tip[0, 0, 0]), int(proj_tip[0, 0, 1])
            tongue_rx = int(cam_mat[0, 0] * (tw * 0.95) / tz)
            tongue_ry = int(cam_mat[1, 1] * (t_drop * 0.50) / tz)
            if tongue_rx > 3 and tongue_ry > 3:
                red_col = self.COLOR_RED_SPRITE if self.tongue_style == self.TONGUE_STYLE_SPRITE else self.COLOR_RED_ORGANIC
                cv2.ellipse(frame_bgr, (px_tip, py_tip), (tongue_rx, tongue_ry), 0, 0, 360, red_col, -1, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # PENCIL / CHALK LINEWORK PASS
        # ----------------------------------------------------------------------
        curves_3d = []

        # 1. Circle Knot Eyes (morphing according to expression)
        for sign, blk, sq in [(1.0, blink_l, squint_l), (-1.0, blink_r, squint_r)]:
            ex = sign * 3.3
            ey = -2.8
            eye_curves = self._make_scribble_knot_eye(
                ex=ex, ey=ey, z_surf=z_surf, radius=1.35,
                gaze_x=look_x, gaze_y=look_y,
                blink=blk, squint=sq, smile=smile,
                is_shocked=is_shocked, is_angry=is_angry, is_laughing=is_laughing,
                is_sleepy=is_sleepy, sign=sign, rng=rng
            )
            curves_3d.extend(eye_curves)

        # 2. Eyebrows (continuous: always present, driven by live blendshape magnitude)
        brow_curves = self._make_eyebrows(
            brow_up=brow_up, brow_down_l=brow_down_l, brow_down_r=brow_down_r,
            brow_outer_up_l=brow_outer_up_l, brow_outer_up_r=brow_outer_up_r,
            z_surf=z_surf, rng=rng
        )
        curves_3d.extend(brow_curves)

        # 3. Mouth (Neutral default, Speech, Happy, Sad, Shocked, or Tongue Out)
        mouth_curves, tongue_curves = self._make_mouth(
            my=my, mw=mw, z_surf=z_surf,
            smile=smile, jaw=jaw, frown=frown,
            pucker=pucker, funnel=funnel, stretch=stretch,
            is_shocked=is_shocked, is_angry=is_angry, is_laughing=is_laughing,
            speech_viseme=speech_viseme, show_tongue=show_tongue,
            tongue_style=self.tongue_style, rng=rng
        )
        curves_3d.extend(mouth_curves)

        # Project and draw all facial sketch strokes — batched into a single
        # projectPoints + polylines call each (matching the vectorized pattern
        # used by scribble_sphere_3d.py) instead of one Python-level call pair
        # per curve, which is real per-frame overhead given ~40-60 small curves.
        stroke_color = self.COLOR_BLACK if style == "white" else self.COLOR_WHITE
        target_img = frame_bgr if alpha >= 0.98 else frame_bgr.copy()

        self._draw_curves_batched(curves_3d, rvec, tvec, cam_mat, stroke_color, line_thickness, target_img)

        # Project and draw tongue strokes with selected tongue style
        if show_tongue and tongue_curves:
            if self.tongue_style in (self.TONGUE_STYLE_ORGANIC, "red_chalk"):
                # Version 2: Red hand-drawn chalk strokes
                tongue_stroke_col = self.COLOR_RED_SPRITE
            else:
                # Version 1: Reference image style (white chalk hatchings matching helmet style)
                tongue_stroke_col = stroke_color
            self._draw_curves_batched(tongue_curves, rvec, tvec, cam_mat, tongue_stroke_col, max(1, line_thickness - 1), target_img)

        if alpha < 0.98:
            cv2.addWeighted(target_img, alpha, frame_bgr, 1.0 - alpha, 0, dst=frame_bgr)

        return frame_bgr
