import time
from collections import deque

import cv2
import numpy as np

from app.tracking.temporal_smoother import OneEuroFilter
from app.tracking.face_geometry import (
    LIPS_INNER_UPPER, LIPS_INNER_LOWER, EYE_R, EYE_L, BROW_R_UPPER, BROW_L_UPPER,
)
from app.effects.scribble_sphere_3d import PALETTES, resolve_style


_EYE_R_IDX = [EYE_R['outer'], EYE_R['inner']] + EYE_R['upper'] + EYE_R['lower']
_EYE_L_IDX = [EYE_L['outer'], EYE_L['inner']] + EYE_L['upper'] + EYE_L['lower']
# Mouth-opening probes: outer lip mid-points (0 / 17), chin (152), nose tip (1, rigid).
_MOUTH_PROBES = [0, 17, 152, 1]
# Every landmark the renderer reads (filtered together as one vector).
_TRACKED = np.array(sorted(set(LIPS_INNER_UPPER + LIPS_INNER_LOWER + _EYE_R_IDX + _EYE_L_IDX +
                               BROW_R_UPPER + BROW_L_UPPER + [468, 473, 61, 291] + _MOUTH_PROBES)), dtype=np.int32)
_POS = {int(i): k for k, i in enumerate(_TRACKED)}


def _ix(idx_list):
    return np.array([_POS[i] for i in idx_list], dtype=np.int32)


I_MU, I_ML = _ix(LIPS_INNER_UPPER), _ix(LIPS_INNER_LOWER)
I_ER, I_EL = _ix(_EYE_R_IDX), _ix(_EYE_L_IDX)
I_ER_UP, I_ER_LO = _ix(EYE_R['upper']), _ix(EYE_R['lower'])
I_EL_UP, I_EL_LO = _ix(EYE_L['upper']), _ix(EYE_L['lower'])
I_ER_CORN, I_EL_CORN = _ix([EYE_R['outer'], EYE_R['inner']]), _ix([EYE_L['outer'], EYE_L['inner']])
I_BR, I_BL = _ix(BROW_R_UPPER), _ix(BROW_L_UPPER)
I_IRIS_R, I_IRIS_L = _POS[468], _POS[473]

EMOTE_COLORS = {
    'blush': (150, 120, 245), 'anger': (40, 40, 225), 'sweat': (235, 175, 70),
    'question': None, 'zzz': None, 'heart': (90, 60, 235), 'shock': None,
}
# Seconds an expression must be held (while not talking) before its sticker pops, how
# long a sticker stays up at least, and the pause before the same sticker can return.
# Speech flickers every blendshape (an "oo" is a pucker, an "ee" a stretch, a laugh a
# smile) — without this, stickers fired on almost every sentence.
EMOTE_SUSTAIN = {'blush': 0.6, 'anger': 0.6, 'sweat': 0.8, 'question': 0.7, 'zzz': 0.0, 'heart': 0.8, 'shock': 0.35}
EMOTE_MIN_SHOW = 1.4
EMOTE_COOLDOWN = 3.0
_EMOTE_BS = ('mouthSmileLeft', 'mouthSmileRight', 'browDownLeft', 'browDownRight', 'browInnerUp',
             'mouthFrownLeft', 'mouthFrownRight', 'browOuterUpLeft', 'browOuterUpRight', 'mouthPucker',
             'jawOpen', 'mouthStretchLeft', 'mouthStretchRight')

# Mouth opening probes, per unit of "openness" (normal conversational speech ~0.3-0.7):
#  static  — ~0 whenever the mouth is shut, at any head pose: inner-lip gap (cm, pose-
#            normalized), jawOpen, mouthLowerDown (blendshapes).
#  dynamic — chin drop and outer-lip height (cm). They catch syllables the inner lips hide
#            (e.g. under a moustache), but their resting value shifts by up to ~1 cm with head
#            pitch/yaw, so they are measured only against the last ~0.6 s and only while the
#            head isn't turning fast.
_STATIC_UNITS = np.array([0.9, 0.22, 0.35])
_DYN_UNITS = np.array([0.7, 0.6])
# Dead zones (in openness units) for [inner, jawOpen, lowerDown] and the dynamic pair: the
# 98th-percentile jitter of each probe on a silent head-turning clip.
_STATIC_DEAD = np.array([0.08, 0.12, 0.05])
_DYN_DEAD = 0.15
_OPEN_ON, _OPEN_OFF = 0.17, 0.09       # hysteresis (drawn mouth opens / closes)


class FaceFeatureRenderer:
    """
    Landmark-driven cartoon face drawn on the 3D helmet surface.

    Every frame the pose-normalized MediaPipe landmarks (see face_geometry) give the
    true shape of your lips, eyelids, irises and brows as if seen head-on. Each feature
    is measured against a slowly-learned neutral baseline of *your* face, exaggerated by
    `gain`, scaled onto the helmet and projected in 3D — so every tiny movement (a half
    smile, one raised brow, a squint, a pout, looking sideways, each syllable) shows up,
    and the features foreshorten and wrap around the head as you turn.

    Styles: eyes = knot (scribble yarn-ball), cartoon (white + pupil), dot (bean eyes).
    Optional emote stickers (blush, anger vein, sweat drop, ?, Zzz, heart, !) triggered
    by sustained expressions — never by talking.

    Talking: MediaPipe's inner-lip points barely part during normal speech (worse with a
    moustache), so the mouth opening is a separate signal combining inner lips, outer lips,
    chin drop and jawOpen, each against *your* closed mouth (a rolling low percentile that
    doesn't drift while you talk), scaled by `talk_sensitivity`. Optionally the microphone
    level (`voice_level`, 0..1, set per frame) opens it too.
    """

    TONGUE_STYLE_SPRITE = "sprite"
    TONGUE_STYLE_ORGANIC = "organic"

    def __init__(self, boil_fps: float = 12.0):
        self.boil_fps = float(boil_fps)
        self.gain = 1.6
        self.feature_scale = 1.0
        self.feature_height = 0.0
        self.eye_style = "knot"
        self.show_brows = True
        self.emotes_enabled = False
        self.tongue_style = self.TONGUE_STYLE_SPRITE
        self.tongue_sensitivity = "medium"
        self._filter = OneEuroFilter(min_cutoff=2.5, beta=0.8, d_cutoff=2.0)
        self._baseline = None
        self._calib_frames = 0
        self._calib_acc = None
        self._emote = {k: 0.0 for k in EMOTE_COLORS}
        self._emote_hold = {k: 0.0 for k in EMOTE_COLORS}
        self._emote_until = {k: -1e9 for k in EMOTE_COLORS}
        self._emote_ready = {k: -1e9 for k in EMOTE_COLORS}
        self._bs_rest = {}
        self._closed_since = None
        self._last_t = None
        self._blink_bs_side = None
        # Mouth opening (talking) state.
        self.talk_sensitivity = 1.2
        self.voice_level = None
        self._mouth_floor = None
        self._open_hist = deque()
        self._open_s = 0.0
        self._mouth_open = False
        self._talk_until = -1e9

    # ------------------------------------------------------------------ control
    def set_tongue_style(self, style: str):
        if style in (self.TONGUE_STYLE_SPRITE, self.TONGUE_STYLE_ORGANIC):
            self.tongue_style = style

    def set_tongue_sensitivity(self, s: str):
        if s in ("off", "low", "medium", "high"):
            self.tongue_sensitivity = s

    def calibrate_neutral(self, frames: int = 20):
        """Re-learns your neutral face from the next ~0.7 s (keep a relaxed face)."""
        self._calib_frames = int(frames)
        self._calib_acc = None
        self._mouth_floor = None
        self._bs_rest = {}

    def reset(self):
        self._filter.reset()
        self._baseline = None
        self._emote = {k: 0.0 for k in EMOTE_COLORS}
        self._emote_hold = {k: 0.0 for k in EMOTE_COLORS}
        self._mouth_floor = None
        self._bs_rest = {}
        self._open_hist.clear()
        self._open_s = 0.0
        self._mouth_open = False
        self._last_pose = None
        self._pose_rate = None

    def is_talking(self, t: float) -> bool:
        return t < self._talk_until

    # ------------------------------------------------------------------ talking
    def _update_mouth(self, P, bs, t, dt, fresh, yaw=0.0, pitch=0.0):
        """
        Updates the mouth-opening state from the raw (unfiltered) pose-normalized landmarks.
        Returns the openness to draw (0 = closed).
        """
        o = 0.0
        if fresh and P is not None and len(P) >= 478:
            iu, il = LIPS_INNER_UPPER[3:8], LIPS_INNER_LOWER[3:8]
            stat = np.array([float(np.mean(P[il, 1] - P[iu, 1])), float(bs.get('jawOpen', 0.0)),
                             max(float(bs.get('mouthLowerDownLeft', 0.0)), float(bs.get('mouthLowerDownRight', 0.0)))])
            dyn = np.array([float(P[152, 1] - P[1, 1]), float(P[17, 1] - P[0, 1])])
            # Landmark geometry is only trustworthy near frontal; blendshapes hold up further.
            w_pose = float(np.clip(1.0 - (abs(yaw) - 20.0) / 15.0, 0.0, 1.0) *
                           np.clip(1.0 - (abs(pitch) - 18.0) / 15.0, 0.0, 1.0))
            w_bs = float(np.clip(1.0 - (abs(yaw) - 32.0) / 15.0, 0.0, 1.0) *
                         np.clip(1.0 - (abs(pitch) - 28.0) / 15.0, 0.0, 1.0))
            prev = getattr(self, '_last_pose', None)
            self._last_pose = (yaw, pitch)
            rate = 0.0 if prev is None else float(np.hypot(yaw - prev[0], pitch - prev[1])) / max(dt, 1e-3)
            self._pose_rate = rate if getattr(self, '_pose_rate', None) is None else \
                self._pose_rate + (rate - self._pose_rate) * min(1.0, dt / 0.15)
            # Your *closed* mouth = a low percentile of recent values. While talking the mouth
            # shuts between syllables, so it can't creep up to a half-open "talking" mouth,
            # and one noisy dip can't drag it down and make a relaxed mouth look open.
            hist = self._mouth_floor
            if hist is None:
                hist = self._mouth_floor = deque()
            hist.append((t, stat, dyn))
            while hist and t - hist[0][0] > 4.0:
                hist.popleft()
            s_hist = np.array([s for _, s, _ in hist])
            d_hist = np.array([d for tt, _, d in hist if t - tt <= 0.6])
            s_open = (stat - np.percentile(s_hist, 8, axis=0)) / _STATIC_UNITS
            s_open *= np.array([w_pose, w_bs, w_bs])
            s_open = np.maximum(0.0, s_open - _STATIC_DEAD)
            d_open = (dyn - np.percentile(d_hist, 10, axis=0)) / _DYN_UNITS
            # Both dynamic probes must move (a jaw drop moves chin *and* lower lip), and a
            # nod or head turn (fast pose change) is not a syllable.
            d_val = float(min(d_open)) * w_pose * float(np.clip(1.0 - (self._pose_rate - 15.0) / 30.0, 0.0, 1.0))
            d_val = max(0.0, d_val - _DYN_DEAD)
            o = max(float(np.max(s_open)), d_val) * self.talk_sensitivity
            # Talking = the opening keeps swinging (syllables), not one held open pose.
            self._open_hist.append((t, o))
            while self._open_hist and t - self._open_hist[0][0] > 0.8:
                self._open_hist.popleft()
            if len(self._open_hist) >= 6:
                vals = [v for _, v in self._open_hist]
                if max(vals) - min(vals) > 0.25 and max(vals) > _OPEN_ON:
                    self._talk_until = t + 0.5
        v = self.voice_level
        if v is not None and v > 0.0:
            o = max(o, 0.85 * float(v))
            if v > 0.3:
                self._talk_until = max(self._talk_until, t + 0.5)
        # Instant attack (every syllable shows), ~70 ms release, hysteresis against flicker.
        a = 1.0 if o > self._open_s else min(1.0, dt / 0.07)
        self._open_s += (o - self._open_s) * a
        self._mouth_open = self._open_s > (_OPEN_OFF if self._mouth_open else _OPEN_ON)
        return self._open_s if self._mouth_open else 0.0

    # ------------------------------------------------------------------ measurements
    @staticmethod
    def _measure(P):
        """Scalar/vector descriptors of the current (pose-normalized) face, all in cm."""
        eye_c_r = P[I_ER, :2].mean(axis=0)
        eye_c_l = P[I_EL, :2].mean(axis=0)
        w_r = float(np.linalg.norm(P[I_ER_CORN[0], :2] - P[I_ER_CORN[1], :2]))
        w_l = float(np.linalg.norm(P[I_EL_CORN[0], :2] - P[I_EL_CORN[1], :2]))
        open_r = float(np.mean(P[I_ER_LO[2:5], 1] - P[I_ER_UP[2:5], 1])) / max(w_r, 1e-3)
        open_l = float(np.mean(P[I_EL_LO[2:5], 1] - P[I_EL_UP[2:5], 1])) / max(w_l, 1e-3)
        gaze_r = (P[I_IRIS_R, :2] - eye_c_r) / max(w_r * 0.5, 1e-3)
        gaze_l = (P[I_IRIS_L, :2] - eye_c_l) / max(w_l * 0.5, 1e-3)
        return {
            'eye_c_r': eye_c_r, 'eye_c_l': eye_c_l, 'eye_w': 0.5 * (w_r + w_l),
            'open_r': open_r, 'open_l': open_l,
            'gaze': 0.5 * (gaze_r + gaze_l),
            'mu': P[I_MU, :2].copy(), 'ml': P[I_ML, :2].copy(),
            'brow_r': P[I_BR, :2] - eye_c_r, 'brow_l': P[I_BL, :2] - eye_c_l,
        }

    def _update_baseline(self, m, bs, talking=False):
        # Never learn "neutral" mid-sentence: the baseline would drift towards a half-open
        # mouth and normal talking would then barely register.
        calm = (not talking and not self._mouth_open and
                bs.get('jawOpen', 0) < 0.10 and max(bs.get('mouthSmileLeft', 0), bs.get('mouthSmileRight', 0)) < 0.25 and
                bs.get('browInnerUp', 0) < 0.25 and max(bs.get('browDownLeft', 0), bs.get('browDownRight', 0)) < 0.25 and
                max(bs.get('eyeBlinkLeft', 0), bs.get('eyeBlinkRight', 0)) < 0.35 and bs.get('mouthPucker', 0) < 0.35)
        if self._calib_frames > 0:
            self._calib_frames -= 1
            if self._calib_acc is None:
                self._calib_acc = {k: np.array(v, dtype=np.float64) for k, v in m.items()}
                self._calib_n = 1
            else:
                for k, v in m.items():
                    self._calib_acc[k] = self._calib_acc[k] + v
                self._calib_n += 1
            if self._calib_frames == 0:
                self._baseline = {k: v / self._calib_n for k, v in self._calib_acc.items()}
            return
        if self._baseline is None:
            self._baseline = {k: np.array(v, dtype=np.float64) for k, v in m.items()}
            return
        if calm:
            a = 0.012
            for k, v in m.items():
                self._baseline[k] = self._baseline[k] * (1 - a) + np.asarray(v) * a

    # ------------------------------------------------------------------ render
    def render(self, frame_bgr, tracking_data, helmet, style="ink", t_sec=None, line_px=3):
        if not tracking_data.get('has_face', False):
            return frame_bgr
        rvec, tvec, cam = tracking_data.get('rvec'), tracking_data.get('tvec'), tracking_data.get('cam_matrix')
        if rvec is None or tvec is None or cam is None or helmet.last_sil_bbox is None:
            return frame_bgr
        pal = PALETTES[resolve_style(style)]
        t = time.time() if t_sec is None else t_sec
        dt = 1 / 30.0 if self._last_t is None else float(np.clip(t - self._last_t, 1e-3, 0.2))
        self._last_t = t
        bs = tracking_data.get('blendshapes') or {}
        P = tracking_data.get('face_pts')
        fresh = tracking_data.get('face_fresh', True)
        openness = self._update_mouth(P, bs, t, dt, fresh, tracking_data.get('yaw', 0.0), tracking_data.get('pitch', 0.0))
        talking = self.is_talking(t)
        m = None
        if P is not None and len(P) >= 478:
            sub = self._filter.filter(P[_TRACKED], t) if fresh else self._filter.x_prev
            if sub is None:
                sub = P[_TRACKED]
            m = self._measure(sub)
            if fresh:
                self._update_baseline(m, bs, talking)
        base = self._baseline
        step = int(t * self.boil_fps)
        rng = np.random.RandomState(step & 0x7FFFFFFF)

        g = self.gain
        rx, ry = helmet.base_rx, helmet.base_ry
        k = (rx / 7.6) * 0.86 * self.feature_scale
        vk = 0.8   # cartoon proportions: features sit closer together vertically
        eye_line_y = -0.12 * ry + self.feature_height
        if base is not None:
            anchor = 0.5 * (base['eye_c_r'] + base['eye_c_l'])
        else:
            anchor = np.array([0.0, -3.2])

        def H(p, center=None):
            """skull-frame (x, y) -> helmet-space (x, y). With `center`, the feature is
            placed by its center (vertically compressed layout) but keeps its true shape."""
            p = np.asarray(p, dtype=np.float64)
            c = p if center is None else np.asarray(center, dtype=np.float64)
            cx = k * (c[..., 0] - anchor[0])
            cy = eye_line_y + k * vk * (c[..., 1] - anchor[1])
            if center is None:
                return np.stack([cx, cy], axis=-1)
            return np.stack([cx + k * (p[..., 0] - c[0]), cy + k * (p[..., 1] - c[1])], axis=-1)

        def amp(cur, ref):
            return ref + g * (np.asarray(cur) - ref)

        fills, strokes, floats = [], [], []
        ink, paper, cavity = pal['ink'], pal['paper'], pal['cavity']
        lw = max(1, int(line_px))
        jit = 0.05 * k

        # ---------------- eyes
        if m is not None and base is not None:
            eyes = [(m['eye_c_r'], base['eye_c_r'], m['open_r'], base['open_r'], -1.0),
                    (m['eye_c_l'], base['eye_c_l'], m['open_l'], base['open_l'], 1.0)]
            gaze = np.clip((m['gaze'] - base['gaze']) * 1.6, -1.0, 1.0)
            eye_w = base['eye_w']
        else:
            eyes = [(np.array([-3.1, -3.2]), np.array([-3.1, -3.2]), 0.3, 0.3, -1.0),
                    (np.array([3.1, -3.2]), np.array([3.1, -3.2]), 0.3, 0.3, 1.0)]
            gaze, eye_w = np.zeros(2), 3.0
        smile = max(bs.get('mouthSmileLeft', 0.0), bs.get('mouthSmileRight', 0.0))
        blink_bs = (bs.get('eyeBlinkRight', 0.0), bs.get('eyeBlinkLeft', 0.0))
        r_eye = eye_w * 0.43 * k
        opens = []
        # While the helmet is holding (face not visible) the last expression is kept, with an
        # occasional idle blink so the character doesn't look frozen.
        idle_blink = (not tracking_data.get('face_fresh', True)) and (t % 3.4) < 0.14
        for (cur_c, ref_c, o_cur, o_ref, side), blink in zip(eyes, blink_bs):
            o = 1.0 + g * (o_cur / max(o_ref, 1e-3) - 1.0)
            if blink > 0.6 or idle_blink:
                o = min(o, 0.08)
            o = float(np.clip(o, 0.0, 1.55))
            opens.append(o)
            c = H(ref_c)
            self._eye(c, r_eye, o, gaze, smile, side, rng, jit, ink, paper, fills, strokes, lw)

        # ---------------- brows
        if self.show_brows:
            for side, key in ((-1.0, 'brow_r'), (1.0, 'brow_l')):
                c = H((base or {}).get('eye_c_r' if side < 0 else 'eye_c_l', np.array([side * 3.1, -3.2])))
                if m is not None and base is not None:
                    delta = (m[key] - base[key]) * g * k   # per-point displacement (outer -> inner)
                else:
                    delta = np.zeros((5, 2))
                self._brow(c, r_eye, side, delta, rng, jit, ink, strokes, lw)
            if bs and max(bs.get('browDownLeft', 0), bs.get('browDownRight', 0)) > 0.45:
                cy = eye_line_y - r_eye * 1.25
                for dx in (-0.35, 0.35):
                    strokes.append((np.array([[dx * k, cy - 0.5 * k], [dx * 0.7 * k, cy + 0.2 * k]]), ink, lw))
            if bs and bs.get('browInnerUp', 0) > 0.55:
                for j in range(2):
                    yy = eye_line_y - r_eye * (2.3 + 0.45 * j)
                    xs = np.linspace(-1.2, 1.2, 7) * k
                    strokes.append((np.column_stack([xs, yy - 0.25 * k * (1 - (xs / (1.2 * k)) ** 2)]), ink, max(1, lw - 1)))

        # ---------------- mouth
        self._mouth(m, base, bs, H, amp, k, rng, jit, pal, fills, strokes, lw, openness,
                    allow_tongue=fresh and not talking)

        # ---------------- emotes (only react to fresh, roughly frontal tracking — blendshapes
        # get unreliable when the face is steeply turned, which would pop random stickers)
        if self.emotes_enabled:
            reliable = fresh and abs(tracking_data.get('pitch', 0.0)) < 30 and abs(tracking_data.get('yaw', 0.0)) < 40
            self._emotes(bs if reliable else {}, opens if reliable else [1.0], dt, H, k, eye_line_y, r_eye, rx, ry,
                         rng, fills, strokes, floats, lw, t, talking)

        self._draw(frame_bgr, helmet, rvec, tvec, cam, fills, strokes, floats)
        return frame_bgr

    # ------------------------------------------------------------------ feature geometry
    def _eye(self, c, r, o, gaze, smile, side, rng, jit, ink, paper, fills, strokes, lw):
        gx, gy = gaze * np.array([0.42, 0.32]) * r
        if o < 0.2:
            # Closed eye: happy "^" when smiling, relaxed arc otherwise.
            t = np.linspace(-1, 1, 9)
            if smile > 0.35:
                y = c[1] - (1 - t ** 2) * r * 0.45 + r * 0.15
            else:
                y = c[1] + (1 - t ** 2) * r * 0.35 - r * 0.05
            pts = np.column_stack([c[0] + t * r * 1.05, y])
            strokes.append((pts + rng.normal(0, jit * 0.5, pts.shape), ink, lw + 1))
            return
        vy = min(1.0, o)
        if self.eye_style == "cartoon":
            th = np.linspace(0, 2 * np.pi, 28, endpoint=False)
            ex, ey = r * 1.0, r * 1.18 * vy
            sclera = np.column_stack([c[0] + ex * np.cos(th), c[1] + ey * np.sin(th)])
            fills.append((sclera, paper))
            pr = r * 0.46
            px = np.clip(gx, -(ex - pr * 0.8), ex - pr * 0.8)
            py = np.clip(gy, -(ey - pr * 0.6), ey - pr * 0.6)
            pry = min(pr, ey * 0.85)
            pupil = np.column_stack([c[0] + px + pr * np.cos(th), c[1] + py + pry * np.sin(th)])
            fills.append((pupil, ink))
            fills.append((np.array([[c[0] + px - pr * 0.35 + 0.18 * r * np.cos(a), c[1] + py - pry * 0.35 + 0.18 * r * np.sin(a)]
                                    for a in th[::3]]), paper))
            strokes.append((np.vstack([sclera, sclera[:1]]) + rng.normal(0, jit * 0.4, (len(sclera) + 1, 2)), ink, lw))
            if o > 1.12:  # wide open: lashes flick
                for a in (-0.6, 0.0, 0.6):
                    base_pt = np.array([c[0] + ex * np.sin(a) * 0.9, c[1] - ey * np.cos(a) * 0.95])
                    strokes.append((np.array([base_pt, base_pt + np.array([np.sin(a), -1.0]) * r * 0.35]), ink, lw))
            return
        if self.eye_style == "dot":
            th = np.linspace(0, 2 * np.pi, 20, endpoint=False)
            dot = np.column_stack([c[0] + gx * 0.6 + r * 0.55 * np.cos(th), c[1] + gy * 0.6 + r * 0.75 * vy * np.sin(th)])
            fills.append((dot + rng.normal(0, jit * 0.3, dot.shape), ink))
            return
        # "knot": dense scribble yarn-ball that squashes with the eyelids and rolls with gaze.
        cx, cy = c[0] + gx, c[1] + gy
        th = np.linspace(0, 2 * np.pi, 17)            # closed loop (last == first angle)
        loops = 12
        li = np.arange(loops)
        ph = li * 1.37 + rng.uniform(0, 2 * np.pi, loops)
        rs = 0.25 + 0.75 * np.sin((li / (loops - 1)) * np.pi / 2)
        freq = rng.choice([3, 4, 5], loops)
        wamp = rng.uniform(0.06, 0.16, loops) * r
        rad = r * 0.82 * rs[:, None] + np.sin(freq[:, None] * th[None, :] + ph[:, None]) * wamp[:, None]
        ang = th[None, :] + ph[:, None] * 0.25
        X = cx + rad * np.cos(ang)
        Y = cy + rad * vy * 1.05 * np.sin(ang)
        for xs, ys in zip(X, Y):
            strokes.append((np.column_stack([xs, ys]), ink, max(1, lw - 1)))
        th2 = np.linspace(0, 2 * np.pi, 18, endpoint=False)
        core = np.column_stack([cx + r * 0.55 * np.cos(th2), cy + r * 0.55 * vy * np.sin(th2)])
        fills.append((core, ink))

    def _brow(self, c, r, side, delta, rng, jit, ink, strokes, lw):
        # Stylized arc (outer -> inner) above the drawn eye, deformed per point by the
        # exaggerated real brow motion (raise, furrow, asymmetric skeptical brow...).
        t = np.linspace(0.0, 1.0, 5)
        xs = c[0] + side * (1.05 - 1.9 * t) * r
        ys = c[1] - r * 1.45 - 0.28 * r * np.sin(np.pi * (0.15 + 0.7 * t))
        pts = np.column_stack([xs, ys]) + np.column_stack([delta[:, 0] * 0.35, np.clip(delta[:, 1], -r * 1.2, r * 0.9)])
        # Densify for a smooth stroke.
        tt = np.linspace(0, 4, 13)
        dense = np.column_stack([np.interp(tt, np.arange(5), pts[:, 0]), np.interp(tt, np.arange(5), pts[:, 1])])
        strokes.append((dense + rng.normal(0, jit * 0.4, dense.shape), ink, lw + 1))
        strokes.append((dense + np.array([0, r * 0.1]) + rng.normal(0, jit * 0.4, dense.shape), ink, max(1, lw - 1)))

    def _mouth(self, m, base, bs, H, amp, k, rng, jit, pal, fills, strokes, lw, openness=0.0, allow_tongue=True):
        ink, paper, cavity = pal['ink'], pal['paper'], pal['cavity']
        # The lip *shape* (smile, frown, pout, lopsided grin) is exaggerated at most ~2.2x:
        # beyond that, landmark jitter makes a closed mouth wriggle as if chewing.
        g = min(self.gain, 2.2)
        if m is not None and base is not None:
            bmu, bml = base['mu'], base['ml']
            center = 0.5 * (bmu.mean(axis=0) + bml.mean(axis=0))
            # Cartoon neutral = a clean straight line as wide as your mouth; everything
            # your lips do relative to your own relaxed mouth is added on top, amplified.
            half_w = 0.5 * (bmu[-1, 0] - bmu[0, 0]) * 1.1
            neutral = np.column_stack([center[0] + np.linspace(-half_w, half_w, 11), np.full(11, center[1])])
            mu = neutral + g * (m['mu'] - bmu)
            ml = neutral + g * (m['ml'] - bml)
        else:
            center = np.array([0.0, 4.4])
            xs = np.linspace(-2.6, 2.6, 11)
            mu = np.column_stack([xs, np.full(11, 4.4)])
            ml = mu.copy()
        ml[:, 1] = np.maximum(ml[:, 1], mu[:, 1])
        U, L = H(mu, center), H(ml, center)
        if openness > 0.0:
            # Open by the talking signal (never less than the lips themselves show): a clean
            # lens around the lips' midline, the lower lip/jaw doing most of the travel.
            mid = 0.5 * (U + L)
            prof = np.sin(np.linspace(0.0, np.pi, len(mid))) ** 0.7
            want = prof * float(np.clip(openness, 0.0, 1.3)) * 2.3 * k
            gp = np.maximum(L[:, 1] - U[:, 1], want)
            U = np.column_stack([mid[:, 0], mid[:, 1] - 0.38 * gp])
            L = np.column_stack([mid[:, 0], mid[:, 1] + 0.62 * gp])
        gap = float(np.max(L[3:8, 1] - U[3:8, 1]))
        jaw = bs.get('jawOpen', 0.0)
        teeth = bs.get('teethShowing', 0.0)
        tongue_thr = {"low": 0.65, "medium": 0.45, "high": 0.28}.get(self.tongue_sensitivity)
        show_tongue = (allow_tongue and tongue_thr is not None and bs.get('tongueOut', 0.0) > tongue_thr and jaw < 0.22
                       and bs.get('speechViseme', 0.0) < 0.05)

        if openness <= 0.0:
            seam = 0.5 * (U + L)
            strokes.append((seam + rng.normal(0, jit * 0.35, seam.shape), ink, lw + 1))
            strokes.append((seam + np.array([0, 0.12 * k]) + rng.normal(0, jit * 0.35, seam.shape), ink, max(1, lw - 1)))
            # Corner ticks follow the real corner lift (smile up / frown down).
            lift = float(seam[5, 1] - 0.5 * (seam[0, 1] + seam[-1, 1]))
            if abs(lift) > 0.18 * k:
                for end, nxt, sgn in ((seam[0], seam[1], -1.0), (seam[-1], seam[-2], 1.0)):
                    d = np.array([sgn * 0.45 * k, -np.sign(lift) * 0.45 * k])
                    strokes.append((np.array([end, end + d]), ink, lw))
        else:
            poly = np.vstack([U, L[::-1]])
            fills.append((poly, cavity))
            if teeth > 0.3 and gap > 0.4 * k:
                band = min(gap * 0.32, 0.9 * k)
                tb = np.vstack([U[1:-1], (U[1:-1] + np.array([0, band]))[::-1]])
                fills.append((tb, paper))
                strokes.append((U[1:-1] + np.array([0, band]), ink, max(1, lw - 1)))
            if show_tongue:
                self._tongue(L, k, rng, pal, fills, strokes, lw)
            outline = np.vstack([poly, poly[:1]])
            strokes.append((outline + rng.normal(0, jit * 0.3, outline.shape), ink, lw + 1))
        if show_tongue and openness <= 0.0:
            self._tongue(0.5 * (U + L), k, rng, pal, fills, strokes, lw)

    def _tongue(self, lower, k, rng, pal, fills, strokes, lw):
        c = lower[5]
        tw = (lower[-1, 0] - lower[0, 0]) * 0.24
        t = np.linspace(-1, 1, 15)
        drop = 1.25 * k
        u = np.column_stack([c[0] + t * tw, c[1] + drop * (1 - np.abs(t) ** 2.4)])
        red = (40, 40, 215) if self.tongue_style == self.TONGUE_STYLE_SPRITE else (30, 25, 200)
        fills.append((np.vstack([u, u[:1]]), red))
        strokes.append((u, pal['ink'], lw))
        strokes.append((np.array([[c[0], c[1] + 0.15 * k], [c[0], c[1] + drop * 0.7]]), pal['ink'], max(1, lw - 1)))

    def _emote_triggers(self, bs, opens, t, dt, talking):
        """Which stickers should be showing now (sustain / min-show / cooldown applied)."""
        # Blendshapes relative to *your* resting face: some faces rest with a slight smile
        # or raised brows, which would otherwise count as an expression all the time.
        rest = self._bs_rest
        for key in (_EMOTE_BS if bs else ()):
            v = float(bs.get(key, 0.0))
            r = rest.get(key, v)
            if v < r:
                r += (v - r) * min(1.0, dt / 0.5)
            elif not talking:
                r += (v - r) * min(1.0, dt / 20.0)
            rest[key] = r

        def d(key):
            return max(0.0, float(bs.get(key, 0.0)) - rest.get(key, 0.0))

        smile = max(d('mouthSmileLeft'), d('mouthSmileRight'))
        brow_down = max(d('browDownLeft'), d('browDownRight'))
        brow_up = d('browInnerUp')
        frown = max(d('mouthFrownLeft'), d('mouthFrownRight'))
        asym = abs(bs.get('browOuterUpLeft', 0) - bs.get('browOuterUpRight', 0))
        closed = max(opens) < 0.2 if opens else False
        if closed:
            self._closed_since = self._closed_since or t
        else:
            self._closed_since = None
        conds = {
            'blush': smile > 0.55,
            'anger': brow_down > 0.5 and smile < 0.2,
            'sweat': brow_up > 0.45 and (frown > 0.25 or d('mouthStretchLeft') > 0.3),
            'question': asym > 0.35,
            'zzz': self._closed_since is not None and (t - self._closed_since) > 1.2,
            'heart': d('mouthPucker') > 0.6,
            'shock': d('jawOpen') > 0.45 and brow_up > 0.4,
        }
        showing = {}
        for key, on in conds.items():
            # Talking suppresses everything except "Zzz" (eyes closed isn't speech).
            held = on and (key == 'zzz' or not talking)
            self._emote_hold[key] = self._emote_hold[key] + dt if held else max(0.0, self._emote_hold[key] - 2.0 * dt)
            if t < self._emote_until[key]:
                if held:   # keep it up while the expression lasts
                    self._emote_until[key] = max(self._emote_until[key], t + 0.25)
                    self._emote_ready[key] = self._emote_until[key] + EMOTE_COOLDOWN
            elif held and self._emote_hold[key] >= EMOTE_SUSTAIN[key] and t >= self._emote_ready[key]:
                self._emote_until[key] = t + EMOTE_MIN_SHOW
                self._emote_ready[key] = self._emote_until[key] + EMOTE_COOLDOWN
            showing[key] = t < self._emote_until[key]
        return showing

    def _emotes(self, bs, opens, dt, H, k, eye_line_y, r_eye, rx, ry, rng, fills, strokes, floats, lw, t,
                talking=False):
        targets = self._emote_triggers(bs, opens, t, dt, talking)
        for key, on in targets.items():
            rate = 5.0 if on else 3.0
            self._emote[key] += ((1.0 if on else 0.0) - self._emote[key]) * min(1.0, rate * dt)
        e = self._emote
        if e['blush'] > 0.3:
            for side in (-1, 1):
                cx, cy = side * r_eye * 2.1, eye_line_y + r_eye * 1.7
                for j in range(3):
                    x = cx + (j - 1) * r_eye * 0.4
                    strokes.append((np.array([[x - r_eye * 0.12, cy + r_eye * 0.18], [x + r_eye * 0.12, cy - r_eye * 0.18]]),
                                    EMOTE_COLORS['blush'], lw + 1))
        if e['anger'] > 0.3:
            cx, cy, s = rx * 0.42, eye_line_y - ry * 0.42, 1.1 * k
            for a, b in (((-1, -0.3), (-0.3, -1)), ((0.3, -1), (1, -0.3)), ((1, 0.3), (0.3, 1)), ((-0.3, 1), (-1, 0.3))):
                strokes.append((np.array([[cx + a[0] * s, cy + a[1] * s], [cx + b[0] * s, cy + b[1] * s]]), EMOTE_COLORS['anger'], lw + 2))
        if e['sweat'] > 0.3:
            cx, cy = -rx * 0.62, eye_line_y - ry * 0.18
            th = np.linspace(0, np.pi, 10)
            drop = np.vstack([[[cx, cy - 1.6 * k]], np.column_stack([cx + 0.7 * k * np.cos(th), cy + 0.7 * k * np.sin(th)])])
            fills.append((drop, EMOTE_COLORS['sweat']))
            strokes.append((np.vstack([drop, drop[:1]]), (120, 70, 20), lw))
        bob = np.sin(t * 3.0) * 0.4 * k
        top = -ry * 1.05
        if e['question'] > 0.3:
            th = np.linspace(np.pi, -0.4 * np.pi, 14)
            q = np.column_stack([rx * 0.75 + 0.9 * k * np.cos(th), top + bob + 0.9 * k * np.sin(th)])
            q = np.vstack([q, [q[-1, 0], q[-1, 1] + 1.0 * k]])
            floats.append((q, None, lw + 2))
            floats.append((np.array([[q[-1, 0], q[-1, 1] + 0.8 * k], [q[-1, 0], q[-1, 1] + 0.95 * k]]), None, lw + 3))
        if e['zzz'] > 0.3:
            for j, s in enumerate((0.7, 1.0, 1.3)):
                x0, y0 = rx * (0.45 + 0.35 * j), top - j * 2.0 * k + bob
                w = s * k
                floats.append((np.array([[x0, y0], [x0 + w, y0], [x0, y0 + w], [x0 + w, y0 + w]]), None, lw + 1))
        if e['heart'] > 0.3:
            th = np.linspace(0, 2 * np.pi, 30)
            hx = 16 * np.sin(th) ** 3
            hy = -(13 * np.cos(th) - 5 * np.cos(2 * th) - 2 * np.cos(3 * th) - np.cos(4 * th))
            rise = (t * 2.0) % 3.0
            heart = np.column_stack([rx * 0.35 + hx * 0.07 * k, -ry * 0.1 - rise * 2.5 * k + hy * 0.07 * k])
            floats.append((heart, EMOTE_COLORS['heart'], -1))
        if e['shock'] > 0.3:
            for dx in (-0.6, 0.6):
                x = dx * k
                floats.append((np.array([[x, top - 2.2 * k + bob], [x, top - 0.6 * k + bob]]), None, lw + 3))
                floats.append((np.array([[x, top - 0.1 * k + bob], [x, top + 0.05 * k + bob]]), None, lw + 4))

    # ------------------------------------------------------------------ rasterize
    def _draw(self, frame, helmet, rvec, tvec, cam, fills, strokes, floats):
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        items = [(pts, 'fill', col, 0) for pts, col in fills] + [(pts, 'stroke', col, th) for pts, col, th in strokes]
        rx0, ry0, rx1, ry1 = helmet.last_sil_bbox
        sil = getattr(helmet, 'last_sil', None)
        if items and sil is not None:
            allxy = np.vstack([p for p, *_ in items])
            pts3, normals = helmet.front_surface(allxy, lift=0.3)
            pc = helmet.to_camera(pts3, R, t)
            uv = helmet.project_cam(pc, cam) - (rx0, ry0)
            vis = np.einsum("ij,ij->i", normals @ R.T, pc) < -0.05 * np.linalg.norm(pc, axis=1)
            roi = frame[ry0:ry1, rx0:rx1]
            canvas = roi.copy()
            groups = {}
            o = 0
            for pts, kind, col, th in items:
                n = len(pts)
                u, v = uv[o:o + n], vis[o:o + n]
                o += n
                if kind == 'fill':
                    if v.mean() > 0.75:
                        cv2.fillPoly(canvas, [np.round(u * 4).astype(np.int32)], col, cv2.LINE_AA, 2)
                    continue
                if not v.any():
                    continue
                if v.all():
                    runs = [u]
                else:
                    idx = np.flatnonzero(np.diff(v.astype(np.int8))) + 1
                    runs = [r for r, rv in zip(np.split(u, idx), np.split(v, idx)) if rv[0] and len(r) >= 2]
                groups.setdefault((col, th), []).extend(np.round(r * 4).astype(np.int32).reshape(-1, 1, 2) for r in runs)
            for (col, th), polys in groups.items():
                cv2.polylines(canvas, polys, False, col, max(1, th), cv2.LINE_AA, 2)
            cv2.copyTo(canvas, sil, roi)
        if floats:
            ink = PALETTES['ink']['ink']
            for pts, col, th in floats:
                pts3 = np.column_stack([pts, np.full(len(pts), -helmet.base_rz * 0.3)])
                uv = helmet.project_cam(helmet.to_camera(pts3, R, t), cam)
                c = col if col is not None else ink
                p = np.round(uv * 4).astype(np.int32).reshape(-1, 1, 2)
                if th < 0:
                    cv2.fillPoly(frame, [p], c, cv2.LINE_AA, 2)
                else:
                    cv2.polylines(frame, [p], False, (250, 250, 250), th + 3, cv2.LINE_AA, 2)
                    cv2.polylines(frame, [p], False, c, th, cv2.LINE_AA, 2)
