import os
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial.transform import Rotation

from app.tracking.temporal_smoother import PoseFilter
from app.tracking.mp_worker import LandmarkWorker, _FaceSet
from app.tracking.hand_tracker import HandTracker
from app.tracking.face_geometry import (
    CANONICAL_SKULL, RIGID_IDX, normalize_landmarks,
)


# ==============================================================================
# Skull frame (metric cm) — see face_geometry.py for the full definition.
#   origin: midpoint between the ear canals, +X screen right, +Y down, -Z forward.
# Facing the camera squarely => rotation = identity, Euler (yaw, pitch, roll) = 0.
# ==============================================================================

CANONICAL_POINTS_3D = {
    'nose_tip':         np.array([ 0.0,   1.2, -11.2], dtype=np.float64),
    'nose_bridge':      np.array([ 0.0,  -1.8,  -8.5], dtype=np.float64),
    'glabella':         np.array([ 0.0,  -4.5,  -7.8], dtype=np.float64),
    'forehead_mid':     np.array([ 0.0,  -7.2,  -6.8], dtype=np.float64),
    'eye_left_outer':   np.array([ 3.6,  -3.2,  -7.0], dtype=np.float64),
    'eye_left_inner':   np.array([ 1.6,  -3.2,  -7.5], dtype=np.float64),
    'eye_left_pupil':   np.array([ 2.6,  -3.2,  -7.2], dtype=np.float64),
    'eye_right_outer':  np.array([-3.6,  -3.2,  -7.0], dtype=np.float64),
    'eye_right_inner':  np.array([-1.6,  -3.2,  -7.5], dtype=np.float64),
    'eye_right_pupil':  np.array([-2.6,  -3.2,  -7.2], dtype=np.float64),
    'ear_left_tragus':  np.array([ 7.8,   0.0,   0.0], dtype=np.float64),
    'ear_right_tragus': np.array([-7.8,   0.0,   0.0], dtype=np.float64),
    'zygoma_left':      np.array([ 4.8,  -0.8,  -5.5], dtype=np.float64),
    'zygoma_right':     np.array([-4.8,  -0.8,  -5.5], dtype=np.float64),
    'chin':             np.array([ 0.0,   9.8,  -7.2], dtype=np.float64),
    'mouth_center':     np.array([ 0.0,   4.8,  -7.0], dtype=np.float64),
    'mouth_left':       np.array([ 2.5,   4.8,  -6.8], dtype=np.float64),
    'mouth_right':      np.array([-2.5,   4.8,  -6.8], dtype=np.float64),
    'ear_left_helix':   np.array([ 8.4,  -2.5,   1.2], dtype=np.float64),
    'ear_right_helix':  np.array([-8.4,  -2.5,   1.2], dtype=np.float64),
    'ear_left_lobule':  np.array([ 8.0,   2.2,   0.5], dtype=np.float64),
    'ear_right_lobule': np.array([-8.0,   2.2,   0.5], dtype=np.float64),
    'back_of_head':     np.array([ 0.0,  -2.0,   9.5], dtype=np.float64),
    'crown':            np.array([ 0.0, -11.5,   1.0], dtype=np.float64),
    'nape':             np.array([ 0.0,   8.5,   7.5], dtype=np.float64),
    'parietal_left':    np.array([ 6.2,  -8.0,   5.0], dtype=np.float64),
    'parietal_right':   np.array([-6.2,  -8.0,   5.0], dtype=np.float64),
}
_FEATURE_NAMES = list(CANONICAL_POINTS_3D.keys())
_FEATURE_PTS = np.array([CANONICAL_POINTS_3D[k] for k in _FEATURE_NAMES], dtype=np.float64)

# Whole-head envelope (face + crown + ears + back of head) used for the head hull.
SKULL_HULL_3D = np.array([CANONICAL_POINTS_3D[k] for k in (
    'chin', 'mouth_left', 'mouth_right', 'nose_tip', 'glabella', 'crown',
    'ear_left_helix', 'ear_left_tragus', 'ear_left_lobule',
    'ear_right_helix', 'ear_right_tragus', 'ear_right_lobule',
    'parietal_left', 'parietal_right', 'back_of_head', 'nape')], dtype=np.float64)

# Fallback PnP set if the canonical model file is missing.
_FALLBACK_PNP = [('nose_tip', 1), ('nose_bridge', 168), ('glabella', 10), ('forehead_mid', 151),
                 ('eye_left_outer', 263), ('eye_left_inner', 362), ('eye_right_outer', 33),
                 ('eye_right_inner', 133), ('ear_left_tragus', 454), ('ear_right_tragus', 234),
                 ('zygoma_left', 447), ('zygoma_right', 227)]

if CANONICAL_SKULL is not None:
    PNP_IDX = RIGID_IDX
    PNP_OBJECT_POINTS = CANONICAL_SKULL[RIGID_IDX].astype(np.float64)
else:
    PNP_IDX = np.array([i for _, i in _FALLBACK_PNP], dtype=np.int32)
    PNP_OBJECT_POINTS = np.array([CANONICAL_POINTS_3D[n] for n, _ in _FALLBACK_PNP], dtype=np.float64)
# Optical-flow seed region while holding: the whole head (face + crown + hair + ears),
# pulled in 15% so few seeds land on the background. When the head pitches down the
# face disappears but the hair keeps being tracked.
_SEED_REGION_3D = SKULL_HULL_3D * np.array([0.85, 0.85, 0.85]) + np.array([0.0, -1.0, 0.0])


class FaceTracker:
    """
    Real-time 3D head tracker built for *privacy first*.

    - MediaPipe FaceLandmarker (478 landmarks + 52 blendshapes) on a small proxy image.
    - Metric head pose via PnP against ~40 rigid points of MediaPipe's canonical face,
      with reprojection / depth / jump sanity checks so garbage landmarks at extreme
      angles can never fling the helmet off your head.
    - Low-lag adaptive smoothing (PoseFilter) so the helmet doesn't trail fast moves.
    - HOLD state: when the face detector drops out (hand over face, looking far down /
      sideways, motion blur) the helmet stays exactly where your head is and keeps
      following it with sparse optical flow (Lucas-Kanade on hair/skin texture) until
      the face is re-acquired. Flow points on or near a tracked hand are ignored, so a
      hand that covered the face can never carry the helmet away with it. It is only
      released once the person is actually gone from that region (presence signal from
      the segmentation matte).
    - Pose-landmarker rescue for heads that enter the frame turned away.
    - Pose-normalized 3D landmarks for the expression renderer.
    """

    STATE_LOST = 'lost'
    STATE_FACE = 'face_dense'
    STATE_HOLD = 'hold'
    STATE_RESCUE = 'pose_rescue'

    def __init__(self, face_model_path: str, pose_model_path: str = None,
                 proxy_width: int = 512, privacy_hold: bool = True, max_coast_frames: int = None,
                 use_worker: bool = False, enable_rescue: bool = False):
        if not os.path.exists(face_model_path):
            raise FileNotFoundError(f"FaceLandmarker model not found at: {face_model_path}")
        self.face_model_path = face_model_path
        self.proxy_width = int(proxy_width)
        self.hold_enabled = bool(privacy_hold)
        # When the privacy hold is OFF we still coast a handful of frames so a single
        # missed detection doesn't flash the face.
        self.short_coast_frames = 6
        self.max_hold_sec_without_presence = 8.0
        self.absent_frames_to_release = 6

        self._face_opts = dict(
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            # Lower than MediaPipe's 0.5 defaults: keeps the landmark tracker locked on
            # through steeper yaw/pitch; PnP sanity checks reject the rare bad fit.
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        # Landmark inference runs in a worker process when possible (true parallelism
        # with hand tracking and rendering); otherwise in-process.
        self.worker = None
        self.face_landmarker = None
        if use_worker:
            try:
                self.worker = LandmarkWorker("face", face_model_path, self._face_opts)
            except Exception as e:
                print(f"[FaceTracker] Worker process unavailable ({e}); running in-process.")
        if self.worker is None:
            self.face_landmarker = _FaceSet(face_model_path, self._face_opts)
        self._job = None
        self.use_crops = True

        # Optional pose-landmarker rescue (off by default). Body-pose head estimates are
        # unreliable at exactly the moments the face tracker fails (extreme angles, hands
        # over the face, backlight) and a mis-sized helmet is worse for privacy than the
        # pipeline's mosaic fallback, so it's only loaded when explicitly enabled.
        self.pose_landmarker = None
        self.enable_rescue = bool(enable_rescue)
        candidates = []
        if pose_model_path and self.enable_rescue:
            candidates.append(os.path.join(os.path.dirname(pose_model_path), "pose_landmarker_lite.task"))
            candidates.append(pose_model_path)
        for p in candidates:
            if p and os.path.exists(p):
                try:
                    po = vision.PoseLandmarkerOptions(
                        base_options=python.BaseOptions(model_asset_path=p),
                        running_mode=vision.RunningMode.VIDEO, num_poses=1,
                        min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4)
                    self.pose_landmarker = vision.PoseLandmarker.create_from_options(po)
                    break
                except Exception as e:
                    print(f"[FaceTracker] PoseLandmarker init warning ({os.path.basename(p)}): {e}")

        self.filter = PoseFilter()
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self.focal_scale = 1.0
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.state = self.STATE_LOST
        self.ever_acquired = False
        self.filter.reset()
        self.rvec_guess = None
        self.tvec_guess = None
        self.last_raw_R = None
        self.last_raw_time = -1.0
        # MediaPipe graphs keep their own clock across our resets: timestamps must keep
        # increasing, so these are never rewound.
        self.last_timestamp_ms = getattr(self, 'last_timestamp_ms', -1)
        self.last_pose_ts = getattr(self, 'last_pose_ts', -1)
        self.last_face_size = 220
        self.last_lmks_2d = None
        self.last_blendshapes = {}
        self.last_face_pts = None
        self.prev_gray = None
        self.prev_pscale = 1.0
        self.lk_pts = None
        self.hold_start_t = 0.0
        self.hold_frames = 0
        self.absent_frames = 0
        self._rescue_counter = 0
        self._last_center = None
        self._crop_size = None

    def set_proxy_width(self, width: int):
        self.proxy_width = int(max(256, min(1280, width)))

    def set_privacy_hold(self, enabled: bool):
        self.hold_enabled = bool(enabled)

    def _camera_matrix(self, w, h):
        f = float(w) * self.focal_scale
        return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)

    # ------------------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int = None, presence=None, hint=None,
                      hands=None) -> dict:
        """
        Tracks one BGR frame (submit + collect in one call).
        presence: None (unknown) / True / False — whether a person currently occupies the
                  last known head region (from the segmentation matte). Used only to decide
                  when a HOLD can be released; never to *start* showing a face.
        hint: optional (x0, y0, x1, y1) full-res box where a head probably is (top of a
              person silhouette) — used to find small / distant faces while lost.
        hands: optional list of (21, 2) full-res hand landmarks (most recent). While holding,
               optical flow ignores everything on and around them.
        """
        self.submit(frame_bgr, timestamp_ms, hint)
        return self.collect(frame_bgr, presence, hands)

    def _choose_crop(self, w, h, hint):
        """
        Region of the frame given to the landmarker. MediaPipe's face detector sees its whole
        input squeezed to 128x128, so a face smaller than ~1/8 of the frame is nearly
        invisible to it. Following the head with a crop ~3x the face size keeps the face
        big in the model's input at any distance (and sharpens the landmarks). The crop
        centre tracks the face every frame, so in normalized coordinates the face moves
        exactly as it would in the full frame and MediaPipe's own tracking stays stable.
        """
        self._crop_tick = not getattr(self, '_crop_tick', False)
        full = (0, 0, w, h)
        cx = cy = size = None
        if self.state == self.STATE_FACE and self.last_lmks_2d is not None:
            lo, hi = self.last_lmks_2d.min(axis=0), self.last_lmks_2d.max(axis=0)
            cx, cy = (lo + hi) / 2.0
            size = float(max(hi - lo)) * 3.2
        elif self.state in (self.STATE_HOLD, self.STATE_RESCUE) and getattr(self, '_last_center', None) is not None:
            if self._crop_tick:            # alternate with full frame to catch big moves
                return full
            cx, cy = self._last_center
            size = self.last_face_size * 4.0
        elif hint and self._crop_tick:
            x0, y0, x1, y1 = self._pick_hint(hint)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            size = max(x1 - x0, y1 - y0) * 1.5
        if size is None:
            return full
        prev = getattr(self, '_crop_size', None)
        size = size if prev is None else prev + (size - prev) * 0.5
        self._crop_size = size
        size = int(max(256, size))
        if size >= 0.85 * min(w, h):
            return full
        x0 = int(np.clip(cx - size / 2, 0, w - size))
        y0 = int(np.clip(cy - size / 2, 0, h - size))
        return (x0, y0, size, size)

    def _recently_tracked(self, t):
        return self._last_center is not None and (t - self.last_raw_time) < 3.0

    def _pick_hint(self, hints):
        """hints: head boxes (largest first). Right after losing you, search where you were —
        not the biggest silhouette, which might be someone else."""
        if isinstance(hints, tuple):
            return hints
        if self._recently_tracked(self.last_timestamp_ms / 1000.0):
            lx, ly = self._last_center
            return min(hints, key=lambda b: np.hypot((b[0] + b[2]) / 2 - lx, (b[1] + b[3]) / 2 - ly))
        # Never tracked yet / long gone: rotate through everyone, biggest first.
        self._hint_rr = (getattr(self, '_hint_rr', -1) + 1) % len(hints)
        return hints[self._hint_rr] if self._hint_rr < 2 else hints[0]

    def submit(self, frame_bgr: np.ndarray, timestamp_ms: int = None, hint=None):
        """Starts landmark inference for a frame (asynchronous when a worker process is used)."""
        h, w = frame_bgr.shape[:2]
        if timestamp_ms is None or timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = (self.last_timestamp_ms + 1) if self.last_timestamp_ms >= 0 else int(time.time() * 1000)
        self.last_timestamp_ms = timestamp_ms
        cx0, cy0, cw, ch = self._choose_crop(w, h, hint) if self.use_crops else (0, 0, w, h)
        region = frame_bgr if (cw, ch) == (w, h) else frame_bgr[cy0:cy0 + ch, cx0:cx0 + cw]
        s = min(1.0, self.proxy_width / float(max(cw, ch) if (cw, ch) != (w, h) else cw))
        small = cv2.resize(region, (max(1, int(round(cw * s))), max(1, int(round(ch * s)))),
                           interpolation=cv2.INTER_AREA) if s < 1.0 else region
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        # Full-frame grayscale proxy for optical-flow following (independent of the crop).
        gs = min(1.0, 480.0 / w)
        gsmall = cv2.resize(frame_bgr, (int(round(w * gs)), int(round(h * gs))), interpolation=cv2.INTER_AREA) if gs < 1.0 else frame_bgr
        # Full-frame searches (acquisition / recovery) use the 2-face model so we can pick you.
        multi = (cw, ch) == (w, h) and self.state != self.STATE_FACE
        submitted = False
        if self.worker is not None:
            try:
                self.worker.submit(np.ascontiguousarray(rgb), timestamp_ms, multi)
                submitted = True
            except Exception as e:
                print(f"[FaceTracker] Worker submit failed ({e}); switching to in-process.")
                self._drop_worker()
        self._job = dict(ts=timestamp_ms, w=w, h=h, pscale=gs, rgb=rgb, remote=submitted, multi=multi,
                         crop=(cx0, cy0, cw, ch), gray=cv2.cvtColor(gsmall, cv2.COLOR_BGR2GRAY))

    def _drop_worker(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        if self.face_landmarker is None:
            self.face_landmarker = _FaceSet(self.face_model_path, self._face_opts)

    def _infer(self, job):
        """Returns (landmarks, blendshapes) of the face that is *you*, or (None, {})."""
        faces = None
        if job['remote'] and self.worker is not None:
            try:
                faces = self.worker.collect()
            except Exception as e:
                print(f"[FaceTracker] Worker failed ({e}); switching to in-process.")
                self._drop_worker()
        if faces is None:
            if self.face_landmarker is None:
                self._drop_worker()
            faces = self.face_landmarker.run(job['rgb'], job['ts'], job.get('multi', False))
        return self._select_face(faces, job)

    def _select_face(self, faces, job):
        if not faces:
            return None, {}
        # Several faces (someone else in frame): stay on the one where your head was a
        # moment ago; after a longer absence prefer the biggest (closest to the camera).
        cx0, cy0, cw, ch = job['crop']
        centers, sizes = [], []
        for lm, _ in faces:
            xy = lm[:, :2] * (cw, ch) + (cx0, cy0)
            lo, hi = xy.min(axis=0), xy.max(axis=0)
            centers.append((lo + hi) / 2.0)
            sizes.append(float(max(hi - lo)))
        if self._recently_tracked(job['ts'] / 1000.0):
            lx, ly = self._last_center
            d = [np.hypot(c[0] - lx, c[1] - ly) for c in centers]
            i = int(np.argmin(d))
            # A face far from where you just were is someone else: don't jump onto them.
            if self.state != self.STATE_FACE and d[i] > 2.5 * max(sizes[i], self.last_face_size):
                return None, {}
            return faces[i]
        return faces[int(np.argmax(sizes))]

    def _retry_full(self, frame_bgr, job):
        h, w = frame_bgr.shape[:2]
        s = min(1.0, self.proxy_width / float(w))
        small = cv2.resize(frame_bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1.0 else frame_bgr
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        ts = max(job['ts'], self.last_timestamp_ms) + 1
        self.last_timestamp_ms = ts
        retry = dict(job, rgb=rgb, ts=ts, remote=False, multi=True, crop=(0, 0, w, h))
        if self.worker is not None:
            try:
                self.worker.submit(rgb, ts, True)
                retry['remote'] = True
            except Exception:
                self._drop_worker()
        self._crop_size = None
        return self._infer(retry)

    def collect(self, frame_bgr: np.ndarray, presence=None, hands=None) -> dict:
        """Finishes tracking for the frame given to `submit()`."""
        job = self._job
        if job is None:
            raise RuntimeError("collect() called without submit()")
        self._job = None
        h, w = job['h'], job['w']
        timestamp_ms = job['ts']
        t = timestamp_ms / 1000.0
        pscale, rgb, gray = job['pscale'], job['rgb'], job['gray']
        cam = self._camera_matrix(w, h)

        lm_norm, bs_raw = self._infer(job)
        cx0, cy0, cw, ch = job['crop']
        if lm_norm is None and (cw, ch) != (w, h) and self.state != self.STATE_LOST:
            # The head left the crop (a fast move / dropped frames): look at the whole frame
            # right now instead of holding the helmet in the wrong place for a frame.
            lm_norm, bs_raw = self._retry_full(frame_bgr, job)
            if lm_norm is not None:
                cx0, cy0, cw, ch = 0, 0, w, h

        fresh = False
        lmks_2d = None
        if lm_norm is not None:
            arr = lm_norm.astype(np.float64)
            # crop-normalized -> full-frame pixels (z shares x's scale in MediaPipe)
            lmks_2d = (arr[:, :2] * (cw, ch) + (cx0, cy0)).astype(np.float32)
            pose = self._solve_pose(lmks_2d, cam, t)
            if pose is not None:
                fresh = True
                rvec, tvec = pose
                self.filter.update(rvec.ravel(), tvec.ravel(), t)
                self.state = self.STATE_FACE
                self.ever_acquired = True
                self.hold_frames = 0
                self.absent_frames = 0
                self.lk_pts = None
                self.last_lmks_2d = lmks_2d
                self.last_face_size = int(max(np.ptp(lmks_2d[:, 0]), np.ptp(lmks_2d[:, 1])))
                bs = dict(bs_raw)
                self._estimate_mouth_extras(frame_bgr, lmks_2d, bs)
                self.last_blendshapes = bs
                pts3 = np.column_stack([lmks_2d.astype(np.float64), arr[:, 2] * cw])
                face_pts, _ = normalize_landmarks(pts3)
                self.last_face_pts = face_pts

        excl = self._hand_exclusion(gray.shape, pscale, hands)
        if not fresh:
            if self.state in (self.STATE_FACE, self.STATE_HOLD, self.STATE_RESCUE):
                if not self._hold_step(gray, pscale, cam, t, presence, excl):
                    self._release()
            if self.state == self.STATE_LOST and presence is not False and self.pose_landmarker is not None:
                full_rgb = cv2.cvtColor(cv2.resize(frame_bgr, (int(w * pscale), int(h * pscale))), cv2.COLOR_BGR2RGB)
                self._try_rescue(full_rgb, gray, pscale, timestamp_ms, cam, w, h, t)

        self.prev_gray = gray
        self.prev_pscale = pscale
        self._prev_excl = excl

        if self.state == self.STATE_LOST:
            return {
                'has_face': False, 'tracking_mode': 'lost', 'confidence': 0.0,
                'is_coasting': False, 'face_size': self.last_face_size,
                'ever_acquired': self.ever_acquired, 'cam_matrix': cam,
            }
        return self._build_result(cam, w, h, lmks_2d, fresh, t)

    # ------------------------------------------------------------------
    def _solve_pose(self, lmks_2d, cam, t):
        img = lmks_2d[PNP_IDX].astype(np.float64)
        face_size = max(1.0, float(max(np.ptp(lmks_2d[:, 0]), np.ptp(lmks_2d[:, 1]))))

        def solve(use_guess):
            if use_guess and self.rvec_guess is not None:
                ok, r, tv = cv2.solvePnP(PNP_OBJECT_POINTS, img, cam, self.dist_coeffs,
                                         rvec=self.rvec_guess.copy(), tvec=self.tvec_guess.copy(),
                                         useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            else:
                ok, r, tv = cv2.solvePnP(PNP_OBJECT_POINTS, img, cam, self.dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                return None
            proj, _ = cv2.projectPoints(PNP_OBJECT_POINTS, r, tv, cam, self.dist_coeffs)
            err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))
            return r, tv, err

        sol = solve(True)
        if sol is None or sol[2] > 0.05 * face_size:
            alt = solve(False)
            if alt is not None and (sol is None or alt[2] < sol[2]):
                sol = alt
        if sol is None:
            return None
        r, tv, err = sol
        if err > 0.09 * face_size or not (8.0 < float(tv[2, 0]) < 600.0):
            return None

        R, _ = cv2.Rodrigues(r)
        # Implausible orientation: facing away from the camera, or a >55 deg flip within
        # 250 ms. That's usually a bad fit (e.g. the mirror solution) — but the landmarks
        # themselves still sit on the real face, so never discard their *position*: keep the
        # last trusted rotation and move the helmet to where the face is now. If the new
        # orientation persists for a few frames it was real, and is accepted.
        forward = R @ np.array([0.0, 0.0, -1.0])
        implausible = forward[2] > 0.35
        if not implausible and self.last_raw_R is not None and (t - self.last_raw_time) < 0.25:
            d = self.last_raw_R.T @ R
            ang = np.degrees(np.arccos(np.clip((np.trace(d) - 1.0) / 2.0, -1.0, 1.0)))
            implausible = ang > 55.0
        if implausible:
            self._implausible_run = getattr(self, '_implausible_run', 0) + 1
            if self._implausible_run < 3 and self.last_raw_R is not None:
                r_old, _ = cv2.Rodrigues(self.last_raw_R)
                self.rvec_guess = None
                return r_old, tv
            if forward[2] > 0.35:
                return None
        self._implausible_run = 0
        self.rvec_guess, self.tvec_guess = r.copy(), tv.copy()
        self.last_raw_R = R
        self.last_raw_time = t
        return r, tv

    # ------------------------------------------------------------------
    @staticmethod
    def _hand_exclusion(shape, pscale, hands):
        """
        Mask (optical-flow proxy resolution) of everything that belongs to a hand: fingers,
        palm and a long forearm, grown generously. Flow points there move with the hand,
        not the head — following them is how a hand could "carry" the helmet off the face.
        """
        if not hands:
            return None
        m = HandTracker.draw_mask(shape, hands, scale=pscale, thickness=1.6, forearm_len=3.0)
        if not m.any():
            return None
        palm = max(float(np.linalg.norm(h[5] - h[17])) for h in hands) * pscale
        k = int(max(3, palm * 0.6)) | 1
        return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    @staticmethod
    def _outside(pts, mask):
        """Boolean per point: not inside `mask` (points as (N, 1, 2) or (N, 2))."""
        p = pts.reshape(-1, 2)
        if mask is None or len(p) == 0:
            return np.ones(len(p), bool)
        h, w = mask.shape
        xi = np.clip(np.round(p[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(p[:, 1]).astype(int), 0, h - 1)
        return mask[yi, xi] == 0

    def _hold_step(self, gray, pscale, cam, t, presence, hand_excl=None) -> bool:
        """
        Keeps the helmet on the head while the face detector has dropped out.
        hand_excl: this frame's hand mask (flow resolution) — see _hand_exclusion().
        """
        if self.filter.rot is None:
            return False
        entering = self.state != self.STATE_HOLD
        if entering:
            self.state = self.STATE_HOLD
            self.hold_start_t = t
            self.hold_frames = 0
            self.absent_frames = 0
            self.lk_pts = None
            self.hold_tz0 = float(self.filter.trans[2])
        self.hold_frames += 1

        if not self.hold_enabled:
            return self.hold_frames <= self.short_coast_frames

        if presence is False:
            self.absent_frames += 1
        elif presence is True:
            self.absent_frames = 0
        if self.absent_frames >= self.absent_frames_to_release:
            return False
        if presence is None and (t - self.hold_start_t) > self.max_hold_sec_without_presence:
            return False

        rot = self.filter.rot.copy()
        trans = self.filter.trans.copy()
        prev = self.prev_gray
        if prev is not None and prev.shape == gray.shape:
            # Hands in either frame are off-limits (a point seeded on the hand last frame
            # would still follow it this frame).
            prev_excl = getattr(self, '_prev_excl', None)
            if prev_excl is not None and prev_excl.shape != gray.shape:
                prev_excl = None
            block = hand_excl if prev_excl is None else (prev_excl if hand_excl is None else cv2.bitwise_or(hand_excl, prev_excl))
            if self.lk_pts is not None and block is not None:
                keep = self._outside(self.lk_pts, block)
                self.lk_pts = self.lk_pts[keep] if keep.any() else None
            if self.lk_pts is None or len(self.lk_pts) < 8:
                self.lk_pts = self._lk_seed(prev, pscale, rot, trans, cam, exclude=block)
            # Too few head points left (the hand covers everything): stay exactly where the
            # head was rather than guess — the head rarely moves far while a hand covers it.
            if self.lk_pts is not None and len(self.lk_pts) >= 4:
                motion = self._lk_track(prev, gray, hand_excl)
                if motion is not None:
                    dx, dy, sc = motion
                    # A real head can't jump more than ~a quarter face per frame.
                    lim = 0.25 * self.last_face_size * pscale
                    mag = float(np.hypot(dx, dy))
                    if mag > lim:
                        dx, dy = dx * lim / mag, dy * lim / mag
                    tz = trans[2]
                    trans[0] += (dx / pscale) * tz / cam[0, 0]
                    trans[1] += (dy / pscale) * tz / cam[1, 1]
                    # Point-spread scale is noisy and compounds over a long hold, so damp
                    # it and keep depth within a sane band around where the hold began —
                    # tighter on the "shrink" side, since a shrinking helmet uncovers the face.
                    sc = 1.0 + (float(np.clip(sc, 0.95, 1.05)) - 1.0) * 0.5
                    tz0 = getattr(self, 'hold_tz0', tz)
                    trans[2] = float(np.clip(tz / sc, tz0 * 0.85, tz0 * 1.10))
                    self.filter.force(rot, trans, t)
        # Keep the rotation guess in sync so re-acquisition PnP starts near the held pose.
        self.rvec_guess = rot.reshape(3, 1).copy()
        self.tvec_guess = self.filter.trans.reshape(3, 1).copy()
        return True

    def _lk_seed(self, gray, pscale, rot, trans, cam, exclude=None):
        proj, _ = cv2.projectPoints(_SEED_REGION_3D, rot.reshape(3, 1), trans.reshape(3, 1), cam, self.dist_coeffs)
        poly = (proj.reshape(-1, 2) * pscale).astype(np.int32)
        hull = cv2.convexHull(poly)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        if exclude is not None:
            mask[exclude > 0] = 0
        if cv2.countNonZero(mask) < 50:
            return None
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=60, qualityLevel=0.01, minDistance=5, mask=mask, blockSize=5)
        return pts

    def _lk_track(self, prev, gray, hand_excl=None):
        p0 = self.lk_pts.astype(np.float32)
        params = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, **params)
        if p1 is None:
            self.lk_pts = None
            return None
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, p1, None, **params)
        fb = np.linalg.norm((pb - p0).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
        good &= self._outside(p1, hand_excl)      # landed on a hand -> it was following the hand
        if good.sum() < 4:
            self.lk_pts = None
            return None
        a = p0.reshape(-1, 2)[good]
        b = p1.reshape(-1, 2)[good]
        d = np.median(b - a, axis=0)
        ca, cb = np.median(a, axis=0), np.median(b, axis=0)
        ra = np.linalg.norm(a - ca, axis=1)
        rb = np.linalg.norm(b - cb, axis=1)
        valid = ra > 3.0
        sc = float(np.median(rb[valid] / ra[valid])) if valid.sum() >= 4 else 1.0
        self.lk_pts = b.reshape(-1, 1, 2)
        return float(d[0]), float(d[1]), sc

    def _release(self):
        self.state = self.STATE_LOST
        self._crop_size = None
        self.lk_pts = None
        self.rvec_guess = None
        self.tvec_guess = None
        self.last_raw_R = None

    # ------------------------------------------------------------------
    def _try_rescue(self, rgb, gray, pscale, ts_ms, cam, w, h, t):
        """Finds a head the face detector can't (e.g. entering frame turned away)."""
        self._rescue_counter += 1
        if self._rescue_counter % 3 != 0 or ts_ms <= self.last_pose_ts:
            return
        self.last_pose_ts = ts_ms
        try:
            pr = self.pose_landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        except Exception:
            return
        if not pr.pose_landmarks:
            return
        pl = pr.pose_landmarks[0]
        nose, eye_l, eye_r, ear_l, ear_r = pl[0], pl[2], pl[5], pl[7], pl[8]

        def vis(p, th=0.5):
            # Pose landmarks can be hallucinated outside the image for an off-frame
            # head; only trust confident keypoints that are actually inside the frame.
            return ((p.visibility or 0.0) > th and 0.01 < p.x < 0.99 and 0.01 < p.y < 0.99)

        head_pts = [p for p in (nose, eye_l, eye_r, ear_l, ear_r) if vis(p, 0.6)]
        if len(head_pts) < 2:
            return
        sh_l, sh_r = pl[11], pl[12]
        head_y = float(np.mean([p.y for p in head_pts]))
        for sh in (sh_l, sh_r):
            if (sh.visibility or 0.0) > 0.5 and sh.y < head_y:
                return  # "head" below the shoulders — not a real head
        pts = [(p.x * w, p.y * h) for p in (nose, eye_l, eye_r, ear_l, ear_r) if vis(p, 0.35)]
        if not pts:
            return
        pts = np.array(pts)
        if vis(ear_l) and vis(ear_r):
            center = np.array([(ear_l.x + ear_r.x) * 0.5 * w, (ear_l.y + ear_r.y) * 0.5 * h])
            ear_span = float(np.hypot((ear_l.x - ear_r.x) * w, (ear_l.y - ear_r.y) * h))
            span_x = max(1e-3, abs(ear_l.x - ear_r.x))
            yaw = float(np.clip((nose.x - (ear_l.x + ear_r.x) * 0.5) / span_x * 90.0, -80, 80)) if vis(nose, 0.3) else 0.0
            head_w_cm = 15.6 * max(0.35, np.cos(np.radians(yaw)))
            tz = float(np.clip(cam[0, 0] * head_w_cm / max(ear_span, 1.0), 25.0, 400.0))
        else:
            center = pts.mean(axis=0)
            yaw = -75.0 if vis(ear_l) else (75.0 if vis(ear_r) else 0.0)
            tz = float(self.filter.trans[2]) if self.filter.trans is not None else 70.0
        tx = (center[0] - cam[0, 2]) * tz / cam[0, 0]
        ty = (center[1] - cam[1, 2]) * tz / cam[1, 1]
        rot = Rotation.from_euler('yxz', [yaw, 0.0, 0.0], degrees=True).as_rotvec()
        self.filter.reset()
        self.filter.force(rot, np.array([tx, ty, tz]), t)
        self.state = self.STATE_RESCUE
        self.hold_start_t = t
        self.hold_frames = 0
        self.absent_frames = 0
        self.lk_pts = self._lk_seed(gray, pscale, rot, np.array([tx, ty, tz]), cam)
        self.last_face_size = int(cam[0, 0] * 16.0 / tz)

    # ------------------------------------------------------------------
    def _estimate_mouth_extras(self, frame_bgr, lmks_2d, bs):
        """Adds teethShowing / tongueOut / speechViseme heuristics to the blendshape dict."""
        h, w = frame_bgr.shape[:2]
        pt13, pt14, pt17, pt18 = lmks_2d[13], lmks_2d[14], lmks_2d[17], lmks_2d[18]
        mouth_w = max(10.0, float(np.linalg.norm(lmks_2d[291] - lmks_2d[61])))
        open_ratio = max(0.0, float(pt14[1] - pt13[1])) / mouth_w

        jaw = bs.get('jawOpen', 0.0)
        smile = max(bs.get('mouthSmileLeft', 0.0), bs.get('mouthSmileRight', 0.0))
        upper = max(bs.get('mouthUpperUpLeft', 0.0), bs.get('mouthUpperUpRight', 0.0))
        lower = max(bs.get('mouthLowerDownLeft', 0.0), bs.get('mouthLowerDownRight', 0.0))
        funnel = bs.get('mouthFunnel', 0.0)
        pucker = bs.get('mouthPucker', 0.0)
        stretch = max(bs.get('mouthStretchLeft', 0.0), bs.get('mouthStretchRight', 0.0))

        speech = float(np.clip(jaw * 2.2 + funnel * 1.5 + lower * 1.2 + stretch * 0.9, 0.0, 1.0))
        teeth = 0.0
        if upper > 0.12 or (smile > 0.18 and (jaw > 0.04 or open_ratio > 0.04)):
            teeth = float(np.clip(upper * 1.5 + smile * 0.85 + jaw * 0.4, 0.0, 1.0))
        elif jaw > 0.15 and smile > 0.12:
            teeth = float(np.clip(smile * 1.1 + jaw * 0.6, 0.0, 1.0))

        tongue = 0.0
        talking = speech > 0.08 or jaw > 0.28 or funnel > 0.12 or pucker > 0.15 or teeth > 0.10
        if not talking and open_ratio > 0.04:
            ymin = int(pt17[1] + 2)
            ymax = int(min(h, pt18[1] + (pt18[1] - pt17[1]) * 0.5))
            xmin = int(max(0, pt17[0] - mouth_w * 0.16))
            xmax = int(min(w, pt17[0] + mouth_w * 0.16))
            if ymax > ymin + 3 and xmax > xmin + 3:
                hsv = cv2.cvtColor(frame_bgr[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2HSV)
                m = (((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 165)) &
                     (hsv[:, :, 1] >= 80) & (hsv[:, :, 1] <= 250) &
                     (hsv[:, :, 2] >= 60) & (hsv[:, :, 2] <= 245))
                ratio = np.count_nonzero(m) / float(m.size)
                if ratio > 0.35:
                    tongue = float(np.clip((ratio - 0.30) * 3.0, 0.0, 1.0))
        bs['teethShowing'] = teeth
        bs['tongueOut'] = tongue
        bs['speechViseme'] = speech
        bs['mouthOpenRatio'] = open_ratio

    # ------------------------------------------------------------------
    def _build_result(self, cam, w, h, lmks_2d, fresh, t):
        rvec = self.filter.rot.reshape(3, 1).copy()
        tvec = self.filter.trans.reshape(3, 1).copy()
        R, _ = cv2.Rodrigues(rvec)
        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = tvec.ravel()
        yaw, pitch, roll = Rotation.from_matrix(R).as_euler('yxz', degrees=True)

        proj_c, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, cam, self.dist_coeffs)
        cx, cy = proj_c.reshape(2)
        self._last_center = (float(cx), float(cy))
        proj_feat, _ = cv2.projectPoints(_FEATURE_PTS, rvec, tvec, cam, self.dist_coeffs)
        proj_feat = proj_feat.reshape(-1, 2)
        proj_hull, _ = cv2.projectPoints(SKULL_HULL_3D, rvec, tvec, cam, self.dist_coeffs)
        hull_pts = proj_hull.reshape(-1, 2)
        if lmks_2d is not None:
            hull_pts = np.vstack([lmks_2d, hull_pts])
        hull_2d = cv2.convexHull(hull_pts.astype(np.float32)).reshape(-1, 2)
        x0, y0 = hull_2d.min(axis=0)
        x1, y1 = hull_2d.max(axis=0)

        holding = self.state in (self.STATE_HOLD, self.STATE_RESCUE)
        if holding:
            conf = float(np.clip(1.0 - self.hold_frames / 90.0, 0.2, 1.0))
        else:
            conf = 1.0
        return {
            'has_face': True,
            'tracking_mode': self.state,
            'is_coasting': holding,
            'face_fresh': fresh,
            'ever_acquired': self.ever_acquired,
            'confidence': conf,
            'matrix_4x4': mat,
            'rotation_matrix': R,
            'translation_skull': tvec.ravel(),
            'rvec': rvec,
            'tvec': tvec,
            'cam_matrix': cam,
            'yaw': float(yaw), 'pitch': float(pitch), 'roll': float(roll),
            'center_x': int(cx), 'center_y': int(cy),
            'face_size': self.last_face_size,
            'landmarks_2d': lmks_2d,
            'face_pts': self.last_face_pts,
            'projected_features': {n: (int(p[0]), int(p[1])) for n, p in zip(_FEATURE_NAMES, proj_feat)},
            'head_hull_2d': hull_2d,
            'head_bbox': (int(x0), int(y0), int(x1), int(y1)),
            'blendshapes': self.last_blendshapes,
        }

    def close(self):
        if getattr(self, 'worker', None) is not None:
            self.worker.close()
            self.worker = None
        for lm in (getattr(self, 'face_landmarker', None), getattr(self, 'pose_landmarker', None)):
            if lm is not None:
                try:
                    lm.close()
                except Exception:
                    pass
