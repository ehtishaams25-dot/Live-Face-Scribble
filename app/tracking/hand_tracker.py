import os
import numpy as np
import cv2

from app.tracking.mp_worker import LandmarkWorker, run_landmarker, _make_landmarker


FINGER_CHAINS = (
    (1, 2, 3, 4),       # thumb
    (5, 6, 7, 8),       # index
    (9, 10, 11, 12),    # middle
    (13, 14, 15, 16),   # ring
    (17, 18, 19, 20),   # pinky
)
PALM_POLY = (0, 1, 5, 9, 13, 17)


class HandTracker:
    """
    Shared MediaPipe HandLandmarker for the whole pipeline.

    It feeds two consumers:
      * the background matte (a "hand prior" that keeps fast-moving or semi-transparent
        hands solid instead of letting the background bleed through), and
      * the helmet renderer (hands raised in front of the face are drawn on top of
        the scribble head).

    The landmarker is the single most expensive CPU model in the pipeline (~15 ms), so it
    is scheduled: every frame while a hand is near the head (where occlusion accuracy
    matters), otherwise every `every_n` frames.

    Privacy: occlusion masks are only ever built from *fresh* (this-frame) landmarks and
    are shrunk inside the fingers, so a stale or loose hand mask can never cut a hole in
    the helmet that shows the face behind it.
    """

    def __init__(self, model_path: str, num_hands: int = 2, proxy_width: int = 384, every_n: int = 2,
                 use_worker: bool = False):
        self.proxy_width = int(proxy_width)
        self.every_n = max(1, int(every_n))
        self.enabled = True
        self.model_path = model_path
        self._opts = dict(
            num_hands=num_hands,
            # Stricter than before (0.18): a false-positive "hand" on the face would be
            # composited over the helmet, i.e. would leak the face.
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = None
        self.worker = None
        if model_path and os.path.exists(model_path):
            if use_worker:
                try:
                    self.worker = LandmarkWorker("hands", model_path, self._opts)
                except Exception as e:
                    print(f"[HandTracker] Worker process unavailable ({e}); running in-process.")
            if self.worker is None:
                try:
                    self.landmarker = _make_landmarker("hands", model_path, self._opts)
                except Exception as e:
                    print(f"[HandTracker] init warning: {e}")
        self.hands = []          # list of (21, 2) float32 arrays in full-res pixels
        self.fresh = False
        self.near_head_frames = 0
        self._frame_idx = 0
        self._last_ts = -1
        self._job = None

    @property
    def available(self) -> bool:
        return (self.landmarker is not None or self.worker is not None) and self.enabled

    def should_run(self) -> bool:
        if not self.available:
            return False
        if self.near_head_frames > 0 or self.every_n <= 1:
            return True
        return (self._frame_idx % self.every_n) == 0

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int, head_bbox=None, force: bool = False):
        """Returns (hands, fresh). When not scheduled this frame, returns the last hands with fresh=False."""
        self.submit(frame_bgr, timestamp_ms, head_bbox, force)
        return self.collect()

    def submit(self, frame_bgr: np.ndarray, timestamp_ms: int, head_bbox=None, force: bool = False) -> bool:
        """Starts hand inference if scheduled this frame (async with a worker). Returns whether it ran."""
        self._frame_idx += 1
        self._job = None
        if not self.available:
            return False
        if not (force or self.should_run()):
            if self.near_head_frames > 0:
                self.near_head_frames -= 1
            return False
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms
        h, w = frame_bgr.shape[:2]
        s = min(1.0, self.proxy_width / float(w))
        small = cv2.resize(frame_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1.0 else frame_bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        remote = False
        if self.worker is not None:
            try:
                self.worker.submit(rgb, timestamp_ms)
                remote = True
            except Exception as e:
                print(f"[HandTracker] Worker submit failed ({e}); switching to in-process.")
                self._drop_worker()
        self._job = dict(w=w, h=h, ts=timestamp_ms, rgb=rgb, remote=remote, head_bbox=head_bbox)
        return True

    def _drop_worker(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        if self.landmarker is None and self.model_path:
            try:
                self.landmarker = _make_landmarker("hands", self.model_path, self._opts)
            except Exception as e:
                print(f"[HandTracker] init warning: {e}")

    def collect(self):
        job, self._job = self._job, None
        if job is None:
            self.fresh = False
            return self.hands, False
        w, h = job['w'], job['h']
        raw = None
        if job['remote'] and self.worker is not None:
            try:
                raw = self.worker.collect()
            except Exception as e:
                print(f"[HandTracker] Worker failed ({e}); switching to in-process.")
                self._drop_worker()
        if raw is None:
            if self.landmarker is None:
                self.hands, self.fresh = [], False
                return self.hands, False
            try:
                raw = run_landmarker("hands", self.landmarker, job['rgb'], job['ts'])
            except Exception:
                self.hands, self.fresh = [], False
                return self.hands, False
        hands = []
        for nrm in raw:
            pts = (nrm * np.array([w, h], dtype=np.float32)).astype(np.float32)
            if float(np.linalg.norm(pts[5] - pts[17])) < 6.0:
                continue
            hands.append(pts)
        self.hands = hands
        self.fresh = True
        head_bbox = job['head_bbox']

        near = False
        if head_bbox is not None and hands:
            hx0, hy0, hx1, hy1 = head_bbox
            mx, my = (hx1 - hx0) * 0.25, (hy1 - hy0) * 0.25
            for pts in hands:
                x0, y0 = pts.min(axis=0)
                x1, y1 = pts.max(axis=0)
                ex, ey = (x1 - x0) * 0.6, (y1 - y0) * 0.6
                if x1 + ex > hx0 - mx and x0 - ex < hx1 + mx and y1 + ey > hy0 - my and y0 - ey < hy1 + my:
                    near = True
                    break
        if near:
            self.near_head_frames = 15
        elif self.near_head_frames > 0:
            self.near_head_frames -= 1
        return self.hands, True

    @staticmethod
    def draw_mask(shape_hw, hands, offset=(0.0, 0.0), scale: float = 1.0, thickness: float = 1.0,
                  include_forearm: bool = True, core_only: bool = False, forearm_len: float = 0.9) -> np.ndarray:
        """
        Rasterizes hand capsules into a uint8 mask of shape `shape_hw`.
        Coordinates are transformed as (pt - offset) * scale, so the same landmarks can be
        drawn into a low-res matte prior or into a full-res ROI around the helmet.
        thickness: multiplier on anatomical finger width (1.0 ~= true finger width).
        core_only: thin skeleton that is guaranteed to lie inside the real hand.
        forearm_len: forearm stub length in palm widths (from the wrist, away from the fingers).
        """
        mask = np.zeros(shape_hw, dtype=np.uint8)
        if not hands:
            return mask
        off = np.asarray(offset, dtype=np.float32)
        for pts_full in hands:
            pts = (pts_full - off) * scale
            palm_w = float(np.linalg.norm(pts[5] - pts[17]))
            if palm_w < 1.0:
                continue
            finger = max(1, int(palm_w * 0.27 * thickness))
            ip = np.round(pts).astype(np.int32)
            if core_only:
                palm_c = np.mean(pts[list(PALM_POLY)], axis=0)
                shrink = (pts[list(PALM_POLY)] - palm_c) * 0.55 + palm_c
                cv2.fillConvexPoly(mask, np.round(shrink).astype(np.int32), 255)
                cw = max(1, finger // 2)
                for chain in FINGER_CHAINS[1:]:
                    for a, b in zip(chain[:-2], chain[1:-1]):   # skip fingertips
                        cv2.line(mask, tuple(ip[a]), tuple(ip[b]), 255, cw)
                continue
            cv2.fillConvexPoly(mask, cv2.convexHull(ip[list(PALM_POLY)]), 255)
            for ci, chain in enumerate(FINGER_CHAINS):
                t = int(finger * (1.2 if ci == 0 else 1.0))
                for a, b in zip(chain[:-1], chain[1:]):
                    cv2.line(mask, tuple(ip[a]), tuple(ip[b]), 255, t, cv2.LINE_8)
                cv2.circle(mask, tuple(ip[chain[-1]]), max(1, t // 2), 255, -1)
            cv2.line(mask, tuple(ip[0]), tuple(ip[1]), 255, int(finger * 1.6))
            if include_forearm:
                v = pts[0] - pts[9]
                n = float(np.linalg.norm(v))
                if n > 1.0:
                    p_arm = np.round(pts[0] + v / n * palm_w * forearm_len).astype(np.int32)
                    cv2.line(mask, tuple(ip[0]), tuple(p_arm), 255, max(1, int(palm_w * 0.7 * thickness)))
        return mask

    @staticmethod
    def bbox(hands, pad: float = 0.25):
        if not hands:
            return None
        allp = np.vstack(hands)
        x0, y0 = allp.min(axis=0)
        x1, y1 = allp.max(axis=0)
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        return int(x0 - px), int(y0 - py), int(x1 + px), int(y1 + py)

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
