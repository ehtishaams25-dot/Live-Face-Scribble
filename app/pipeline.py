import time
import threading

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


class PipelineWorker(QThread):
    """
    Per-frame real-time pipeline (one frame in flight, lowest possible latency):

        capture ─┬─> RVM matting queued on the GPU (async, CUDA Graph)
                 ├─> face tracking (CPU, this thread)          ┐ run concurrently
                 └─> hand tracking (CPU, helper thread)        ┘
                 -> matte finish + background (+ person isolation, hand prior)
                 -> privacy guard -> 3D scribble helmet + face -> hands on top
                 -> recorder / virtual camera / preview mailbox

    Privacy guard: a frame with a person in it is never output with a bare face while the
    guard is on. If the face tracker has the head (fresh or holding — a hand that covered
    the face can't drag a held helmet away, see FaceTracker), the helmet covers it;
    if no head is known but a person is present, their head region is mosaicked; the
    "Be Right Back" curtain replaces everything on demand.

    The preview uses a single-slot mailbox: the UI always shows the newest frame and
    queued-up paint events can never accumulate into growing latency.
    """

    frame_ready = Signal()
    status_updated = Signal(str)

    def __init__(self, capture_thread, tracker, hand_tracker, scribble_engine, bg_engine, recorder, virtual_cam):
        super().__init__()
        self.capture = capture_thread
        self.tracker = tracker
        self.hands = hand_tracker
        self.scribble_engine = scribble_engine
        self.bg_engine = bg_engine
        self.recorder = recorder
        self.virtual_cam = virtual_cam
        self.running = True

        # --- settings written by the UI -------------------------------------------------
        self.enable_scribble = True
        self.enable_expressions = True
        self.enable_boil = True
        self.sketch_fps = 12.0
        self.sketch_noise = 0.85
        self.scribble_scale = 1.60
        self.privacy_guard = True
        self.curtain = False
        self.curtain_text = "BE RIGHT BACK"
        self.voice = None          # optional VoiceMeter (voice-assisted lip sync)

        self._mail_lock = threading.Lock()
        self._mail = None
        self._mail_pending = False
        self._presence = None
        self._quick_alpha = None
        self._quick_alpha_age = 99
        self._last_alpha = None
        self._t_start = time.monotonic()
        self.last_frame_id = -1
        self.metrics = {
            'fps': 0.0, 'total_ms': 0.0, 'track_ms': 0.0, 'bg_ms': 0.0, 'scribble_ms': 0.0,
            'latency_ms': 0.0, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'has_face': False,
            'is_coasting': False, 'confidence': 0.0, 'tracking_mode': 'lost', 'privacy': 'clear',
            'is_recording': False, 'is_vcam': False, 'people': [], 'engine': '', 'hands': 0,
        }

    # ------------------------------------------------------------------ mailbox
    def take_latest(self):
        """Called by the UI thread: returns (frame, metrics) or None."""
        with self._mail_lock:
            item, self._mail = self._mail, None
            self._mail_pending = False
            return item

    def _post(self, frame, metrics):
        with self._mail_lock:
            self._mail = (frame, metrics)
            notify = not self._mail_pending
            self._mail_pending = True
        if notify:
            self.frame_ready.emit()

    # ------------------------------------------------------------------ main loop
    def run(self):
        fps_n, fps_t = 0, time.monotonic()
        while self.running:
            fid, frame, t_cap = self.capture.wait_for_frame(self.last_frame_id, timeout=0.2)
            if frame is None:
                continue
            self.last_frame_id = fid
            try:
                out, m = self._process(frame, t_cap)
            except Exception as e:  # never let one bad frame kill the stream
                import traceback
                traceback.print_exc()
                self.status_updated.emit(f"Frame error: {e}")
                continue
            fps_n += 1
            now = time.monotonic()
            if now - fps_t >= 0.5:
                self.metrics['fps'] = fps_n / (now - fps_t)
                fps_n, fps_t = 0, now
            self.metrics.update(m)
            self._post(out, dict(self.metrics))

    def _process(self, frame, t_cap):
        t0 = time.perf_counter()
        ts_ms = int(t_cap * 1000)
        h, w = frame.shape[:2]
        rec = self.recorder.is_recording
        want_matte = rec and self.recorder.record_matte

        # 1. Kick off face + hand landmark inference (worker processes => truly parallel),
        #    then queue GPU matting so all three run at the same time.
        head_bbox_prev = self.metrics.get('_head_bbox')
        hint = None
        if head_bbox_prev is None and self._last_alpha is not None:
            hint = self._head_hints(self._last_alpha, w, h) or None
        t_tr = time.perf_counter()
        self.tracker.submit(frame, ts_ms, hint)
        if self.hands is not None:
            self.hands.submit(frame, ts_ms, head_bbox_prev)
        bg_handle = self.bg_engine.begin(frame, ts_ms, self.bg_engine.needs_matte(extra=want_matte))

        # 2. Gather tracking results (pose solve / hold logic runs here). Hands first: while
        #    the helmet is holding, the tracker must ignore anything that moves with a hand.
        if self.hands is not None:
            hands, hands_fresh = self.hands.collect()
        else:
            hands, hands_fresh = [], False
        track = self.tracker.collect(frame, presence=self._presence, hands=hands or None)
        track_ms = (time.perf_counter() - t_tr) * 1000.0

        # 3. Finish the matte / background.
        head_bbox = track.get('head_bbox') if track.get('has_face') else None
        res = self.bg_engine.finish(bg_handle, frame, hands, hands_fresh, head_bbox=head_bbox, want_alpha_full=want_matte)
        comp = res.composite
        if comp is frame:
            comp = frame.copy()  # never draw into the camera's buffer

        # 4. Presence sensing for the privacy guard (matte if we have one, else a cheap probe).
        alpha_small = res.alpha_small
        needs_probe = self.privacy_guard and (not track.get('face_fresh', False))
        if alpha_small is None and needs_probe:
            self._quick_alpha_age += 1
            if self._quick_alpha is None or self._quick_alpha_age >= 2:
                self._quick_alpha = self.bg_engine.quick_alpha(frame, ts_ms)
                self._quick_alpha_age = 0
            alpha_small = self._quick_alpha
        self._presence = self._presence_at(alpha_small, head_bbox, w, h) if alpha_small is not None else None
        self._last_alpha = alpha_small

        # 5. Helmet / privacy.
        t_sc = time.perf_counter()
        privacy = 'clear'
        if track.get('has_face'):
            privacy = 'helmet' if not track.get('is_coasting') else 'holding'
            if self.enable_scribble:
                self._feed_voice(t_cap)
                comp = self.scribble_engine.apply(
                    comp, track, size_scale=self.scribble_scale,
                    enable_boil=self.enable_boil, t_sec=t_cap - self._t_start, sketch_fps=self.sketch_fps,
                    sketch_noise=self.sketch_noise, enable_expressions=self.enable_expressions,
                    clean_camera_frame=res.composite, hands=hands, hands_fresh=hands_fresh)
        elif self.privacy_guard and alpha_small is not None:
            if self._mosaic_heads(comp, alpha_small):
                privacy = 'mosaic'
        if self.curtain:
            comp = self._curtain(comp)
            privacy = 'curtain'
        scribble_ms = (time.perf_counter() - t_sc) * 1000.0

        # 6. Outputs.
        if rec:
            self.recorder.write_frame(comp, t_cap, res.alpha_full)
        if self.virtual_cam.active:
            self.virtual_cam.send_frame(comp)

        m = {
            'total_ms': (time.perf_counter() - t0) * 1000.0,
            'latency_ms': (time.monotonic() - t_cap) * 1000.0,
            'track_ms': track_ms, 'bg_ms': res.bg_ms, 'scribble_ms': scribble_ms,
            'has_face': track.get('has_face', False), 'is_coasting': track.get('is_coasting', False),
            'confidence': track.get('confidence', 0.0), 'tracking_mode': track.get('tracking_mode', 'lost'),
            'privacy': privacy, 'is_recording': self.recorder.is_recording, 'is_vcam': self.virtual_cam.active,
            'people': res.people, 'engine': res.engine, 'hands': len(hands) if hands_fresh else 0,
            '_head_bbox': head_bbox,
        }
        if track.get('has_face'):
            m.update(yaw=track['yaw'], pitch=track['pitch'], roll=track['roll'])
        return comp, m

    def _feed_voice(self, t_cap):
        """Mic loudness for this frame -> the face renderer (None when voice lip-sync is off)."""
        ff = getattr(self.scribble_engine, 'features', None)
        if ff is None:
            return
        v = self.voice
        if v is not None and v.running:
            # Audio reaches us before the camera frame showing the same moment; the recorder's
            # Audio Delay is exactly that offset, so the drawn mouth matches the recorded voice.
            v.delay_ms = float(getattr(self.recorder, 'audio_offset_ms', 0) or 0)
            ff.voice_level = v.level(t_cap)
        else:
            ff.voice_level = None

    # ------------------------------------------------------------------ privacy helpers
    @staticmethod
    def _presence_at(alpha_small, head_bbox, w, h):
        """Is a person still occupying the head region? (None-safe, low-res)."""
        sh, sw = alpha_small.shape
        if head_bbox is None:
            return bool((alpha_small > 0.5).mean() > 0.01)
        x0, y0, x1, y1 = head_bbox
        mx, my = (x1 - x0) * 0.15, (y1 - y0) * 0.15
        ax0, ax1 = int(max(0, (x0 - mx) / w * sw)), int(min(sw, np.ceil((x1 + mx) / w * sw)))
        ay0, ay1 = int(max(0, (y0 - my) / h * sh)), int(min(sh, np.ceil((y1 + my) / h * sh)))
        if ax1 <= ax0 or ay1 <= ay0:
            return False
        return bool((alpha_small[ay0:ay1, ax0:ax1] > 0.5).mean() > 0.10)

    @staticmethod
    def _head_regions(alpha_small, W, H):
        """Full-res boxes where heads probably are: the top of each person silhouette, largest first."""
        fg = (alpha_small > 0.5).astype(np.uint8)
        if fg.mean() < 0.01:
            return []
        sh, sw = fg.shape
        n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 0.004 * sw * sh:
                continue
            head_h = int(min(bh, max(bw * 1.0, bh * 0.5)))
            x0, y0 = int(x / sw * W), int(y / sh * H)
            x1, y1 = int((x + bw) / sw * W), int(min(H, (y + head_h) / sh * H))
            pad = int((x1 - x0) * 0.12)
            x0, x1, y0 = max(0, x0 - pad), min(W, x1 + pad), max(0, y0 - pad)
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                boxes.append((area, (x0, y0, x1, y1)))
        return [b for _, b in sorted(boxes, reverse=True)]

    @staticmethod
    def _head_hints(alpha_small, W, H):
        """
        Tight guesses of where each person's head is (for the face tracker to zoom into):
        the horizontal extent of the top of each silhouette, squared up. Largest first.
        """
        fg = (alpha_small > 0.5).astype(np.uint8)
        sh, sw = fg.shape
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        out = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 0.004 * sw * sh or bh < 4:
                continue
            band = labels[y:y + max(2, int(bh * 0.12)), x:x + bw] == i
            cols = np.flatnonzero(band.any(axis=0))
            if cols.size == 0:
                continue
            hx0, hx1 = x + cols[0], x + cols[-1] + 1
            hw = max(hx1 - hx0, 2)
            cx = (hx0 + hx1) / 2.0
            x0, x1 = (cx - hw * 0.8) / sw * W, (cx + hw * 0.8) / sw * W
            y0, y1 = (y - hw * 0.2) / sh * H, (y + hw * 1.5) / sh * H
            out.append((area, (int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1)))))
        return [b for _, b in sorted(out, reverse=True)]

    @classmethod
    def _mosaic_heads(cls, frame, alpha_small):
        """
        No tracked head but people present (e.g. entering turned away, before the first
        detection): blur-pixelate the top of every person silhouette — where a head would be.
        """
        H, W = frame.shape[:2]
        done = False
        for (x0, y0, x1, y1) in cls._head_regions(alpha_small, W, H):
            roi = frame[y0:y1, x0:x1]
            # Coarse blocks (a face spans ~2 of them) smeared with a blur: unrecognizable,
            # not just "pixel art".
            bx = max(2, (x1 - x0) // 64)
            cell = max(16, (x1 - x0) // 7)
            small = cv2.resize(roi, (max(2, (x1 - x0) // cell), max(2, (y1 - y0) // cell)), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (3, 3), 0)
            roi[:] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (40, 40, 40), bx, cv2.LINE_AA)
            done = True
        return done

    def _curtain(self, frame):
        H, W = frame.shape[:2]
        small = cv2.resize(frame, (max(8, W // 24), max(8, H // 24)), interpolation=cv2.INTER_AREA)
        out = cv2.resize(cv2.GaussianBlur(small, (5, 5), 0), (W, H), interpolation=cv2.INTER_LINEAR)
        out = cv2.addWeighted(out, 0.45, np.zeros_like(out), 0.55, 0)
        text = self.curtain_text
        scale = max(0.8, W / 900.0)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
        cv2.putText(out, text, ((W - tw) // 2, (H + th) // 2), cv2.FONT_HERSHEY_DUPLEX, scale, (245, 245, 245), 2, cv2.LINE_AA)
        return out

    def stop(self):
        self.running = False
        self.wait(2000)
