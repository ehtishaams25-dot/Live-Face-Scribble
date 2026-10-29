import os
import threading
import time
import cv2


def list_camera_devices():
    """Returns a list of available DirectShow video input device names."""
    devices = []
    try:
        from pygrabber.dshow_graph import FilterGraph
        devices = FilterGraph().get_input_devices()
    except Exception:
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                devices.append(f"Camera Device {i}")
                cap.release()
    return devices


class CameraCaptureThread(threading.Thread):
    """
    Dedicated capture thread.

    - Reads continuously so the driver never builds up a backlog of stale frames
      (DirectShow ignores CAP_PROP_BUFFERSIZE for most virtual cameras like DroidCam).
    - Consumers block on `wait_for_frame()` and wake the instant a new frame lands
      (no polling sleep adding up to 5 ms of latency per frame).
    - Each frame carries its monotonic capture time, used as the MediaPipe timestamp
      and for end-to-end latency measurement.
    - Auto-reconnects if the device stops delivering (DroidCam over Wi-Fi dropping).
    - Optional horizontal mirror.
    """

    def __init__(self, source=0, target_fps=30, width=1280, height=720, mirror=False):
        super().__init__(daemon=True)
        self.source = source
        self.target_fps = target_fps
        self.width = width
        self.height = height
        self.mirror = bool(mirror)
        self.cap = None
        self.running = False
        self.cond = threading.Condition()
        self.current_frame = None
        self.frame_id = 0
        self.frame_time = 0.0
        self.last_frame_time = 0.0
        self.actual_fps = 0.0
        self.status = "starting"
        self.is_video_file = isinstance(source, str) and os.path.isfile(source)
        self.lock = self.cond  # backwards-compatible name

    def open_camera(self):
        if self.cap is not None:
            self.cap.release()
        if self.is_video_file:
            self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.source)
            # MJPG lets USB webcams deliver 720p/1080p at full frame rate; harmless if unsupported.
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            return False
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.is_video_file:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps and 1 < fps < 240:
                self.target_fps = fps
        print(f"[CameraCaptureThread] Source {self.source} active at {self.width}x{self.height} @ {self.target_fps:.0f} FPS")
        return True

    def run(self):
        self.running = True
        if not self.open_camera():
            print(f"[CameraCaptureThread] Failed to open source: {self.source}")
            self.status = "failed"
        else:
            self.status = "live"
        fps_counter, fps_timer = 0, time.monotonic()
        failures = 0
        next_due = time.monotonic()
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.status = "reconnecting"
                time.sleep(1.0)
                if self.running and self.open_camera():
                    self.status = "live"
                    failures = 0
                continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                if self.is_video_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                failures += 1
                if failures > 60:          # ~2 s of nothing: reopen the device
                    print("[CameraCaptureThread] Source stalled; reconnecting...")
                    self.cap.release()
                    self.cap = None
                    failures = 0
                else:
                    time.sleep(0.03)
                continue
            failures = 0
            if self.mirror:
                frame = cv2.flip(frame, 1)
            now = time.monotonic()
            with self.cond:
                self.current_frame = frame
                self.frame_id += 1
                self.frame_time = now
                self.last_frame_time = time.time()
                self.cond.notify_all()
            fps_counter += 1
            if now - fps_timer >= 1.0:
                self.actual_fps = fps_counter / (now - fps_timer)
                fps_counter, fps_timer = 0, now
            if self.is_video_file:
                next_due += 1.0 / max(1.0, self.target_fps)
                delay = next_due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_due = time.monotonic()
        if self.cap:
            self.cap.release()

    def wait_for_frame(self, last_id: int, timeout: float = 0.2):
        """Blocks until a frame newer than `last_id` arrives. Returns (id, frame, t_capture) or (id, None, 0)."""
        with self.cond:
            if self.frame_id == last_id or self.current_frame is None:
                self.cond.wait(timeout)
            if self.current_frame is None or self.frame_id == last_id:
                return self.frame_id, None, 0.0
            return self.frame_id, self.current_frame, self.frame_time

    def get_latest_frame(self):
        """Returns (frame_id, frame_bgr). The array is never written to by the capture thread."""
        with self.cond:
            return self.frame_id, self.current_frame

    def stop(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()
        self.join(timeout=1.5)
