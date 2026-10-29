import threading
import cv2
import numpy as np


class VirtualCameraOutput:
    """
    Virtual camera broadcaster (pyvirtualcam -> OBS Virtual Camera / Unity Capture) for
    Zoom, Meet, Discord, OBS, etc.

    Never blocks the processing pipeline: frames are handed to a sender thread that
    always transmits the newest frame (older ones are simply replaced), so a slow
    consumer can't add latency or stall tracking.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width, self.height, self.fps = width, height, fps
        self.cam = None
        self.active = False
        self.last_error = None
        self.device = None
        self._cond = threading.Condition()
        self._pending = None
        self._thread = None

    def start(self, width: int = None, height: int = None, fps: int = None) -> bool:
        if width:
            self.width = width
        if height:
            self.height = height
        if fps:
            self.fps = fps
        try:
            import pyvirtualcam
            self.cam = pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps,
                                           fmt=pyvirtualcam.PixelFormat.BGR, print_fps=False)
            self.device = self.cam.device
            self.active = True
            self.last_error = None
            self._thread = threading.Thread(target=self._loop, name="VirtualCam", daemon=True)
            self._thread.start()
            print(f"[VirtualCameraOutput] Virtual camera started on device: {self.cam.device}")
            return True
        except Exception as e:
            self.last_error = str(e)
            print(f"[VirtualCameraOutput] Failed to start virtual camera: {e}")
            self.active = False
            self.cam = None
            return False

    def send_frame(self, frame_bgr: np.ndarray):
        if not self.active:
            return
        with self._cond:
            self._pending = frame_bgr
            self._cond.notify()

    def _loop(self):
        while self.active:
            with self._cond:
                if self._pending is None:
                    self._cond.wait(0.5)
                frame, self._pending = self._pending, None
            if frame is None or self.cam is None:
                continue
            try:
                h, w = frame.shape[:2]
                if w != self.width or h != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                self.cam.send(frame)
            except Exception as e:
                print(f"[VirtualCameraOutput] Error sending frame: {e}")

    def stop(self):
        self.active = False
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.cam:
            try:
                self.cam.close()
            except Exception:
                pass
            self.cam = None
