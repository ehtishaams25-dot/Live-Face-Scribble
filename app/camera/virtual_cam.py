import threading
import numpy as np

class VirtualCameraOutput:
    """
    DirectShow Virtual Camera broadcaster using pyvirtualcam.
    Allows Zoom, Google Meet, Discord, and OBS to receive the processed Scribble Camera feed.
    """
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cam = None
        self.active = False
        self.lock = threading.Lock()
        self.last_error = None

    def start(self, width: int = None, height: int = None, fps: int = None) -> bool:
        if width: self.width = width
        if height: self.height = height
        if fps: self.fps = fps

        try:
            import pyvirtualcam
            # On Windows, pyvirtualcam connects to OBS Virtual Camera or DirectShow
            self.cam = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
                print_fps=False
            )
            self.active = True
            self.last_error = None
            print(f"[VirtualCameraOutput] Virtual camera started on device: {self.cam.device}")
            return True
        except Exception as e:
            self.last_error = str(e)
            print(f"[VirtualCameraOutput] Failed to start virtual camera: {e}")
            self.active = False
            self.cam = None
            return False

    def send_frame(self, frame_bgr: np.ndarray):
        """Sends a frame to the virtual camera if active."""
        if not self.active or self.cam is None:
            return

        with self.lock:
            try:
                h, w = frame_bgr.shape[:2]
                if w != self.width or h != self.height:
                    import cv2
                    frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
                self.cam.send(frame_bgr)
                self.cam.sleep_until_next_frame()
            except Exception as e:
                print(f"[VirtualCameraOutput] Error sending frame: {e}")

    def stop(self):
        with self.lock:
            if self.cam:
                try:
                    self.cam.close()
                except Exception:
                    pass
                self.cam = None
            self.active = False
