import os
import threading
import time
import cv2
import numpy as np

def list_camera_devices():
    """Returns a list of available DirectShow video input device names."""
    devices = []
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        devices = graph.get_input_devices()
    except Exception as e:
        # Fallback to scanning cv2 device indices
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                devices.append(f"Camera Device {i}")
                cap.release()
    return devices


class CameraCaptureThread(threading.Thread):
    """
    Dedicated high-performance capture thread.
    Features:
    - Zero-latency frame delivery via cv2.CAP_PROP_BUFFERSIZE = 1
    - DirectShow backend for Windows
    - Supports live webcams/DroidCam and video file test playback with looping
    - Thread-safe frame reading
    """
    def __init__(self, source=0, target_fps=30, width=1920, height=1080):
        super().__init__(daemon=True)
        self.source = source
        self.target_fps = target_fps
        self.width = width
        self.height = height
        
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        
        self.current_frame = None
        self.frame_id = 0
        self.last_frame_time = 0.0
        self.actual_fps = 0.0
        self.is_video_file = isinstance(source, str) and os.path.isfile(source)

    def open_camera(self):
        """Opens camera or video file with 1080p negotiation."""
        if self.cap is not None:
            self.cap.release()
            
        if self.is_video_file:
            self.cap = cv2.VideoCapture(self.source)
        else:
            # Live webcam / DroidCam / Iriun: Try DirectShow with MSMF fallback
            opened = False
            try:
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    opened = True
            except Exception:
                opened = False

            if not opened or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.source)

            # Request 1080p Full HD (or requested dimensions)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            return False
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CameraCaptureThread] Source {self.source} active at {self.width}x{self.height} @ {self.target_fps} FPS")
        return True

    def run(self):
        self.running = True
        if not self.open_camera():
            print(f"[CameraCaptureThread] Failed to open source: {self.source}")
            self.running = False
            return

        fps_counter = 0
        fps_timer = time.time()

        while self.running:
            t0 = time.time()
            ret, frame = self.cap.read()

            if not ret:
                if self.is_video_file:
                    # Loop video playback seamlessly
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.01)
                    continue

            with self.lock:
                self.current_frame = frame
                self.frame_id += 1
                self.last_frame_time = time.time()

            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                self.actual_fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()

            # Pace loop for video playback or cap frame rate
            elapsed = time.time() - t0
            sleep_time = max(0.001, (1.0 / self.target_fps) - elapsed)
            if self.is_video_file:
                time.sleep(sleep_time)

        if self.cap:
            self.cap.release()

    def get_latest_frame(self):
        """Returns (frame_id, frame_bgr) thread-safely."""
        with self.lock:
            if self.current_frame is None:
                return self.frame_id, None
            return self.frame_id, self.current_frame.copy()

    def stop(self):
        self.running = False
        self.join(timeout=1.0)
