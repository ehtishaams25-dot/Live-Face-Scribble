import os
import queue
import subprocess
import threading
import time
from datetime import datetime
import cv2
import numpy as np

class VideoRecorder:
    """
    Asynchronous high-performance video recorder.
    Features:
    - Zero-overhead queue-based async frame encoding
    - Hardware NVENC encoding via FFmpeg (C:\ffmpeg\bin\ffmpeg.exe)
    - Automatic fallback to OpenCV VideoWriter (mp4v / avc1) if FFmpeg is unavailable
    - Non-blocking frame ingestion preserving 30+ FPS UI performance
    """
    def __init__(self, output_dir: str = "recordings", fps: int = 30):
        self.output_dir = output_dir
        self.fps = fps
        os.makedirs(self.output_dir, exist_ok=True)

        self.is_recording = False
        self.frame_queue = queue.Queue(maxsize=180)
        self.worker_thread = None
        self.ffmpeg_proc = None
        self.cv2_writer = None
        
        self.current_filename = None
        self.start_time = 0.0
        self.recorded_frames = 0
        self.ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe" if os.path.exists(r"C:\ffmpeg\bin\ffmpeg.exe") else "ffmpeg"

    def start(self, width: int, height: int, fps: int = None) -> str:
        if self.is_recording:
            return self.current_filename

        if fps: self.fps = fps
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_filename = os.path.join(self.output_dir, f"scribble_rec_{timestamp}.mp4")
        
        self.is_recording = True
        self.recorded_frames = 0
        self.start_time = time.time()
        self.frame_queue = queue.Queue(maxsize=180)

        # Attempt to launch FFmpeg with NVENC hardware acceleration
        use_ffmpeg = False
        try:
            # Check if FFmpeg with nvenc is accessible
            bitrate = "14M" if width >= 1920 else "8M"
            maxrate = "18M" if width >= 1920 else "12M"
            bufsize = "28M" if width >= 1920 else "16M"
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "bgr24",
                "-r", str(self.fps),
                "-i", "-",
                "-c:v", "h264_nvenc",
                "-preset", "p5",
                "-cq", "18",
                "-b:v", bitrate,
                "-maxrate", maxrate,
                "-bufsize", bufsize,
                "-pix_fmt", "yuv420p",
                self.current_filename
            ]
            self.ffmpeg_proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            use_ffmpeg = True
            print(f"[VideoRecorder] Started FFmpeg NVENC recording: {self.current_filename}")
        except Exception as e:
            print(f"[VideoRecorder] FFmpeg NVENC failed ({e}), falling back to OpenCV VideoWriter")
            use_ffmpeg = False

        if not use_ffmpeg:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.cv2_writer = cv2.VideoWriter(self.current_filename, fourcc, self.fps, (width, height))
            print(f"[VideoRecorder] Started OpenCV VideoWriter: {self.current_filename}")

        # Start asynchronous writer thread
        self.worker_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.worker_thread.start()
        return self.current_filename

    def _writer_loop(self):
        while self.is_recording or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if self.ffmpeg_proc and self.ffmpeg_proc.stdin:
                    self.ffmpeg_proc.stdin.write(frame.tobytes())
                elif self.cv2_writer:
                    self.cv2_writer.write(frame)
                self.recorded_frames += 1
            except Exception as e:
                print(f"[VideoRecorder] Error writing frame: {e}")
                break

    def write_frame(self, frame_bgr: np.ndarray):
        """Pushes a frame into the recording queue without blocking."""
        if not self.is_recording:
            return
        try:
            self.frame_queue.put_nowait(frame_bgr)
        except queue.Full:
            # Drop frame rather than blocking the real-time pipeline
            pass

    def stop(self) -> dict:
        if not self.is_recording:
            return {}

        self.is_recording = False
        if self.worker_thread:
            self.worker_thread.join(timeout=3.0)
            self.worker_thread = None

        if self.ffmpeg_proc:
            try:
                if self.ffmpeg_proc.stdin:
                    self.ffmpeg_proc.stdin.close()
                if self.ffmpeg_proc.stderr:
                    self.ffmpeg_proc.stderr.close()
                self.ffmpeg_proc.wait(timeout=3.0)
            except Exception:
                self.ffmpeg_proc.kill()
            self.ffmpeg_proc = None

        if self.cv2_writer:
            self.cv2_writer.release()
            self.cv2_writer = None

        duration = max(0.1, time.time() - self.start_time)
        file_size_mb = 0.0
        if os.path.exists(self.current_filename):
            file_size_mb = os.path.getsize(self.current_filename) / (1024 * 1024)

        result = {
            'file_path': self.current_filename,
            'duration_sec': duration,
            'frame_count': self.recorded_frames,
            'avg_fps': self.recorded_frames / duration,
            'file_size_mb': file_size_mb
        }
        print(f"[VideoRecorder] Recording finalized: {result['file_path']} ({result['file_size_mb']:.2f} MB, {result['frame_count']} frames, {result['avg_fps']:.1f} FPS)")
        return result

    def get_status(self) -> dict:
        if not self.is_recording:
            return {'is_recording': False}
        elapsed = time.time() - self.start_time
        return {
            'is_recording': True,
            'elapsed_sec': elapsed,
            'frame_count': self.recorded_frames,
            'file_path': self.current_filename
        }
