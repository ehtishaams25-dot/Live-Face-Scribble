import os
import re
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime

import cv2
import numpy as np

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def find_ffmpeg():
    for p in (r"C:\ffmpeg\bin\ffmpeg.exe", shutil.which("ffmpeg")):
        if p and os.path.exists(p):
            return p
    return None


def list_audio_devices(ffmpeg_path=None):
    """DirectShow audio capture device names (e.g. 'Microphone (DroidCam Audio)')."""
    ffmpeg_path = ffmpeg_path or find_ffmpeg()
    if not ffmpeg_path:
        return []
    try:
        out = subprocess.run([ffmpeg_path, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                             capture_output=True, text=True, timeout=8, creationflags=_CREATE_NO_WINDOW)
        text = (out.stderr or "") + (out.stdout or "")
    except Exception:
        return []
    return re.findall(r'"([^"]+)"\s*\(audio\)', text)


_ENCODER_CACHE = {}


def detect_video_encoder(ffmpeg_path):
    """First hardware H.264 encoder that actually works on this machine, else libx264."""
    if ffmpeg_path in _ENCODER_CACHE:
        return _ENCODER_CACHE[ffmpeg_path]
    choice = "libx264"
    for enc in ("h264_nvenc", "h264_amf", "h264_qsv"):
        try:
            r = subprocess.run([ffmpeg_path, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                                "color=c=black:s=256x256:d=0.2", "-c:v", enc, "-f", "null", "-"],
                               capture_output=True, timeout=10, creationflags=_CREATE_NO_WINDOW)
            if r.returncode == 0:
                choice = enc
                break
        except Exception:
            continue
    _ENCODER_CACHE[ffmpeg_path] = choice
    return choice


def _encoder_args(enc, width, quality="high"):
    br = "16M" if width >= 1920 else "10M"
    mx = "24M" if width >= 1920 else "16M"
    if enc == "h264_nvenc":
        return ["-c:v", enc, "-preset", "p4", "-rc", "vbr", "-cq", "19", "-b:v", br, "-maxrate", mx, "-bufsize", mx]
    if enc == "h264_amf":
        return ["-c:v", enc, "-quality", "balanced", "-rc", "vbr_peak", "-b:v", br, "-maxrate", mx]
    if enc == "h264_qsv":
        return ["-c:v", enc, "-preset", "medium", "-global_quality", "21", "-b:v", br, "-maxrate", mx]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-maxrate", mx, "-bufsize", mx]


class VideoRecorder:
    """
    Real-time-accurate MP4 recorder.

    Frames are placed on a constant-frame-rate timeline by their *capture* time (frames are
    duplicated to fill gaps and dropped if they arrive faster than the output rate), so a
    recording always plays back at true speed and lines up with audio even when processing
    dips below 30 FPS. Encoding runs in FFmpeg on a hardware encoder when available
    (NVENC / AMF / QuickSync, else x264), fed from a background thread so the pipeline
    never waits on disk or encoder.

    Optional:
      * microphone audio muxed in (any DirectShow mic, incl. 'Microphone (DroidCam Audio)'),
        with an A/V offset to compensate camera latency;
      * a matte track (<name>_matte.mp4, white = you) for re-doing the background in
        After Effects / Premiere as a luma matte.
    """

    def __init__(self, output_dir: str = "recordings", fps: int = 30):
        self.output_dir = output_dir
        self.fps = fps
        os.makedirs(self.output_dir, exist_ok=True)
        self.ffmpeg_path = find_ffmpeg()
        self.audio_device = None
        self.audio_offset_ms = 0
        self.record_matte = False
        self.is_recording = False
        self.current_filename = None
        self.matte_filename = None
        self.encoder = None
        self.recorded_frames = 0
        self.dropped_frames = 0
        self.start_time = 0.0
        self._t0 = None
        self._queue = None
        self._thread = None
        self._procs = []
        self._cv_writer = None
        self._log = None
        self.last_error = None

    # ------------------------------------------------------------------
    def _spawn(self, args):
        return subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=self._log or subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW)

    def _audio_input_args(self):
        return ["-f", "dshow", "-audio_buffer_size", "50", "-thread_queue_size", "1024",
                "-i", f"audio={self.audio_device}"]

    def start(self, width: int, height: int, fps: int = None) -> str:
        if self.is_recording:
            return self.current_filename
        if fps:
            self.fps = int(fps)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_filename = os.path.join(self.output_dir, f"scribble_rec_{stamp}.mp4")
        self.matte_filename = os.path.join(self.output_dir, f"scribble_rec_{stamp}_matte.mp4") if self.record_matte else None
        self.recorded_frames = 0
        self.dropped_frames = 0
        self._t0 = None
        self._queue = queue.Queue(maxsize=90)
        self._procs = []
        self._cv_writer = None
        self.last_error = None
        self.width, self.height = int(width), int(height)

        if self.ffmpeg_path:
            try:
                self._log = open(os.path.join(self.output_dir, "last_recording_ffmpeg.log"), "wb")
            except Exception:
                self._log = None
            self.encoder = detect_video_encoder(self.ffmpeg_path)
            main = [self.ffmpeg_path, "-y", "-hide_banner", "-loglevel", "warning",
                    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(self.fps),
                    "-thread_queue_size", "512", "-i", "-"]
            if self.audio_device:
                if self.audio_offset_ms:
                    main += ["-itsoffset", f"{self.audio_offset_ms / 1000.0:.3f}"]
                main += self._audio_input_args() + ["-map", "0:v", "-map", "1:a",
                                                    "-c:a", "aac", "-b:a", "192k", "-shortest"]
            main += _encoder_args(self.encoder, width) + ["-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                                          self.current_filename]
            try:
                self._procs.append(self._spawn(main))
                if self.matte_filename:
                    matte = [self.ffmpeg_path, "-y", "-hide_banner", "-loglevel", "warning",
                             "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}", "-r", str(self.fps),
                             "-i", "-"] + _encoder_args(self.encoder, width) + ["-pix_fmt", "yuv420p", self.matte_filename]
                    self._procs.append(self._spawn(matte))
                print(f"[VideoRecorder] Recording {self.current_filename} ({self.encoder}, audio={self.audio_device or 'off'})")
            except Exception as e:
                self.last_error = str(e)
                print(f"[VideoRecorder] FFmpeg failed to start ({e}); using OpenCV writer.")
                self._procs = []
        if not self._procs:
            self._cv_writer = cv2.VideoWriter(self.current_filename, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
            self.matte_filename = None

        self.is_recording = True
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._writer_loop, name="Recorder", daemon=True)
        self._thread.start()
        return self.current_filename

    def write_frame(self, frame_bgr: np.ndarray, t_capture: float = None, matte: np.ndarray = None):
        """Queue a frame (never blocks). t_capture: monotonic capture time of the source frame."""
        if not self.is_recording:
            return
        if t_capture is None:
            t_capture = time.monotonic()
        try:
            self._queue.put_nowait((frame_bgr, t_capture, matte))
        except queue.Full:
            self.dropped_frames += 1

    def _write(self, frame, matte, repeats):
        vid = self._procs[0] if self._procs else None
        mat = self._procs[1] if len(self._procs) > 1 else None
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        buf = frame.tobytes()
        if mat is not None:
            if matte is None:
                matte = np.zeros((self.height, self.width), np.uint8)
            elif matte.shape[:2] != (self.height, self.width):
                matte = cv2.resize(matte, (self.width, self.height))
            mbuf = np.ascontiguousarray(matte).tobytes()
        for _ in range(repeats):
            if vid is not None:
                vid.stdin.write(buf)
            elif self._cv_writer is not None:
                self._cv_writer.write(frame)
            if mat is not None:
                mat.stdin.write(mbuf)
            self.recorded_frames += 1

    def _writer_loop(self):
        written = 0
        while self.is_recording or not self._queue.empty():
            try:
                frame, t_cap, matte = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._t0 is None:
                self._t0 = t_cap
            # Slot this frame belongs to on the constant-rate output timeline.
            slot = int(round((t_cap - self._t0) * self.fps)) + 1
            repeats = slot - written
            if repeats <= 0:
                continue                       # arrived faster than the output rate
            repeats = min(repeats, self.fps * 2)  # never pad more than 2 s after a stall
            try:
                self._write(frame, matte, repeats)
                written += repeats
            except Exception as e:
                self.last_error = str(e)
                print(f"[VideoRecorder] Write error: {e}")
                break

    def stop(self) -> dict:
        if not self.is_recording:
            return {}
        self.is_recording = False
        if self._thread:
            self._thread.join(timeout=10.0)
            self._thread = None
        for p in self._procs:
            try:
                p.stdin.close()
            except Exception:
                pass
        for p in self._procs:
            try:
                p.wait(timeout=10.0)
            except Exception:
                p.kill()
        self._procs = []
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None
        if self._log:
            try:
                self._log.close()
            except Exception:
                pass
            self._log = None
        duration = max(0.1, time.time() - self.start_time)
        size = os.path.getsize(self.current_filename) / (1024 * 1024) if os.path.exists(self.current_filename) else 0.0
        result = {
            'file_path': self.current_filename,
            'matte_path': self.matte_filename if self.matte_filename and os.path.exists(self.matte_filename) else None,
            'duration_sec': duration,
            'frame_count': self.recorded_frames,
            'dropped_frames': self.dropped_frames,
            'avg_fps': self.recorded_frames / duration,
            'file_size_mb': size,
            'encoder': self.encoder or 'opencv',
            'audio': self.audio_device,
        }
        print(f"[VideoRecorder] Saved {result['file_path']} ({size:.1f} MB, {result['frame_count']} frames)")
        return result

    def get_status(self) -> dict:
        if not self.is_recording:
            return {'is_recording': False}
        return {'is_recording': True, 'elapsed_sec': time.time() - self.start_time,
                'frame_count': self.recorded_frames, 'file_path': self.current_filename}
