"""
MediaPipe landmarkers in worker processes.

MediaPipe's Python bindings hold the GIL for most of an inference, so running the face
and hand landmarkers on two *threads* barely overlaps them. Separate processes give true
multi-core parallelism: the main process hands each worker a downscaled RGB frame through
shared memory (a ~0.4 MB memcpy), keeps working on the GPU matte meanwhile, and collects
the small landmark arrays back over a pipe.

Everything degrades gracefully: if a worker can't start (or dies), callers fall back to
running the landmarker in-process.
"""
import multiprocessing as _mp
from multiprocessing import shared_memory

import numpy as np

_MAX_BYTES = 1920 * 1920 * 3


def _make_landmarker(kind, model_path, options):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    base = python.BaseOptions(model_asset_path=model_path)
    if kind == "face":
        opts = vision.FaceLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.VIDEO, **options)
        return vision.FaceLandmarker.create_from_options(opts)
    if kind == "hands":
        opts = vision.HandLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.VIDEO, **options)
        return vision.HandLandmarker.create_from_options(opts)
    raise ValueError(kind)


def run_landmarker(kind, landmarker, rgb, ts_ms):
    """
    Runs one inference and converts the result to plain numpy/dicts (picklable, cheap).
    face  -> list of (landmarks (478,3) normalized, blendshapes dict), one per face
    hands -> list of (21,2) normalized landmark arrays
    """
    import mediapipe as mp
    res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
    if kind == "face":
        faces = []
        for i, lm in enumerate(res.face_landmarks or []):
            bs = {}
            if res.face_blendshapes and i < len(res.face_blendshapes):
                bs = {c.category_name: float(c.score) for c in res.face_blendshapes[i]}
            faces.append((np.array([(p.x, p.y, p.z) for p in lm], dtype=np.float32), bs))
        return faces
    return [np.array([(p.x, p.y) for p in l], dtype=np.float32) for l in (res.hand_landmarks or [])]


class _FaceSet:
    """Single-face landmarker for the head-following fast path + a 2-face one for
    full-frame (re)acquisition, so we can pick *you* when someone else is in frame."""

    def __init__(self, model_path, options):
        self.model_path, self.options = model_path, dict(options)
        self.single = _make_landmarker("face", model_path, dict(options, num_faces=1))
        self.multi = None

    def run(self, rgb, ts_ms, multi=False):
        if multi:
            if self.multi is None:
                self.multi = _make_landmarker("face", self.model_path, dict(self.options, num_faces=2))
            return run_landmarker("face", self.multi, rgb, ts_ms)
        return run_landmarker("face", self.single, rgb, ts_ms)

    def close(self):
        for lm in (self.single, self.multi):
            if lm is not None:
                lm.close()


def _worker_main(kind, model_path, options, conn, shm_name):
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        lm = _FaceSet(model_path, options) if kind == "face" else _make_landmarker(kind, model_path, options)
        conn.send(("ready", None))
    except Exception as e:  # pragma: no cover - reported to parent
        conn.send(("error", repr(e)))
        return
    try:
        while True:
            msg = conn.recv()
            if msg is None:
                break
            h, w, ts, multi = msg
            rgb = np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf).copy()
            try:
                out = lm.run(rgb, ts, multi) if kind == "face" else run_landmarker(kind, lm, rgb, ts)
                conn.send(("ok", out))
            except Exception as e:
                conn.send(("error", repr(e)))
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        try:
            lm.close()
        except Exception:
            pass
        shm.close()


class LandmarkWorker:
    """One MediaPipe landmarker living in its own process."""

    def __init__(self, kind: str, model_path: str, options: dict, start_timeout: float = 60.0):
        """Spawns the worker without waiting: it loads its model in parallel with everything
        else, and the first submit() waits for it (raising if it failed to start)."""
        ctx = _mp.get_context("spawn")
        self.kind = kind
        self.start_timeout = start_timeout
        self.shm = shared_memory.SharedMemory(create=True, size=_MAX_BYTES)
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker_main, args=(kind, model_path, options, child, self.shm.name),
                                name=f"mp-{kind}", daemon=True)
        self.proc.start()
        self.busy = False
        self.ready = False

    def wait_ready(self, timeout: float = None):
        if self.ready:
            return
        if self.proc is None or not self.conn.poll(self.start_timeout if timeout is None else timeout):
            raise RuntimeError(f"{self.kind} worker did not start in time")
        status, info = self.conn.recv()
        if status != "ready":
            raise RuntimeError(f"{self.kind} worker failed: {info}")
        self.ready = True

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def submit(self, rgb: np.ndarray, ts_ms: int, multi: bool = False):
        self.wait_ready()
        h, w = rgb.shape[:2]
        if h * w * 3 > _MAX_BYTES:
            raise ValueError("frame too large for worker buffer")
        np.ndarray((h, w, 3), dtype=np.uint8, buffer=self.shm.buf)[...] = rgb
        self.conn.send((h, w, int(ts_ms), bool(multi)))
        self.busy = True

    def collect(self, timeout: float = 2.0):
        if not self.busy:
            return None
        if not self.conn.poll(timeout):
            self.busy = False
            raise TimeoutError(f"{self.kind} worker timed out")
        self.busy = False
        status, payload = self.conn.recv()
        if status != "ok":
            raise RuntimeError(payload)
        return payload

    def close(self):
        try:
            if self.proc is not None and self.proc.is_alive():
                if not self.ready and self.conn.poll(0.0):
                    self.conn.recv()
                if self.busy:
                    self.conn.poll(1.0) and self.conn.recv()
                self.conn.send(None)
                self.proc.join(1.5)
                if self.proc.is_alive():
                    self.proc.terminate()
        except Exception:
            pass
        self.proc = None
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception:
            pass
