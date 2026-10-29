import threading
import time
from collections import deque

import numpy as np

try:
    import sounddevice as sd
except Exception:  # PortAudio missing: voice lip-sync is simply unavailable
    sd = None


def _match_device(name):
    """sounddevice index for a DirectShow mic name (MME truncates names to 31 chars)."""
    if sd is None:
        return None
    try:
        devs = sd.query_devices()
        apis = [a['name'] for a in sd.query_hostapis()]
    except Exception:
        return None
    if not name:
        try:
            idx = sd.default.device[0]
            return idx if idx is not None and idx >= 0 else None
        except Exception:
            return None
    ranked = []
    for i, d in enumerate(devs):
        if d.get('max_input_channels', 0) <= 0:
            continue
        api = apis[d['hostapi']] if d['hostapi'] < len(apis) else ""
        dn = d['name']
        if dn == name or (api == "MME" and len(dn) >= 20 and name.startswith(dn)):
            # MME and DirectSound share the mic with other apps (the recorder's FFmpeg).
            ranked.append(({"MME": 0, "Windows DirectSound": 1}.get(api, 2), i))
    return min(ranked)[1] if ranked else None


class VoiceMeter:
    """
    Microphone loudness envelope for voice-assisted lip sync.

    Reads the mic in 10 ms blocks on PortAudio's thread and keeps a short history of
    levels with their monotonic timestamps. `level(t)` returns 0..1 speech loudness for
    the moment the camera frame captured at `t` shows — audio arrives earlier than video,
    so it looks back by `delay_ms` (the recorder's Audio Delay, which lines the two up).

    The scale adapts to the room: a noise floor that settles on the quietest recent level
    and a peak that follows your loudest syllables, so it works for a whisper-quiet phone
    mic and a loud desk mic alike, without a sensitivity knob.
    """

    def __init__(self):
        self.delay_ms = 0.0
        self.error = None
        self.device_name = None
        self._stream = None
        self._lock = threading.Lock()
        self._hist = deque(maxlen=300)          # (t, dB) — 3 s
        self._noise_db = -55.0
        self._peak_db = -25.0
        self._sr = 16000

    @property
    def running(self):
        return self._stream is not None

    @staticmethod
    def available():
        return sd is not None

    def start(self, device_name=None) -> bool:
        """Opens the mic ('' / None = system default). Returns False (and sets .error) on failure."""
        self.stop()
        self.error = None
        self.device_name = device_name or None
        if sd is None:
            self.error = "sounddevice / PortAudio not installed"
            return False
        idx = _match_device(device_name)
        if idx is None:
            self.error = f"microphone not found: {device_name or 'default'}"
            return False
        try:
            info = sd.query_devices(idx)
            self._sr = int(info.get('default_samplerate') or 44100)
            self._stream = sd.InputStream(device=idx, channels=1, samplerate=self._sr, dtype='float32',
                                          blocksize=max(64, self._sr // 100), latency='low', callback=self._callback)
            self._stream.start()
        except Exception as e:
            self._stream = None
            self.error = f"could not open microphone ({e})"
            return False
        return True

    def stop(self):
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        with self._lock:
            self._hist.clear()

    def _callback(self, indata, frames, time_info, status):
        t = time.monotonic() - frames / float(self._sr)
        rms = float(np.sqrt(np.mean(np.square(indata[:, 0], dtype=np.float64)) + 1e-12))
        db = 20.0 * np.log10(rms + 1e-9)
        dt = frames / float(self._sr)
        # Noise floor: drops fast to the quietest level, creeps up ~2 dB/s.
        if db < self._noise_db:
            self._noise_db += (db - self._noise_db) * min(1.0, dt / 0.3)
        else:
            self._noise_db += min(db - self._noise_db, 2.0 * dt)
        # Peak: jumps to loud syllables, relaxes ~3 dB/s, stays >= 18 dB above the floor.
        if db > self._peak_db:
            self._peak_db = db
        else:
            self._peak_db -= 3.0 * dt
        self._peak_db = max(self._peak_db, self._noise_db + 18.0)
        with self._lock:
            self._hist.append((t, db, self._noise_db, self._peak_db))

    def level(self, t=None) -> float:
        """0..1 voice loudness around monotonic time `t` (default: now) minus the delay."""
        if self._stream is None:
            return 0.0
        target = (time.monotonic() if t is None else t) - self.delay_ms / 1000.0
        with self._lock:
            near = [h for h in self._hist if target - 0.06 <= h[0] <= target + 0.01]
            if not near and self._hist:
                near = [min(self._hist, key=lambda h: abs(h[0] - target))]
        if not near:
            return 0.0
        _, db, noise, peak = max(near, key=lambda h: h[1])
        lo = noise + 12.0           # speech starts well above the room noise
        return float(np.clip((db - lo) / max(6.0, peak - 3.0 - lo), 0.0, 1.0))

    def close(self):
        self.stop()
