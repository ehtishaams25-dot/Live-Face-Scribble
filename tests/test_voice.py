import time
import unittest

import numpy as np

from tests._common import have_video, load_frames, model


class TestVoiceMeter(unittest.TestCase):
    def _feed(self, vm, db, seconds, t_start):
        """Pushes 10 ms blocks of noise at `db` dBFS straight into the meter's audio callback."""
        rng = np.random.RandomState(1)
        n = vm._sr // 100
        amp = 10 ** (db / 20.0)
        for k in range(int(seconds * 100)):
            block = (rng.normal(0.0, amp, (n, 1))).astype(np.float32)
            vm._callback(block, n, None, None)
            # stamp the block as if it arrived at t_start + k*10ms
            t, d, lo, hi = vm._hist[-1]
            vm._hist[-1] = (t_start + k * 0.01, d, lo, hi)

    def test_quiet_room_reads_zero_and_speech_reads_high(self):
        from app.tracking.voice_meter import VoiceMeter
        vm = VoiceMeter()
        vm._sr = 16000
        vm._stream = object()          # pretend the mic is open (level() needs a running stream)
        t0 = 1000.0
        self._feed(vm, -62.0, 2.0, t0)            # room noise
        quiet = vm.level(t0 + 1.9)
        self._feed(vm, -22.0, 0.5, t0 + 2.0)      # talking
        loud = vm.level(t0 + 2.4)
        self.assertLess(quiet, 0.1)
        self.assertGreater(loud, 0.5)
        vm.delay_ms = 300                          # the lookup honours the audio delay
        self.assertLess(vm.level(t0 + 2.25), 0.1)  # 300 ms earlier it was still quiet


@unittest.skipUnless(have_video(), "test video not available")
class TestVoiceOpensMouth(unittest.TestCase):
    def test_voice_level_opens_drawn_mouth(self):
        from app.tracking.face_tracker import FaceTracker
        from app.effects.scribble_mask import ScribbleMaskEngine
        tr = FaceTracker(model("face_landmarker.task"))
        t0 = int(time.monotonic() * 1000)
        for i, f in enumerate(load_frames(280, 10)):
            track = tr.process_frame(f, t0 + i * 33)
        tr.close()
        sc = ScribbleMaskEngine()
        t = 0.0
        for _ in range(10):
            closed = sc.apply(f.copy(), track, t_sec=t, enable_boil=False)
            t += 1 / 30
        x0, y0, x1, y1 = sc.sphere_3d.last_sil_bbox
        dark = lambda im: int((im[y0:y1, x0:x1].max(axis=2) < 70).sum())
        sc.features.voice_level = 0.8
        opened = sc.apply(f.copy(), track, t_sec=t, enable_boil=False)
        self.assertGreater(dark(opened), dark(closed) + 150)


if __name__ == "__main__":
    unittest.main()
