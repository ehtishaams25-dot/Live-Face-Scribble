import time
import unittest

import cv2
import numpy as np

from tests._common import have_video, load_frames, model


def _blob(h, w, cx, cy, r):
    a = np.zeros((h, w), np.float32)
    cv2.circle(a, (int(cx), int(cy)), int(r), 1.0, -1)
    cv2.rectangle(a, (int(cx - r), int(cy)), (int(cx + r), h), 1.0, -1)   # torso
    return a


class TestPersonIsolation(unittest.TestCase):
    def test_only_me_hides_bystander_and_follows_them(self):
        from app.effects.person_isolation import PersonIsolator
        iso = PersonIsolator()
        iso.enabled = True
        iso.set_mode(iso.MODE_ONLY_ME)
        h, w = 108, 192
        me_head = (60 / w, 20 / h, 100 / w, 50 / h)
        keep = None
        # Bystander walks in from the right, then overlaps (touches) me.
        for step, bx in enumerate(np.linspace(170, 112, 12)):
            alpha = np.maximum(_blob(h, w, 80, 35, 18), _blob(h, w, bx, 40, 14))
            keep = iso.update(alpha, me_head)
            if keep is not None:
                self.assertGreater(keep[40, 80], 0.9, "I must always be kept")
                self.assertLess(keep[40, int(bx) + 6], 0.2, f"bystander visible at step {step}")
        self.assertIsNotNone(keep)
        self.assertEqual(sum(1 for p in iso.people() if not p[2]), 1)

    def test_click_toggles_person(self):
        from app.effects.person_isolation import PersonIsolator
        iso = PersonIsolator()
        iso.enabled = True
        iso.set_mode(iso.MODE_EVERYONE)
        h, w = 108, 192
        alpha = np.maximum(_blob(h, w, 50, 35, 16), _blob(h, w, 150, 35, 16))
        self.assertIsNone(iso.update(alpha, (34 / w, 19 / h, 66 / w, 51 / h)))   # everyone kept
        tid, kept = iso.click(150 / w, 40 / h)
        self.assertFalse(kept)
        keep = iso.update(alpha, (34 / w, 19 / h, 66 / w, 51 / h))
        self.assertLess(keep[40, 150], 0.2)
        self.assertGreater(keep[40, 50], 0.9)
        iso.click(150 / w, 40 / h)
        self.assertIsNone(iso.update(alpha, (34 / w, 19 / h, 66 / w, 51 / h)))

    def test_keep_zone_forces_foreground(self):
        from app.effects.person_isolation import PersonIsolator
        iso = PersonIsolator()
        iso.enabled = True
        iso.add_keep_zone(0.7, 0.1, 0.9, 0.5)
        keep = iso.update(np.zeros((108, 192), np.float32), None)
        self.assertGreater(keep[30, 160], 1.5)


@unittest.skipUnless(have_video(), "test video not available")
class TestBackgroundEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.effects.background_engine import BackgroundEngine
        cls.BE = BackgroundEngine
        cls.bg = BackgroundEngine(model("selfie_segmenter.tflite"), models_dir=model(""))
        cls.frames = load_frames(200, 20)

    @classmethod
    def tearDownClass(cls):
        cls.bg.close()

    def _run(self, mode, engine):
        self.bg.set_engine_type(engine)
        self.bg.mode = mode
        res = None
        t0 = int(time.monotonic() * 1000)
        for i, f in enumerate(self.frames):
            res = self.bg.finish(self.bg.begin(f, t0 + i * 33), f)
        return res

    def test_modes_and_engines(self):
        engines = [self.BE.ENGINE_MEDIAPIPE]
        if self.bg.has_cuda:
            engines += [self.BE.ENGINE_RVM_MOBILENET, self.BE.ENGINE_RVM_RESNET50]
        for eng in engines:
            for mode in (self.BE.MODE_BLUR, self.BE.MODE_STUDIO, self.BE.MODE_SOLID):
                res = self._run(mode, eng)
                self.assertEqual(res.composite.shape, self.frames[0].shape)
                a = res.alpha_small
                self.assertIsNotNone(a)
                # person occupies the middle, the top corners are background
                self.assertGreater(a[a.shape[0] // 2:, a.shape[1] // 2].mean(), 0.6, f"{eng}/{mode}")
                self.assertLess(a[:5, :5].mean(), 0.2, f"{eng}/{mode}")

    def test_original_mode_skips_matting(self):
        self.bg.mode = self.BE.MODE_ORIGINAL
        self.assertIsNone(self.bg.begin(self.frames[0], 1))

    @unittest.skipUnless(__import__("torch").cuda.is_available(), "needs CUDA")
    def test_cuda_graph_matches_eager(self):
        from app.effects.rvm_runner import RVMRunner
        g, e = RVMRunner(model("")), RVMRunner(model(""))
        e.use_graphs = False
        g.load("mobilenetv3")
        e.load("mobilenetv3")
        for f in self.frames[:8]:
            _, _, pg = g.run(f, 0.4)
            _, _, pe = e.run(f, 0.4)
        diff = (pg.float() - pe.float()).abs().mean().item()
        self.assertLess(diff, 0.01)

    @unittest.skipUnless(__import__("torch").cuda.is_available(), "needs CUDA")
    def test_rvm_is_fast(self):
        self._run(self.BE.MODE_STUDIO, self.BE.ENGINE_RVM_MOBILENET)
        t = time.perf_counter()
        for f in self.frames:
            self.bg.finish(self.bg.begin(f, int(time.monotonic() * 1000)), f)
        ms = (time.perf_counter() - t) / len(self.frames) * 1000
        print(f"\n[background] RVM MobileNet end-to-end {ms:.1f} ms/frame")
        self.assertLess(ms, 20.0)


if __name__ == "__main__":
    unittest.main()
