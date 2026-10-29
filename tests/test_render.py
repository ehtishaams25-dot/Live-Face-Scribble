import time
import unittest

import cv2
import numpy as np

from tests._common import have_video, load_frames, model


@unittest.skipUnless(have_video(), "test video not available")
class TestScribbleRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.tracking.face_tracker import FaceTracker
        tr = FaceTracker(model("face_landmarker.task"))
        t0 = int(time.monotonic() * 1000)
        cls.frames = load_frames(280, 12)
        for i, f in enumerate(cls.frames):
            cls.track = tr.process_frame(f, t0 + i * 33)
        tr.close()
        cls.frame = cls.frames[-1]
        assert cls.track['has_face']

    def _engine(self):
        from app.effects.scribble_mask import ScribbleMaskEngine
        return ScribbleMaskEngine()

    def test_helmet_is_opaque_over_face(self):
        sc = self._engine()
        out = sc.apply(self.frame.copy(), self.track, size_scale=1.6, t_sec=1.0)
        hull = cv2.convexHull(self.track['landmarks_2d']).astype(np.int32)
        m = np.zeros(self.frame.shape[:2], np.uint8)
        cv2.fillConvexPoly(m, hull, 255)
        same = (np.abs(out.astype(int) - self.frame.astype(int)).max(axis=2) < 10)[m > 0].mean()
        self.assertLess(same, 0.01)

    def test_all_shapes_and_styles_render_fast(self):
        sc = self._engine()
        for shape in ("head", "cube", "star", "diamond", "pill"):
            for style in ("ink", "chalk", "neon", "sunny"):
                sc.set_shape(shape)
                sc.set_style(style)
                t = time.perf_counter()
                out = sc.apply(self.frame.copy(), self.track, size_scale=1.6, t_sec=0.5)
                dt = (time.perf_counter() - t) * 1000
                self.assertEqual(out.shape, self.frame.shape)
                self.assertLess(dt, 60.0, f"{shape}/{style} took {dt:.1f} ms")

    def test_boil_changes_strokes_over_time(self):
        sc = self._engine()
        a = sc.apply(self.frame.copy(), self.track, t_sec=0.0)
        b = sc.apply(self.frame.copy(), self.track, t_sec=0.25)
        self.assertGreater(np.abs(a.astype(int) - b.astype(int)).mean(), 0.05)

    def test_mouth_opening_is_mimicked(self):
        """Opening the (real) mouth must open the drawn mouth: more dark cavity pixels."""
        sc = self._engine()
        ff = sc.features
        for _ in range(3):  # learn the neutral baseline
            sc.apply(self.frame.copy(), self.track, t_sec=0.0)
        from app.tracking.face_geometry import LIPS_INNER_LOWER
        open_track = dict(self.track)
        pts = self.track['face_pts'].copy()
        pts[LIPS_INNER_LOWER[1:-1], 1] += 1.2   # drop the lower lip 1.2 cm
        open_track['face_pts'] = pts
        open_track['blendshapes'] = dict(self.track['blendshapes'], jawOpen=0.6)
        ff._filter.reset()
        closed = sc.apply(self.frame.copy(), self.track, t_sec=0.0)
        ff._filter.reset()
        opened = sc.apply(self.frame.copy(), open_track, t_sec=0.0)
        x0, y0, x1, y1 = sc.sphere_3d.last_sil_bbox
        dark = lambda im: (im[y0:y1, x0:x1].max(axis=2) < 70).sum()
        self.assertGreater(dark(opened), dark(closed) + 200)

    def _talk_track(self, amount, smile=None):
        """The tracked face with a *subtle* speech-sized opening: the jaw drops a few mm while
        the inner-lip landmarks barely part (what MediaPipe reports for normal talking)."""
        from app.tracking.face_geometry import LIPS_INNER_LOWER
        tr = dict(self.track)
        pts = self.track['face_pts'].copy()
        pts[152, 1] += 0.45 * amount                   # chin
        pts[17, 1] += 0.30 * amount                    # outer lower lip
        pts[LIPS_INNER_LOWER[3:8], 1] += 0.06 * amount
        tr['face_pts'] = pts
        bs = dict(self.track['blendshapes'])
        bs['jawOpen'] = bs.get('jawOpen', 0.0) + 0.08 * amount
        if smile is not None:
            bs['mouthSmileLeft'] = bs['mouthSmileRight'] = smile
        tr['blendshapes'] = bs
        return tr

    def test_normal_talking_opens_mouth(self):
        """Review: 'I'm talking normally and the mouth doesn't move — I have to over-exaggerate'."""
        sc = self._engine()
        t = 0.0
        for _ in range(15):                            # relaxed, closed mouth (static strokes:
            closed = sc.apply(self.frame.copy(), self.track, t_sec=t, enable_boil=False)   # only the mouth changes)
            t += 1 / 30
        x0, y0, x1, y1 = sc.sphere_3d.last_sil_bbox
        dark = lambda im: int((im[y0:y1, x0:x1].max(axis=2) < 70).sum())
        base = dark(closed)
        opened = shut = 0
        for i in range(48):                            # ~4 syllables per second
            phase = (i // 4) % 2
            out = sc.apply(self.frame.copy(), self._talk_track(1.0) if phase else self.track, t_sec=t, enable_boil=False)
            t += 1 / 30
            if phase and dark(out) > base + 150:
                opened += 1
            if not phase and i % 4 == 3 and dark(out) < base + 80:
                shut += 1
        self.assertGreaterEqual(opened, 20, f"mouth opened on only {opened}/24 talking frames")
        self.assertGreaterEqual(shut, 4, "mouth must close again between syllables")
        self.assertTrue(sc.features.is_talking(t))

    def test_emotes_ignore_talking_but_react_to_held_expressions(self):
        sc = self._engine()
        ff = sc.features
        ff.emotes_enabled = True
        t = 0.0
        for _ in range(10):
            sc.apply(self.frame.copy(), self.track, t_sec=t)
            t += 1 / 30
        # 3 s of talking with a laugh-y smile spike on every other syllable: no sticker.
        for i in range(90):
            phase = (i // 4) % 2
            tr = self._talk_track(1.0, smile=0.85) if phase else self.track
            sc.apply(self.frame.copy(), tr, t_sec=t)
            t += 1 / 30
            self.assertLess(ff._emote['blush'], 0.3, f"blush popped while talking (frame {i})")
        # A held grin with a still mouth is a real expression: the blush appears.
        held = dict(self.track, blendshapes=dict(self.track['blendshapes'], mouthSmileLeft=0.85, mouthSmileRight=0.85))
        for _ in range(75):
            sc.apply(self.frame.copy(), held, t_sec=t)
            t += 1 / 30
        self.assertGreater(ff._emote['blush'], 0.3)

    def test_eye_styles(self):
        sc = self._engine()
        for style in ("knot", "cartoon", "dot"):
            sc.features.eye_style = style
            out = sc.apply(self.frame.copy(), self.track, t_sec=0.0)
            self.assertEqual(out.shape, self.frame.shape)

    def test_emotes_do_not_crash(self):
        sc = self._engine()
        sc.features.emotes_enabled = True
        tr = dict(self.track, blendshapes=dict(self.track['blendshapes'], mouthSmileLeft=0.9, mouthSmileRight=0.9,
                                               browDownLeft=0.8, mouthPucker=0.8, jawOpen=0.6, browInnerUp=0.7))
        for i in range(10):
            sc.apply(self.frame.copy(), tr, t_sec=i / 30)


if __name__ == "__main__":
    unittest.main()
