import time
import unittest

import cv2
import numpy as np

from tests._common import have_video, load_frames, model


def _synthetic_hand(cx, cy, s):
    """21 hand landmarks in MediaPipe order: palm centred at (cx, cy), fingers up, s ~ palm width."""
    pts = np.zeros((21, 2), np.float32)
    pts[0] = (cx, cy + 0.9 * s)
    bases = [(-0.55, 0.35), (-0.3, -0.35), (-0.05, -0.45), (0.2, -0.4), (0.42, -0.3)]
    lens = [0.9, 1.25, 1.35, 1.25, 1.0]
    for f, ((bx, by), ln) in enumerate(zip(bases, lens)):
        for j in range(4):
            if f == 0:
                pts[1 + j] = (cx + (bx - 0.25 * j) * s, cy + (by - 0.2 * j) * s)
            else:
                pts[1 + f * 4 + j] = (cx + bx * s * (1 + 0.1 * j), cy + by * s - j * ln * 0.3 * s)
    return pts


class TestGeometry(unittest.TestCase):
    def test_umeyama_recovers_similarity(self):
        from app.tracking.face_geometry import umeyama
        from scipy.spatial.transform import Rotation
        rng = np.random.RandomState(0)
        src = rng.normal(size=(40, 3))
        R = Rotation.from_euler('xyz', [20, -35, 10], degrees=True).as_matrix()
        dst = 2.5 * src @ R.T + np.array([1.0, -2.0, 3.0])
        s, R2, t = umeyama(src, dst)
        self.assertAlmostEqual(s, 2.5, places=6)
        self.assertTrue(np.allclose(R2, R, atol=1e-6))
        self.assertTrue(np.allclose(t, [1.0, -2.0, 3.0], atol=1e-6))

    def test_canonical_face_aligned_to_skull_frame(self):
        from app.tracking.face_geometry import CANONICAL_SKULL, SKULL_ANCHORS
        self.assertIsNotNone(CANONICAL_SKULL, "canonical_face_model.npy missing")
        idx = list(SKULL_ANCHORS)
        err = np.linalg.norm(CANONICAL_SKULL[idx] - np.array([SKULL_ANCHORS[i] for i in idx]), axis=1)
        self.assertLess(err.mean(), 1.5)
        # nose forward (-z), forehead up (-y), subject's left eye at +x
        self.assertLess(CANONICAL_SKULL[1, 2], CANONICAL_SKULL[234, 2])
        self.assertLess(CANONICAL_SKULL[10, 1], CANONICAL_SKULL[152, 1])
        self.assertGreater(CANONICAL_SKULL[263, 0], 0)

    def test_pose_filter_follows_fast_motion(self):
        from app.tracking.temporal_smoother import PoseFilter
        f = PoseFilter()
        t = 0.0
        for _ in range(10):
            f.update([0, 0, 0], [0, 0, 60], t)
            t += 1 / 30
        # a fast 15 cm move must be followed within ~3 frames (low lag = no exposed face)
        for _ in range(3):
            r, tr = f.update([0, 0, 0], [15, 0, 60], t)
            t += 1 / 30
        self.assertGreater(tr[0], 12.0)


@unittest.skipUnless(have_video(), "test video not available")
class TestFaceTracker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.tracking.face_tracker import FaceTracker
        cls.FaceTracker = FaceTracker
        cls.tr = FaceTracker(model("face_landmarker.task"), pose_model_path=model("pose_landmarker_lite.task"))

    @classmethod
    def tearDownClass(cls):
        cls.tr.close()

    def setUp(self):
        self.tr.reset()
        self.t0 = int(time.monotonic() * 1000)

    def _feed(self, frames, start=0, presence=None):
        out = []
        for i, f in enumerate(frames):
            out.append(self.tr.process_frame(f, self.t0 + (start + i) * 33, presence=presence))
        return out

    def test_frontal_pose(self):
        res = self._feed(load_frames(200, 10))[-1]
        self.assertTrue(res['has_face'])
        self.assertEqual(res['tracking_mode'], 'face_dense')
        self.assertLess(abs(res['yaw']), 25)
        self.assertLess(abs(res['roll']), 25)
        self.assertTrue(30 < res['translation_skull'][2] < 250)
        self.assertEqual(res['face_pts'].shape, (478, 3))

    def test_hold_when_face_disappears(self):
        frames = load_frames(200, 10)
        self._feed(frames)
        # Occlude the face completely (e.g. a hand over it) -> detector drops out.
        blocked = frames[-1].copy()
        blocked[:, :] = cv2.GaussianBlur(blocked, (0, 0), 25)
        res = self._feed([blocked] * 5, start=10, presence=True)
        for r in res:
            self.assertTrue(r['has_face'], "helmet must be held while the person is still there")
            self.assertTrue(r['is_coasting'])

    def test_hold_does_not_follow_a_hand(self):
        """Review bug: a hand covers the face (helmet holds), then moves away while the face
        stays undetectable (head tilted back) — the helmet went with the hand and uncovered
        the face. The held helmet must stay with the head."""
        from app.tracking.hand_tracker import HandTracker
        frames = load_frames(200, 40)
        ref = self.FaceTracker(model("face_landmarker.task"))
        try:
            truth = [ref.process_frame(f, self.t0 + 10_000_000 + i * 33) for i, f in enumerate(frames)]
        finally:
            ref.close()
        rng = np.random.RandomState(3)
        tex = cv2.GaussianBlur(rng.randint(0, 255, frames[0].shape[:2]).astype(np.uint8), (0, 0), 1.2)
        tex = cv2.merge([tex // 3 + 60, tex // 3 + 100, tex // 3 + 150])   # skin-ish, very textured
        lm = fsize = None
        worst = 0.0
        for i, f in enumerate(frames):
            f, hands = f.copy(), None
            if i >= 10:
                lo, hi = lm.min(axis=0).astype(int) - 10, lm.max(axis=0).astype(int) + 10
                f[lo[1]:hi[1], lo[0]:hi[0]] = cv2.GaussianBlur(f[lo[1]:hi[1], lo[0]:hi[0]], (0, 0), 25)
                move = max(0, i - 16) * 14                       # hand covers, then leaves up-left
                hand = _synthetic_hand(c0[0] - move * 0.6, c0[1] - move, fsize * 0.55)
                m = HandTracker.draw_mask(f.shape[:2], [hand])
                f[m > 0] = tex[m > 0]
                hands = [hand]
            r = self.tr.process_frame(f, self.t0 + i * 33, presence=True, hands=hands)
            if i < 10:
                self.assertTrue(r['face_fresh'])
                lm, fsize = r['landmarks_2d'], r['face_size']
                c0 = np.array([r['center_x'], r['center_y']], float)
                continue
            self.assertTrue(r['has_face'], "helmet must be held while the person is there")
            d = np.hypot(r['center_x'] - truth[i]['center_x'], r['center_y'] - truth[i]['center_y']) / fsize
            worst = max(worst, d)
        self.assertLess(worst, 0.35, f"held helmet drifted {worst:.2f} face widths off the head")

    def test_release_when_person_leaves(self):
        frames = load_frames(200, 10)
        self._feed(frames)
        empty = np.full_like(frames[0], 90)
        res = self._feed([empty] * 12, start=10, presence=False)
        self.assertFalse(res[-1]['has_face'], "hold must end once nobody occupies the head region")

    def test_worker_process_matches_in_process(self):
        tw = self.FaceTracker(model("face_landmarker.task"), use_worker=True)
        try:
            frames = load_frames(200, 6)
            a = [tw.process_frame(f, self.t0 + i * 33) for i, f in enumerate(frames)]
            b = self._feed(frames)
            self.assertIsNotNone(tw.worker, "worker process should be running")
            for ra, rb in zip(a, b):
                self.assertEqual(ra['has_face'], rb['has_face'])
                self.assertLess(abs(ra['yaw'] - rb['yaw']), 2.0)
        finally:
            tw.close()


@unittest.skipUnless(have_video(), "test video not available")
class TestHandTracker(unittest.TestCase):
    def test_masks(self):
        from app.tracking.hand_tracker import HandTracker
        hands = [np.array([[100 + 3 * i, 200 - 6 * i] for i in range(21)], np.float32)]
        hands[0][5] = (80, 150)
        hands[0][17] = (130, 150)
        full = HandTracker.draw_mask((300, 300), hands)
        core = HandTracker.draw_mask((300, 300), hands, core_only=True)
        self.assertGreater(full.sum(), core.sum())
        self.assertTrue(((core > 0) <= (cv2.dilate(full, np.ones((3, 3), np.uint8)) > 0)).all(),
                        "core must lie inside the full hand mask")

    def test_scheduling(self):
        from app.tracking.hand_tracker import HandTracker
        ht = HandTracker(model("hand_landmarker.task"), every_n=3)
        try:
            f = load_frames(60, 1)[0]
            ran = [ht.submit(f, 1000 + i * 33) for i in range(9)]
            for _ in ran:
                pass
            self.assertEqual(sum(ran), 3)
        finally:
            ht.close()


if __name__ == "__main__":
    unittest.main()
