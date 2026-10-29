"""
The one test that matters most: does any of the real face ever reach the output?

Runs the full per-frame pipeline over a real clip. For every frame, an independent face
landmarker finds the real face on the ORIGINAL frame; we then measure how many pixels
inside that face region are unchanged in the OUTPUT frame. An exposed face shows up as
a large unchanged fraction.
"""
import time
import unittest

import cv2
import numpy as np

from tests._common import have_video, load_frames, model, ReferenceFaceDetector


@unittest.skipUnless(have_video(), "test video not available")
class TestPrivacy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from app.tracking.face_tracker import FaceTracker
        from app.tracking.hand_tracker import HandTracker
        from app.effects.scribble_mask import ScribbleMaskEngine
        from app.effects.background_engine import BackgroundEngine
        from app.recording.video_recorder import VideoRecorder
        from app.camera.virtual_cam import VirtualCameraOutput
        from app.pipeline import PipelineWorker
        import tempfile
        cls.tmp = tempfile.mkdtemp()
        cls.tracker = FaceTracker(model("face_landmarker.task"), pose_model_path=model("pose_landmarker_lite.task"))
        cls.hands = HandTracker(model("hand_landmarker.task"))
        cls.bg = BackgroundEngine(model("selfie_segmenter.tflite"), models_dir=model(""))
        cls.worker = PipelineWorker(None, cls.tracker, cls.hands, ScribbleMaskEngine(), cls.bg,
                                    VideoRecorder(cls.tmp), VirtualCameraOutput())
        cls.ref = ReferenceFaceDetector()

    @classmethod
    def tearDownClass(cls):
        cls.tracker.close()
        cls.hands.close()
        cls.bg.close()
        cls.ref.close()

    def _run(self, frames, mode, label, step=1):
        """step: frame spacing in the source clip (timestamps stay honest when subsampling)."""
        self.bg.mode = mode
        self.tracker.reset()
        t0 = time.monotonic() + 1000.0 * getattr(self, '_runs', 0)
        self._runs = getattr(self, '_runs', 0) + 1
        worst, audited = 0.0, 0
        for i, f in enumerate(frames):
            out, m = self.worker._process(f, t0 + i * step / 30.0)
            self.worker.metrics.update(m)
            for mask in self.ref.face_masks(f):
                area = mask > 0
                if area.sum() < 400:
                    continue
                audited += 1
                same = (np.abs(out.astype(np.int16) - f.astype(np.int16)).max(axis=2) < 10)[area].mean()
                worst = max(worst, float(same))
                self.assertLess(same, 0.05, f"[{label}] frame {i}: {same:.1%} of the real face is visible "
                                            f"(privacy={m['privacy']}, mode={m['tracking_mode']})")
        print(f"\n[privacy/{label}] audited {audited} face detections, worst unchanged fraction {worst:.2%}")
        self.assertGreater(audited, 10)

    def test_original_background(self):
        self._run(load_frames(120, 150), "original", "original")

    def test_replaced_background(self):
        self._run(load_frames(380, 120), "studio", "studio")

    def test_every_shape_covers_face(self):
        frames = load_frames(120, 150)[::10]
        for shape in ("head", "cube", "star", "diamond", "pill"):
            self.worker.scribble_engine.set_shape(shape)
            self._run(frames, "original", shape, step=10)
        self.worker.scribble_engine.set_shape("head")

    def test_stays_on_you_with_a_second_person(self):
        """You on the left, someone else on the right. Your face disappears for a few frames
        (hand over it); when tracking recovers the helmet must return to *you*."""
        me = load_frames(200, 50, (405, 720))
        other = load_frames(560, 50, (405, 720))
        frames = []
        for i, (a, b) in enumerate(zip(me, other)):
            canvas = np.full((720, 1280, 3), 70, np.uint8)
            canvas[:, 60:465] = a
            if i >= 15:        # someone walks in on the right
                canvas[:, 815:1220] = b
            if 25 <= i < 30:   # your face vanishes (blurred out) for 5 frames
                canvas[:, 60:465] = cv2.GaussianBlur(canvas[:, 60:465], (0, 0), 30)
            frames.append(canvas)
        self.bg.mode = "original"
        self.tracker.reset()
        t0 = time.monotonic() + 5000.0
        for i, f in enumerate(frames):
            out, m = self.worker._process(f, t0 + i / 30.0)
            self.worker.metrics.update(m)
            bbox = m.get('_head_bbox')
            if bbox is not None and i >= 15:
                cx = (bbox[0] + bbox[2]) / 2
                self.assertLess(cx, 640, f"frame {i}: helmet jumped to the other person (cx={cx:.0f})")
        self.assertIsNotNone(bbox, "should be tracking you again at the end")

    def test_curtain_hides_everything(self):
        self.worker.curtain = True
        try:
            f = load_frames(120, 150)[0]
            out, m = self.worker._process(f, time.monotonic())
            self.assertEqual(m['privacy'], 'curtain')
            self.assertEqual(len(self.ref.face_masks(out)), 0)
        finally:
            self.worker.curtain = False


if __name__ == "__main__":
    unittest.main()
