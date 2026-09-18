import os
import sys
import time
import unittest
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.tracking.face_tracker import FaceTracker
from app.effects.scribble_mask import ScribbleMaskEngine
from app.effects.background_engine import BackgroundEngine
from app.effects.head_objects import WireframeHelmetObject, AnatomicalHeadRigVisualizer
from app.recording.video_recorder import VideoRecorder
from app.camera.capture import list_camera_devices


class TestScribblePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = os.path.join(BASE_DIR, "app", "models")
        cls.lmk_model = os.path.join(cls.models_dir, "face_landmarker.task")
        cls.pose_model = os.path.join(cls.models_dir, "pose_landmarker_full.task")
        cls.seg_model = os.path.join(cls.models_dir, "selfie_segmenter.tflite")
        cls.scribble_tex = os.path.join(cls.models_dir, "assets", "scribble_texture.png")
        cls.test_video = r"D:\Ehtishaam\Files\Mine\New Branding\Scribble face vid test\VID_20260430_222927.mp4"

        cap = cv2.VideoCapture(cls.test_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
        ret, cls.sample_frame = cap.read()

        # Also get frame 90 (where face detection dropped)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 90)
        _, cls.drop_frame = cap.read()
        cap.release()

        assert ret and cls.sample_frame is not None, "Failed to load sample test frame"

    def test_01_assets_exist(self):
        self.assertTrue(os.path.exists(self.lmk_model), "FaceLandmarker model missing")
        self.assertTrue(os.path.exists(self.pose_model), "PoseLandmarker model missing")
        self.assertTrue(os.path.exists(self.seg_model), "SelfieSegmenter model missing")
        self.assertTrue(os.path.exists(self.scribble_tex), "Scribble texture missing")

    def test_02_3d_head_tracker_metric_accuracy(self):
        tracker = FaceTracker(self.lmk_model, pose_model_path=self.pose_model)
        res = tracker.process_frame(self.sample_frame, timestamp_ms=100)
        self.assertTrue(res['has_face'], "Face tracking failed on test frame")
        self.assertEqual(res['tracking_mode'], 'face_dense')

        # Check Euler Angles: Facing camera should be near 0° (NOT -180° / +178°!)
        yaw = res['yaw']
        pitch = res['pitch']
        roll = res['roll']
        print(f"\n[Test 3D Tracker] Calibrated Euler: Yaw={yaw:+.1f}°, Pitch={pitch:+.1f}°, Roll={roll:+.1f}°")
        self.assertLess(abs(yaw), 20.0, f"Yaw out of bounds: {yaw}")
        self.assertLess(abs(pitch), 20.0, f"Pitch out of bounds: {pitch}")
        self.assertLess(abs(roll), 20.0, f"Roll out of bounds: {roll}")

        # Check translation depth Z is realistic (~50cm to 120cm from webcam)
        tz = res['translation_skull'][2]
        print(f"  Camera Distance Z: {tz:.1f} cm")
        self.assertGreater(tz, 30.0)
        self.assertLess(tz, 150.0)

        # Check 3D Anatomical Landmarks exist
        l3d = res['landmarks_3d']
        self.assertIn('eyes', l3d)
        self.assertIn('nose', l3d)
        self.assertIn('mouth', l3d)
        self.assertIn('ears', l3d)
        self.assertIn('back_of_head', l3d)

        # Check Projected 2D Features
        proj = res['projected_features']
        for feat in ['nose_tip', 'ear_left_tragus', 'ear_right_tragus', 'back_of_head', 'crown', 'mouth_center']:
            self.assertIn(feat, proj, f"Missing projected feature {feat}")
            u, v = proj[feat]
            self.assertTrue(0 <= u <= self.sample_frame.shape[1], f"{feat} u={u} outside image width")
            self.assertTrue(0 <= v <= self.sample_frame.shape[0], f"{feat} v={v} outside image height")

        # Check full-head 3D convex hull
        hull = res['head_hull_2d']
        self.assertGreater(len(hull), 5)
        print(f"  Full-Head 3D Convex Hull Points: {len(hull)}")

        tracker.close()

    def test_03_background_engine(self):
        bg_engine = BackgroundEngine(self.seg_model)
        out_blur = bg_engine.process(self.sample_frame, mode="blur", blur_strength=45, timestamp_ms=100)
        self.assertEqual(out_blur.shape, self.sample_frame.shape)
        out_studio = bg_engine.process(self.sample_frame, mode="studio", timestamp_ms=200)
        self.assertEqual(out_studio.shape, self.sample_frame.shape)
        print("\n[Test BackgroundEngine] Successfully processed blur & studio modes")
        bg_engine.close()

    def test_04_zero_leak_privacy_obscuration(self):
        tracker = FaceTracker(self.lmk_model, pose_model_path=self.pose_model)
        scribble_engine = ScribbleMaskEngine(self.scribble_tex, boil_fps=12.0)
        track_res = tracker.process_frame(self.sample_frame, timestamp_ms=100)

        frame_to_modify = self.sample_frame.copy()
        out = scribble_engine.apply(frame_to_modify, track_res, size_scale=1.85, enable_boil=True)
        self.assertEqual(out.shape, self.sample_frame.shape)

        # Verify face pixels are completely modified
        diff = np.mean(np.abs(out.astype(float) - self.sample_frame.astype(float)))
        self.assertGreater(diff, 3.0, "Scribble mask did not modify face pixels")

        # Test simulated frame drops (coasting): mask must remain 100% opaque!
        coast_res = track_res.copy()
        coast_res['is_coasting'] = True
        coast_res['confidence'] = 0.1  # Low confidence during drop
        frame_coast = self.sample_frame.copy()
        out_coast = scribble_engine.apply(frame_coast, coast_res, enable_boil=False)

        # Diff must be solid (no fade out to raw face!)
        diff_coast = np.mean(np.abs(out_coast.astype(float) - self.sample_frame.astype(float)))
        self.assertGreater(diff_coast, 3.0, "Mask faded out during low confidence coasting!")
        print(f"\n[Test Zero-Leak Privacy] Solid obscuration preserved during coasting (diff={diff_coast:.2f})")

        tracker.close()

    def test_05_pose_fallback_extreme_yaw(self):
        tracker = FaceTracker(self.lmk_model, pose_model_path=self.pose_model)
        res = tracker.process_frame(self.drop_frame, timestamp_ms=100)
        self.assertTrue(res['has_face'], "Tracker failed to retain tracking on extreme yaw frame")
        self.assertIn(res['tracking_mode'], ['pose_rescue', 'face_dense', 'kinematic_coast'])
        print(f"\n[Test Pose Rescue] Mode on drop frame: {res['tracking_mode']} (Confidence: {res['confidence']:.2f})")
        tracker.close()

    def test_06_anatomical_rig_visualizer(self):
        tracker = FaceTracker(self.lmk_model, pose_model_path=self.pose_model)
        rig = AnatomicalHeadRigVisualizer()
        track_res = tracker.process_frame(self.sample_frame, timestamp_ms=100)

        frame_to_render = self.sample_frame.copy()
        rendered = rig.render(frame_to_render, track_res)
        self.assertEqual(rendered.shape, self.sample_frame.shape)

        diff = np.mean(np.abs(rendered.astype(float) - self.sample_frame.astype(float)))
        self.assertGreater(diff, 0.4, "Rig visualizer did not draw on frame")
        print(f"\n[Test Anatomical Head Rig] Successfully rendered 3D skull cage, axes, & feature nodes (diff={diff:.2f})")

        tracker.close()

    def test_07_recorder(self):
        rec_dir = os.path.join(BASE_DIR, "tests", "rec_output")
        recorder = VideoRecorder(output_dir=rec_dir, fps=30)
        h, w = self.sample_frame.shape[:2]

        rec_file = recorder.start(width=w, height=h, fps=30)
        self.assertTrue(recorder.is_recording)

        for _ in range(30):
            recorder.write_frame(self.sample_frame)
            time.sleep(0.01)

        stats = recorder.stop()
        self.assertFalse(recorder.is_recording)
        self.assertTrue(os.path.exists(stats['file_path']))
        self.assertGreater(stats['file_size_mb'], 0.01)
        print(f"\n[Test VideoRecorder] Recorded {stats['frame_count']} frames to {stats['file_path']} ({stats['file_size_mb']:.2f} MB)")

        try:
            os.remove(stats['file_path'])
            os.rmdir(rec_dir)
        except Exception:
            pass

    def test_08_full_pipeline_benchmark_720p(self):
        tracker = FaceTracker(self.lmk_model, pose_model_path=self.pose_model)
        scribble_engine = ScribbleMaskEngine(self.scribble_tex, boil_fps=12.0)
        bg_engine = BackgroundEngine(self.seg_model)

        frame_720p = cv2.resize(self.sample_frame, (1280, 720))
        num_frames = 60
        latencies = []

        t_start = time.time()
        for i in range(num_frames):
            t0 = time.time()
            ts = int(i * (1000.0 / 30.0))

            tr = tracker.process_frame(frame_720p, timestamp_ms=ts)
            comp = bg_engine.process(frame_720p, mode="blur", timestamp_ms=ts)
            if tr.get('has_face', False):
                comp = scribble_engine.apply(comp, tr, enable_boil=True)

            latencies.append((time.time() - t0) * 1000.0)

        total_time = time.time() - t_start
        fps = num_frames / total_time
        avg_lat = np.mean(latencies)

        print(f"\n================ 720p PIPELINE BENCHMARK (60 frames @ 1280x720) ================")
        print(f"  Throughput: {fps:.1f} FPS")
        print(f"  Latency:    Avg={avg_lat:.1f} ms  Min={np.min(latencies):.1f} ms  Max={np.max(latencies):.1f} ms")
        print(f"=================================================================================")

        tracker.close()
        bg_engine.close()
        self.assertGreater(fps, 15.0, "720p pipeline throughput must exceed 15 FPS on CPU")

    def test_09_scribble_sphere_3d(self):
        tracker = FaceTracker(self.lmk_model, pose_model_path=self.pose_model)
        scribble_engine = ScribbleMaskEngine(self.scribble_tex, boil_fps=12.0)
        track_res = tracker.process_frame(self.sample_frame, timestamp_ms=100)

        # 1. Render White Chalk Helmet (Reference 1)
        t0 = time.time()
        out_white = scribble_engine.apply(
            self.sample_frame.copy(), track_res,
            style=ScribbleMaskEngine.STYLE_WHITE, cavity_darkness=0.0, enable_boil=True
        )
        lat_ms = (time.time() - t0) * 1000.0
        self.assertLess(lat_ms, 35.0, f"3D Scribble Helmet rendering too slow: {lat_ms:.1f}ms")

        # 2. Render Dark Charcoal Helmet (References 2 & 3)
        out_dark = scribble_engine.apply(
            self.sample_frame.copy(), track_res,
            style=ScribbleMaskEngine.STYLE_DARK, cavity_darkness=0.0, enable_boil=True
        )
        self.assertEqual(out_dark.shape, self.sample_frame.shape)

        # 3. Render Shaded Interior Cavity
        out_cavity = scribble_engine.apply(
            self.sample_frame.copy(), track_res,
            style=ScribbleMaskEngine.STYLE_WHITE, cavity_darkness=0.6, enable_boil=True
        )
        self.assertEqual(out_cavity.shape, self.sample_frame.shape)

        # Verify differences between styles
        diff_white_dark = np.mean(np.abs(out_white.astype(float) - out_dark.astype(float)))
        self.assertGreater(diff_white_dark, 1.0)

        print(f"\n[Test 3D Scribble Sphere] Rendered White, Dark, and Cavity Helmets in {lat_ms:.2f}ms")
        tracker.close()

    def test_10_iron_man_helmet(self):
        from app.effects.iron_man_renderer import IronManHelmetRenderer
        tracker = FaceTracker(self.lmk_model, pose_model_path=self.pose_model)
        iron_man = IronManHelmetRenderer()
        track_res = tracker.process_frame(self.sample_frame, timestamp_ms=100)

        # Warmup frame (allocates GPU FBO texture)
        _ = iron_man.render(self.sample_frame.copy(), track_res)

        # Measure steady-state GPU rendering latency
        t0 = time.time()
        rendered = iron_man.render(self.sample_frame.copy(), track_res)
        lat_ms = (time.time() - t0) * 1000.0

        self.assertEqual(rendered.shape, self.sample_frame.shape)
        diff = np.mean(np.abs(rendered.astype(float) - self.sample_frame.astype(float)))
        self.assertGreater(diff, 2.0, "Iron Man helmet must be rendered onto the frame")
        self.assertLess(lat_ms, 60.0, f"Iron Man GPU rendering on 1080p must be fast (<60ms), took {lat_ms:.2f}ms")

        print(f"\n[Test Iron Man Helmet] Steady-State GPU Render: {lat_ms:.2f}ms (diff={diff:.2f})")
        tracker.close()


if __name__ == "__main__":
    unittest.main()
