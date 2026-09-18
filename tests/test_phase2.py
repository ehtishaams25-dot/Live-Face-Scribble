import os
import sys
import time
import unittest
import numpy as np
import cv2

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.tracking.face_tracker import FaceTracker
from app.tracking.temporal_smoother import KinematicHeadPoseTracker
from app.effects.head_objects import WireframeHelmetObject
from app.effects.environment_engine import EnvironmentRelightingEngine
from app.effects.background_engine import BackgroundEngine
from app.effects.scribble_mask import ScribbleMaskEngine


class TestPhase2Architecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = os.path.join(BASE_DIR, "app", "models")
        cls.lmk_model = os.path.join(cls.models_dir, "face_landmarker.task")
        cls.seg_model = os.path.join(cls.models_dir, "selfie_segmenter.tflite")
        cls.scribble_tex = os.path.join(cls.models_dir, "assets", "scribble_texture.png")
        cls.video_path = r"D:\Ehtishaam\Files\Mine\New Branding\Scribble face vid test\VID_20260430_222927.mp4"

        # Ensure models exist
        assert os.path.exists(cls.lmk_model), f"Missing {cls.lmk_model}"
        assert os.path.exists(cls.seg_model), f"Missing {cls.seg_model}"
        assert os.path.exists(cls.scribble_tex), f"Missing {cls.scribble_tex}"

        # Load real test frame (Frame 200: front face looking directly at camera)
        cap = cv2.VideoCapture(cls.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
        ret, frame = cap.read()
        cap.release()
        assert ret and frame is not None, "Failed to read test frame from video"
        # Proportional resize maintaining aspect ratio (540x960 ~ 0.52 MP, standard live HD proxy)
        cls.test_frame = cv2.resize(frame, (540, 960))

    def test_01_kinematic_smoother_and_prediction(self):
        """Verify 1-Euro smoothing, velocity estimation, and damped prediction during tracking drops."""
        tracker = KinematicHeadPoseTracker(max_coast_frames=10, damping=0.88)
        
        # Simulate moving head: yaw increasing by 10 deg/s
        t = 0.0
        for i in range(5):
            euler = [i * 3.0, 2.0, -1.0]
            trans = [0.0, 5.0, 50.0 + i * 0.5]
            smooth_e, smooth_t, conf, coast = tracker.update(euler, trans, t=t)
            self.assertFalse(coast)
            self.assertEqual(conf, 1.0)
            t += 0.033

        # Simulate sudden loss (e.g. occlusion or extreme yaw past 70 deg)
        for step in range(1, 11):
            pred_e, pred_t, conf, active = tracker.predict(dt=0.033)
            self.assertTrue(active)
            self.assertGreater(conf, 0.0)
            self.assertLessEqual(conf, 1.0)
            # Velocity damping should continue the rotation trend gently
            self.assertGreaterEqual(pred_e[0], 5.0)

        # 11th frame exceeds max_coast_frames -> should gracefully report inactive
        pred_e, pred_t, conf, active = tracker.predict(dt=0.033)
        self.assertFalse(active)
        self.assertEqual(conf, 0.0)
        print("\n[OK] Kinematic prediction & graceful degradation verified.")

    def test_02_3d_head_space_skull_center(self):
        """Verify FaceTracker returns persistent 3D Head Space anchored to True Skull Center."""
        tracker = FaceTracker(self.lmk_model)
        res = tracker.process_frame(self.test_frame, timestamp_ms=100)
        
        self.assertTrue(res['has_face'])
        self.assertIn('matrix_4x4', res)
        self.assertIn('rvec', res)
        self.assertIn('tvec', res)
        self.assertIn('cam_matrix', res)
        self.assertIn('translation_skull', res)
        
        # Check skull center depth is realistic metric distance (30..150 cm)
        # MediaPipe camera coordinates look down -Z, so depth = abs(tz)
        depth_cm = abs(float(res['translation_skull'][2]))
        self.assertGreater(depth_cm, 20.0, f"Unrealistic depth: {depth_cm}")
        self.assertLess(depth_cm, 200.0, f"Unrealistic depth: {depth_cm}")

        # Check projected center is within frame boundaries
        h, w = self.test_frame.shape[:2]
        cx, cy = res['center_x'], res['center_y']
        self.assertGreater(cx, 0)
        self.assertLess(cx, w)
        self.assertGreater(cy, 0)
        self.assertLess(cy, h)

        tracker.close()
        print(f"\n[OK] 3D Skull Space verified: Center=({cx}, {cy}), Depth={depth_cm:.1f} cm, Yaw={res['yaw']:.1f}°")

    def test_03_wireframe_helmet_projection(self):
        """Verify 3D wireframe helmet and coordinate axes project properly into 2D."""
        tracker = FaceTracker(self.lmk_model)
        helmet = WireframeHelmetObject(radius_cm=10.5)

        track_res = tracker.process_frame(self.test_frame, timestamp_ms=100)
        frame_copy = self.test_frame.copy()

        out_frame = helmet.render(frame_copy, track_res)
        self.assertEqual(out_frame.shape, self.test_frame.shape)
        # Ensure rendered pixels changed
        diff = np.sum(np.abs(out_frame.astype(np.int32) - self.test_frame.astype(np.int32)))
        self.assertGreater(diff, 10000, "Helmet was not drawn onto the frame")

        tracker.close()
        print("\n[OK] 3D Wireframe Helmet / Visor spatial attachment verified.")

    def test_04_environment_relighting_engine(self):
        """Verify ambient tinting, light wrap rim lighting, and atmospheric spill."""
        h, w = self.test_frame.shape[:2]
        bg_plate = np.full((h, w, 3), [180, 50, 220], dtype=np.uint8) # Vibrant magenta/blue background
        # Create a circle mask representing subject
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_u8, (w // 2, h // 2), min(w, h) // 3, 255, -1)

        engine = EnvironmentRelightingEngine(enabled=True, rim_intensity=0.8, ambient_intensity=0.15)
        
        # Warmup
        engine.integrate(self.test_frame, bg_plate, mask_u8)

        t0 = time.time()
        comp = engine.integrate(self.test_frame, bg_plate, mask_u8)
        dt = (time.time() - t0) * 1000.0

        self.assertEqual(comp.shape, self.test_frame.shape)
        self.assertLess(dt, 20.0, f"Environment relighting too slow: {dt:.2f} ms")
        print(f"\n[OK] Environment Relighting & Light Wrap verified ({dt:.2f} ms).")

    def test_05_guided_filter_background_engine(self):
        """Verify BackgroundEngine with guided filter edge refinement and relighting."""
        bg_engine = BackgroundEngine(self.seg_model, use_guided_filter=True, enable_relighting=True)
        
        # Warmup
        bg_engine.process(self.test_frame, mode=BackgroundEngine.MODE_STUDIO, timestamp_ms=50)

        t0 = time.time()
        res = bg_engine.process(self.test_frame, mode=BackgroundEngine.MODE_STUDIO, timestamp_ms=100)
        dt = (time.time() - t0) * 1000.0

        self.assertEqual(res.shape, self.test_frame.shape)
        self.assertLess(dt, 30.0, f"Background engine too slow: {dt:.2f} ms")
        bg_engine.close()
        print(f"\n[OK] Guided Filter Background Engine verified ({dt:.2f} ms).")

    def test_06_end_to_end_pipeline_benchmark(self):
        """Run full 3D tracking + guided bg + relight + scribble + helmet pipeline at solid 30+ FPS."""
        tracker = FaceTracker(self.lmk_model)
        bg_engine = BackgroundEngine(self.seg_model, use_guided_filter=True, enable_relighting=True)
        scribble_engine = ScribbleMaskEngine(self.scribble_tex)
        helmet = WireframeHelmetObject()

        # Warmup
        for i in range(3):
            tr = tracker.process_frame(self.test_frame, timestamp_ms=i * 33)
            bg = bg_engine.process(self.test_frame, mode=BackgroundEngine.MODE_BLUR, blur_strength=45, timestamp_ms=i * 33)
            if tr.get('has_face'):
                sc = scribble_engine.apply(bg, tr)
                helmet.render(sc, tr)

        # Benchmark 20 iterations
        times = []
        for i in range(20):
            t0 = time.time()
            tr = tracker.process_frame(self.test_frame, timestamp_ms=(i + 3) * 33)
            bg = bg_engine.process(self.test_frame, mode=BackgroundEngine.MODE_BLUR, blur_strength=45, timestamp_ms=(i + 3) * 33)
            if tr.get('has_face'):
                sc = scribble_engine.apply(bg, tr)
                fin = helmet.render(sc, tr)
            else:
                fin = bg
            times.append((time.time() - t0) * 1000.0)

        avg_ms = np.mean(times)
        fps = 1000.0 / avg_ms

        tracker.close()
        bg_engine.close()

        print(f"\n==========================================")
        print(f"End-to-End Phase 2 Pipeline (540x960):")
        print(f"Average Total Latency: {avg_ms:.2f} ms per frame")
        print(f"Effective Throughput:  {fps:.1f} FPS")
        print(f"==========================================")
        self.assertGreaterEqual(fps, 28.0, f"Target 30 FPS not met: {fps:.1f} FPS")


if __name__ == "__main__":
    unittest.main()
