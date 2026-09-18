import os
import sys
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.effects.hand_drawn_expressions import HandDrawnExpressionsRenderer


class TestMatureExpressions(unittest.TestCase):
    def setUp(self):
        self.renderer = HandDrawnExpressionsRenderer(boil_fps=12.0)
        self.tracking_data = {
            'has_face': True,
            'rvec': np.array([0.0, 0.0, 0.0], dtype=np.float64),
            'tvec': np.array([0.0, 0.0, 75.0], dtype=np.float64),
            'cam_matrix': np.array([
                [800.0, 0.0, 320.0],
                [0.0, 800.0, 240.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64),
            'blendshapes': {}
        }
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_expressions_rendering(self):
        expressions = [
            ("neutral", {}),
            ("smile", {"mouthSmileLeft": 0.7, "mouthSmileRight": 0.7, "eyeSquintLeft": 0.4}),
            ("laugh", {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8, "jawOpen": 0.5}),
            ("shocked", {"jawOpen": 0.6, "browInnerUp": 0.8}),
            ("angry", {"browDownLeft": 0.8, "browDownRight": 0.8, "mouthFrownLeft": 0.3}),
            ("sad", {"mouthFrownLeft": 0.5, "browInnerUp": 0.5}),
            ("smirk_wink", {"mouthSmileRight": 0.7, "eyeBlinkLeft": 0.9})
        ]

        for name, bs in expressions:
            self.tracking_data['blendshapes'] = bs
            self.renderer.reset()
            # Render on white style (black strokes)
            out_white = self.renderer.render(self.frame.copy(), self.tracking_data, style="white", t_sec=1.0)
            self.assertEqual(out_white.shape, self.frame.shape)
            # Render on dark style (white strokes)
            out_dark = self.renderer.render(self.frame.copy(), self.tracking_data, style="dark", t_sec=1.0)
            self.assertEqual(out_dark.shape, self.frame.shape)

    def test_contrast_colors(self):
        # White style should use black strokes (18, 18, 24)
        self.assertEqual(self.renderer.COLOR_BLACK, (18, 18, 24))
        # Dark style should use white strokes (255, 255, 255)
        self.assertEqual(self.renderer.COLOR_WHITE, (255, 255, 255))


if __name__ == '__main__':
    unittest.main()
