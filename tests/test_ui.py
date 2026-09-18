import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from PySide6.QtWidgets import QApplication
from app.main import ScribbleCameraWindow

class TestUI(unittest.TestCase):
    def test_window_initialization(self):
        app = QApplication.instance() or QApplication(sys.argv)
        win = ScribbleCameraWindow()
        self.assertIsNotNone(win)
        win.show()
        self.assertTrue(win.isVisible())
        # Clean shutdown
        win.close()
        app.processEvents()
        print("\n[Test UI] Window and controls initialized and destroyed cleanly.")

if __name__ == "__main__":
    unittest.main()
