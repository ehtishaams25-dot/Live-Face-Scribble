"""
Scribble Studio launcher.

Kept deliberately import-light: MediaPipe runs in worker processes (Windows "spawn"),
and each child re-executes this module's top level. Heavy imports (Qt, PyTorch, the UI)
therefore live inside main() so the workers start fast and stay small.
"""
import os
import sys
import multiprocessing

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def main():
    multiprocessing.freeze_support()
    os.environ.setdefault("GLOG_minloglevel", "2")       # quieter MediaPipe logs
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from PySide6.QtWidgets import QApplication
    from app.ui_window import ScribbleCameraWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Scribble Studio")
    window = ScribbleCameraWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
