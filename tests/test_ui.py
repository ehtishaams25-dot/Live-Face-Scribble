import os
import tempfile
import unittest

from tests._common import have_video


@unittest.skipUnless(have_video(), "test video not available")
class TestUI(unittest.TestCase):
    def test_window_runs_on_demo_video(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        ini = os.path.join(tempfile.mkdtemp(), "settings.ini")
        os.environ["SCRIBBLE_SETTINGS_INI"] = ini
        from PySide6.QtCore import QSettings, QTimer
        from PySide6.QtWidgets import QApplication
        s = QSettings(ini, QSettings.IniFormat)
        s.setValue("cam/device", "Demo Video File")
        s.sync()
        from app.ui_window import ScribbleCameraWindow
        app = QApplication.instance() or QApplication([])
        win = ScribbleCameraWindow()
        win.show()
        seen = {}

        def check():
            seen.update(win.video.metrics)
            for i in range(4):
                win.tab_nav.set_current_index(i)
            res = win.res_combo.currentIndex()
            win._apply_preset(0)
            # Review: Quality switched the camera to 1080p and FPS collapsed. Presets must keep
            # the resolution and differ in what they really set (matte detail, Solid Hands).
            seen['res_kept'] = win.res_combo.currentIndex() == res
            seen['quality_detail'] = win.bg_engine.downsample_ratio
            win._apply_preset(2)
            seen['lowend_detail'] = win.bg_engine.downsample_ratio
            seen['solid_hands'] = win.bg_engine.hand_prior
            win._apply_voice(True)      # opens the default mic if there is one; must never raise
            win._apply_voice(False)
            seen['voice_off'] = not win.voice.running
            win.close()
            app.quit()
        QTimer.singleShot(8000, check)
        app.exec()
        self.assertGreater(seen.get('fps', 0), 5.0, "pipeline should be producing frames")
        self.assertIn(seen.get('privacy'), ('helmet', 'holding', 'mosaic', 'clear'))
        self.assertTrue(seen['res_kept'], "a preset must not change the camera resolution")
        self.assertAlmostEqual(seen['quality_detail'], 0.50)
        self.assertAlmostEqual(seen['lowend_detail'], 0.25)
        self.assertTrue(seen['solid_hands'])
        self.assertTrue(seen['voice_off'])


if __name__ == "__main__":
    unittest.main()
