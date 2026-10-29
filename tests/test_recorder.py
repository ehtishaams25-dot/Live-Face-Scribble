import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest

import numpy as np

from tests._common import BASE_DIR  # noqa: F401  (sys.path setup)
from app.recording.video_recorder import VideoRecorder, find_ffmpeg


def _probe(path):
    ffprobe = os.path.join(os.path.dirname(find_ffmpeg() or ""), "ffprobe.exe")
    if not os.path.exists(ffprobe):
        ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    out = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames,r_frame_rate", "-show_entries", "format=duration", "-of", "json", path],
                         capture_output=True, text=True)
    return json.loads(out.stdout)


class TestRecorder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_real_time_speed_when_fps_drops(self):
        """Frames arriving at 15 fps must still produce a real-time 30 fps file (2 s -> 2 s)."""
        rec = VideoRecorder(self.tmp)
        rec.record_matte = True
        rec.start(320, 240, fps=30)
        t0 = time.monotonic()
        for i in range(30):                    # 2 seconds at 15 fps
            f = np.full((240, 320, 3), i * 8 % 255, np.uint8)
            rec.write_frame(f, t0 + i / 15.0, np.full((240, 320), 255, np.uint8))
        stats = rec.stop()
        self.assertTrue(os.path.exists(stats['file_path']))
        self.assertGreater(stats['frame_count'], 55)
        self.assertLess(stats['frame_count'], 65)
        if find_ffmpeg():
            self.assertIsNotNone(stats['matte_path'])
            info = _probe(stats['file_path'])
            if info:
                n = int(info['streams'][0]['nb_read_frames'])
                self.assertTrue(55 <= n <= 65, n)
                self.assertAlmostEqual(float(info['format']['duration']), 2.0, delta=0.2)

    @unittest.skipUnless(find_ffmpeg(), "needs FFmpeg")
    def test_audio_is_muxed(self):
        rec = VideoRecorder(self.tmp)
        rec.audio_device = "test"
        rec.audio_offset_ms = 120
        # Silent generated audio instead of a real microphone.
        rec._audio_input_args = lambda: ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        rec.start(320, 240, fps=30)
        t0 = time.monotonic()
        for i in range(45):
            rec.write_frame(np.zeros((240, 320, 3), np.uint8), t0 + i / 30.0)
            time.sleep(1 / 60)
        stats = rec.stop()
        ffprobe = os.path.join(os.path.dirname(find_ffmpeg()), "ffprobe.exe")
        out = subprocess.run([ffprobe, "-v", "error", "-show_entries", "stream=codec_type", "-of", "json",
                              stats['file_path']], capture_output=True, text=True)
        kinds = [st['codec_type'] for st in json.loads(out.stdout)['streams']]
        self.assertIn('audio', kinds)
        self.assertIn('video', kinds)


if __name__ == "__main__":
    unittest.main()
