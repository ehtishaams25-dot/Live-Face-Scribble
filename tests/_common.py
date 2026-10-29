import os
import sys
import functools

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MODELS = os.path.join(BASE_DIR, "app", "models")
# Any portrait/landscape clip of a person talking to the camera works; override with SCRIBBLE_TEST_VIDEO.
TEST_VIDEO = os.environ.get(
    "SCRIBBLE_TEST_VIDEO",
    r"D:\Ehtishaam\Files\Mine\New Branding\Scribble face vid test\VID_20260430_222927.mp4")


def model(name):
    return os.path.join(MODELS, name)


def have_video():
    return os.path.exists(TEST_VIDEO)


@functools.lru_cache(maxsize=4)
def load_frames(start=0, count=60, size=(720, 1280)):
    cap = cv2.VideoCapture(TEST_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA))
    cap.release()
    return tuple(frames)


class ReferenceFaceDetector:
    """Independent MediaPipe face landmarker (IMAGE mode) used to audit outputs."""

    def __init__(self):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model("face_landmarker.task")),
                                            running_mode=vision.RunningMode.IMAGE, num_faces=2,
                                            min_face_detection_confidence=0.5, min_face_presence_confidence=0.5)
        self.lm = vision.FaceLandmarker.create_from_options(opts)

    def face_masks(self, frame_bgr):
        import mediapipe as mp
        h, w = frame_bgr.shape[:2]
        res = self.lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        masks = []
        for lms in res.face_landmarks or []:
            pts = np.array([(p.x * w, p.y * h) for p in lms], dtype=np.float32)
            m = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(m, cv2.convexHull(pts).astype(np.int32), 255)
            masks.append(m)
        return masks

    def close(self):
        self.lm.close()
