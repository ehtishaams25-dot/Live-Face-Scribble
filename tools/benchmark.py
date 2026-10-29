"""
Headless pipeline benchmark — measures what the live app would do per frame on THIS machine.

    .venv\\Scripts\\python.exe tools\\benchmark.py --video my_clip.mp4 --res 720 --engine rvm_mobilenetv3 --mode studio

Portrait clips are letterboxed into the requested landscape size (like a phone held
sideways would be). Prints per-stage timings, the FPS the pipeline can sustain, and how
often the helmet was locked / holding / mosaicked.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
DEFAULT_VIDEO = r"D:\Ehtishaam\Files\Mine\New Branding\Scribble face vid test\VID_20260430_222927.mp4"


def fit(frame, w, h):
    fh, fw = frame.shape[:2]
    s = min(w / fw, h / fh)
    small = cv2.resize(frame, (int(fw * s), int(fh * s)), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(cv2.resize(frame, (w // 8, h // 8)), (9, 9), 0), (w, h))
    y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    bg[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--res", type=int, default=720, choices=(720, 1080))
    ap.add_argument("--engine", default="rvm_mobilenetv3", choices=("rvm_mobilenetv3", "rvm_resnet50", "mediapipe"))
    ap.add_argument("--mode", default="studio", choices=("original", "blur", "studio", "solid"))
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--proxy", type=int, default=512, help="face tracking proxy width (384 = Low-End)")
    ap.add_argument("--hands-every", type=int, default=2)
    ap.add_argument("--detail", type=float, default=0.40, help="matte detail / RVM downsample ratio (Quality 0.5, Low-End 0.25)")
    ap.add_argument("--no-workers", action="store_true")
    a = ap.parse_args()

    from app.tracking.face_tracker import FaceTracker
    from app.tracking.hand_tracker import HandTracker
    from app.effects.scribble_mask import ScribbleMaskEngine
    from app.effects.background_engine import BackgroundEngine
    from app.recording.video_recorder import VideoRecorder
    from app.camera.virtual_cam import VirtualCameraOutput
    from app.pipeline import PipelineWorker

    m = os.path.join(BASE_DIR, "app", "models")
    tr = FaceTracker(os.path.join(m, "face_landmarker.task"), proxy_width=a.proxy, use_worker=not a.no_workers)
    hd = HandTracker(os.path.join(m, "hand_landmarker.task"), every_n=a.hands_every, use_worker=not a.no_workers)
    bg = BackgroundEngine(os.path.join(m, "selfie_segmenter.tflite"), models_dir=m, engine_type=a.engine)
    bg.mode = a.mode
    bg.downsample_ratio = a.detail
    w = PipelineWorker(None, tr, hd, ScribbleMaskEngine(), bg, VideoRecorder(os.path.join(BASE_DIR, "recordings")),
                       VirtualCameraOutput())
    W, H = (1280, 720) if a.res == 720 else (1920, 1080)
    cap = cv2.VideoCapture(a.video)
    times, stages, states = [], {"track_ms": [], "bg_ms": [], "scribble_ms": []}, {}
    t0 = time.monotonic()
    for i in range(a.frames):
        ok, f = cap.read()
        if not ok:
            break
        f = fit(f, W, H)
        s = time.perf_counter()
        _, met = w._process(f, t0 + i / 30.0)
        w.metrics.update(met)
        times.append((time.perf_counter() - s) * 1000)
        if i >= 15:
            for k in stages:
                stages[k].append(met[k])
        states[met["privacy"]] = states.get(met["privacy"], 0) + 1
    tr.close()
    hd.close()
    bg.close()
    t = np.array(times[15:])
    print(f"{W}x{H} {a.engine}/{a.mode} detail={a.detail} proxy={a.proxy} hands/{a.hands_every} workers={not a.no_workers}")
    print(f"  per frame: mean {t.mean():.1f} ms | p50 {np.median(t):.1f} | p95 {np.percentile(t, 95):.1f} "
          f"-> sustains {min(1000 / t.mean(), 999):.0f} FPS")
    print("  stages (ms):", {k: round(float(np.mean(v)), 1) for k, v in stages.items()})
    print("  privacy states:", states)


if __name__ == "__main__":
    main()
