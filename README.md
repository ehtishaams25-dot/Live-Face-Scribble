# Scribble Studio — private 3D scribble head + live background

A desktop app that hides your real face behind an animated, hand-drawn **3D scribble head** that
moves with your head and mirrors your expressions (lips, eyelids, gaze, brows), while removing or
replacing your background in real time. Built for recording YouTube videos and going live without
ever showing your face.

Works with **DroidCam / Iriun** (phone as webcam), USB webcams, or video files. Outputs to the
built-in recorder (MP4 with optional mic audio) and to a **virtual camera** for OBS, Zoom, Discord,
Meet.

---

## Review fixes (19 Sep 2026)

| What you said in the review | What was wrong | Fix |
| :--- | :--- | :--- |
| **"When I tilt my head back I can remove the helmet from my face"** | With a hand over your face the helmet *holds* and follows your head with optical flow — but the flow points it picked were on your hand, so the helmet rode away on the hand (and shrank) and uncovered your face. | While holding, everything on and around a tracked hand (fingers, palm, forearm) is ignored; if no head points are left the helmet stays exactly where your head was. Per-frame moves and shrinking are capped. Regression test: `test_hold_does_not_follow_a_hand`. |
| **"I talk normally and the mouth doesn't move — I have to exaggerate"** | The mouth only opened when MediaPipe's inner-lip points parted, which they barely do in normal speech (worse with a moustache). And the "neutral face" kept re-learning *while you talked*, drifting to a half-open mouth. | A dedicated talking signal: inner lips, `jawOpen` and lower-lip blendshapes, plus chin drop and outer-lip height for syllables the lips hide — measured against your *closed* mouth (a floor that can't drift while you talk) and made robust to head turns. New **Talk Sensitivity** slider. Neutral isn't re-learned mid-sentence. |
| **"My mic is working and its mouth is not moving"** | The mic wasn't used for the face at all. | Optional **Voice Lip-Sync** (Face tab): your microphone's loudness opens the mouth too, lined up with the video using the recorder's Audio Delay. Uses the Recording microphone, or your default mic. Off by default — turn it off if others or music are audible. |
| **"It's emoting every time I talk"** | Stickers fired the instant a blendshape crossed a threshold — and speech flickers them all ("oo" = pucker, laugh = smile). | A sticker needs the expression **held ~0.6-0.8 s while you're not talking**, shows for at least 1.4 s, then has a 3 s cooldown. Expressions are measured against your resting face. |
| **"Looks like earthworms crawling all over my face" / "what I wanted is this" (scribble ball)** | Short, thick, jagged squiggles that wobbled vertex by vertex at 12 fps. | A **pen-tangle**: ~230 long, smooth ballpoint loops wrapped around the head. Their density follows the 3D lighting — light on the lit side, filled in on the shadow side — so the shading *is* the scribble (the 3D shading you liked is kept). Line Boil now re-draws each stroke a hair off instead of wiggling it. Face area stays clean. Best on **Cube**. |
| **"Quality: the frame rate absolutely plummets" / "these three settings aren't doing much"** | Quality switched the camera to **1080p** (phone over Wi-Fi + 2.25x the pixels everywhere). The presets also computed a matte-detail value and never applied it. | Presets no longer touch the resolution. They now set the matte network, **matte detail** (Quality 50% · Balanced 40% · Low-End 25%), face-tracking detail and hand-tracking rate. At 720p: Quality ~30 FPS, Balanced ~36, Low-End ~40 (this PC). |
| **"If I move my hands very fast it vanishes"** | *Solid Hands* was switched off. | Every preset turns **Solid Hands** back on. |

**CorridorKey?** It's an offline keyer for **green/blue-screen** footage (it needs a coloured backdrop plus a rough
alpha hint, and processes clips after the fact), so it can't run live on a normal room background. If you ever shoot
on a green screen, the **Export Matte Track** file is exactly the kind of alpha hint it wants.

---

## What changed in the previous version (rebuild, Sep 2026)

| Problem you reported | What was wrong | Fix |
| :--- | :--- | :--- |
| **Face visible through the helmet** | The helmet was a wire mesh of lines — your face showed through every gap. | The helmet is now a **solid, shaded 3D head**. Its silhouette is the exact projected solid, unioned with an anatomical head core and your live face outline, so no shape or size can uncover you. |
| **"One dropped frame shows my face"** | When face tracking failed (hand over face, looking down/sideways) the helmet disappeared. | **Privacy Guard**: the helmet is *held* on your head and follows it with optical flow until tracking recovers; any head that isn't tracked yet is **blurred**; hotkey **B** swaps the whole output for a *Be Right Back* screen. |
| **Low FPS / not responsive** | ~93 ms of work per frame, all in sequence; the neural matte alone took ~20–40 ms of pure Python overhead. | ~**22 ms/frame at 720p**. Neural matte runs as a **CUDA Graph** (3× faster), face and hand AI run in **separate processes** in parallel with the GPU, and the preview never queues stale frames. |
| **Loses my face when I move** | MediaPipe's face detector sees the whole frame shrunk to 128×128, so a face farther from the camera is almost invisible to it. | **Head-crop tracking**: the tracker feeds the AI a crop that follows your head (and the top of your body silhouette when lost), with an instant full-frame retry if you move out of it. ~60% more frames tracked on small/distant faces. |
| **Expressions not good** | A handful of canned mouth shapes chosen by thresholds. | **Landmark-driven face**: your real lip contour, eyelid openness, iris gaze and brow shape — pose-normalized, compared to *your* neutral face and exaggerated — drawn onto the 3D helmet surface. Every half-smile, raised brow, squint, pout and syllable shows up. |
| **Looks flat / clips when turning** | Face features floated inside the helmet; strokes had no occlusion. | Features sit **on** the helmet surface and foreshorten/disappear correctly as you turn; strokes wrap around the back of the head and vanish at the silhouette. |
| **Background not real-time, loses hands** | Slow matte + blobby hand patches. | 6–11 ms neural matte, a **hand prior** that keeps fast hands solid without filling the background between fingers, and matte hardness/choke controls. |
| **Helmet could jump to someone else** | One-face tracking re-locked onto whichever face it found first. | When you're briefly lost with someone else in frame, the tracker re-acquires the face **nearest where your head was** and refuses to jump to a face far away. |
| **Keep / remove specific people** | Only fixed screen points. | **Per-person control**: click anyone in the preview to hide or show them — the choice *follows them as they move*, even when they walk right behind you. "Only Me" mode hides newcomers automatically; hidden people are replaced by a background plate the app learns. Keep Zones force-show anything the matte misses. |
| **Recording speed** | Recordings sped up whenever FPS dipped (fixed 30 fps stamp). | Frames are placed on a real-time timeline — files always play at true speed, and stay in sync with **microphone audio** (e.g. *Microphone (DroidCam Audio)*). |

### New features
* **4 looks** — Ink (black pen on paper), Chalk, Neon, Sunny (emoji yellow); **3 eye styles** — scribble knots, cartoon eyes with moving pupils, dots; **5 shapes** — Head, Cube, Star, Diamond, Pill.
* **Expression controls** — exaggeration, **talk sensitivity**, optional **voice lip-sync**, face size/height, eyebrows on/off, **Calibrate Neutral Face** (hotkey **C**).
* **Emote stickers** (optional) — blush when you grin, anger vein, sweat drop, "?", "Zzz" when your eyes stay closed, hearts on a pout, "!!" when shocked. Held expressions only — talking never triggers them.
* **Hands in front of the helmet** — raise a hand in front of your face and it's drawn over the scribble head (built only from this-frame hand landmarks, shrunk inside the fingers, so it can never reveal your face).
* **Be Right Back screen** (hotkey **B**), **Snapshot** PNG for thumbnails (hotkey **S**).
* **Matte track export** — `*_matte.mp4` (white = you) alongside the recording, to redo the background in After Effects / Premiere as a luma matte.
* **Performance presets** — Quality / Balanced / Low-End (matte network + detail, tracking detail; they never change your camera resolution), plus Smart hand tracking (every frame only while a hand is near your face).
* **Settings are remembered** between launches; mirror toggle; camera auto-reconnect when DroidCam drops.
* While the helmet is holding (face hidden from the camera), the drawn face keeps your last expression and blinks now and then.

---

## Getting started

1. Double-click **`run.bat`** (first run creates `.venv` and installs `requirements.txt`).
2. **Studio → Camera**: pick *DroidCam Video* (it's picked automatically if present). 720p is recommended.
3. Relax your face, look at the camera and press **C** (Calibrate Neutral Face).
4. **Space** to record, **V** for the virtual camera (appears as *OBS Virtual Camera* — install OBS Studio if you don't have it).

**Hotkeys:** `Space` record · `V` virtual cam · `B` be-right-back · `C` calibrate face · `S` snapshot · `H` HUD

### DroidCam tips
* **USB mode** has noticeably lower latency than Wi-Fi. Set DroidCam to 720p.
* In **Studio → Recording**, choose *Microphone (DroidCam Audio)* to record the phone's mic, and use **Audio Delay** to line your voice up with the video (Wi-Fi ≈ 100–200 ms).

### Keeping other people out (or in)
**Stage → People in Frame → Person Control**. With **Only Me**, anyone who walks in is hidden automatically.
Click a person in the preview to toggle them (blue = you, green = shown, red dashed = hidden). **Add Keep Zone**
and drag a box over anything that must always show. Hidden people are painted over with the empty background the
app learns while you use it — works even with *Original* background.

---

## Performance

Measured with `tools/benchmark.py` on a Ryzen 5 5600G + RTX 3060 (whole pipeline: tracking, matte, helmet, face):

| Setup | ms / frame | Sustains |
| :--- | :--- | :--- |
| 720p, Quality (RVM ResNet50, detail 50%, hands every frame) | ~33 | ~30 FPS |
| 720p, Balanced (RVM MobileNet, detail 40%) | ~28 | ~36 FPS |
| 720p, Low-End (RVM MobileNet, detail 25%, lighter tracking) | ~25 | ~40 FPS |
| 1080p, Quality | ~43 | ~23 FPS (less with a Wi-Fi phone camera) |

Neural matte alone: MobileNet **20 → 6 ms**, ResNet50 **~40 → 10.5 ms** (CUDA Graphs). Previous build: ~93 ms/frame.

Run it on your own machine:
```
.venv\Scripts\python.exe tools\benchmark.py --res 720 --engine rvm_resnet50 --detail 0.5 --proxy 640 --hands-every 1
```

---

## How it works

```
Camera thread (latest frame only, auto-reconnect)
   │
   ├─► face landmarks  ── worker process 1 (MediaPipe, head-following crop)
   ├─► hand landmarks  ── worker process 2 (MediaPipe, scheduled)
   └─► neural matte    ── GPU, CUDA Graph (RVM), async
   ▼
Pose (PnP on ~40 rigid points of MediaPipe's canonical face) → low-lag filter → HOLD + optical flow on dropouts
   ▼
Background (blur / studio / solid / image / learned plate) + per-person isolation + hand prior
   ▼
Privacy guard → solid 3D helmet (shaded, scribbles, outline) → face features on the surface → hands on top
   ▼
Recorder (real-time timeline, NVENC/AMF/QSV/x264, mic, matte) · Virtual camera · Preview (newest frame only)
```

| Module | Role |
| :--- | :--- |
| `app/tracking/face_tracker.py` | Head pose, privacy HOLD with optical-flow following, head-crop tracking |
| `app/tracking/face_geometry.py` | Canonical face model, pose-normalized landmarks, landmark groups |
| `app/tracking/hand_tracker.py`, `mp_worker.py` | Hand tracking; MediaPipe in worker processes |
| `app/tracking/voice_meter.py` | Microphone loudness for voice lip-sync |
| `app/effects/scribble_sphere_3d.py` | Solid 3D helmet: silhouette, shading, strokes, outline, shapes |
| `app/effects/face_features.py` | Landmark-driven expressions, talking mouth, eye styles, emotes |
| `app/effects/background_engine.py`, `rvm_runner.py` | Matting engines, compositing, clean plate |
| `app/effects/person_isolation.py` | Per-person identities, click-to-hide, keep zones |
| `app/pipeline.py` | Per-frame orchestration + privacy guard |
| `app/recording/video_recorder.py` | Real-time MP4 recorder, audio, matte track |
| `app/ui_window.py` | The UI |

## Tests

```
.venv\Scripts\python.exe -m unittest discover -s tests -t .
```
`tests/test_privacy.py` runs the full pipeline over a real clip and, for every frame, re-detects the real face on the
*original* frame with an independent model and checks how much of it survives into the output. Current worst case:
**≤0.2%** of face pixels unchanged (coincidental colour matches) across all five shapes and both background modes —
including a 3-FPS stress test with large jumps between frames, and a two-person scene where your face disappears and
the helmet has to find *you* again.

## Models & licenses
* MediaPipe Face/Hand/Pose Landmarker, Selfie Segmenter, canonical face model — Apache 2.0
* Robust Video Matting (RVM) — GPL-3.0 (weights and network code in `app/models`)
* OpenCV — Apache 2.0 · PySide6 — LGPLv3 · pyvirtualcam — GPL-2.0 · FFmpeg — LGPL/GPL (external binary)
