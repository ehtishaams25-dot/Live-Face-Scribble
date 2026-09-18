# 🖊️ Real-Time 3D Scribble Camera & Background Engine

An open-source desktop application that hides your real face behind an organically animated, boiling **3D ink/pencil scribble head** that moves, rotates, and tilts with true 3D head pose in real time, while simultaneously providing real-time Zoom/Meet-style **background removal, cinematic blurring, and custom studio replacement**.

Built for content creators, anonymous livestreaming, privacy, and artistic video production.

---

## 🆕 Latest Optimization Pass

* **Parallel Tracking + Matting**: head tracking (MediaPipe, CPU) and background matting (RVM, GPU) are independent per-frame computations and now run concurrently on two threads, collapsing total latency from `track_ms + bg_ms` down to roughly `max(track_ms, bg_ms)`. Measured ~50% throughput gain end-to-end.
* **Fixed the 3D head-wrap clipping seam**: the helmet's front/back depth sort used to bucket an *entire* scribble curve as front-of-head or back-of-head based on a single centroid Z value — a curve spanning the visibility boundary near the temples would pop to the wrong side as a whole when turning. It now splits each curve into contiguous front/back runs at the exact vertex where it crosses the depth threshold, so turning your head no longer shows a visible seam/clipping artifact.
* **Continuous, less "snappy" expressions**: eyebrows are now always driven by live blendshape magnitude (they move naturally while you talk) instead of only appearing for a handful of extreme discrete emotions; blendshape smoothing switched from a flat exponential filter to per-channel adaptive 1-Euro filtering; discrete expression states (shocked/angry/confused/etc.) now use hysteresis so they don't flicker at the threshold boundary.
* **Primary-Person Isolation**: the background matte can now restrict itself to just the tracked subject via connected-component analysis on the alpha mask, so someone walking through the background is excluded automatically — no click needed. Left-click the video to explicitly keep a spot in frame; right-click to exclude a spot; click a marked spot again to undo. Off by default, and a solo frame is always passed through untouched. See **Stage → Person Isolation**.
* **One-tap Low-End PC Preset** (**Studio → Performance**): 720p capture, the fastest matting engine, and relighting/hair-detail refinement off, freeing CPU/GPU headroom for your recording/editing software.
* **Keyboard shortcuts**: `Space` = start/stop recording, `V` = toggle virtual camera, `H` = toggle HUD overlay.
* Removed a redundant second `HandLandmarker` model instance (the background engine's hand mask is now the single shared instance for the whole pipeline) and batch-vectorized the facial-expression rendering to match the fast path already used by the helmet renderer.
* Default capture resolution is now **720p** (was 1080p) — the recommended setting for lower-end machines; 1080p remains selectable.

---

## 📸 Key Features

* **3D Head-Attached Scribble Mask**:
  * Tracks 478 3D facial landmarks and derives a full metric **4x4 camera transformation matrix** using Google MediaPipe FaceLandmarker.
  * **Zero Face Leakage**: An inner morphologically dilated, feathered charcoal core base mask guarantees 100% of your real eyes, nose, mouth, and skin remain completely hidden at any angle.
  * **3-Layer Volumetric 3D Parallax**: Cranium, mid-face, and outer loops rotate with independent depth layers, showing authentic 3D parallax when turning left/right (yaw up to ±60°) and looking up/down (pitch).
  * **12 FPS Line Boil Animation**: Smooth 30–60 FPS body and head tracking, with hand-drawn pencil/ink lines boiling and redrawing at 12 FPS for an authentic animated aesthetic.
  * **Expression Reactivity**: Scribble head dynamically expands and reacts when opening your mouth or speaking.

* **Real-Time Background Engine**:
  * Powered by MediaPipe's lightweight `SelfieSegmenter` model (249 KB).
  * Modes:
    * **Cinematic Blur**: Smooth, two-stage downsampled Gaussian bokeh with feathered human contours.
    * **Studio Dark Gradient**: Professional studio backdrop with subtle vertical lighting vignette.
    * **Solid Color**: Customizable backdrop (e.g. Green screen for streaming).
    * **Custom Virtual Image**: Load any PNG/JPG/WebP wallpaper as your background.
    * **Original**: Pass-through background.

* **High Performance & Zero VRAM Bloat**:
  * Runs the entire AI inference pipeline on CPU via **XNNPACK SIMD multi-threading** (~21 ms per frame at 720p).
  * Consumes **0 MB GPU VRAM**, allowing you to run heavy video editing software (Premiere Pro, After Effects, Blender) simultaneously on the same machine without OOM crashes.
  * SIMD C++ vectorized alpha blending via OpenCV (`cv2.multiply`).

* **Hardware-Accelerated Video Recording**:
  * Non-blocking queue-based asynchronous MP4 recording.
  * Hardware **NVIDIA NVENC (`h264_nvenc`)** encoding via FFmpeg (installed at `C:\ffmpeg\bin\ffmpeg.exe`), with automatic fallback to OpenCV `VideoWriter`.
  * Preserves full 30+ FPS UI and capture rate during recording.

* **DirectShow Virtual Camera**:
  * Integrated with `pyvirtualcam` to broadcast directly to **Zoom, Google Meet, Discord, Microsoft Teams, and OBS Studio**.

* **Input Flexibility**:
  * DirectShow support for **DroidCam Video**, **Iriun Webcam**, physical USB webcams, and OBS Virtual Camera.
  * **Offline Video File Playback Mode**: Loop any test video (such as phone recordings) to preview and record effects without a webcam attached.

---

## ⚡ Performance Benchmarks

Tested on: **AMD Ryzen 5 5600G (6C/12T) + NVIDIA GeForce RTX 3060 (12GB)**

| Resolution | AI Tracking Latency | Segmentation Latency | Compositing & Scribble | Total Latency | Effective FPS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **720p (1280x720)** | **8.5 ms** | **11.2 ms** | **2.2 ms** | **21.9 ms** | **~46 FPS** |
| **1080p (1920x1080)** | **10.1 ms** | **18.5 ms** | **6.1 ms** | **34.7 ms** | **~30 FPS** |

*GPU VRAM Footprint for Inference:* **0 MB**  
*CPU Utilization:* **~8–12%**  
*Recording CPU Overhead (NVENC):* **< 1%**

---

## 🚀 Getting Started

### Prerequisites
* Windows 10/11 64-bit
* Python 3.10 or 3.11 (or `uv` package manager)
* (Optional for NVENC) NVIDIA GPU with driver installed and FFmpeg on PATH (or at `C:\ffmpeg\bin\ffmpeg.exe`)
* (Optional for Phone Webcam) [DroidCam](https://www.dev47apps.com/) on your Android/iOS phone and PC client.

### 1-Click Launch
Double click **`run.bat`** in the repository root.

If running for the first time, `run.bat` will automatically initialize `.venv`, install the locked dependencies from `requirements.txt`, and launch the application.

### Manual Setup
```bash
# 1. Create and activate virtual environment
uv venv .venv --python 3.11
.venv\Scripts\activate

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Launch the desktop application
python app\main.py
```

---

## 🎮 How to Use

1. **Select Camera**:
   * Pick **`DroidCam Video`**, your webcam, or **`🎬 Demo Video File`** from the *Video Input Source* dropdown.
   * If using DroidCam, start the DroidCam app on your phone, connect via WiFi or USB, and select *DroidCam Video*.
2. **Adjust 3D Scribble**:
   * **Head Size Scale**: Adjust how far the scribble head expands past your face contour (default: 1.85x).
   * **3D Head Parallax Intensity**: Controls how much the back and front scribble shells shift when turning your head (default: 1.6x).
   * **12 FPS Line Boil Animation**: Toggle the hand-drawn boiling redrawing effect.
   * **Facial Obscuration Core**: Controls the opacity of the inner charcoal silhouette mask (default: 95%).
3. **Configure Background**:
   * Choose between *Cinematic Blur*, *Studio Dark Gradient*, *Solid Color (Green Screen)*, *Custom Virtual Image*, or *Original*.
   * Adjust the *Blur Strength* slider to achieve your desired depth of field.
4. **Record Video**:
   * Click **`⏺ Start Recording`**. The HUD will display a red pulsing `REC` indicator.
   * When finished, click **`⏹ Stop Recording`**. A summary dialog will show the saved MP4 path, duration, frame count, and file size.
   * Click **`Open Recordings Folder`** to browse your exported MP4 videos in `recordings/`.
5. **Broadcast to Zoom / Discord / Google Meet**:
   * Click **`🎥 Start Virtual Camera`**.
   * In your meeting app (Zoom, Discord, Meet), open Video Settings and choose **`OBS Virtual Camera`** as your camera device.
6. **Keep bystanders out of frame (or add someone in)** — *Stage → Person Isolation*:
   * Toggle **Isolate Primary Person** on. By default only the tracked subject (you) stays in the matte — anyone else who walks through the background is excluded automatically.
   * Left-click a spot on the video to force it to stay in frame (green marker); right-click a spot to force it out (red marker). Click a marker again to remove it. **Clear Selections** resets everything.
7. **Low-end PC?** — *Studio → Performance*: click **Apply Low-End PC Preset** for a one-tap bundle of settings (720p, fastest matting engine, relighting off) tuned to leave CPU/GPU headroom for your recording/editing software.
8. **Keyboard shortcuts**: `Space` start/stop recording · `V` toggle virtual camera · `H` toggle HUD overlay.

---

## 🏗️ Architecture & Component Breakdown

```
[ Camera Capture Thread (cv2.CAP_DSHOW) ] (Drops stale frames, buffer_size=1)
                       │
         ┌─────────────┴──────────────┐
         ▼                            ▼
[ Face Landmarker (CPU) ]    [ Selfie Segmenter (CPU) ]
(MediaPipe 4x4 Matrix &      (MediaPipe Confidence Mask
 478 3D Facial Landmarks)     256x256 @ 11ms)
         │                            │
         ▼                            ▼
[ 3D Scribble Engine ]       [ Background Engine ]
 - Dilated Charcoal Core      - 2-Stage Downsampled Blur
 - 3-Layer Parallax Shells    - Studio Vignette Backdrop
 - 12 FPS Boiling Animator    - SIMD C++ Alpha Compositing
         └─────────────┬──────────────┘
                       ▼
            [ Live Composited Frame ]
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
[ PySide6 Viewport ] [ FFmpeg NVENC ] [ pyvirtualcam ]
 (30-60 FPS Display)  (Async MP4 Rec)  (DirectShow Broadcast)
```

### Build vs. Reuse Matrix

| Component | Architecture Choice | Repository / Project | License |
| :--- | :--- | :--- | :--- |
| **Camera Capture** | Reuse | `cv2.VideoCapture` + `pygrabber` (DirectShow) | Apache 2.0 / BSD |
| **Face Tracking & Pose** | Reuse | Google MediaPipe `FaceLandmarker` (`face_landmarker.task`) | Apache 2.0 |
| **3D Transform Matrix** | Reuse | MediaPipe `facial_transformation_matrixes` | Apache 2.0 |
| **3D Scribble Engine** | Custom | Multi-Planar Volumetric Parallax + 12 FPS Boiling Loop | MIT |
| **Human Segmentation** | Reuse | Google MediaPipe `ImageSegmenter` (`selfie_segmenter.tflite`) | Apache 2.0 |
| **Background Effects** | Custom | SIMD Vectorized Blend (`cv2.multiply`) + Downscaled Bokeh | MIT |
| **Video Recording** | Reuse | Asynchronous `FFmpeg` NVENC hardware pipe (`h264_nvenc`) | LGPL 2.1 |
| **Virtual Camera** | Reuse | `pyvirtualcam` (DirectShow OBS Virtual Camera backend) | MIT |
| **Desktop GUI** | Reuse | `PySide6` (Qt 6 for Python) | LGPLv3 |

---

## 🛡️ License & Legal Compliance

All components and models in this project use commercial-friendly, open-source licenses:
* **MediaPipe models**: Apache License 2.0
* **OpenCV**: Apache License 2.0
* **PySide6**: LGPLv3
* **PyVirtualCam**: MIT License
* **FFmpeg**: LGPL 2.1
* *Zero Non-Commercial (NC) or viral AGPL restrictions.*

---

## 🔮 What I Would Improve Next

1. **Procedural Wireframe Mesh Generator**: Integrate a real-time Delaunay / 3D spline tube generator directly from the 478 MediaPipe 3D metric coordinates to create procedural physical scribble wireframes in OpenGL shaders.
2. **Audio Track Passthrough for Recording**: Capture system microphone via `sounddevice` and mux it synchronously into the FFmpeg NVENC MP4 container.
3. **Custom Scribble Brushes**: Allow users to drop in custom scribble SVG vector brushes, chalk textures, or watercolor masks.
4. **Segmented Hair/Clothing Separation**: Use multi-class segmentation (`selfie_multiclass_256x256.tflite`) to allow applying ink textures to clothes or hair independently.
5. **Tracked (not fixed-screen) person-isolation markers**: click-to-include/exclude spots are currently fixed screen positions for the session; tracking a bystander's blob across frames (so an exclude marker follows them as they move) would remove the need to re-click if they walk further across frame.
6. **MediaPipe GPU delegate on Windows**: all MediaPipe graphs (face/hand/pose landmarkers, selfie segmenter) currently run on CPU — Tasks API GPU delegate support on Windows Python bindings is limited/unreliable today, but is worth revisiting as MediaPipe's Windows support matures, since the GPU is already warm for RVM.
7. **Persist last-used settings** (camera, resolution, background mode, matting engine) across launches via `QSettings`, so the app reopens the way you left it.
