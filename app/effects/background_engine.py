import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app.effects.environment_engine import EnvironmentRelightingEngine
from app.effects.person_isolation import PersonIsolator
from app.effects.rvm_runner import RVMRunner, HAS_TORCH, torch, F
from app.tracking.hand_tracker import HandTracker


class BgResult:
    """Output of one background pass."""
    __slots__ = ("composite", "alpha_small", "alpha_full", "bg_ms", "engine", "people")

    def __init__(self, composite, alpha_small=None, alpha_full=None, bg_ms=0.0, engine="none", people=None):
        self.composite = composite
        self.alpha_small = alpha_small
        self.alpha_full = alpha_full
        self.bg_ms = bg_ms
        self.engine = engine
        self.people = people or []


class BackgroundEngine:
    """
    Real-time person matting + background replacement.

    Engines
      * RVM ResNet50 / MobileNetV3 on CUDA (fp16, CUDA Graphs) — best quality, 6-11 ms.
      * RVM on CPU (slow; only if there is no GPU and you insist).
      * MediaPipe Selfie Segmenter on CPU — fastest on low-end machines; temporally
        smoothed and edge-refined with a guided filter.

    Two-phase API so the GPU works while the CPU tracks the face:
        handle = engine.begin(frame, ts)          # queues GPU work, returns immediately
        ... face / hand tracking on the CPU ...
        result = engine.finish(handle, frame, hands=..., head_bbox=...)

    Features: hand prior (fast hands stay solid; background between fingers stays clear),
    matte hardness & choke, per-person isolation (click to hide/show people, Only-Me mode,
    keep zones), and a learned clean background plate so hidden bystanders are erased
    even in Original mode.
    """

    ENGINE_RVM_RESNET50 = "rvm_resnet50"
    ENGINE_RVM_MOBILENET = "rvm_mobilenetv3"
    ENGINE_MEDIAPIPE = "mediapipe"

    MODE_ORIGINAL = "original"
    MODE_BLUR = "blur"
    MODE_STUDIO = "studio"
    MODE_SOLID = "solid"
    MODE_IMAGE = "image"

    SMALL_W = PersonIsolator.WORK_W

    def __init__(self, model_path: str = None, models_dir: str = None, use_guided_filter: bool = True,
                 enable_relighting: bool = False, engine_type: str = None):
        if models_dir and os.path.isdir(models_dir):
            self.models_dir = models_dir
        elif model_path and os.path.exists(model_path):
            self.models_dir = os.path.dirname(model_path)
        else:
            self.models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        seg_path = model_path if (model_path and os.path.isfile(model_path)) else os.path.join(self.models_dir, "selfie_segmenter.tflite")

        self.rvm = RVMRunner(self.models_dir)
        self.has_cuda = self.rvm.cuda
        self.engine_type = engine_type or (self.ENGINE_RVM_MOBILENET if self.has_cuda else self.ENGINE_MEDIAPIPE)

        self.segmenter = None
        if os.path.exists(seg_path):
            try:
                opts = vision.ImageSegmenterOptions(
                    base_options=python.BaseOptions(model_asset_path=seg_path),
                    running_mode=vision.RunningMode.VIDEO,
                    output_confidence_masks=True)
                self.segmenter = vision.ImageSegmenter.create_from_options(opts)
            except Exception as e:
                print(f"[BackgroundEngine] MediaPipe segmenter init warning: {e}")
        self._seg_ts = -1
        self._mp_prev = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Segmenter")

        # Settings (the pipeline/UI writes these directly)
        self.mode = self.MODE_ORIGINAL
        self.blur_strength = 55
        self.solid_color = (0, 255, 0)          # BGR — chroma green by default
        self.downsample_ratio = 0.4
        self.edge_choke = 0
        self.hardness = 0.5
        self.enable_decontamination = True
        self.use_guided_filter = use_guided_filter
        self.hand_prior = True
        self.relighting_engine = EnvironmentRelightingEngine(enabled=enable_relighting, rim_intensity=0.2,
                                                             ambient_intensity=0.0, spill_intensity=0.0)
        self.isolator = PersonIsolator()

        self.custom_bg_img = None
        self._cpu_cache = {}
        self._gpu_cache = {}
        self._plate_t = None
        self._plate_valid_t = None
        self._plate_np = None
        self._plate_valid_np = None
        self._has_guided = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter")

        if self.engine_type != self.ENGINE_MEDIAPIPE:
            if not self.rvm.load(self._variant(self.engine_type)):
                self.engine_type = self.ENGINE_MEDIAPIPE

    # ------------------------------------------------------------------ settings
    @staticmethod
    def _variant(engine_type):
        return "resnet50" if engine_type == BackgroundEngine.ENGINE_RVM_RESNET50 else "mobilenetv3"

    @property
    def person_isolation_enabled(self):
        return self.isolator.enabled

    def set_person_isolation(self, enabled: bool):
        self.isolator.enabled = bool(enabled)
        if not enabled:
            self.isolator.reset()

    def set_engine_type(self, engine_type: str):
        if engine_type == self.engine_type:
            return
        if engine_type in (self.ENGINE_RVM_RESNET50, self.ENGINE_RVM_MOBILENET):
            if self.rvm.load(self._variant(engine_type)):
                self.engine_type = engine_type
        else:
            self.engine_type = self.ENGINE_MEDIAPIPE
        self.reset_states()

    def reset_states(self):
        self.rvm.reset_state()
        self._mp_prev = None
        self._plate_t = self._plate_valid_t = None
        self._plate_np = self._plate_valid_np = None

    def reset_clean_plate(self):
        self._plate_t = self._plate_valid_t = None
        self._plate_np = self._plate_valid_np = None

    def set_custom_image(self, image_path: str) -> bool:
        if image_path and os.path.exists(image_path):
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self.custom_bg_img = img
                self._cpu_cache.pop("image", None)
                self._gpu_cache.pop("image", None)
                return True
        return False

    def set_solid_color(self, bgr):
        self.solid_color = tuple(int(c) for c in bgr)
        self._cpu_cache.pop("solid", None)
        self._gpu_cache.pop("solid", None)

    def set_relighting_params(self, enabled=None, rim_intensity=None, ambient_intensity=None):
        if enabled is not None:
            self.relighting_engine.enabled = bool(enabled)
        if rim_intensity is not None:
            self.relighting_engine.rim_intensity = float(rim_intensity)
        if ambient_intensity is not None:
            self.relighting_engine.ambient_intensity = float(ambient_intensity)

    def uses_gpu_matting(self) -> bool:
        return self.engine_type != self.ENGINE_MEDIAPIPE and self.rvm.model is not None

    def needs_matte(self, extra: bool = False) -> bool:
        return self.mode != self.MODE_ORIGINAL or self.isolator.enabled or extra

    # ------------------------------------------------------------------ phase 1
    def begin(self, frame_bgr: np.ndarray, timestamp_ms: int, need_matte: bool = None):
        if need_matte is None:
            need_matte = self.needs_matte()
        if not need_matte:
            return None
        t0 = time.perf_counter()
        if self.engine_type != self.ENGINE_MEDIAPIPE and self.rvm.load(self._variant(self.engine_type)):
            ds = float(np.clip(self.downsample_ratio, 0.15, 1.0))
            src, fgr, pha = self.rvm.run(frame_bgr, ds)
            return {"kind": "rvm", "src": src, "fgr": fgr, "pha": pha, "t0": t0}
        if self.segmenter is None:
            return None
        fut = self._pool.submit(self._mp_confidence, frame_bgr, timestamp_ms)
        return {"kind": "mp", "future": fut, "t0": t0}

    def quick_alpha(self, frame_bgr: np.ndarray, timestamp_ms: int):
        """Cheap low-res person mask (MediaPipe) — used as a presence sensor for the privacy guard."""
        if self.segmenter is None:
            return None
        try:
            conf = self._pool.submit(self._mp_confidence, frame_bgr, timestamp_ms).result()
        except Exception:
            return None
        sh = max(8, int(round(self.SMALL_W * conf.shape[0] / conf.shape[1])))
        return cv2.resize(conf, (self.SMALL_W, sh), interpolation=cv2.INTER_AREA)

    def _mp_confidence(self, frame_bgr, timestamp_ms):
        h, w = frame_bgr.shape[:2]
        pw = 320
        ph = max(8, int(round(h * pw / w)))
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (pw, ph), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        if timestamp_ms <= self._seg_ts:
            timestamp_ms = self._seg_ts + 1
        self._seg_ts = timestamp_ms
        res = self.segmenter.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        if not res.confidence_masks:
            return np.zeros((ph, pw), np.float32)
        m = res.confidence_masks[-1].numpy_view()
        if m.ndim == 3:
            m = m[:, :, 0]
        return np.ascontiguousarray(m, dtype=np.float32)

    # ------------------------------------------------------------------ phase 2
    def finish(self, handle, frame_bgr: np.ndarray, hands=None, hands_fresh: bool = False,
               head_bbox=None, want_alpha_full: bool = False) -> BgResult:
        if handle is None:
            return BgResult(frame_bgr)
        h, w = frame_bgr.shape[:2]
        head_n = None
        if head_bbox is not None:
            x0, y0, x1, y1 = head_bbox
            head_n = (x0 / w, y0 / h, x1 / w, y1 / h)
        if handle["kind"] == "rvm":
            res = self._finish_rvm(handle, frame_bgr, hands, hands_fresh, head_n, want_alpha_full)
        else:
            res = self._finish_mp(handle, frame_bgr, hands, hands_fresh, head_n, want_alpha_full)
        res.bg_ms = (time.perf_counter() - handle["t0"]) * 1000.0
        if self.isolator.enabled:
            res.people = self.isolator.people()
        return res

    def process(self, frame_bgr, mode=None, blur_strength=None, timestamp_ms=None, engine_type=None,
                downsample_ratio=None, edge_choke=None, enable_decontamination=None, hand_mask=None,
                face_center_norm=None, **_):
        """Single-call convenience wrapper (tests / scripts)."""
        if mode is not None:
            self.mode = mode
        if blur_strength is not None:
            self.blur_strength = blur_strength
        if engine_type is not None:
            self.set_engine_type(engine_type)
        if downsample_ratio is not None:
            self.downsample_ratio = downsample_ratio
        if edge_choke is not None:
            self.edge_choke = edge_choke
        if enable_decontamination is not None:
            self.enable_decontamination = enable_decontamination
        ts = int(time.time() * 1000) if timestamp_ms is None else timestamp_ms
        return self.finish(self.begin(frame_bgr, ts), frame_bgr).composite

    # ------------------------------------------------------------------ GPU (RVM)
    def _t(self, arr_u8, size_hw=None):
        """uint8 HxW numpy -> (1,1,H,W) tensor in [0,1] on the matting device, optionally resized."""
        t = torch.from_numpy(arr_u8).to(self.rvm.device, non_blocking=False).to(self.rvm.dtype).div_(255.0)
        t = t.view(1, 1, *arr_u8.shape[:2])
        if size_hw is not None and tuple(arr_u8.shape[:2]) != tuple(size_hw):
            t = F.interpolate(t, size=size_hw, mode="bilinear", align_corners=False)
        return t

    def _finish_rvm(self, hd, frame_bgr, hands, hands_fresh, head_n, want_alpha_full):
        src, fgr, pha = hd["src"], hd["fgr"], hd["pha"]
        H, W = frame_bgr.shape[:2]
        sw = self.SMALL_W
        sh = max(8, int(round(sw * H / W)))
        with torch.no_grad():
            choke = int(self.edge_choke)
            if choke < 0:
                k = -choke * 2 + 1
                pha = -F.max_pool2d(-pha, k, stride=1, padding=k // 2)
            elif choke > 0:
                k = choke * 2 + 1
                pha = F.max_pool2d(pha, k, stride=1, padding=k // 2)

            if self.hand_prior and hands:
                s = 0.25
                hs, ws = max(8, int(H * s)), max(8, int(W * s))
                boost = HandTracker.draw_mask((hs, ws), hands, scale=s, thickness=1.35)
                if boost.any():
                    bt = self._t(boost, (H, W))
                    pha = torch.where(bt > 0.5, torch.maximum(pha, (pha * 3.0).clamp(0.0, 1.0)), pha)
                    if hands_fresh:
                        core = HandTracker.draw_mask((hs, ws), hands, scale=s, core_only=True)
                        pha = torch.maximum(pha, (self._t(core, (H, W)) - 0.5).clamp(min=0.0) * 2.0)

            hd_ = float(np.clip(self.hardness, 0.0, 1.0))
            if hd_ > 0.01:
                lo, hi = 0.12 * hd_, 1.0 - 0.3 * hd_
                pha = ((pha - lo) / (hi - lo)).clamp(0.0, 1.0)

            pha_pre = pha
            alpha_small = F.interpolate(pha.float(), size=(sh, sw), mode="area")[0, 0].cpu().numpy()

            removal = None
            if self.isolator.enabled:
                keep = self.isolator.update(alpha_small, head_n)
                self._update_plate_gpu(src, pha_pre)
                if keep is not None:
                    kt = torch.from_numpy(keep).to(self.rvm.device, dtype=self.rvm.dtype).view(1, 1, sh, sw)
                    kt = F.interpolate(kt, size=(H, W), mode="bilinear", align_corners=False)
                    force = kt > 1.5
                    kk = kt.clamp(0.0, 1.0)
                    removal = torch.where(force, torch.zeros_like(pha), pha * (1.0 - kk))
                    pha = torch.where(force, torch.ones_like(pha), pha * kk)
                    alpha_small = np.where(keep > 1.5, 1.0, alpha_small * np.clip(keep, 0.0, 1.0)).astype(np.float32)

            alpha_full = None
            if want_alpha_full:
                alpha_full = (pha[0, 0] * 255.0).to(torch.uint8).cpu().numpy()

            if self.mode == self.MODE_ORIGINAL:
                if removal is None:
                    return BgResult(frame_bgr, alpha_small, alpha_full, engine="rvm")
                fill = self._fill_gpu(src, pha_pre)
                out = src + (fill - src) * removal
                return BgResult(self._download(out), alpha_small, alpha_full, engine="rvm")

            src_clean = src if removal is None else src + (self._fill_gpu(src, pha_pre) - src) * removal
            bg = self._bg_gpu(src_clean, H, W)
            if bg is None:
                return BgResult(frame_bgr, alpha_small, alpha_full, engine="rvm")

            fg = torch.where(pha > 0.85, src, fgr) if self.enable_decontamination else src
            if removal is not None:
                fg = src_clean if not self.enable_decontamination else torch.where(pha > 0.85, src_clean, fgr)
            rl = self.relighting_engine
            if rl.enabled and rl.rim_intensity > 0.02:
                inner = -F.max_pool2d(-pha, 5, stride=1, padding=2)
                rim = (pha - inner).clamp(0.0, 1.0)
                fg = (fg + bg * rim * min(0.35, 0.3 * rl.rim_intensity)).clamp(0.0, 1.0)
            comp = bg + (fg - bg) * pha
            return BgResult(self._download(comp), alpha_small, alpha_full, engine="rvm")

    def _download(self, t_rgb):
        return (t_rgb[0].flip(0).permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).to(torch.uint8).contiguous().cpu().numpy()

    def _bg_gpu(self, src, H, W):
        mode = self.mode
        dev, dt = self.rvm.device, self.rvm.dtype
        if mode == self.MODE_BLUR:
            f = int(max(4, min(12, self.blur_strength / 7)))
            small = F.interpolate(src, size=(max(16, H // f), max(16, W // f)), mode="area")
            k = 7 if self.blur_strength < 70 else 9
            small = F.avg_pool2d(small, k, stride=1, padding=k // 2, count_include_pad=False)
            small = F.avg_pool2d(small, k, stride=1, padding=k // 2, count_include_pad=False)
            return F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
        if mode in (self.MODE_STUDIO, self.MODE_SOLID, self.MODE_IMAGE):
            key = (mode, H, W, self.solid_color if mode == self.MODE_SOLID else None)
            cached = self._gpu_cache.get(mode)
            if cached is not None and cached[0] == key:
                return cached[1]
            plate = self._cpu_plate(mode, H, W)
            if plate is None:
                return None
            t = torch.from_numpy(np.ascontiguousarray(plate[:, :, ::-1])).to(dev).permute(2, 0, 1).unsqueeze(0).to(dt).div_(255.0)
            self._gpu_cache[mode] = (key, t)
            return t
        return None

    def _update_plate_gpu(self, src, pha):
        """Learns the empty background wherever nobody is standing (for erasing bystanders)."""
        bgm = F.max_pool2d(pha, 9, stride=1, padding=4) < 0.03
        if self._plate_t is None or self._plate_t.shape != src.shape:
            self._plate_t = src.clone()
            self._plate_valid_t = bgm.to(src.dtype)
            return
        fresh = bgm & (self._plate_valid_t < 0.5)
        blended = self._plate_t + (src - self._plate_t) * 0.08
        self._plate_t = torch.where(fresh, src, torch.where(bgm, blended, self._plate_t))
        self._plate_valid_t = torch.maximum(self._plate_valid_t, bgm.to(src.dtype))

    def _fill_gpu(self, src, pha):
        """Background to paint where a hidden person stands: learned plate, else a smeared fill."""
        H, W = src.shape[-2:]
        wgt = (1.0 - F.max_pool2d(pha, 9, stride=1, padding=4)).clamp(0.0, 1.0)
        s = (max(8, H // 16), max(8, W // 16))
        num = F.avg_pool2d(F.interpolate(src * wgt, size=s, mode="area"), 5, stride=1, padding=2)
        den = F.avg_pool2d(F.interpolate(wgt, size=s, mode="area"), 5, stride=1, padding=2)
        fill = F.interpolate(num / den.clamp(min=1e-3), size=(H, W), mode="bilinear", align_corners=False)
        if self._plate_t is not None and self._plate_t.shape == src.shape:
            fill = torch.where(self._plate_valid_t > 0.5, self._plate_t, fill)
        return fill

    # ------------------------------------------------------------------ CPU (MediaPipe)
    def _finish_mp(self, hd, frame_bgr, hands, hands_fresh, head_n, want_alpha_full):
        H, W = frame_bgr.shape[:2]
        try:
            conf = hd["future"].result()
        except Exception as e:
            print(f"[BackgroundEngine] segmentation failed: {e}")
            return BgResult(frame_bgr)
        ph, pw = conf.shape
        # Temporal smoothing kills the frame-to-frame edge flicker of the per-frame segmenter.
        if self._mp_prev is not None and self._mp_prev.shape == conf.shape:
            m = cv2.addWeighted(conf, 0.65, self._mp_prev, 0.35, 0.0)
        else:
            m = conf
        self._mp_prev = m

        if self.hand_prior and hands:
            s = pw / float(W)
            boost = HandTracker.draw_mask((ph, pw), hands, scale=s, thickness=1.35)
            if boost.any():
                m = np.where(boost > 0, np.maximum(m, np.minimum(m * 3.0, 1.0)), m).astype(np.float32)
                if hands_fresh:
                    core = HandTracker.draw_mask((ph, pw), hands, scale=s, core_only=True)
                    m = np.maximum(m, core.astype(np.float32) / 255.0)
        hd_ = float(np.clip(self.hardness, 0.0, 1.0))
        if hd_ > 0.01:
            lo, hi = 0.15 * hd_, 1.0 - 0.3 * hd_
            m = np.clip((m - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        choke = int(self.edge_choke)
        if choke != 0:
            k = np.ones((abs(choke) * 2 + 1,) * 2, np.uint8)
            m = cv2.erode(m, k) if choke < 0 else cv2.dilate(m, k)

        sw = self.SMALL_W
        sh = max(8, int(round(sw * H / W)))
        alpha_small = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA)
        m_pre = m
        removal = None
        if self.isolator.enabled:
            keep = self.isolator.update(alpha_small, head_n)
            if keep is not None:
                kp = cv2.resize(keep, (pw, ph), interpolation=cv2.INTER_LINEAR)
                force = kp > 1.5
                kk = np.clip(kp, 0.0, 1.0)
                removal = np.where(force, 0.0, m * (1.0 - kk)).astype(np.float32)
                m = np.where(force, 1.0, m * kk).astype(np.float32)
                alpha_small = np.where(keep > 1.5, 1.0, alpha_small * np.clip(keep, 0.0, 1.0)).astype(np.float32)

        a_full = self._upsample_alpha(m, frame_bgr)
        alpha_full = (a_full * 255.0).astype(np.uint8) if want_alpha_full else None

        if self.isolator.enabled:
            self._update_plate_cpu(frame_bgr, m_pre)
        src_clean = frame_bgr
        if removal is not None:
            r_full = cv2.resize(removal, (W, H), interpolation=cv2.INTER_LINEAR)
            fill = self._fill_cpu(frame_bgr, m_pre)
            src_clean = cv2.blendLinear(fill, frame_bgr, r_full, 1.0 - r_full)

        if self.mode == self.MODE_ORIGINAL:
            return BgResult(src_clean, alpha_small, alpha_full, engine="mediapipe")
        if self.mode == self.MODE_BLUR:
            sw_, sh_ = max(160, W // 4), max(90, H // 4)
            small = cv2.resize(src_clean, (sw_, sh_), interpolation=cv2.INTER_AREA)
            k = max(3, (int(self.blur_strength) // 2) | 1)
            bg = cv2.resize(cv2.GaussianBlur(small, (k, k), 0), (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            bg = self._cpu_plate(self.mode, H, W)
            if bg is None:
                return BgResult(frame_bgr, alpha_small, alpha_full, engine="mediapipe")
        if self.relighting_engine.enabled and self.mode != self.MODE_BLUR:
            comp = self.relighting_engine.integrate(src_clean, bg, (a_full * 255.0).astype(np.uint8))
        else:
            comp = cv2.blendLinear(src_clean, bg, a_full, 1.0 - a_full)
        return BgResult(comp, alpha_small, alpha_full, engine="mediapipe")

    def _upsample_alpha(self, m, frame_bgr):
        H, W = frame_bgr.shape[:2]
        if self.use_guided_filter and self._has_guided:
            hw, hh = max(64, W // 2), max(36, H // 2)
            guide = cv2.cvtColor(cv2.resize(frame_bgr, (hw, hh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            a = cv2.resize(m, (hw, hh), interpolation=cv2.INTER_LINEAR)
            a = cv2.ximgproc.guidedFilter(guide=guide, src=a, radius=4, eps=2e-3 * 255 * 255)
            return np.clip(cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
        return np.clip(cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)

    def _update_plate_cpu(self, frame_bgr, m_small):
        H, W = frame_bgr.shape[:2]
        bgm = cv2.resize((cv2.dilate(m_small, np.ones((5, 5), np.uint8)) < 0.05).astype(np.uint8),
                         (W, H), interpolation=cv2.INTER_NEAREST)
        if self._plate_np is None or self._plate_np.shape[:2] != (H, W):
            self._plate_np = frame_bgr.astype(np.float32)
            self._plate_valid_np = bgm.copy()
            return
        cv2.accumulateWeighted(frame_bgr.astype(np.float32), self._plate_np, 0.08, mask=bgm)
        self._plate_valid_np |= bgm

    def _fill_cpu(self, frame_bgr, m_small):
        H, W = frame_bgr.shape[:2]
        sw, sh = max(8, W // 16), max(8, H // 16)
        wgt = 1.0 - cv2.resize(cv2.dilate(m_small, np.ones((5, 5), np.uint8)), (sw, sh), interpolation=cv2.INTER_AREA)
        small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        num = cv2.blur(small * wgt[:, :, None], (5, 5))
        den = cv2.blur(wgt, (5, 5))[:, :, None]
        fill = cv2.resize(np.clip(num / np.maximum(den, 1e-3), 0, 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
        if self._plate_np is not None and self._plate_np.shape[:2] == (H, W):
            valid = self._plate_valid_np.astype(bool)
            fill[valid] = self._plate_np[valid].astype(np.uint8)
        return fill

    def _cpu_plate(self, mode, H, W):
        key = (mode, H, W, self.solid_color if mode == self.MODE_SOLID else None)
        cached = self._cpu_cache.get(mode)
        if cached is not None and cached[0] == key:
            return cached[1]
        if mode == self.MODE_STUDIO:
            y = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
            x = np.linspace(-1.0, 1.0, W, dtype=np.float32)[None, :]
            glow = np.exp(-((x * 0.9) ** 2 + ((y - 0.35) * 1.6) ** 2) * 1.8)
            top = np.array([62, 46, 38], np.float32)       # BGR: cool slate
            bot = np.array([22, 18, 16], np.float32)
            base = top[None, None, :] * (1.0 - y[..., None]) + bot[None, None, :] * y[..., None]
            plate = np.clip(base * (0.55 + 0.75 * glow[..., None]), 0, 255).astype(np.uint8)
        elif mode == self.MODE_SOLID:
            plate = np.empty((H, W, 3), np.uint8)
            plate[:] = self.solid_color
        elif mode == self.MODE_IMAGE:
            if self.custom_bg_img is None:
                return None
            bh, bw = self.custom_bg_img.shape[:2]
            s = max(W / bw, H / bh)
            nw, nh = int(np.ceil(bw * s)), int(np.ceil(bh * s))
            r = cv2.resize(self.custom_bg_img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            x0, y0 = (nw - W) // 2, (nh - H) // 2
            plate = np.ascontiguousarray(r[y0:y0 + H, x0:x0 + W])
        else:
            return None
        self._cpu_cache[mode] = (key, plate)
        return plate

    def close(self):
        if self.segmenter is not None:
            try:
                self.segmenter.close()
            except Exception:
                pass
        self._pool.shutdown(wait=False)
        self.rvm.model = None
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
