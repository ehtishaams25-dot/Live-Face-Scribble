import os
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    torch = None
    F = None
    HAS_TORCH = False


class RVMRunner:
    """
    Robust Video Matting inference with CUDA Graphs.

    RVM (MobileNetV3 / ResNet50) is a few hundred small kernels per frame; in eager
    PyTorch the per-kernel Python/launch overhead dominates (~18 ms per frame on an
    RTX 3060 regardless of resolution, with the GPU mostly idle). Capturing the whole
    network once into a CUDA Graph and replaying it removes that overhead:
        MobileNetV3: ~20 ms -> ~6 ms,  ResNet50: ~40 ms -> ~10.5 ms  (720p, fp16)
    and frees the CPU/GIL for tracking. Graphs are re-captured automatically when the
    input size, downsample ratio, or variant changes; if capture ever fails we fall
    back to eager execution transparently.
    """

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.available = HAS_TORCH
        self.cuda = HAS_TORCH and torch.cuda.is_available()
        self.device = torch.device("cuda") if self.cuda else (torch.device("cpu") if HAS_TORCH else None)
        self.dtype = torch.float16 if self.cuda else (torch.float32 if HAS_TORCH else None)
        self.use_graphs = self.cuda
        self.model = None
        self.variant = None
        self._graph = None
        self._graph_key = None
        self._static_src = None
        self._rec_in = None
        self._out = None
        self._eager_rec = [None, None, None, None]
        self._host = None
        self.last_error = None
        if self.cuda:
            torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------
    def load(self, variant: str) -> bool:
        if not self.available:
            return False
        variant = "resnet50" if "resnet" in variant else "mobilenetv3"
        if self.model is not None and self.variant == variant:
            return True
        weight_path = os.path.join(self.models_dir, f"rvm_{variant}.pth")
        if not os.path.exists(weight_path):
            alt = os.path.expanduser(f"~/.cache/torch/hub/checkpoints/rvm_{variant}.pth")
            weight_path = alt if os.path.exists(alt) else None
        try:
            from app.models.rvm_net.model import MattingNetwork
            model = MattingNetwork(variant)
            if weight_path:
                model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
            else:
                hub = torch.hub.load("PeterL1n/RobustVideoMatting", variant)
                model.load_state_dict(hub.state_dict())
            model = model.to(self.device, dtype=self.dtype).eval()
            self.model = model
            self.variant = variant
            self._drop_graph()
            self.reset_state()
            print(f"[RVMRunner] Loaded RVM {variant} on {self.device} (fp16={self.dtype == torch.float16}, graphs={self.use_graphs})")
            return True
        except Exception as e:
            self.last_error = str(e)
            print(f"[RVMRunner] Failed to load RVM {variant}: {e}")
            self.model = None
            return False

    def _drop_graph(self):
        self._graph = None
        self._graph_key = None
        self._static_src = None
        self._rec_in = None
        self._out = None

    def reset_state(self):
        self._eager_rec = [None, None, None, None]
        if self._rec_in is not None:
            for r in self._rec_in:
                r.zero_()

    # ------------------------------------------------------------------
    def _upload(self, frame_bgr: np.ndarray):
        """BGR uint8 HWC numpy -> RGB [0,1] NCHW tensor on device (async on CUDA)."""
        h, w = frame_bgr.shape[:2]
        if self.cuda:
            if self._host is None or self._host.shape != (h, w, 3):
                self._host = torch.empty((h, w, 3), dtype=torch.uint8).pin_memory()
            self._host.numpy()[...] = frame_bgr
            t = self._host.to(self.device, non_blocking=True)
        else:
            t = torch.from_numpy(np.ascontiguousarray(frame_bgr))
        return t.permute(2, 0, 1).flip(0).unsqueeze(0).to(self.dtype).div_(255.0)

    def _capture(self, h, w, ds):
        src = torch.zeros((1, 3, h, w), device=self.device, dtype=self.dtype)
        with torch.no_grad():
            outs = self.model(src, downsample_ratio=ds)
        rec_in = [r.detach().clone().zero_() for r in outs[2:]]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(2):
                self.model(src, *rec_in, downsample_ratio=ds)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            out = self.model(src, *rec_in, downsample_ratio=ds)
        self._graph = g
        self._static_src = src
        self._rec_in = rec_in
        self._out = out
        for r in self._rec_in:
            r.zero_()

    def run(self, frame_bgr: np.ndarray, downsample_ratio: float):
        """
        Queues RVM on the GPU for one frame. Returns (src, fgr, pha) tensors
        (NCHW, [0,1]) — valid until the next call. Asynchronous on CUDA: nothing
        blocks until the caller reads data back to the CPU.
        """
        h, w = frame_bgr.shape[:2]
        ds = float(downsample_ratio)
        src = self._upload(frame_bgr)
        if self.use_graphs:
            key = (h, w, round(ds, 3), self.variant)
            try:
                if key != self._graph_key:
                    self._drop_graph()
                    self._capture(h, w, ds)
                    self._graph_key = key
                self._static_src.copy_(src)
                self._graph.replay()
                fgr, pha, *rec_out = self._out
                for a, b in zip(self._rec_in, rec_out):
                    a.copy_(b)
                return self._static_src, fgr, pha
            except Exception as e:
                print(f"[RVMRunner] CUDA graph path failed ({e}); falling back to eager execution.")
                self.last_error = str(e)
                self.use_graphs = False
                self._drop_graph()
                torch.cuda.synchronize()
        eager_key = (h, w, round(ds, 3), self.variant)
        if eager_key != getattr(self, "_eager_key", None):
            self._eager_rec = [None, None, None, None]
            self._eager_key = eager_key
        with torch.no_grad():
            fgr, pha, *self._eager_rec = self.model(src, *self._eager_rec, downsample_ratio=ds)
        return src, fgr, pha
