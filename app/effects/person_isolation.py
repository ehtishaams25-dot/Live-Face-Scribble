import numpy as np
import cv2


class _Track:
    __slots__ = ('tid', 'mask', 'bbox', 'centroid', 'area', 'last_seen', 'age')

    def __init__(self, tid):
        self.tid = tid
        self.mask = None
        self.bbox = (0, 0, 0, 0)
        self.centroid = (0.0, 0.0)
        self.area = 0
        self.last_seen = 0
        self.age = 0


class PersonIsolator:
    """
    Per-person keep/remove control on top of the segmentation matte.

    Every foreground blob in the (low-res) matte gets a persistent ID. IDs survive frame
    to frame by seeding each new frame with the previous frame's (eroded) per-person
    regions, and when two people's blobs touch or overlap (someone walking right behind
    you) the merged blob is split by nearest-seed (Voronoi) assignment, so they keep
    separate identities instead of fusing into one.

    The person wearing the helmet (seeded from the tracked head) is "you" and is always
    kept. Everyone else follows the mode (Everyone -> kept, Only Me -> removed) unless you
    click them in the preview, which flips that specific person and follows them around.
    Keep Zones force a rectangle to always show (e.g. a friend or object the matte misses).
    """

    MODE_EVERYONE = "everyone"
    MODE_ONLY_ME = "only_me"

    WORK_W = 192

    def __init__(self):
        self.mode = self.MODE_EVERYONE
        self.enabled = False
        self.tracks = {}
        self.overrides = {}        # tid -> bool (True keep, False remove)
        self.keep_zones = []       # list of (x0, y0, x1, y1) normalized
        self.me_tid = None
        self._next_tid = 1
        self._frame = 0
        self._labels = None
        self._shape = None
        self.forget_after = 45     # frames an unseen person is remembered

    # ------------------------------------------------------------------ control
    def set_mode(self, mode: str):
        if mode in (self.MODE_EVERYONE, self.MODE_ONLY_ME):
            self.mode = mode

    def reset(self):
        self.tracks.clear()
        self.overrides.clear()
        self.me_tid = None
        self._labels = None
        self._shape = None

    def clear_selections(self):
        self.overrides.clear()
        self.keep_zones = []

    def add_keep_zone(self, x0, y0, x1, y1):
        x0, x1 = sorted((float(np.clip(x0, 0, 1)), float(np.clip(x1, 0, 1))))
        y0, y1 = sorted((float(np.clip(y0, 0, 1)), float(np.clip(y1, 0, 1))))
        if (x1 - x0) > 0.01 and (y1 - y0) > 0.01:
            self.keep_zones.append((x0, y0, x1, y1))

    def remove_zone_at(self, xn, yn) -> bool:
        for i, (x0, y0, x1, y1) in enumerate(self.keep_zones):
            if x0 <= xn <= x1 and y0 <= yn <= y1:
                self.keep_zones.pop(i)
                return True
        return False

    def is_kept(self, tid) -> bool:
        if tid == self.me_tid:
            return True
        if tid in self.overrides:
            return self.overrides[tid]
        return self.mode == self.MODE_EVERYONE

    def click(self, xn: float, yn: float):
        """Toggles the person under a normalized click. Returns (tid, kept) or None."""
        if self._labels is None:
            return None
        h, w = self._labels.shape
        x = int(np.clip(xn * w, 0, w - 1))
        y = int(np.clip(yn * h, 0, h - 1))
        tid = int(self._labels[y, x])
        if tid == 0:
            # Snap to the nearest person within a small radius (clicks near edges).
            r = max(3, w // 30)
            win = self._labels[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
            ids = win[win > 0]
            if ids.size == 0:
                return None
            tid = int(np.bincount(ids).argmax())
        if tid == self.me_tid:
            return tid, True
        new_state = not self.is_kept(tid)
        self.overrides[tid] = new_state
        return tid, new_state

    def people(self):
        """[(tid, bbox_norm, kept, is_me)] for drawing the selection overlay."""
        if self._shape is None:
            return []
        h, w = self._shape
        out = []
        for t in self.tracks.values():
            if t.last_seen != self._frame or t.area < 12:
                continue
            x0, y0, x1, y1 = t.bbox
            out.append((t.tid, (x0 / w, y0 / h, x1 / w, y1 / h), self.is_kept(t.tid), t.tid == self.me_tid))
        return out

    # ------------------------------------------------------------------ core
    def update(self, alpha_small: np.ndarray, head_bbox_norm=None):
        """
        alpha_small: float32 (h, w) matte in [0, 1] at low resolution.
        head_bbox_norm: (x0, y0, x1, y1) normalized bbox of the tracked head, or None.
        Returns a float32 keep mask (h, w) in [0, 1], or None if nothing needs removing
        and no keep zones apply (fast path: matte passes through untouched).
        """
        self._frame += 1
        h, w = alpha_small.shape
        if self._shape != (h, w):
            self.tracks.clear()
            self.me_tid = None
        self._shape = (h, w)

        fg = (alpha_small > 0.4).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # ---- seeds: previous identities (eroded) + the tracked head ("you")
        seeds = np.zeros((h, w), dtype=np.int32)
        k = np.ones((3, 3), np.uint8)
        for t in self.tracks.values():
            if t.mask is None:
                continue
            sm = cv2.erode(t.mask, k) & fg
            seeds[sm > 0] = t.tid
        if head_bbox_norm is not None:
            x0, y0, x1, y1 = head_bbox_norm
            cx0, cy0 = int(np.clip(x0 * w, 0, w - 1)), int(np.clip(y0 * h, 0, h - 1))
            cx1, cy1 = int(np.clip(x1 * w, 0, w - 1)), int(np.clip(y1 * h, 0, h - 1))
            if cx1 > cx0 and cy1 > cy0:
                if self.me_tid is None or self.me_tid not in self.tracks:
                    # Adopt whichever existing identity dominates the head region, else a new one.
                    region = seeds[cy0:cy1 + 1, cx0:cx1 + 1]
                    ids = region[region > 0]
                    self.me_tid = int(np.bincount(ids).argmax()) if ids.size else self._new_tid()
                # The head region is authoritative for "you".
                head_seed = np.zeros_like(fg)
                head_seed[cy0:cy1 + 1, cx0:cx1 + 1] = 1
                seeds[(head_seed & fg) > 0] = self.me_tid

        n_cc, cc = cv2.connectedComponents(fg, connectivity=8)
        labels = np.zeros((h, w), dtype=np.int32)
        need_split = []
        for c in range(1, n_cc):
            comp = cc == c
            ids = np.unique(seeds[comp])
            ids = ids[ids > 0]
            if ids.size == 0:
                labels[comp] = self._new_tid()
            elif ids.size == 1:
                labels[comp] = int(ids[0])
            else:
                need_split.append(comp)
        if need_split:
            # Nearest-seed (Voronoi) partition of merged blobs.
            zero_at_seeds = np.where(seeds > 0, 0, 255).astype(np.uint8)
            _, lab = cv2.distanceTransformWithLabels(zero_at_seeds, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
            seed_ys, seed_xs = np.nonzero(seeds > 0)
            # Label i (1-based) corresponds to the i-th zero pixel in row-major order.
            lut = np.zeros(len(seed_ys) + 1, dtype=np.int32)
            lut[1:] = seeds[seed_ys, seed_xs]
            nearest = lut[np.clip(lab, 0, len(seed_ys))]
            for comp in need_split:
                labels[comp] = nearest[comp]

        # ---- update tracks
        present = set(np.unique(labels).tolist()) - {0}
        for tid in present:
            m = (labels == tid)
            area = int(m.sum())
            ys, xs = np.nonzero(m)
            t = self.tracks.get(tid)
            if t is None:
                t = _Track(tid)
                self.tracks[tid] = t
            t.mask = m.astype(np.uint8)
            t.area = area
            t.bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            t.centroid = (float(xs.mean()), float(ys.mean()))
            t.last_seen = self._frame
            t.age += 1
        for tid in list(self.tracks.keys()):
            if self._frame - self.tracks[tid].last_seen > self.forget_after:
                self.tracks.pop(tid)
                self.overrides.pop(tid, None)
                if tid == self.me_tid:
                    self.me_tid = None
        # Without a head, fall back to the biggest person as "you" in Only-Me mode.
        if self.me_tid is None and present and self.mode == self.MODE_ONLY_ME:
            self.me_tid = max(present, key=lambda i: self.tracks[i].area)
        self._labels = labels

        removed = [tid for tid in present if not self.is_kept(tid)]
        if not removed and not self.keep_zones:
            return None

        keep = np.ones((h, w), dtype=np.float32)
        if removed:
            rm = np.isin(labels, removed).astype(np.uint8)
            # Grow removed regions over their soft matte fringe, but never into a kept person.
            kept_core = ((labels > 0) & (rm == 0)).astype(np.uint8)
            rm = cv2.dilate(rm, np.ones((5, 5), np.uint8)) & (1 - kept_core)
            keep[rm > 0] = 0.0
            keep = cv2.GaussianBlur(keep, (5, 5), 0)
        for (x0, y0, x1, y1) in self.keep_zones:
            keep[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))] = 2.0  # 2 => force foreground
        return keep

    def _new_tid(self):
        tid = self._next_tid
        self._next_tid += 1
        return tid
