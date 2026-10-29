import time
import cv2
import numpy as np
from scipy.spatial import cKDTree


PALETTES = {
    # fill / shade / deep: body tones (BGR). line: scribble strokes. outline: silhouette.
    # ink: features. paper: eye whites / teeth. cavity: open mouth interior.
    "ink": dict(fill=(236, 236, 232), shade=(196, 196, 190), deep=(160, 160, 156),
                line=(34, 34, 36), outline=(22, 22, 24), ink=(20, 20, 24),
                paper=(252, 252, 250), cavity=(38, 32, 34), back_line=(120, 120, 118)),
    "chalk": dict(fill=(46, 44, 42), shade=(34, 33, 31), deep=(24, 23, 22),
                  line=(232, 232, 228), outline=(248, 248, 245), ink=(248, 248, 245),
                  paper=(70, 68, 66), cavity=(14, 12, 12), back_line=(120, 120, 118)),
    "neon": dict(fill=(52, 26, 16), shade=(38, 18, 10), deep=(26, 12, 7),
                 line=(255, 222, 40), outline=(255, 245, 140), ink=(255, 245, 140),
                 paper=(80, 44, 26), cavity=(18, 8, 4), back_line=(150, 110, 30)),
    "sunny": dict(fill=(70, 205, 255), shade=(45, 165, 228), deep=(30, 128, 196),
                  line=(40, 52, 76), outline=(28, 36, 56), ink=(26, 30, 42),
                  paper=(250, 252, 255), cavity=(40, 30, 70), back_line=(60, 120, 170)),
}
# Legacy style keys from the previous UI.
STYLE_ALIASES = {"white": "chalk", "dark": "ink", "cyber": "neon"}


def resolve_style(style: str) -> str:
    s = STYLE_ALIASES.get(style, style)
    return s if s in PALETTES else "ink"


class ScribbleSphere3DEngine:
    """
    Solid 3D scribble helmet.

    Rendering (all per-pixel work is confined to the helmet's bounding box):
      1. Silhouette: every mesh quad (front and back) is filled into a mask, so the
         covered area is exactly the projected solid — no gaps, ever. An optional
         "privacy core" (the live face hull, dilated) is unioned in so the face stays
         covered even if the helmet pose lags a fast head move by a frame.
      2. Body: cel-style shading from a camera-space key light (per-quad Lambert ->
         low-res shade layer, painter-sorted, blurred) blended between the style's
         fill/shade/deep tones. This is what makes it read as a 3D object.
      3. Scribble strokes: a tangle of long, smooth ballpoint loops wrapped on the surface.
         Their density follows the lighting (sparse on the lit side, filled in on the shadow
         side) so the shading itself reads as pen scribble. Each vertex is visibility-tested
         against the surface normal so strokes disappear exactly at the silhouette as they
         wrap around the back of the head (no see-through).
      4. Hand-drawn outline traced from the silhouette with boil jitter.

    Shapes: Head, Cube, Star, Diamond, Pill. Features (eyes/mouth) are placed on the
    front surface via `front_surface()`, which ray-casts a precomputed depth map.
    """

    STYLE_WHITE = "chalk"
    STYLE_DARK = "ink"
    STYLE_CYBER = "neon"

    SHAPE_HEAD = "head"
    SHAPE_CUBE = "cube"
    SHAPE_STAR = "star"
    SHAPE_DIAMOND = "diamond"
    SHAPE_PILL = "pill"
    ALL_SHAPES = [SHAPE_HEAD, SHAPE_CUBE, SHAPE_STAR, SHAPE_DIAMOND, SHAPE_PILL]

    # Key light in camera space, pointing towards the light (up-left, in front).
    LIGHT = np.array([-0.45, -0.6, -0.66]) / np.linalg.norm([-0.45, -0.6, -0.66])

    def __init__(self, semi_width_x: float = 14.5, semi_height_y: float = 18.5, semi_depth_z: float = 15.5,
                 center_offset_x: float = 0.0, center_offset_y: float = 1.0, center_offset_z: float = -0.5,
                 line_thickness: int = 3, density: int = 90, cavity_darkness: float = 0.85,
                 shape: str = SHAPE_HEAD):
        self.base_rx = float(semi_width_x)
        self.base_ry = float(semi_height_y)
        self.base_rz = float(semi_depth_z)
        self.scale_factor = 1.0
        self.offset = np.array([center_offset_x, center_offset_y, center_offset_z], dtype=np.float64)
        self.offset_x, self.offset_y, self.offset_z = self.offset
        self.line_thickness = int(line_thickness)
        self.density = int(density)
        self.cavity_darkness = float(cavity_darkness)
        self.shading = 0.75
        self.fill_opacity = 1.0
        self.show_outline = True
        # Face clearing (center_y, half_width, half_height as fractions of rx/ry): the
        # scribble strokes stop around the drawn face so expressions read clearly.
        self.face_clear = (0.14, 0.58, 0.62)
        # Anatomical head core (skull frame, NOT scaled with the helmet): a head-with-hair
        # ellipsoid that is always part of the silhouette, so no shape (star notches,
        # diamond corners) or small helmet size can ever uncover the real head.
        self.core_scale = 1.0
        u = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        v = np.linspace(-np.pi / 2, np.pi / 2, 12)
        uu, vv = np.meshgrid(u, v)
        self._core_pts = np.column_stack([9.0 * (np.cos(vv) * np.sin(uu)).ravel(),
                                          12.5 * np.sin(vv).ravel(),
                                          -11.5 * (np.cos(vv) * np.cos(uu)).ravel()])
        self._core_center = np.array([0.0, -1.5, 0.0])
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self.start_time = time.time()
        self.current_shape = shape if shape in self.ALL_SHAPES else self.SHAPE_HEAD
        self._build(self.current_shape)
        self.last_sil_bbox = None

    # ------------------------------------------------------------------ settings
    def set_shape(self, shape_name: str):
        s = (shape_name or "").lower().strip()
        if s in self.ALL_SHAPES and s != self.current_shape:
            self.current_shape = s
            self._build(s)

    def set_density(self, density: int):
        """10..84 (the UI's "Scribbles" slider): how much of the pen tangle is drawn."""
        self.density = int(np.clip(density, 10, 84))

    def set_line_thickness(self, thickness: int):
        self.line_thickness = max(1, min(8, int(thickness)))

    def set_offsets(self, offset_x=None, offset_y=None, offset_z=None):
        if offset_x is not None:
            self.offset_x = float(offset_x)
        if offset_y is not None:
            self.offset_y = float(offset_y)
        if offset_z is not None:
            self.offset_z = float(offset_z)
        self.offset = np.array([self.offset_x, self.offset_y, self.offset_z], dtype=np.float64)

    def set_scale(self, scale_ratio: float):
        self.scale_factor = max(0.4, float(scale_ratio))

    def set_cavity_darkness(self, darkness: float):
        self.cavity_darkness = float(np.clip(darkness, 0.0, 1.0))

    # ------------------------------------------------------------------ geometry
    def _map_point_to_shape(self, theta, phi, shape, radial_jitter=0.0):
        """Maps spherical params (theta in [-pi, pi], phi in [-pi/2, pi/2]) onto the helmet surface."""
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        rx, ry, rz = self.base_rx, self.base_ry, self.base_rz
        if shape == self.SHAPE_CUBE:
            dx, dy, dz = cos_p * sin_t, sin_p, -cos_p * cos_t
            m = max(abs(dx), abs(dy), abs(dz), 1e-4)
            blend = 0.88 / m + 0.12 * np.linalg.norm([dx, dy, dz])
            x, y, z = rx * dx * blend * 0.9, ry * dy * blend * 0.82, rz * dz * blend * 0.9
        elif shape == self.SHAPE_STAR:
            # Puffy 5-point star (a plush star cushion): the front silhouette is a star,
            # radius modulated by the angle around the viewing axis; depth stays rounded.
            x0, y0, z0 = cos_p * sin_t, sin_p, -cos_p * cos_t
            ang = np.arctan2(y0, x0) + np.pi / 2          # 0 = pointing straight up
            c = 0.5 + 0.5 * np.cos(5.0 * ang)
            f = 0.70 + 0.55 * c ** 1.6
            x, y = rx * 1.05 * x0 * f, ry * 0.92 * y0 * f
            z = rz * 0.85 * z0 * (0.75 + 0.25 * f)
        elif shape == self.SHAPE_DIAMOND:
            dx, dy, dz = cos_p * sin_t, sin_p, cos_p * cos_t
            f = 1.0 / max(1e-4, abs(dx) + abs(dy) + abs(dz)) * 1.3
            x, y, z = rx * dx * f, ry * dy * f, -rz * dz * f
        elif shape == self.SHAPE_PILL:
            # Capsule: straight sides (|u| <= s) joined continuously to elliptical caps.
            u = phi / (np.pi / 2)
            s, half_len = 0.4, 0.45 * ry
            if abs(u) <= s:
                y, rf = u / s * half_len, 1.0
            else:
                a = (abs(u) - s) / (1.0 - s) * (np.pi / 2)
                y, rf = np.sign(u) * (half_len + (ry * 1.05 - half_len) * np.sin(a)), np.cos(a)
            x, z = rx * 0.95 * sin_t * rf, -rz * 0.95 * cos_t * rf
        else:  # anatomical head
            x, y, z = rx * cos_p * sin_t, ry * sin_p, -rz * cos_p * cos_t
            if z > 0:
                x *= 1.0 + 0.10 * (z / rz)
                if y < 0:
                    y *= 1.05
            elif y > 2.0:
                chin = min(1.0, (y - 2.0) / max(1.0, ry - 2.0))
                x *= 1.0 - 0.20 * chin
                z *= 1.0 - 0.12 * chin
            if y > 5.0 and z > 0:
                z -= 3.0 * min(1.0, (y - 5.0) / 10.0)
        if radial_jitter:
            n = np.sqrt(x * x + y * y + z * z)
            if n > 1e-3:
                x, y, z = x + x / n * radial_jitter, y + y / n * radial_jitter, z + z / n * radial_jitter
        return np.array([x, y, z], dtype=np.float64)

    def _build(self, shape):
        self._build_mesh(shape)
        self._build_shape_curves(shape, seed=2026)
        self._build_depth_map()
        # Per-stroke-vertex normals from the nearest mesh vertex (for visibility culling).
        _, nn = cKDTree(self.mesh_v).query(self.all_base_pts)
        self.stroke_normals = self.mesh_vn[nn]

    def _build_mesh(self, shape, n_theta=36, n_phi=21):
        thetas = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
        phis = np.linspace(-np.pi / 2, np.pi / 2, n_phi)
        V = np.array([[self._map_point_to_shape(t, p, shape) for t in thetas] for p in phis]).reshape(-1, 3)
        quads = []
        for i in range(n_phi - 1):
            for j in range(n_theta):
                j2 = (j + 1) % n_theta
                quads.append((i * n_theta + j, i * n_theta + j2, (i + 1) * n_theta + j2, (i + 1) * n_theta + j))
        Q = np.array(quads, dtype=np.int32)
        p = V[Q]
        n = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
        c = p.mean(axis=1)
        center = V.mean(axis=0)
        flip = np.einsum("ij,ij->i", n, c - center) < 0
        n[flip] *= -1
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        vn = np.zeros_like(V)
        for k in range(4):
            np.add.at(vn, Q[:, k], n)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
        self.mesh_v, self.mesh_q, self.mesh_n, self.mesh_c, self.mesh_vn = V, Q, n, c, vn

    def _build_depth_map(self, res=96):
        """Front-surface depth z(x, y) over the helmet's frontal footprint (for placing features)."""
        th = np.linspace(-np.pi * 0.62, np.pi * 0.62, 120)
        ph = np.linspace(-np.pi * 0.49, np.pi * 0.49, 120)
        pts = np.array([self._map_point_to_shape(t, p, self.current_shape) for p in ph for t in th])
        xr = max(np.abs(pts[:, 0]).max(), 1e-3) * 1.02
        yr = max(np.abs(pts[:, 1]).max(), 1e-3) * 1.02
        zmap = np.full((res, res), np.nan, dtype=np.float32)
        ix = np.clip(((pts[:, 0] / xr + 1) * 0.5 * (res - 1)).round().astype(int), 0, res - 1)
        iy = np.clip(((pts[:, 1] / yr + 1) * 0.5 * (res - 1)).round().astype(int), 0, res - 1)
        for x_, y_, z_ in zip(ix, iy, pts[:, 2]):
            if np.isnan(zmap[y_, x_]) or z_ < zmap[y_, x_]:
                zmap[y_, x_] = z_
        # Fill holes by repeated neighbour averaging.
        for _ in range(res):
            nanm = np.isnan(zmap)
            if not nanm.any():
                break
            filled = np.where(nanm, 0, zmap)
            cnt = cv2.blur((~nanm).astype(np.float32), (3, 3))
            avg = cv2.blur(filled, (3, 3)) / np.maximum(cnt, 1e-6)
            zmap = np.where(nanm & (cnt > 0), avg, zmap).astype(np.float32)
        zmap = cv2.GaussianBlur(zmap, (5, 5), 0)
        self._zmap, self._zmap_xr, self._zmap_yr, self._zmap_res = zmap, xr, yr, res
        gy, gx = np.gradient(zmap)
        self._zgx = gx / (2.0 * xr / (res - 1))
        self._zgy = gy / (2.0 * yr / (res - 1))

    def front_surface(self, xy: np.ndarray, lift: float = 0.25):
        """
        xy: (N, 2) helmet-space (unscaled) x/y. Returns (pts (N,3), normals (N,3)) on the
        front surface, lifted `lift` cm outward so strokes sit on top of the body.
        """
        xy = np.asarray(xy, dtype=np.float64)
        res = self._zmap_res
        u = np.clip((xy[:, 0] / self._zmap_xr + 1) * 0.5 * (res - 1), 0, res - 1).astype(np.float32)
        v = np.clip((xy[:, 1] / self._zmap_yr + 1) * 0.5 * (res - 1), 0, res - 1).astype(np.float32)
        um, vm = u.reshape(1, -1), v.reshape(1, -1)
        z = cv2.remap(self._zmap, um, vm, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
        dzx = cv2.remap(self._zgx.astype(np.float32), um, vm, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
        dzy = cv2.remap(self._zgy.astype(np.float32), um, vm, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
        n = np.stack([dzx, dzy, -np.ones_like(dzx)], axis=1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        pts = np.column_stack([xy[:, 0], xy[:, 1], z]) + n * lift
        return pts, n

    # Pen-tangle scribble: how many strokes, points per stroke and the step between points
    # (radians on the unit sphere that is then mapped onto the shape).
    N_STROKES = 230
    STROKE_PTS = 76
    STROKE_STEP = 0.045

    def _build_shape_curves(self, shape, seed=2026):
        """
        The scribble texture: long, smooth ballpoint loops tangled over the whole surface —
        the "scribble ball" look (not short wavy squiggles). Each stroke is a curvature-driven
        walk on the unit sphere: it circles with a loop radius of ~2-4 cm whose curvature
        slowly swings (sometimes reversing), so consecutive loops drift apart and cross each
        other like a real pen scribble. Directions are then mapped onto the helmet shape.

        Every stroke gets a random `rank`; at render time a vertex is inked only where
        rank < local tone (see _stroke_threshold), so the lit side stays light and the
        shadow side fills in with scribbles — shading made of pen lines.
        """
        rng = np.random.RandomState(seed)
        n, m, ds = self.N_STROKES, self.STROKE_PTS, self.STROKE_STEP
        # Start points evenly spread (Fibonacci sphere) so every side of the head gets ink.
        k = np.arange(n) + 0.5
        zz = 1.0 - 2.0 * k / n
        az = np.pi * (1.0 + 5 ** 0.5) * k
        starts = np.column_stack([np.sqrt(1 - zz * zz) * np.cos(az), zz, np.sqrt(1 - zz * zz) * np.sin(az)])
        starts = starts[rng.permutation(n)]   # random order: the density slider thins evenly
        curves = []
        for s0 in starts:
            p = s0 + rng.normal(0.0, 0.08, 3)
            p /= np.linalg.norm(p)
            a = rng.normal(size=3)
            u = a - a.dot(p) * p
            u /= np.linalg.norm(u)
            r_loop = rng.uniform(0.12, 0.34)
            spin = rng.choice([-1.0, 1.0])
            # Curvature swings about once per loop (egg-shaped, precessing loops) plus a
            # slow random drift — so the loops wander and cross instead of stacking into rings.
            ph, wf, wa = rng.uniform(0, 2 * np.pi), rng.uniform(0.12, 0.30), rng.uniform(0.4, 1.1)
            drift = 0.0
            dirs = np.empty((m, 3))
            for j in range(m):
                dirs[j] = p
                drift = 0.9 * drift + rng.normal(0.0, 0.45)
                ang = (spin * (1.0 + wa * np.sin(wf * j + ph)) + drift) / r_loop * ds
                w = np.cross(p, u)
                u = u * np.cos(ang) + w * np.sin(ang)
                p = p + u * ds
                p /= np.linalg.norm(p)
                u = u - u.dot(p) * p
                u /= np.linalg.norm(u)
            phi = np.arcsin(np.clip(dirs[:, 1], -1.0, 1.0))
            theta = np.arctan2(dirs[:, 0], -dirs[:, 2])
            curves.append(np.array([self._map_point_to_shape(t, f, shape, radial_jitter=0.2)
                                    for t, f in zip(theta, phi)]))
        self.all_base_pts = np.vstack(curves)
        self.curve_offsets = np.arange(n + 1) * m
        self.stroke_id = np.repeat(np.arange(n), m)
        self.stroke_s = np.tile(np.linspace(-0.5, 0.5, m), n)[:, None]
        self.stroke_rank = rng.uniform(0.0, 1.0, n)
        self.stroke_heavy = rng.uniform(0.0, 1.0, n) < 0.35          # a third drawn with more pressure
        self.curve_thicks = list(np.where(self.stroke_heavy, 2, 1))
        # Line boil = each stroke re-drawn a hair off (shift + slight shear), not wiggled.
        rng_noise = np.random.RandomState(999)
        self._boil_a = [rng_noise.normal(0.0, 1.0, (n, 3)) for _ in range(12)]
        self._boil_b = [rng_noise.normal(0.0, 1.0, (n, 3)) for _ in range(12)]
        self._outline_noise = [rng_noise.normal(0.0, 1.0, 512) for _ in range(24)]
        self._fz_key = None

    def get_evolved_points(self, t_sec, sketch_fps=12.0, sketch_noise=0.85):
        pts = self.all_base_pts
        if sketch_noise > 0.01:
            step = int(np.floor(t_sec * sketch_fps)) if sketch_fps > 0 else 0
            a = self._boil_a[step % len(self._boil_a)][self.stroke_id]
            b = self._boil_b[step % len(self._boil_b)][self.stroke_id]
            pts = pts + (a + b * self.stroke_s) * (0.2 * sketch_noise)
        return pts * self.scale_factor + self.offset

    def _face_zone_weight(self, n_pts):
        """0..1 per stroke point: how deep inside the face clearing it is (1 = keep clear)."""
        cy, ax, ay = self.face_clear
        key = (n_pts, cy, ax, ay, self.current_shape)
        if getattr(self, '_fz_key', None) != key:
            p = self.all_base_pts[:n_pts]
            r = np.sqrt((p[:, 0] / (ax * self.base_rx)) ** 2 + ((p[:, 1] - cy * self.base_ry) / (ay * self.base_ry)) ** 2)
            w = np.clip((1.05 - r) / 0.35, 0.0, 1.0)
            w = w * w * (3.0 - 2.0 * w)
            self._fz = np.where(p[:, 2] < -0.35 * self.base_rz, w, 0.0)
            self._fz_key = key
        return self._fz

    def _stroke_threshold(self, ncam, pc, n_pts):
        """
        Per stroke point, the fraction of strokes that are inked there: sparse where the
        key light hits, dense in shadow and towards the silhouette, none in the middle of
        the face. `shading` blends between that and an even texture; `density` scales it.
        """
        lam = np.clip(ncam @ self.LIGHT, 0.0, 1.0)
        facing = np.clip(-np.einsum("ij,ij->i", ncam, pc) / np.linalg.norm(pc, axis=1), 0.0, 1.0)

        def smooth(e0, e1, x):
            tt = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
            return tt * tt * (3.0 - 2.0 * tt)
        tone = np.clip(0.85 * smooth(0.30, 0.95, 1.0 - lam) + 0.6 * smooth(0.55, 0.15, facing), 0.0, 1.0)
        w = float(np.clip(self.shading, 0.0, 1.0))
        level = (1.0 - w) * 0.55 + w * (0.22 + 0.78 * tone)
        dens = float(np.clip(self.density / 84.0, 0.05, 1.0))
        return dens * level * (1.0 - self._face_zone_weight(n_pts))

    # ------------------------------------------------------------------ projection
    def to_camera(self, pts_helmet, R, tvec):
        """Helmet-space (unscaled) points -> camera-space points."""
        p = pts_helmet * self.scale_factor + self.offset
        return p @ R.T + tvec.reshape(1, 3)

    @staticmethod
    def project_cam(pc, cam):
        z = np.maximum(pc[:, 2], 1e-3)
        return np.column_stack([cam[0, 0] * pc[:, 0] / z + cam[0, 2], cam[1, 1] * pc[:, 1] / z + cam[1, 2]])

    # ------------------------------------------------------------------ render
    def apply(self, frame_bgr, tracking_data, t_sec=None, style="ink", custom_color=None,
              cavity_darkness=None, line_width=None, enable_boil=True, sketch_fps=12.0,
              sketch_noise=0.85, hand_mask=None, privacy_hull=None):
        if not tracking_data.get('has_face', False):
            return frame_bgr
        rvec, tvec, cam = tracking_data.get('rvec'), tracking_data.get('tvec'), tracking_data.get('cam_matrix')
        if rvec is None or tvec is None or cam is None:
            return frame_bgr
        H, W = frame_bgr.shape[:2]
        pal = PALETTES[resolve_style(style)]
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        elapsed = t_sec if t_sec is not None else (time.time() - self.start_time)
        px_scale = max(1.0, H / 720.0) if H < W else max(1.0, W / 720.0)
        lw = max(1, int(round((line_width or self.line_thickness) * px_scale)))

        # --- mesh projection
        vc = self.to_camera(self.mesh_v, R, t)
        if (vc[:, 2] < 1.0).any():
            return frame_bgr  # helmet crosses the camera plane — nothing sensible to draw
        uv = self.project_cam(vc, cam)
        core_pc = (self._core_pts * self.core_scale + self._core_center) @ R.T + t
        core_uv = self.project_cam(core_pc, cam) if (core_pc[:, 2] > 1.0).all() else None
        extra = [a for a in (core_uv, privacy_hull) if a is not None and len(a)]
        allpts = np.vstack([uv] + [np.asarray(a, np.float64) for a in extra])
        x0, y0 = np.floor(allpts.min(axis=0)).astype(int) - 4 * lw
        x1, y1 = np.ceil(allpts.max(axis=0)).astype(int) + 4 * lw
        rx0, ry0, rx1, ry1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if rx1 - rx0 < 2 or ry1 - ry0 < 2:
            return frame_bgr
        self.last_sil_bbox = (rx0, ry0, rx1, ry1)
        rw, rh = rx1 - rx0, ry1 - ry0
        roi = frame_bgr[ry0:ry1, rx0:rx1]

        SH = 2  # fixed-point subpixel bits for polygon rasterization
        quads_px = np.round((uv[self.mesh_q] - (rx0, ry0)) * (1 << SH)).astype(np.int32)

        # --- 1. silhouette = union of every quad that can touch the visible solid, plus
        # the privacy core. Filled one convex quad at a time: a single multi-polygon
        # fillPoly uses even-odd filling, where overlapping front/back quads cancel out
        # and leave holes — unacceptable for something whose job is to hide a face.
        nc = self.mesh_n @ R.T
        cc = self.to_camera(self.mesh_c, R, t)
        cos_view = np.einsum("ij,ij->i", nc, cc) / np.linalg.norm(cc, axis=1)
        sil = np.zeros((rh, rw), np.uint8)
        # Every quad, front and back: culling by quad-centre normal can drop a sliver
        # of a creased quad at the silhouette, and a sliver is enough to show skin.
        for q in quads_px:
            cv2.fillConvexPoly(sil, q, 255, cv2.LINE_8, SH)
        if core_uv is not None:
            cp = np.round((core_uv - (rx0, ry0)) * (1 << SH)).astype(np.int32)
            cv2.fillConvexPoly(sil, cv2.convexHull(cp), 255, cv2.LINE_8, SH)
        if privacy_hull is not None and len(privacy_hull) >= 3:
            ph = np.round((np.asarray(privacy_hull, np.float64) - (rx0, ry0)) * (1 << SH)).astype(np.int32)
            cv2.fillConvexPoly(sil, cv2.convexHull(ph), 255, cv2.LINE_8, SH)
        self.last_sil = sil

        # --- 2. shaded body
        facing = cos_view < 0
        body = self._shade_body(rw, rh, quads_px, nc, cc, facing, pal, SH)
        if self.fill_opacity >= 0.995:
            cv2.copyTo(body, sil, roi)
        else:
            mixed = cv2.addWeighted(body, self.fill_opacity, roi, 1.0 - self.fill_opacity, 0.0)
            cv2.copyTo(mixed, sil, roi)

        # --- 3. scribble strokes: front-facing, and only as many as the local tone asks for
        if enable_boil:
            pts_w = self.get_evolved_points(elapsed, sketch_fps=sketch_fps, sketch_noise=sketch_noise)
        else:
            pts_w = self.all_base_pts * self.scale_factor + self.offset
        n_pts = len(pts_w)
        pc = pts_w @ R.T + t
        spx = np.round(self.project_cam(pc, cam) * 4).astype(np.int32)
        ncam = self.stroke_normals @ R.T
        ink = np.einsum("ij,ij->i", ncam, pc) < -0.02 * np.linalg.norm(pc, axis=1)
        ink &= self.stroke_rank[self.stroke_id] < self._stroke_threshold(ncam, pc, n_pts)
        if custom_color is not None:
            line_col = custom_color
        else:
            line_col = tuple(int(0.85 * a + 0.15 * b) for a, b in zip(pal['line'], pal['fill']))
        # Runs break wherever inking flips or a new stroke starts.
        brk = np.zeros(n_pts, bool)
        brk[1:] = ink[1:] != ink[:-1]
        brk[self.curve_offsets[1:-1]] = True
        starts = np.flatnonzero(brk)
        ends = np.append(starts[1:], n_pts)
        keep = ink[starts] & (ends - starts >= 2)
        light, heavy = [], []
        for s0, e0 in zip(starts[keep], ends[keep]):
            (heavy if self.stroke_heavy[self.stroke_id[s0]] else light).append(spx[s0:e0].reshape(-1, 1, 2))
        # Pen width follows Line Width but stays pen-thin: dense tangles need fine lines.
        for polys, width in ((light, max(1, int(lw * 0.25))), (heavy, max(1, int(round(lw * 0.42))))):
            if polys:
                cv2.polylines(frame_bgr, polys, False, line_col, width, cv2.LINE_AA, 2)

        # --- 4. outline
        if self.show_outline:
            self._draw_outline(frame_bgr, sil, rx0, ry0, pal['outline'], lw, elapsed, sketch_fps, enable_boil)
        return frame_bgr

    def _palette_lut(self, pal):
        key = (tuple(pal['fill']), tuple(pal['shade']), tuple(pal['deep']), round(self.shading, 3))
        if getattr(self, '_lut_key', None) != key:
            s = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, None] * self.shading
            fill, shade, deep = (np.array(pal[k], np.float32)[None, :] for k in ('fill', 'shade', 'deep'))
            s1 = np.clip(s * 2.0, 0.0, 1.0)
            s2 = np.clip(s * 2.0 - 1.0, 0.0, 1.0)
            lut = fill + (shade - fill) * s1 + (deep - shade) * s2
            self._lut = np.clip(lut, 0, 255).astype(np.uint8).reshape(1, 256, 3)
            self._lut_key = key
        return self._lut

    def _shade_body(self, rw, rh, quads_px, nc, cc, facing, pal, SH):
        """
        Smooth toon shading. Quad normals are rasterized (painter-sorted, far -> near) into
        a low-res normal map, blurred so they interpolate smoothly across quads (Phong-like),
        then lit per pixel: key light + a darker rim where the surface turns away. The shade
        amount indexes a palette LUT (fill -> shade -> deep).
        """
        if self.shading <= 0.01:
            body = np.empty((rh, rw, 3), np.uint8)
            body[:] = pal['fill']
            return body
        f_idx = np.flatnonzero(facing)
        order = f_idx[np.argsort(-cc[f_idx, 2])]
        ds = 3
        lw_, lh_ = max(2, rw // ds), max(2, rh // ds)
        nmap = np.zeros((lh_, lw_, 3), np.uint8)
        cols = np.clip(nc[order] * 127.0 + 128.0, 0, 255).astype(np.int32).tolist()
        q_small = quads_px[order] // ds
        for q, col in zip(q_small, cols):
            cv2.fillConvexPoly(nmap, q, col, cv2.LINE_8, SH)
        kb = max(3, (min(lw_, lh_) // 7) | 1)
        n = cv2.GaussianBlur(nmap, (kb, kb), 0).astype(np.float32) * (1.0 / 127.0) - (128.0 / 127.0)
        n /= np.maximum(np.sqrt((n * n).sum(axis=2, keepdims=True)), 1e-3)
        lam = np.clip(n @ self.LIGHT.astype(np.float32), 0.0, 1.0)
        facing_amt = np.clip(-n[..., 2], 0.0, 1.0)

        def smooth(e0, e1, x):
            tt = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
            return tt * tt * (3.0 - 2.0 * tt)
        dark = 1.0 - lam
        s = 0.5 * smooth(0.38, 0.52, dark) + 0.5 * smooth(0.72, 0.84, dark)
        s = np.maximum(s, 0.85 * smooth(0.42, 0.18, facing_amt))
        s_u8 = cv2.resize((s * 255.0).astype(np.uint8), (rw, rh), interpolation=cv2.INTER_LINEAR)
        return cv2.LUT(cv2.merge([s_u8, s_u8, s_u8]), self._palette_lut(pal))

    def _draw_outline(self, frame, sil, rx0, ry0, color, lw, t_sec, fps, boil):
        cnts, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, max(1.0, peri * 0.004), True).reshape(-1, 2).astype(np.float32)
        if len(c) < 3:
            return
        # Resample evenly so the jitter is uniform along the contour.
        seg = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        n = int(np.clip(peri / 9.0, 24, 400))
        s = np.linspace(0, cum[-1], n, endpoint=False)
        pts = np.column_stack([np.interp(s, cum, np.append(c[:, 0], c[0, 0])), np.interp(s, cum, np.append(c[:, 1], c[0, 1]))])
        step = int(np.floor(t_sec * fps)) if boil and fps > 0 else 0
        noise = self._outline_noise[step % len(self._outline_noise)]
        cen = pts.mean(axis=0)
        d = pts - cen
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        amp = max(0.6, lw * 0.45)
        k = np.arange(n) % len(noise)
        p1 = pts + d * (noise[k] * amp)[:, None] + (rx0, ry0)
        p2 = pts + d * (noise[(k + 97) % len(noise)] * amp - lw * 0.6)[:, None] + (rx0, ry0)
        cv2.polylines(frame, [np.round(p1 * 4).astype(np.int32).reshape(-1, 1, 2)], True, color, max(1, int(lw * 1.2)), cv2.LINE_AA, 2)
        cv2.polylines(frame, [np.round(p2 * 4).astype(np.int32).reshape(-1, 1, 2)], True, color, max(1, lw // 2), cv2.LINE_AA, 2)
