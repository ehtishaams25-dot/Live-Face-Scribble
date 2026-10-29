"""
Face geometry helpers shared by the tracker and the expression renderer.

Everything here lives in the "skull frame" used by the whole app (metric cm):
    origin = midpoint between the ear canals
    +X = screen right (subject's left), +Y = down (towards chin),
    -Z = forward (towards the nose / the camera when facing it).

MediaPipe's canonical face model (468 metric vertices, Apache-2.0) is loaded once
and rigidly aligned into this frame, which gives us:
  * ~40 rigid 3D anchor points for PnP head pose (vs. 12 hand-placed points before),
    so the pose is far less jittery and needs less smoothing -> less lag.
  * a reference shape for pose-normalizing the live 3D landmarks, so the
    expression renderer sees lips / eyelids / brows independent of head rotation.
"""
import os
import numpy as np


# Hand-placed anthropometric anchors (the original rig). Used to define the skull
# frame the helmet is authored in, and to align MediaPipe's canonical model into it.
SKULL_ANCHORS = {
    1:   np.array([0.0,   1.2, -11.2]),   # nose tip
    168: np.array([0.0,  -1.8,  -8.5]),   # nose bridge
    10:  np.array([0.0,  -9.5,  -7.2]),   # upper forehead
    151: np.array([0.0,  -7.2,  -6.8]),   # forehead mid
    263: np.array([3.6,  -3.2,  -7.0]),   # left eye outer (screen right)
    362: np.array([1.6,  -3.2,  -7.5]),   # left eye inner
    33:  np.array([-3.6, -3.2,  -7.0]),   # right eye outer
    133: np.array([-1.6, -3.2,  -7.5]),   # right eye inner
    454: np.array([7.8,   0.0,   0.0]),   # left tragus region
    234: np.array([-7.8,  0.0,   0.0]),   # right tragus region
}

# Landmarks that barely move with speech/expressions (forehead, nose, cheekbones,
# temples, eye corners). Lips, jaw, eyelids and brows are deliberately excluded so
# talking never wobbles the head pose.
RIGID_IDX = np.array([
    10, 151, 9, 8, 108, 337, 109, 338, 67, 297,              # forehead
    168, 6, 197, 195, 5, 4, 1, 98, 327, 48, 278, 64, 294,    # nose
    234, 454, 127, 356, 227, 447, 93, 323, 116, 345,         # temples / cheekbones
    117, 346, 123, 352, 50, 280, 34, 264, 143, 372,
    33, 133, 263, 362, 130, 359,                             # eye corners
], dtype=np.int32)

# ---- Feature landmark groups (MediaPipe face mesh indices) -------------------
# Lips, ordered left-corner -> right-corner in image space for each contour.
LIPS_OUTER_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIPS_OUTER_LOWER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LIPS_INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIPS_INNER_LOWER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# Eyes. "R" = subject's right eye (image left, around 33/133), "L" = subject's left.
EYE_R = {'outer': 33, 'inner': 133,
         'upper': [246, 161, 160, 159, 158, 157, 173],
         'lower': [7, 163, 144, 145, 153, 154, 155],
         'iris': [468, 469, 470, 471, 472]}
EYE_L = {'outer': 263, 'inner': 362,
         'upper': [466, 388, 387, 386, 385, 384, 398],
         'lower': [249, 390, 373, 374, 380, 381, 382],
         'iris': [473, 474, 475, 476, 477]}

# Brows, ordered outer -> inner. Upper contour carries the visible brow line.
BROW_R_UPPER = [70, 63, 105, 66, 107]
BROW_R_LOWER = [46, 53, 52, 65, 55]
BROW_L_UPPER = [300, 293, 334, 296, 336]
BROW_L_LOWER = [276, 283, 282, 295, 285]

# Face oval (for the privacy core / hull).
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
             378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
             162, 21, 54, 103, 67, 109]


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """
    Least-squares similarity transform (Umeyama 1991): dst ~= s * R @ src + t.
    src, dst: (N, 3). Returns (s, R, t). Guaranteed proper rotation (det R = +1).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    if with_scale:
        var_s = (xs ** 2).sum() / len(src)
        s = float(np.trace(np.diag(S) @ D) / max(var_s, 1e-12))
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def _load_canonical_skull() -> np.ndarray:
    """Loads MediaPipe's canonical face (468x3, cm) and aligns it into the skull frame."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "models", "canonical_face_model.npy")
    if not os.path.exists(path):
        return None
    v = np.load(path).astype(np.float64)
    # Canonical model is y-up / z-towards-viewer; skull frame is y-down / z-away.
    v = v * np.array([1.0, -1.0, -1.0])
    idx = list(SKULL_ANCHORS.keys())
    dst = np.array([SKULL_ANCHORS[i] for i in idx])
    s, R, t = umeyama(v[idx], dst, with_scale=True)
    return (s * (R @ v.T)).T + t


CANONICAL_SKULL = _load_canonical_skull()


def normalize_landmarks(pts_img3d: np.ndarray):
    """
    Pose-normalizes live MediaPipe landmarks.

    pts_img3d: (N>=468, 3) landmarks as (x*W, y*H, z*W) pixels.
    Returns (pts_face, scale_px_per_cm) where pts_face is (N, 3) in skull-frame cm
    with head rotation/translation/scale removed — i.e. how the face looks when
    viewed straight on — so lip/eyelid/brow geometry is comparable across poses.
    """
    if CANONICAL_SKULL is None or pts_img3d is None or len(pts_img3d) < 468:
        return None, None
    s, R, t = umeyama(CANONICAL_SKULL[RIGID_IDX], pts_img3d[RIGID_IDX], with_scale=True)
    if s <= 1e-6:
        return None, None
    pts_face = ((pts_img3d - t) @ R) / s
    return pts_face, s
