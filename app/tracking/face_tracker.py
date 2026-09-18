import os
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial.transform import Rotation
from app.tracking.temporal_smoother import KinematicHeadPoseTracker


# ==============================================================================
# Anthropometric 3D Skull Rig Model (Metric centimeters)
# Coordinate Frame:
# Origin (0,0,0): Skull Center (midpoint between left and right auditory canals/tragus)
#   +X: Screen Right (Subject's Left)
#   -X: Screen Left  (Subject's Right)
#   +Y: Inferior Down (Towards Chin / Jaw)
#   -Y: Cranial Up    (Towards Crown / Vertex)
#   -Z: Anterior Forward (Towards Nose / Face Front)
#   +Z: Posterior Backward (Towards Occiput / Back of Head)
# When facing camera directly, Rotation matrix = Identity, Euler = (0, 0, 0).
# ==============================================================================

CANONICAL_POINTS_3D = {
    'nose_tip':         np.array([ 0.0,   1.2, -11.2], dtype=np.float64),
    'nose_bridge':      np.array([ 0.0,  -1.8,  -8.5], dtype=np.float64),
    'glabella':         np.array([ 0.0,  -4.5,  -7.8], dtype=np.float64),
    'forehead_mid':     np.array([ 0.0,  -7.2,  -6.8], dtype=np.float64),
    'eye_left_outer':   np.array([ 3.6,  -3.2,  -7.0], dtype=np.float64),  # Screen right
    'eye_left_inner':   np.array([ 1.6,  -3.2,  -7.5], dtype=np.float64),
    'eye_left_pupil':   np.array([ 2.6,  -3.2,  -7.2], dtype=np.float64),
    'eye_right_outer':  np.array([-3.6,  -3.2,  -7.0], dtype=np.float64),  # Screen left
    'eye_right_inner':  np.array([-1.6,  -3.2,  -7.5], dtype=np.float64),
    'eye_right_pupil':  np.array([-2.6,  -3.2,  -7.2], dtype=np.float64),
    'ear_left_tragus':  np.array([ 7.8,   0.0,   0.0], dtype=np.float64),  # Screen right ear
    'ear_right_tragus': np.array([-7.8,   0.0,   0.0], dtype=np.float64),  # Screen left ear
    'zygoma_left':      np.array([ 4.8,  -0.8,  -5.5], dtype=np.float64),
    'zygoma_right':     np.array([-4.8,  -0.8,  -5.5], dtype=np.float64),
    'chin':             np.array([ 0.0,   9.8,  -7.2], dtype=np.float64),
    'mouth_center':     np.array([ 0.0,   4.8,  -7.0], dtype=np.float64),
    'mouth_left':       np.array([ 2.5,   4.8,  -6.8], dtype=np.float64),
    'mouth_right':      np.array([-2.5,   4.8,  -6.8], dtype=np.float64),
    'ear_left_helix':   np.array([ 8.4,  -2.5,   1.2], dtype=np.float64),
    'ear_right_helix':  np.array([-8.4,  -2.5,   1.2], dtype=np.float64),
    'ear_left_lobule':  np.array([ 8.0,   2.2,   0.5], dtype=np.float64),
    'ear_right_lobule': np.array([-8.0,   2.2,   0.5], dtype=np.float64),
    'back_of_head':     np.array([ 0.0,  -2.0,   9.5], dtype=np.float64),  # Occiput
    'crown':            np.array([ 0.0, -11.5,   1.0], dtype=np.float64),  # Top of skull
    'nape':             np.array([ 0.0,   8.5,   7.5], dtype=np.float64),  # Base of skull/neck
    'parietal_left':    np.array([ 6.2,  -8.0,   5.0], dtype=np.float64),  # Upper back left
    'parietal_right':   np.array([-6.2,  -8.0,   5.0], dtype=np.float64),  # Upper back right
}

# Rigid cranial landmarks in MediaPipe Face Mesh (IMMUTABLE TO SPEECH & EXPRESSIONS)
# Excludes movable jaw (chin 152) and lips (13) so head pose NEVER wobbles or scales when talking!
PNP_LANDMARK_MAP = [
    ('nose_tip', 1),
    ('nose_bridge', 168),
    ('glabella', 10),
    ('forehead_mid', 151),
    ('eye_left_outer', 263),
    ('eye_left_inner', 362),
    ('eye_right_outer', 33),
    ('eye_right_inner', 133),
    ('ear_left_tragus', 454),
    ('ear_right_tragus', 234),
    ('zygoma_left', 447),
    ('zygoma_right', 227),
]

PNP_OBJECT_POINTS = np.array([CANONICAL_POINTS_3D[name] for name, _ in PNP_LANDMARK_MAP], dtype=np.float64)

# Full 3D Cranium Bounding Geometry (16 anchor points encompassing the full 3D skull)
SKULL_HULL_3D = np.array([
    CANONICAL_POINTS_3D['chin'],
    CANONICAL_POINTS_3D['mouth_left'],
    CANONICAL_POINTS_3D['mouth_right'],
    CANONICAL_POINTS_3D['nose_tip'],
    CANONICAL_POINTS_3D['glabella'],
    CANONICAL_POINTS_3D['crown'],
    CANONICAL_POINTS_3D['ear_left_helix'],
    CANONICAL_POINTS_3D['ear_left_tragus'],
    CANONICAL_POINTS_3D['ear_left_lobule'],
    CANONICAL_POINTS_3D['ear_right_helix'],
    CANONICAL_POINTS_3D['ear_right_tragus'],
    CANONICAL_POINTS_3D['ear_right_lobule'],
    CANONICAL_POINTS_3D['parietal_left'],
    CANONICAL_POINTS_3D['parietal_right'],
    CANONICAL_POINTS_3D['back_of_head'],
    CANONICAL_POINTS_3D['nape']
], dtype=np.float64)


class FaceTracker:
    """
    High-Precision 3D Head Space & Zero-Leak Privacy Tracker.
    Features:
    - Calibrated Metric Perspective-n-Point (cv2.solvePnP) against anthropometric skull rig
    - Full 3D spatial coordinates: Eyes, Nose, Mouth, Left/Right Ears, and Back of Head
    - Zero-inversion Euler angles (Yaw, Pitch, Roll = 0° facing forward)
    - Dual Face + Pose Fusion: PoseLandmarker fallback on extreme yaw or face occlusion
    - 1-Euro adaptive temporal smoothing with kinematic momentum
    - Complete full-head 3D convex hull for zero-leak privacy obscuration
    """
    def __init__(self, face_model_path: str, pose_model_path: str = None, max_coast_frames: int = 45):
        self.face_model_path = face_model_path
        self.pose_model_path = pose_model_path
        self.kinematic_tracker = KinematicHeadPoseTracker(max_coast_frames=max_coast_frames)
        self.max_coast_frames = max_coast_frames

        if not os.path.exists(face_model_path):
            raise FileNotFoundError(f"FaceLandmarker model not found at: {face_model_path}")

        # 1. Initialize MediaPipe FaceLandmarker
        base_options_face = python.BaseOptions(model_asset_path=face_model_path)
        options_face = vision.FaceLandmarkerOptions(
            base_options=base_options_face,
            running_mode=vision.RunningMode.VIDEO,
            output_facial_transformation_matrixes=True,
            output_face_blendshapes=True,
            num_faces=1
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options_face)

        # 2. Initialize MediaPipe PoseLandmarker (if model provided)
        self.pose_landmarker = None
        if pose_model_path and os.path.exists(pose_model_path):
            try:
                base_options_pose = python.BaseOptions(model_asset_path=pose_model_path)
                options_pose = vision.PoseLandmarkerOptions(
                    base_options=base_options_pose,
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1
                )
                self.pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)
            except Exception as e:
                print(f"[FaceTracker] Warning: Could not initialize PoseLandmarker: {e}")

        self.last_timestamp_ms = -1
        self.last_face_size = 220
        self.last_lmks_2d = None
        self.last_rvec = None
        self.last_tvec = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int = None) -> dict:
        """
        Processes a BGR frame and returns precise 3D Head Space tracking data.
        """
        h, w = frame_bgr.shape[:2]
        now = time.time()

        if timestamp_ms is None or timestamp_ms <= self.last_timestamp_ms:
            self.last_timestamp_ms = self.last_timestamp_ms + 1 if self.last_timestamp_ms >= 0 else int(now * 1000)
            timestamp_ms = self.last_timestamp_ms
        else:
            self.last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # High-Speed Neural Proxy (640px wide proxy executes in 6-8ms vs 25ms at 1080p)
        if w > 640:
            pw = 640
            ph = int(h * 640 / w)
            rgb_proxy = cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_LINEAR)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_proxy)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Camera intrinsic matrix (focal length fx=fy=w, principal point at optical center)
        fx = float(w)
        fy = float(w)
        cx = float(w) / 2.0
        cy = float(h) / 2.0
        cam_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # 1. Run FaceLandmarker
        face_res = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

        tracking_mode = 'lost'
        rvec = None
        tvec = None
        lmks_2d = None
        blendshapes_dict = {}
        raw_conf = 0.0

        if face_res.face_landmarks and len(face_res.face_landmarks) > 0:
            # ------------------------------------------------------------------
            # Mode A: Dense Facial 3D Landmark Tracking
            # ------------------------------------------------------------------
            lmks = face_res.face_landmarks[0]
            lmks_2d = np.array([[l.x * w, l.y * h] for l in lmks], dtype=np.float32)
            self.last_lmks_2d = lmks_2d

            # Extract corresponding 2D points for 3D SolvePnP
            img_pts = np.array([[lmks[idx].x * w, lmks[idx].y * h] for _, idx in PNP_LANDMARK_MAP], dtype=np.float64)

            # Solve metric 6DoF Pose via Perspective-n-Point
            flags = cv2.SOLVEPNP_ITERATIVE if self.last_rvec is not None else cv2.SOLVEPNP_EPNP
            use_extrinsic_guess = self.last_rvec is not None
            r_guess = self.last_rvec.copy() if use_extrinsic_guess else None
            t_guess = self.last_tvec.copy() if use_extrinsic_guess else None

            success, r_pnp, t_pnp = cv2.solvePnP(
                PNP_OBJECT_POINTS,
                img_pts,
                cam_matrix,
                self.dist_coeffs,
                rvec=r_guess,
                tvec=t_guess,
                useExtrinsicGuess=use_extrinsic_guess,
                flags=flags
            )

            if success:
                tracking_mode = 'face_dense'
                raw_conf = 1.0
                rvec = r_pnp
                tvec = t_pnp
                self.last_rvec = rvec.copy()
                self.last_tvec = tvec.copy()

                # Update face size
                xs = lmks_2d[:, 0]
                ys = lmks_2d[:, 1]
                self.last_face_size = int(max(np.max(xs) - np.min(xs), np.max(ys) - np.min(ys)))

                # Blendshapes
                if face_res.face_blendshapes and len(face_res.face_blendshapes) > 0:
                    for cat in face_res.face_blendshapes[0]:
                        blendshapes_dict[cat.category_name] = cat.score

                # Multi-modal Tongue, Teeth, and Speech Viseme Estimator
                if lmks_2d is not None and len(lmks_2d) > 308:
                    pt13 = lmks_2d[13]  # Upper inner lip
                    pt14 = lmks_2d[14]  # Lower inner lip
                    pt17 = lmks_2d[17]  # Lower outer lip / chin border
                    pt61 = lmks_2d[61]  # Left mouth corner
                    pt291 = lmks_2d[291] # Right mouth corner

                    mouth_w = max(10.0, float(np.linalg.norm(pt291 - pt61)))
                    inner_open = max(0.0, float(pt14[1] - pt13[1]))
                    open_ratio = inner_open / mouth_w

                    jaw_bs = blendshapes_dict.get('jawOpen', 0.0)
                    smile_bs = max(blendshapes_dict.get('mouthSmileLeft', 0.0), blendshapes_dict.get('mouthSmileRight', 0.0))
                    upper_bs = max(blendshapes_dict.get('mouthUpperUpLeft', 0.0), blendshapes_dict.get('mouthUpperUpRight', 0.0))
                    lower_bs = max(blendshapes_dict.get('mouthLowerDownLeft', 0.0), blendshapes_dict.get('mouthLowerDownRight', 0.0))
                    funnel_bs = blendshapes_dict.get('mouthFunnel', 0.0)
                    pucker_bs = blendshapes_dict.get('mouthPucker', 0.0)
                    stretch_bs = max(blendshapes_dict.get('mouthStretchLeft', 0.0), blendshapes_dict.get('mouthStretchRight', 0.0))

                    # 1. Continuous speech viseme index (fluid speech tracking)
                    speech_score = float(np.clip(jaw_bs * 2.2 + funnel_bs * 1.5 + lower_bs * 1.2 + stretch_bs * 0.9, 0.0, 1.0))

                    # 2. Teeth showing score
                    teeth_score = 0.0
                    if upper_bs > 0.12 or (smile_bs > 0.18 and (jaw_bs > 0.04 or open_ratio > 0.04)):
                        teeth_score = float(np.clip(upper_bs * 1.5 + smile_bs * 0.85 + jaw_bs * 0.4, 0.0, 1.0))
                    elif jaw_bs > 0.15 and smile_bs > 0.12:
                        teeth_score = float(np.clip(smile_bs * 1.1 + jaw_bs * 0.6, 0.0, 1.0))

                    # 3. Tongue protrusion score (STRICT SPEECH IMMUNITY)
                    # Normal speech, talking, laughing, or mouth opening must NEVER trigger tongue protrusion!
                    tongue_score = 0.0
                    pt18 = lmks_2d[18]  # Chin groove directly below lower lip

                    # Lock out tongue completely whenever mouth is active for speech or expression
                    is_speech_or_talking = (speech_score > 0.08 or jaw_bs > 0.28 or funnel_bs > 0.12 or pucker_bs > 0.15 or teeth_score > 0.10)

                    if not is_speech_or_talking and open_ratio > 0.04:
                        # Sample STRICTLY BELOW lower lip vermilion border (Landmark 17 down into chin Landmark 18)
                        # During speech or smiling, this area is 100% chin skin or beard (zero tongue!).
                        # ONLY when the tongue physically curls down past the lower lip does tongue tissue cover this area!
                        ymin = int(pt17[1] + 2)
                        ymax = int(min(h, pt18[1] + (pt18[1] - pt17[1]) * 0.5))
                        xmin = int(max(0, pt17[0] - mouth_w * 0.16))
                        xmax = int(min(w, pt17[0] + mouth_w * 0.16))

                        if ymax > ymin + 3 and xmax > xmin + 3:
                            patch = frame_bgr[ymin:ymax, xmin:xmax]
                            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                            # True tongue tissue: saturated red / pink
                            mask1 = (hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 165)
                            mask2 = (hsv[:, :, 1] >= 80) & (hsv[:, :, 1] <= 250)
                            mask3 = (hsv[:, :, 2] >= 60) & (hsv[:, :, 2] <= 245)
                            tongue_mask = mask1 & mask2 & mask3
                            ratio = np.count_nonzero(tongue_mask) / float(patch.shape[0] * patch.shape[1])
                            # Must cover at least 35% of the chin zone below lower lip
                            if ratio > 0.35:
                                tongue_score = float(np.clip((ratio - 0.30) * 3.0, 0.0, 1.0))

                    blendshapes_dict['teethShowing'] = teeth_score
                    blendshapes_dict['tongueOut'] = tongue_score
                    blendshapes_dict['speechViseme'] = speech_score

        # ----------------------------------------------------------------------
        # Mode B: Pose Landmark Fallback (Profile / Turned Away / Occlusion)
        # ----------------------------------------------------------------------
        if tracking_mode == 'lost' and self.pose_landmarker is not None:
            pose_res = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            if pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
                pl = pose_res.pose_landmarks[0]
                # Keypoints: 7 = Left Ear, 8 = Right Ear, 0 = Nose, 11 = Left Shoulder, 12 = Right Shoulder
                ear_l = pl[7]
                ear_r = pl[8]
                nose = pl[0]

                # If at least one ear is reliably visible
                if ear_l.visibility > 0.45 or ear_r.visibility > 0.45:
                    tracking_mode = 'pose_rescue'
                    raw_conf = 0.85

                    # Approximate skull 2D center
                    if ear_l.visibility > 0.45 and ear_r.visibility > 0.45:
                        head_center_2d = np.array([(ear_l.x + ear_r.x) * 0.5 * w, (ear_l.y + ear_r.y) * 0.5 * h])
                        ear_dist_2d = np.linalg.norm(np.array([ear_l.x * w, ear_l.y * h]) - np.array([ear_r.x * w, ear_r.y * h]))
                        est_face_size = max(160, int(ear_dist_2d * 1.5))
                    elif ear_l.visibility > 0.45:
                        head_center_2d = np.array([ear_l.x * w - self.last_face_size * 0.35, ear_l.y * h])
                        est_face_size = self.last_face_size
                    else:
                        head_center_2d = np.array([ear_r.x * w + self.last_face_size * 0.35, ear_r.y * h])
                        est_face_size = self.last_face_size

                    self.last_face_size = est_face_size

                    # Unproject skull center to 3D translation at current distance
                    cur_z = self.last_tvec[2, 0] if self.last_tvec is not None else 75.0
                    tz = float(cur_z)
                    tx = float((head_center_2d[0] - cx) * tz / fx)
                    ty = float((head_center_2d[1] - cy) * tz / fy)
                    tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)

                    # Estimate yaw, pitch, and roll from ear, nose, and shoulder geometry
                    span_x = max(1e-3, abs(ear_l.x - ear_r.x))
                    if nose.visibility > 0.35 and ear_l.visibility > 0.35 and ear_r.visibility > 0.35:
                        mid_x = (ear_l.x + ear_r.x) * 0.5
                        est_yaw_deg = float(np.clip((nose.x - mid_x) / span_x * 90.0, -85.0, 85.0))
                    elif ear_l.visibility > 0.45 and ear_r.visibility < 0.3:
                        est_yaw_deg = -75.0  # Turned showing left ear (facing screen right)
                    elif ear_r.visibility > 0.45 and ear_l.visibility < 0.3:
                        est_yaw_deg = 75.0   # Turned showing right ear (facing screen left)
                    else:
                        est_yaw_deg = 0.0

                    # Dynamic Pitch & Roll: When head tilts up (looking at ceiling, Image 4):
                    # Ears drop lower on screen than nose (ears_y > nose_y), pitch tilts up (up to +75°).
                    if ear_l.visibility > 0.35 and ear_r.visibility > 0.35 and nose.visibility > 0.35:
                        ears_y = (ear_l.y + ear_r.y) * 0.5
                        dy = ears_y - nose.y
                        pitch_deg = float(np.clip((dy / span_x) * 95.0, -65.0, 75.0))
                        roll_deg = float(np.degrees(np.arctan2(ear_l.y - ear_r.y, max(1e-3, ear_l.x - ear_r.x))))
                    else:
                        pitch_deg = float(np.degrees(self.last_rvec[0, 0])) if self.last_rvec is not None else 0.0
                        roll_deg = 0.0

                    rot_mat = Rotation.from_euler('yxz', [est_yaw_deg, pitch_deg, roll_deg], degrees=True).as_matrix()
                    rvec, _ = cv2.Rodrigues(rot_mat)
                    self.last_rvec = rvec.copy()
                    self.last_tvec = tvec.copy()

        # ----------------------------------------------------------------------
        # Temporal Smoothing & Kinematic Dead-Reckoning
        # ----------------------------------------------------------------------
        if rvec is not None and tvec is not None:
            # Active tracking update
            smooth_rot, smooth_trans, confidence, is_coasting = self.kinematic_tracker.update(
                rotvec=rvec.flatten(),
                trans=tvec.flatten(),
                t=now
            )
        else:
            # Mode C: Kinematic Coasting (Lost detection -> Predict forward)
            smooth_rot, smooth_trans, confidence, is_active = self.kinematic_tracker.predict(dt=0.033)
            if not is_active:
                return {
                    'has_face': False,
                    'tracking_mode': 'lost',
                    'confidence': 0.0,
                    'is_coasting': False,
                    'face_size': self.last_face_size
                }
            tracking_mode = 'kinematic_coast'
            is_coasting = True

        # Reconstruct smoothed rotation and translation vectors
        rvec_smooth = smooth_rot.reshape(3, 1)
        tvec_smooth = smooth_trans.reshape(3, 1)
        R_smooth, _ = cv2.Rodrigues(rvec_smooth)

        # True metric 4x4 matrix
        mat_4x4 = np.eye(4, dtype=np.float64)
        mat_4x4[:3, :3] = R_smooth
        mat_4x4[:3, 3] = smooth_trans

        # Continuous Euler angles (yxz: yaw, pitch, roll)
        # Facing camera: Yaw=0°, Pitch=0°, Roll=0°
        euler = Rotation.from_matrix(R_smooth).as_euler('yxz', degrees=True)
        yaw = float(euler[0])
        pitch = float(euler[1])
        roll = float(euler[2])

        # Project 3D Skull Center onto screen
        proj_center, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), rvec_smooth, tvec_smooth, cam_matrix, self.dist_coeffs)
        skull_2d = proj_center.reshape(-1, 2)[0]
        center_x, center_y = int(skull_2d[0]), int(skull_2d[1])

        # ----------------------------------------------------------------------
        # Project Full 3D Anatomical Features (Eyes, Nose, Mouth, Ears, Back of Head)
        # ----------------------------------------------------------------------
        feature_names = list(CANONICAL_POINTS_3D.keys())
        feature_coords_3d = np.array([CANONICAL_POINTS_3D[k] for k in feature_names], dtype=np.float64)
        proj_features, _ = cv2.projectPoints(feature_coords_3d, rvec_smooth, tvec_smooth, cam_matrix, self.dist_coeffs)
        proj_features = proj_features.reshape(-1, 2)

        feature_projected_dict = {}
        for name, pt2 in zip(feature_names, proj_features):
            feature_projected_dict[name] = (int(pt2[0]), int(pt2[1]))

        # Grouped feature points for high-level consumers
        landmarks_3d_dict = {
            'eyes': {
                'left': CANONICAL_POINTS_3D['eye_left_pupil'],
                'right': CANONICAL_POINTS_3D['eye_right_pupil'],
                'outer_left': CANONICAL_POINTS_3D['eye_left_outer'],
                'outer_right': CANONICAL_POINTS_3D['eye_right_outer']
            },
            'nose': {
                'tip': CANONICAL_POINTS_3D['nose_tip'],
                'bridge': CANONICAL_POINTS_3D['nose_bridge']
            },
            'mouth': {
                'center': CANONICAL_POINTS_3D['mouth_center'],
                'left': CANONICAL_POINTS_3D['mouth_left'],
                'right': CANONICAL_POINTS_3D['mouth_right'],
                'chin': CANONICAL_POINTS_3D['chin']
            },
            'ears': {
                'left_tragus': CANONICAL_POINTS_3D['ear_left_tragus'],
                'right_tragus': CANONICAL_POINTS_3D['ear_right_tragus'],
                'left_helix': CANONICAL_POINTS_3D['ear_left_helix'],
                'right_helix': CANONICAL_POINTS_3D['ear_right_helix'],
                'left_lobule': CANONICAL_POINTS_3D['ear_left_lobule'],
                'right_lobule': CANONICAL_POINTS_3D['ear_right_lobule']
            },
            'back_of_head': {
                'occiput': CANONICAL_POINTS_3D['back_of_head'],
                'crown': CANONICAL_POINTS_3D['crown'],
                'nape': CANONICAL_POINTS_3D['nape'],
                'parietal_left': CANONICAL_POINTS_3D['parietal_left'],
                'parietal_right': CANONICAL_POINTS_3D['parietal_right']
            }
        }

        # ----------------------------------------------------------------------
        # Compute Full 3D Cranium Convex Hull for Zero-Leak Privacy Concealment
        # ----------------------------------------------------------------------
        proj_skull_hull, _ = cv2.projectPoints(SKULL_HULL_3D, rvec_smooth, tvec_smooth, cam_matrix, self.dist_coeffs)
        skull_hull_pts_2d = proj_skull_hull.reshape(-1, 2)

        if lmks_2d is not None and len(lmks_2d) > 0:
            # Combine 2D face landmarks with projected 3D cranium points for absolute coverage
            combined_points = np.vstack([lmks_2d, skull_hull_pts_2d])
        else:
            combined_points = skull_hull_pts_2d

        # 2D Convex Hull enclosing face, ears, crown, and back of head
        hull_2d = cv2.convexHull(combined_points.astype(np.float32)).reshape(-1, 2)

        return {
            'has_face': True,
            'tracking_mode': tracking_mode,
            'is_coasting': is_coasting,
            'confidence': float(confidence),
            'matrix_4x4': mat_4x4,
            'rotation_matrix': R_smooth,
            'translation_skull': smooth_trans,
            'rvec': rvec_smooth,
            'tvec': tvec_smooth,
            'cam_matrix': cam_matrix,
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll,
            'center_x': center_x,
            'center_y': center_y,
            'face_size': self.last_face_size,
            'landmarks_2d': lmks_2d,
            'landmarks_3d': landmarks_3d_dict,
            'projected_features': feature_projected_dict,
            'head_hull_2d': hull_2d,
            'blendshapes': blendshapes_dict
        }

    def close(self):
        if hasattr(self, 'face_landmarker') and self.face_landmarker:
            self.face_landmarker.close()
        if hasattr(self, 'pose_landmarker') and self.pose_landmarker:
            self.pose_landmarker.close()
