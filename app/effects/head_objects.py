import cv2
import numpy as np


class HeadObject:
    """
    Abstract base class for 3D objects attached to the 3D Head Space coordinate system.
    Origin (0, 0, 0) is the anatomical skull center between the ears.
    Coordinate axes in head space:
      +X: Screen Right (Subject's Left)
      -X: Screen Left  (Subject's Right)
      +Y: Inferior Down (Towards Chin / Neck)
      -Y: Cranial Up    (Towards Crown / Vertex)
      -Z: Anterior Forward (Towards Nose / Face Front)
      +Z: Posterior Backward (Towards Occiput / Back of Head)
    Units: Centimeters (cm)
    """
    def __init__(self, name: str = "HeadObject"):
        self.name = name
        self.enabled = True

    def render(self, frame_bgr: np.ndarray, tracking_data: dict) -> np.ndarray:
        raise NotImplementedError


class AnatomicalHeadRigVisualizer(HeadObject):
    """
    High-Precision 3D Anatomical Head Rig Inspector.
    Visualizes:
    - 3D Metric Coordinate Axes from True Skull Origin
    - Eyes (Pupils, Eye Corners, Gaze Vector)
    - Nose (Tip, Bridge, Projection)
    - Mouth (Lips, Oral Aperture, Jawline)
    - Ears (Left & Right Ear Tragus, Helix, Auricular Ring)
    - Back of Head (Occipital Protuberance, Parietal Plates, Cranial Hull)
    - Full 3D Cranium Bounding Cage
    """
    def __init__(self):
        super().__init__(name="AnatomicalHeadRig")
        self.show_axes = True
        self.show_labels = True
        self.show_cranium_cage = True
        self._build_3d_geometry()

    def _build_3d_geometry(self):
        # 1. 3D Coordinate Axes at Skull Center (cm)
        axis_len = 14.0
        self.axes_verts = np.array([
            [0.0, 0.0, 0.0],          # Origin (Skull Center)
            [axis_len, 0.0, 0.0],     # +X (Lateral: Screen Right)
            [0.0, -axis_len, 0.0],    # -Y (Cranial Up: Green)
            [0.0, 0.0, -axis_len],    # -Z (Forward Face: Blue)
            [0.0, 0.0, axis_len],     # +Z (Back of Head: Cyan)
        ], dtype=np.float64)

        # 2. 3D Cranium Ribbed Cage Vertices (16 key skull landmarks)
        # Dimensions: Width ~16cm (-8 to +8), Height ~23cm (-12 to +11), Depth ~21cm (-11 to +10)
        self.cage_verts = np.array([
            # Crown & Top
            [0.0, -11.5, 1.0],        # 0: Crown top
            [5.5, -8.5, 4.5],         # 1: Parietal left
            [-5.5, -8.5, 4.5],        # 2: Parietal right
            [0.0, -4.5, -7.8],        # 3: Glabella / Brow
            # Back of Head & Nape
            [0.0, -2.0, 9.5],         # 4: Occipital bone (Back of Head)
            [0.0, 8.5, 7.5],          # 5: Nape / Base of Skull
            # Ears (Biauricular axis)
            [8.4, -2.5, 1.2],         # 6: Left Ear Top Helix
            [7.8, 0.0, 0.0],          # 7: Left Ear Tragus
            [8.0, 2.5, 0.5],          # 8: Left Ear Lobe
            [-8.4, -2.5, 1.2],        # 9: Right Ear Top Helix
            [-7.8, 0.0, 0.0],         # 10: Right Ear Tragus
            [-8.0, 2.5, 0.5],         # 11: Right Ear Lobe
            # Face Anterior
            [0.0, 1.2, -11.2],        # 12: Nose Tip
            [0.0, 9.8, -7.2],         # 13: Chin / Jaw
            [2.5, 4.8, -6.8],         # 14: Mouth Left
            [-2.5, 4.8, -6.8],        # 15: Mouth Right
        ], dtype=np.float64)

        # Edges connecting the 3D cranium cage
        self.cage_edges = [
            # Cranium Crown Ring
            (0, 3), (0, 1), (0, 2), (0, 4),
            # Back of Head / Occipital Connections
            (1, 4), (2, 4), (4, 5),
            (1, 6), (2, 9),
            # Left Ear Ring
            (6, 7), (7, 8), (6, 8),
            # Right Ear Ring
            (9, 10), (10, 11), (9, 11),
            # Face Connections
            (3, 12), (12, 13), (13, 14), (13, 15), (14, 15),
            # Temporal Bridges (Ears to Brow & Chin)
            (3, 6), (3, 9),
            (8, 14), (11, 15),
            (5, 8), (5, 11), (5, 13)
        ]

    def render(self, frame_bgr: np.ndarray, tracking_data: dict) -> np.ndarray:
        if not tracking_data.get('has_face', False):
            return frame_bgr

        rvec = tracking_data.get('rvec')
        tvec = tracking_data.get('tvec')
        cam_matrix = tracking_data.get('cam_matrix')
        proj_features = tracking_data.get('projected_features', {})
        tracking_mode = tracking_data.get('tracking_mode', 'face_dense')
        is_coasting = tracking_data.get('is_coasting', False)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        if rvec is None or tvec is None or cam_matrix is None:
            return frame_bgr

        h, w = frame_bgr.shape[:2]

        # ----------------------------------------------------------------------
        # 1. Draw 3D Cranium Cage Edges
        # ----------------------------------------------------------------------
        if self.show_cranium_cage:
            proj_cage, _ = cv2.projectPoints(self.cage_verts, rvec, tvec, cam_matrix, dist_coeffs)
            cage_2d = proj_cage.reshape(-1, 2).astype(np.int32)

            cage_color = (0, 220, 120) if tracking_mode == 'face_dense' else (0, 180, 255)
            for i1, i2 in self.cage_edges:
                p1 = tuple(cage_2d[i1])
                p2 = tuple(cage_2d[i2])
                if (-w < p1[0] < 2 * w) and (-h < p1[1] < 2 * h):
                    cv2.line(frame_bgr, p1, p2, cage_color, 1, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # 2. Draw 3D Coordinate Axes from Skull Origin
        # ----------------------------------------------------------------------
        if self.show_axes:
            proj_axes, _ = cv2.projectPoints(self.axes_verts, rvec, tvec, cam_matrix, dist_coeffs)
            axes_2d = proj_axes.reshape(-1, 2).astype(np.int32)
            origin = tuple(axes_2d[0])

            # +X (Red = Lateral / Subject Left)
            cv2.arrowedLine(frame_bgr, origin, tuple(axes_2d[1]), (50, 50, 255), 3, cv2.LINE_AA, tipLength=0.18)
            # -Y (Green = Cranial Up)
            cv2.arrowedLine(frame_bgr, origin, tuple(axes_2d[2]), (50, 240, 50), 3, cv2.LINE_AA, tipLength=0.18)
            # -Z (Blue = Forward Face / Gaze)
            cv2.arrowedLine(frame_bgr, origin, tuple(axes_2d[3]), (255, 140, 30), 3, cv2.LINE_AA, tipLength=0.18)
            # +Z (Cyan = Back of Head)
            cv2.line(frame_bgr, origin, tuple(axes_2d[4]), (220, 220, 0), 2, cv2.LINE_AA)

            # Skull Center Origin Marker
            cv2.circle(frame_bgr, origin, 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, origin, 8, (50, 50, 50), 1, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # 3. Highlight Named 3D Anatomical Nodes
        # ----------------------------------------------------------------------
        def draw_node(key, color, label=None, radius=6):
            pt = proj_features.get(key)
            if pt and (0 <= pt[0] < w) and (0 <= pt[1] < h):
                cv2.circle(frame_bgr, pt, radius, color, -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, pt, radius + 2, (20, 20, 20), 1, cv2.LINE_AA)
                if label and self.show_labels:
                    cv2.putText(frame_bgr, label, (pt[0] + 8, pt[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        # Eyes (Cyan)
        draw_node('eye_left_pupil', (255, 240, 0), "L-Eye", 5)
        draw_node('eye_right_pupil', (255, 240, 0), "R-Eye", 5)

        # Nose (Orange)
        draw_node('nose_tip', (0, 140, 255), "Nose", 6)

        # Mouth & Chin (Magenta)
        draw_node('mouth_center', (220, 50, 220), "Mouth", 5)
        draw_node('chin', (180, 50, 200), "Chin", 4)

        # Ears (Yellow - Left & Right)
        draw_node('ear_left_tragus', (0, 255, 255), "L-Ear", 7)
        draw_node('ear_right_tragus', (0, 255, 255), "R-Ear", 7)

        # Back of Head & Crown (Electric Blue)
        draw_node('back_of_head', (255, 100, 50), "Back of Head", 8)
        draw_node('crown', (100, 255, 200), "Crown", 5)
        draw_node('nape', (200, 100, 50), "Nape", 5)

        # ----------------------------------------------------------------------
        # 4. Status Badge Overlay
        # ----------------------------------------------------------------------
        yaw = tracking_data.get('yaw', 0.0)
        pitch = tracking_data.get('pitch', 0.0)
        roll = tracking_data.get('roll', 0.0)
        tx, ty, tz = tracking_data.get('translation_skull', [0, 0, 0])

        badge_x, badge_y = max(16, origin[0] - 110), max(40, origin[1] - 110)
        cv2.rectangle(frame_bgr, (badge_x, badge_y), (badge_x + 220, badge_y + 44), (16, 18, 22), -1)
        cv2.rectangle(frame_bgr, (badge_x, badge_y), (badge_x + 220, badge_y + 44), (60, 65, 75), 1)

        mode_color = (80, 240, 120) if tracking_mode == 'face_dense' else ((0, 180, 255) if tracking_mode == 'pose_rescue' else (0, 100, 255))
        cv2.putText(frame_bgr, f"3D RIG: {tracking_mode.upper()}",
                    (badge_x + 8, badge_y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.42, mode_color, 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Y {yaw:+.1f} P {pitch:+.1f} R {roll:+.1f} | Z:{tz:.1f}cm",
                    (badge_x + 8, badge_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 205, 215), 1, cv2.LINE_AA)

        return frame_bgr


class WireframeHelmetObject(HeadObject):
    """
    A futuristic 3D wireframe helmet and HUD visor attached rigidly to the user's
    calibrated 3D Head Space coordinate system.
    """
    def __init__(self, radius_cm: float = 10.5):
        super().__init__(name="WireframeHelmet")
        self.radius = radius_cm
        self.show_axes = True
        self.color_frame = (0, 220, 100)      # Neon green/emerald
        self.color_visor = (0, 240, 255)      # Cyberpunk cyan/yellow
        self.color_jaw = (255, 160, 40)       # Tech blue/amber
        self._build_geometry()

    def _build_geometry(self):
        s = self.radius
        # In new coordinate frame:
        # X: -s to +s (lateral)
        # Y: -s - 2 to +s (cranial up is -Y, chin down is +Y)
        # Z: -s - 2 to +s (nose front is -Z, occiput back is +Z)
        y_top = -s - 1.5
        y_bot = s + 1.0
        z_front = -s - 1.8
        z_back = s + 1.2

        # Main Helmet Cage Vertices
        self.box_verts = np.array([
            # Back quad (0..3)
            [-s, y_top, z_back],
            [ s, y_top, z_back],
            [ s, y_bot, z_back],
            [-s, y_bot, z_back],
            # Front quad (4..7)
            [-s, y_top, z_front],
            [ s, y_top, z_front],
            [ s, y_bot, z_front],
            [-s, y_bot, z_front],
            # Cranium top crown (8..11)
            [-s * 0.6, y_top - 1.8, z_back * 0.5],
            [ s * 0.6, y_top - 1.8, z_back * 0.5],
            [ s * 0.6, y_top - 1.8, z_front * 0.5],
            [-s * 0.6, y_top - 1.8, z_front * 0.5],
        ], dtype=np.float64)

        self.box_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Back rim
            (4, 5), (5, 6), (6, 7), (7, 4),  # Front rim
            (0, 4), (1, 5), (2, 6), (3, 7),  # Struts connecting front to back
            (8, 9), (9, 10), (10, 11), (11, 8), # Top crown rim
            (0, 8), (1, 9), (4, 11), (5, 10)   # Crown struts
        ]

        # Visor Shield Vertices (Curved protective eye-visor)
        vz = z_front - 1.0
        self.visor_verts = np.array([
            [-s * 0.85, -4.5, vz + 1.5],
            [-s * 0.45, -4.5, vz],
            [ 0.0,      -4.5, vz - 0.5],
            [ s * 0.45, -4.5, vz],
            [ s * 0.85, -4.5, vz + 1.5],
            [ s * 0.85, -1.0, vz + 1.5],
            [ s * 0.45, -1.0, vz],
            [ 0.0,       -1.0, vz - 0.5],
            [-s * 0.45,  -1.0, vz],
            [-s * 0.85,  -1.0, vz + 1.5],
        ], dtype=np.float64)

        self.visor_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),   # Top visor curve
            (5, 6), (6, 7), (7, 8), (8, 9),   # Bottom visor curve
            (4, 5), (9, 0),                   # Outer temples
            (1, 8), (2, 7), (3, 6)            # Vertical grid separators
        ]

    def render(self, frame_bgr: np.ndarray, tracking_data: dict) -> np.ndarray:
        if not tracking_data.get('has_face', False):
            return frame_bgr

        rvec = tracking_data.get('rvec')
        tvec = tracking_data.get('tvec')
        cam_matrix = tracking_data.get('cam_matrix')
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        if rvec is None or tvec is None or cam_matrix is None:
            return frame_bgr

        h, w = frame_bgr.shape[:2]

        # Draw wireframe helmet box
        proj_box, _ = cv2.projectPoints(self.box_verts, rvec, tvec, cam_matrix, dist_coeffs)
        box_2d = proj_box.reshape(-1, 2).astype(np.int32)

        for i1, i2 in self.box_edges:
            p1 = tuple(box_2d[i1])
            p2 = tuple(box_2d[i2])
            if (-w < p1[0] < 2 * w) and (-h < p1[1] < 2 * h):
                cv2.line(frame_bgr, p1, p2, self.color_frame, 2, cv2.LINE_AA)

        # Draw curved visor
        proj_visor, _ = cv2.projectPoints(self.visor_verts, rvec, tvec, cam_matrix, dist_coeffs)
        visor_2d = proj_visor.reshape(-1, 2).astype(np.int32)

        for i1, i2 in self.visor_edges:
            p1 = tuple(visor_2d[i1])
            p2 = tuple(visor_2d[i2])
            if (-w < p1[0] < 2 * w) and (-h < p1[1] < 2 * h):
                cv2.line(frame_bgr, p1, p2, self.color_visor, 2, cv2.LINE_AA)

        return frame_bgr
