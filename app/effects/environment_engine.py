import cv2
import numpy as np


class EnvironmentRelightingEngine:
    """
    Lightweight, high-speed Environment Integration & Relighting Engine.
    Features:
    - Dominant environment ambient color extraction (zero GPU overhead)
    - Ambient color tinting to harmoniously seat the subject in the environment lighting
    - Physically convincing Light Wrap (silhouette edge / rim lighting)
    - Soft atmospheric light spill along the perimeter to eliminate hard cutout artifacts
    - Sub-5ms execution via proxy morphological processing and OpenCV SIMD C++ operations
    """
    def __init__(self,
                 enabled: bool = True,
                 rim_intensity: float = 0.75,
                 ambient_intensity: float = 0.12,
                 spill_intensity: float = 0.20):
        self.enabled = enabled
        self.rim_intensity = float(rim_intensity)
        self.ambient_intensity = float(ambient_intensity)
        self.spill_intensity = float(spill_intensity)

        # Cached background light plate
        self._last_bg_id = None
        self._cached_bg_blurred = None
        self._cached_bg_mean = None
        self._cached_bg_shape = None

    def analyze_background(self, bg_plate: np.ndarray):
        """
        Analyzes the background plate to precompute blurred ambient light
        and dominant tone using proxy resolution.
        """
        h, w = bg_plate.shape[:2]
        bg_id = (id(bg_plate), bg_plate.shape, bg_plate.data.tobytes()[:32] if hasattr(bg_plate.data, 'tobytes') else None)

        if self._cached_bg_shape != (h, w) or self._last_bg_id != bg_id:
            # 1. High-speed blurred light plate (samples incoming environmental radiance)
            small_w, small_h = max(160, w // 4), max(90, h // 4)
            small_bg = cv2.resize(bg_plate, (small_w, small_h), interpolation=cv2.INTER_AREA)
            small_light = cv2.GaussianBlur(small_bg, (11, 11), 0)
            self._cached_bg_blurred = cv2.resize(small_light, (w, h), interpolation=cv2.INTER_LINEAR)

            # 2. Extract dominant ambient color (8x8 micro sample)
            micro = cv2.resize(small_bg, (8, 8), interpolation=cv2.INTER_AREA)
            self._cached_bg_mean = np.mean(micro, axis=(0, 1)).astype(np.float32) # BGR
            self._cached_bg_shape = (h, w)
            self._last_bg_id = bg_id

        return self._cached_bg_blurred, self._cached_bg_mean

    def integrate(self,
                  fg_frame: np.ndarray,
                  bg_plate: np.ndarray,
                  mask_u8: np.ndarray) -> np.ndarray:
        """
        Integrates the foreground subject with the background environment.
        fg_frame: Raw camera frame (BGR, uint8)
        bg_plate: Background plate (BGR, uint8)
        mask_u8: Refined foreground alpha mask (single channel, uint8 0..255)
        """
        if not self.enabled:
            # Standard clean alpha blend
            m_3ch = cv2.merge([mask_u8, mask_u8, mask_u8])
            inv_m = cv2.bitwise_not(m_3ch)
            part_fg = cv2.multiply(fg_frame, m_3ch, scale=1.0/255.0)
            part_bg = cv2.multiply(bg_plate, inv_m, scale=1.0/255.0)
            return cv2.add(part_fg, part_bg)

        h, w = fg_frame.shape[:2]
        bg_blurred_light, bg_mean = self.analyze_background(bg_plate)

        # 1. AMBIENT COLOR TINTING
        if self.ambient_intensity > 0.001:
            tint_overlay = np.full_like(fg_frame, bg_mean.astype(np.uint8))
            alpha_fg = 1.0 - self.ambient_intensity
            fg_tinted = cv2.addWeighted(fg_frame, alpha_fg, tint_overlay, self.ambient_intensity, 0)
        else:
            fg_tinted = fg_frame

        # 2. SILHOUETTE LIGHT WRAP (RIM LIGHTING)
        # Compute edge ribbon at proxy resolution (half-res) for sub-millisecond execution
        if self.rim_intensity > 0.001:
            pw, ph = max(160, w // 2), max(90, h // 2)
            m_proxy = cv2.resize(mask_u8, (pw, ph), interpolation=cv2.INTER_AREA)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            er = cv2.erode(m_proxy, k)
            edge_proxy = cv2.subtract(m_proxy, er)
            edge_blur_proxy = cv2.GaussianBlur(edge_proxy, (9, 9), 0)
            edge_full = cv2.resize(edge_blur_proxy, (w, h), interpolation=cv2.INTER_LINEAR)

            edge_3ch = cv2.merge([edge_full, edge_full, edge_full])
            edge_light = cv2.multiply(bg_blurred_light, edge_3ch, scale=self.rim_intensity / 255.0)
            fg_relit = cv2.add(fg_tinted, edge_light)
        else:
            fg_relit = fg_tinted

        # 3. BASE COMPOSITING
        m_3ch = cv2.merge([mask_u8, mask_u8, mask_u8])
        inv_m = cv2.bitwise_not(m_3ch)

        comp_fg = cv2.multiply(fg_relit, m_3ch, scale=1.0/255.0)
        comp_bg = cv2.multiply(bg_plate, inv_m, scale=1.0/255.0)
        composite = cv2.add(comp_fg, comp_bg)

        # 4. ATMOSPHERIC LIGHT SPILL (Bloom over silhouette border)
        if self.spill_intensity > 0.001:
            pw, ph = max(160, w // 2), max(90, h // 2)
            m_proxy = cv2.resize(mask_u8, (pw, ph), interpolation=cv2.INTER_AREA)
            spill_proxy = cv2.GaussianBlur(m_proxy, (7, 7), 0)
            diff_proxy = cv2.absdiff(spill_proxy, m_proxy)
            diff_full = cv2.resize(diff_proxy, (w, h), interpolation=cv2.INTER_LINEAR)

            spill_3ch = cv2.merge([diff_full, diff_full, diff_full])
            spill_light = cv2.multiply(bg_blurred_light, spill_3ch, scale=self.spill_intensity / 255.0)
            composite = cv2.add(composite, spill_light)

        return composite
