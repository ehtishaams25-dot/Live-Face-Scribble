import math
import time
import numpy as np


class OneEuroFilter:
    """
    1-Euro Filter for adaptive jitter reduction and zero-lag dynamic tracking.
    Reference: Casiez et al., '1 € Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Human-Computer Interaction'
    """
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t=None):
        if t is None:
            t = time.time()
            
        x = np.asarray(x, dtype=np.float64)
        
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-4)
        self.t_prev = t

        dx = (x - self.x_prev) / dt
        edx = self._alpha(self.d_cutoff, dt)
        dx_hat = edx * dx + (1.0 - edx) * self.dx_prev
        self.dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        
        if np.isscalar(cutoff):
            a = self._alpha(cutoff, dt)
        else:
            tau = 1.0 / (2.0 * math.pi * cutoff)
            a = 1.0 / (1.0 + tau / dt)
            
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class KinematicHeadPoseTracker:
    """
    Persistent 3D Head Space tracker featuring:
    - 1-Euro adaptive temporal smoothing on continuous Rodrigues rotation vectors
    - Zero gimbal lock / zero Euler angle distortion at any rotation angle
    - Kinematic dead-reckoning & velocity prediction when landmarks are occluded or turned
    - Graceful degradation through extreme yaw (60° to 90°)
    - Smooth re-acquisition blending
    """
    def __init__(self, max_coast_frames: int = 45, damping: float = 0.92):
        # Smoothing filters (1-Euro for zero-lag and rock-solid jitter removal)
        self.rot_filter = OneEuroFilter(min_cutoff=1.0, beta=0.02)
        self.trans_filter = OneEuroFilter(min_cutoff=1.2, beta=0.03)

        # State tracking
        self.rot = np.zeros(3, dtype=np.float64)
        self.trans = np.zeros(3, dtype=np.float64)
        self.vel_rot = np.zeros(3, dtype=np.float64)
        self.vel_trans = np.zeros(3, dtype=np.float64)
        
        self.last_time = None
        self.frames_lost = 999
        self.max_coast_frames = max_coast_frames
        self.damping = damping
        self.confidence = 0.0

    def update(self, rotvec, trans, t=None):
        """
        Called when physical face landmarks are detected.
        rotvec: Rodrigues rotation vector (rx, ry, rz)
        trans: [tx, ty, tz] in cm
        """
        if t is None:
            t = time.time()

        raw_rot = np.asarray(rotvec, dtype=np.float64)
        raw_trans = np.asarray(trans, dtype=np.float64)

        if self.last_time is not None:
            dt = max(t - self.last_time, 1e-4)
            # Estimate angular and linear velocities (capped to prevent noise spikes)
            inst_vel_r = (raw_rot - self.rot) / dt
            inst_vel_t = (raw_trans - self.trans) / dt
            
            # Smooth velocity estimates
            self.vel_rot = 0.6 * self.vel_rot + 0.4 * np.clip(inst_vel_r, -15.0, 15.0)
            self.vel_trans = 0.6 * self.vel_trans + 0.4 * np.clip(inst_vel_t, -200.0, 200.0)
        else:
            self.vel_rot = np.zeros(3, dtype=np.float64)
            self.vel_trans = np.zeros(3, dtype=np.float64)

        self.last_time = t

        # Apply 1-Euro smoothing
        smooth_rot = self.rot_filter.filter(raw_rot, t)
        smooth_trans = self.trans_filter.filter(raw_trans, t)

        # If re-acquiring after coasting, blend gently to avoid snap
        if 0 < self.frames_lost <= self.max_coast_frames:
            blend = 0.65
            self.rot = blend * smooth_rot + (1.0 - blend) * self.rot
            self.trans = blend * smooth_trans + (1.0 - blend) * self.trans
        else:
            self.rot = smooth_rot
            self.trans = smooth_trans

        self.frames_lost = 0
        self.confidence = 1.0
        return self.rot.copy(), self.trans.copy(), self.confidence, False

    def predict(self, dt: float = 0.033):
        """
        Called when landmark detection drops (e.g. extreme yaw or hand occlusion).
        Predicts 3D head pose forward using damped kinematic velocity.
        """
        self.frames_lost += 1
        
        if self.frames_lost <= self.max_coast_frames:
            # Predict forward with damping
            self.rot += self.vel_rot * dt * (self.damping ** self.frames_lost)
            self.trans += self.vel_trans * dt * (self.damping ** self.frames_lost)
            # Confidence decays smoothly from 1.0 down to ~0.05 on the final coast frame
            decay = (self.frames_lost - 1) / float(self.max_coast_frames)
            self.confidence = float(np.clip(1.0 - decay, 0.05, 1.0))
            is_active = True
        else:
            # Lost completely
            self.confidence = 0.0
            is_active = False

        return self.rot.copy(), self.trans.copy(), self.confidence, is_active

    def reset(self):
        self.rot = np.zeros(3, dtype=np.float64)
        self.trans = np.zeros(3, dtype=np.float64)
        self.vel_rot = np.zeros(3, dtype=np.float64)
        self.vel_trans = np.zeros(3, dtype=np.float64)
        self.last_time = None
        self.frames_lost = 999
        self.confidence = 0.0
        self.rot_filter.reset()
        self.trans_filter.reset()
