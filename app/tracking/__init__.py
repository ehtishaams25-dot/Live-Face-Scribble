from app.tracking.face_tracker import FaceTracker
from app.tracking.temporal_smoother import OneEuroFilter, KinematicHeadPoseTracker

# Alias for backwards compatibility
HeadPoseSmoother = KinematicHeadPoseTracker

__all__ = ["FaceTracker", "OneEuroFilter", "KinematicHeadPoseTracker", "HeadPoseSmoother"]
