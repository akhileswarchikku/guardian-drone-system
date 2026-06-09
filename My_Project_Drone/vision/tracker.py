"""
Phase 4.3 — ByteTrack Multi-Object Tracker + Role Assignment

Wraps supervision.ByteTrack (spec parameters: max_age=30, min_hits=3, iou_threshold=0.3).
Assigns one of three roles to each tracked person:
  - "victim"    — person whose screen position is closest to the frame centre
                  (drone hovers above victim GPS on arrival)
  - "suspect"   — person moving fastest toward the victim
  - "bystander" — everyone else
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


# ─── Spec parameters (Section 4.3) ────────────────────────────────────────────
MAX_AGE    = 30   # frames to keep a lost track
MIN_HITS   = 3    # consecutive detections before track is confirmed
IOU_THRESH = 0.3  # matching threshold

ROLE_VICTIM    = "victim"
ROLE_SUSPECT   = "suspect"
ROLE_BYSTANDER = "bystander"


class PersonTracker:
    """
    ByteTrack wrapper with per-frame role assignment.

    Usage
    -----
    tracker = PersonTracker(frame_rate=10)   # 10 fps for drone camera mock
    for frame in frames:
        detections = detector.detect(frame)
        tracked    = tracker.update(detections, frame)
        roles      = tracker.roles              # dict[track_id -> role_str]
    """

    def __init__(self, frame_rate: int = 10):
        import supervision as sv

        # TODO: ByteTrack deprecated in supervision 0.28.0, removed in 0.30.0.
        # When upgrading supervision past 0.29, replace with sv.InertiaTracker or sv.SORT
        # using equivalent params: max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESH.
        self._tracker = sv.ByteTrack(
            track_activation_threshold = IOU_THRESH,
            lost_track_buffer          = MAX_AGE,
            minimum_matching_threshold = 0.8,
            frame_rate                 = frame_rate,
            minimum_consecutive_frames = MIN_HITS,
        )
        # track_id -> deque of (cx, cy) centroids for velocity calculation
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        self.roles: dict[int, str] = {}

    def update(self, detections: "sv.Detections", frame: np.ndarray) -> "sv.Detections":
        """
        Update tracker with new detections; assign roles.
        Returns tracked Detections with .tracker_id populated.
        """
        tracked = self._tracker.update_with_detections(detections)
        self._update_roles(tracked, frame)
        return tracked

    def reset(self) -> None:
        self._tracker.reset()
        self._history.clear()
        self.roles.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _centroid(self, box: np.ndarray) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _update_roles(self, tracked: "sv.Detections", frame: np.ndarray) -> None:
        if tracked.tracker_id is None or len(tracked) == 0:
            self.roles = {}
            return

        h, w = frame.shape[:2]
        frame_cx, frame_cy = w / 2, h / 2

        # Record centroid history for each track
        centroids: dict[int, tuple[float, float]] = {}
        for tid, box in zip(tracked.tracker_id, tracked.xyxy):
            cx, cy = self._centroid(box)
            centroids[tid] = (cx, cy)
            self._history[tid].append((cx, cy))

        # Victim = track whose centroid is closest to frame centre
        if not centroids:
            return

        victim_id = min(
            centroids,
            key=lambda tid: (centroids[tid][0] - frame_cx) ** 2
                          + (centroids[tid][1] - frame_cy) ** 2,
        )
        victim_cx, victim_cy = centroids[victim_id]

        # Suspect = track approaching victim fastest (most negative distance delta)
        # Needs at least 2 history points to compute velocity
        best_approach = 0.0
        suspect_id    = None

        for tid, (cx, cy) in centroids.items():
            if tid == victim_id:
                continue
            hist = self._history[tid]
            if len(hist) < 2:
                continue
            prev_cx, prev_cy = hist[-2]
            prev_dist = ((prev_cx - victim_cx) ** 2 + (prev_cy - victim_cy) ** 2) ** 0.5
            curr_dist = ((cx      - victim_cx) ** 2 + (cy      - victim_cy) ** 2) ** 0.5
            approach  = prev_dist - curr_dist   # positive = moving toward victim
            if approach > best_approach:
                best_approach = approach
                suspect_id    = tid

        # Assign roles (only flag a suspect if approach is meaningful: > 5px/frame)
        new_roles = {}
        for tid in centroids:
            if tid == victim_id:
                new_roles[tid] = ROLE_VICTIM
            elif tid == suspect_id and best_approach > 5.0:
                new_roles[tid] = ROLE_SUSPECT
            else:
                new_roles[tid] = ROLE_BYSTANDER

        self.roles = new_roles
