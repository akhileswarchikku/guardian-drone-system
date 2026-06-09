"""
Phase 4 — Occupancy Grid + A* Path Planner

Converts a binary obstacle mask into a 2D grid and finds the optimal
flyable path from a start point to a goal using A* search.

The grid works entirely in image-pixel coordinates.  GPS conversion happens
in tracking_node (Phase 5) once real depth data is available.

Usage
-----
    from vision.occupancy_grid import plan_path, draw_path

    waypoints = plan_path(obstacle_mask, start=(h-1, w//2), goal=(0, w//2))
    frame = draw_path(frame, waypoints)
"""
from __future__ import annotations

import heapq
from collections import deque

import cv2
import numpy as np


DOWNSAMPLE = 4      # work at 1/4 resolution for A* speed (57k cells → 3.6k)
SMOOTH_WIN = 7      # moving-average window for path smoothing


# ─── Public API ───────────────────────────────────────────────────────────────

def plan_path(
    obstacle_mask: np.ndarray,
    start: tuple[int, int],
    goal:  tuple[int, int],
) -> list[tuple[int, int]]:
    """
    A* pathfinding on a binary obstacle mask.

    Parameters
    ----------
    obstacle_mask : H x W bool array (True = blocked, drone cannot enter)
    start         : (row, col) in original resolution — drone's position in frame
    goal          : (row, col) in original resolution — target direction in frame

    Returns
    -------
    List of (row, col) waypoints in original resolution, ordered start→goal.
    Empty list if no path found (completely blocked).
    """
    h, w = obstacle_mask.shape

    # Clamp start/goal to valid range
    start = (_clamp(start[0], 0, h-1), _clamp(start[1], 0, w-1))
    goal  = (_clamp(goal[0],  0, h-1), _clamp(goal[1],  0, w-1))

    # Downsample grid for A* performance — max-pool so blocked cell always wins
    ds_h, ds_w = h // DOWNSAMPLE, w // DOWNSAMPLE
    trimmed = obstacle_mask[:ds_h * DOWNSAMPLE, :ds_w * DOWNSAMPLE]
    ds_mask = trimmed.reshape(ds_h, DOWNSAMPLE, ds_w, DOWNSAMPLE).max(axis=(1, 3)).astype(bool)

    ds_start = (start[0] // DOWNSAMPLE, start[1] // DOWNSAMPLE)
    ds_goal  = (goal[0]  // DOWNSAMPLE, goal[1]  // DOWNSAMPLE)

    # If start/goal land inside a blocked cell, find nearest free cell
    ds_start = _nearest_free(ds_mask, ds_start) or ds_start
    ds_goal  = _nearest_free(ds_mask, ds_goal)  or ds_goal

    path_ds = _astar(ds_mask, ds_start, ds_goal)
    if not path_ds:
        return []

    # Scale waypoints back to original resolution + smooth
    path = [(r * DOWNSAMPLE, c * DOWNSAMPLE) for r, c in path_ds]
    return _smooth_path(path)


def draw_path(
    frame: np.ndarray,
    waypoints: list[tuple[int, int]],
    colour: tuple = (0, 230, 60),
    thickness: int = 3,
    dot_r: int = 5,
) -> np.ndarray:
    """
    Draw the A* path on the frame as a polyline with endpoint dots.
    waypoints are (row, col); converted to (x, y) = (col, row) for OpenCV.
    """
    if len(waypoints) < 2:
        return frame

    pts = [(c, r) for r, c in waypoints]

    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i+1], colour, thickness, cv2.LINE_AA)

    cv2.circle(frame, pts[0],  dot_r, (0, 200, 255), -1)   # start = orange
    cv2.circle(frame, pts[-1], dot_r, colour, -1)           # goal  = green
    return frame


def assign_altitudes(
    waypoints_2d: list[tuple[int, int]],
    objects: list,
    frame_h: int,
    frame_w: int,
    default_alt_m: float = 10.0,
    clearance_m:   float = 2.0,
    min_alt_m:     float = 5.0,
    max_alt_m:     float = 30.0,
) -> list[tuple[int, int, float]]:
    """
    Extend 2D A* waypoints to 3D by attaching an altitude to each point.

    Rule per waypoint (row, col):
      - Start at default_alt_m (cruise height).
      - For every detected obstacle whose bbox footprint overlaps this waypoint's
        column range: alt = max(alt, obstacle.estimated_height_m + clearance_m).
      - Smooth altitude transitions so the drone doesn't spike up/down sharply.
      - Clamp to [min_alt_m, max_alt_m].

    Parameters
    ----------
    waypoints_2d   : list of (row, col) from plan_path()
    objects        : list[ObjectInfo] from ObstacleAnalysis.objects
    frame_h/w      : original frame dimensions
    default_alt_m  : cruise altitude when no obstacle is nearby
    clearance_m    : extra vertical margin above obstacle top
    min_alt_m      : lowest allowed drone altitude
    max_alt_m      : highest allowed drone altitude

    Returns
    -------
    list of (row, col, alt_m) — 3D waypoints ready for drone controller.
    Phase 5: convert (row,col) → (lat,lon) using GPS + camera FoV.
    """
    if not waypoints_2d:
        return []

    alts = [default_alt_m] * len(waypoints_2d)

    for wp_idx, (row, col) in enumerate(waypoints_2d):
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox
            # Check if this waypoint's column falls within the object's horizontal span
            if x1 <= col <= x2:
                needed = obj.estimated_height_m + clearance_m
                if needed > alts[wp_idx]:
                    alts[wp_idx] = needed

    # Smooth altitude: moving average so transitions are gradual
    win = 5
    smoothed = list(alts)
    for i in range(len(alts)):
        lo = max(0, i - win)
        hi = min(len(alts), i + win + 1)
        smoothed[i] = sum(alts[lo:hi]) / (hi - lo)

    # Clamp
    smoothed = [max(min_alt_m, min(max_alt_m, a)) for a in smoothed]

    return [(r, c, round(a, 1)) for (r, c), a in zip(waypoints_2d, smoothed)]


def draw_obstacle_overlay(
    frame: np.ndarray,
    obstacle_mask: np.ndarray,
    colour: tuple = (0, 0, 200),
    alpha: float  = 0.35,
) -> np.ndarray:
    """
    Overlay the blocked region (dilated obstacle mask) in semi-transparent colour.
    """
    overlay = frame.copy()
    overlay[obstacle_mask] = colour
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_clear_overlay(
    frame: np.ndarray,
    obstacle_mask: np.ndarray,
    colour: tuple = (0, 180, 0),
    alpha: float  = 0.15,
) -> np.ndarray:
    """
    Highlight the flyable (non-blocked) corridor in a faint green tint.
    """
    clear_mask = ~obstacle_mask
    overlay = frame.copy()
    overlay[clear_mask] = colour
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


# ─── A* internals ─────────────────────────────────────────────────────────────

_DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
_COST = [1.414,  1.0,  1.414,  1.0,  1.0,  1.414,  1.0, 1.414]


def _astar(
    grid:  np.ndarray,
    start: tuple[int, int],
    goal:  tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = grid.shape

    def heuristic(a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    open_set: list = [(0.0, start)]
    came_from: dict[tuple, tuple] = {}
    g: dict[tuple, float] = {start: 0.0}

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            return path[::-1]

        for (dr, dc), cost in zip(_DIRS, _COST):
            nr, nc = cur[0]+dr, cur[1]+dc
            nb = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if grid[nr, nc]:
                continue
            tg = g[cur] + cost
            if tg < g.get(nb, float('inf')):
                came_from[nb] = cur
                g[nb] = tg
                f = tg + heuristic(nb, goal)
                heapq.heappush(open_set, (f, nb))

    return []


def _nearest_free(
    mask: np.ndarray,
    pos:  tuple[int, int],
) -> tuple[int, int] | None:
    """BFS to the nearest unblocked cell from pos."""
    if not mask[pos]:
        return pos
    h, w = mask.shape
    visited: set = set()
    q: deque = deque([pos])
    while q:
        r, c = q.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        if not mask[r, c]:
            return (r, c)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                q.append((nr, nc))
    return None


def _smooth_path(
    path: list[tuple[int, int]],
    window: int = SMOOTH_WIN,
) -> list[tuple[int, int]]:
    if len(path) <= window:
        return path
    result = [path[0]]
    half = window // 2
    for i in range(1, len(path) - 1):
        lo = max(0, i - half)
        hi = min(len(path), i + half + 1)
        chunk = path[lo:hi]
        avg_r = int(round(sum(p[0] for p in chunk) / len(chunk)))
        avg_c = int(round(sum(p[1] for p in chunk) / len(chunk)))
        result.append((avg_r, avg_c))
    result.append(path[-1])
    return result


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))
