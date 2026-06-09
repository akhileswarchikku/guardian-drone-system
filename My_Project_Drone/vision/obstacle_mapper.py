"""
Phase 4 — Obstacle Passability Mapper

USER REQUIREMENT (2026-06-08):
  "It detected a car but it cannot pass from it — it should find the space.
   Based on the size and shape, if there is no space or it is a solid object
   it should not pass from that way."

What this module does
---------------------
  1. Run YOLOv8-seg to get pixel-level masks for every detected object.
  2. Estimate per-object proximity (how close it is to the drone).
  3. For each object, measure gaps in 4 directions (left, right, above, below).
  4. Mark each object as PASSABLE (gap >= drone clearance) or BLOCKED.
  5. Dilate combined obstacle mask by drone clearance → occupancy grid input.
  6. vision/occupancy_grid.py then runs A* through the free space.

Gap types
---------
  - left / right : lateral gap between object edge and frame border
  - above        : vertical gap above object (fly over it)
  - below        : gap below object (fly under — rare for drone use)

Drone clearance
---------------
  Defaults to 8% of frame width (~102 px for 1280px video).
  Represents drone body width + 40 cm safety margin.

Phase 5 upgrade
---------------
  Replace proximity_map from DepthEstimator (heuristic/MiDaS) with
  RealSense D435i metric depth stream. The rest of this module is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from vision.detector import get_category, class_name, CATEGORY_MAP
from vision.depth_estimator import DepthEstimator


# Typical real-world object heights in metres (COCO class ID → height)
# Used to estimate safe fly-over altitude. Phase 5: replace with RealSense metric depth.
COCO_TYPICAL_HEIGHT_M: dict[int, float] = {
    0:  1.8,   # person
    1:  1.0,   # bicycle
    2:  1.5,   # car
    3:  1.2,   # motorcycle
    4: 10.0,   # airplane
    5:  3.5,   # bus
    6:  4.0,   # train
    7:  4.0,   # truck
    8:  2.0,   # boat
    9:  4.0,   # traffic light
    10: 0.8,   # fire hydrant
    11: 2.0,   # stop sign
    13: 0.5,   # bench
    14: 0.3,   # bird
    15: 0.4,   # cat
    16: 0.6,   # dog
    17: 1.8,   # horse
    18: 1.0,   # sheep
    19: 1.5,   # cow
    20: 3.0,   # elephant
    21: 1.8,   # bear
}
_DEFAULT_HEIGHT_M = 2.0   # fallback for unknown classes


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ObjectInfo:
    class_id:    int
    label:       str
    category:    str
    bbox:        tuple[int, int, int, int]    # x1 y1 x2 y2
    has_mask:    bool
    proximity:   float                        # 0=far  1=very close
    gap_left:    int                          # pixels from left edge to x1
    gap_right:   int                          # pixels from x2 to right edge
    gap_above:   int                          # pixels from top to y1
    gap_below:   int                          # pixels from y2 to bottom
    passable:          bool
    best_dir:          str                    # "left"|"right"|"above"|"below"|"blocked"
    best_gap_px:       int                    # size of the chosen gap
    estimated_height_m: float                 # typical real-world height of this object class


@dataclass
class ObstacleAnalysis:
    obstacle_mask:  np.ndarray               # H x W bool — dilated (A* input)
    raw_mask:       np.ndarray               # H x W bool — undilated (visualization)
    proximity_map:  np.ndarray               # H x W float32 [0,1]
    objects:        list[ObjectInfo]
    frame_h:        int
    frame_w:        int
    clearance_px:   int
    n_blocked:      int
    n_passable:     int

    @property
    def has_clear_path(self) -> bool:
        return not self.obstacle_mask.all()

    @property
    def summary(self) -> str:
        parts = []
        for obj in self.objects:
            s = f"{obj.label}({'PASS:'+obj.best_dir if obj.passable else 'BLOCK'})"
            parts.append(s)
        return "  ".join(parts) if parts else "no obstacles"


# ─── Mapper ───────────────────────────────────────────────────────────────────

class ObstacleMapper:
    """
    Full obstacle passability pipeline.

    Parameters
    ----------
    model_size      : YOLOv8 size char "n"|"s"|"m"|"l" (shared with SceneDetector)
    clearance_frac  : drone clearance as fraction of frame width (default 0.08 = 8%)
    prox_threshold  : objects with proximity < this value are considered far & ignored
    use_midas       : attempt MiDaS depth (falls back to heuristic if unavailable)
    """

    def __init__(
        self,
        model_size:     str   = "n",
        clearance_frac: float = 0.08,
        prox_threshold: float = 0.10,
        use_midas:      bool  = True,
    ):
        self._model_size    = model_size
        self._clearance_frac = clearance_frac
        self._prox_threshold = prox_threshold
        self._seg_model     = None          # lazy — loads yolov8n-seg.pt on first call
        self._depth_est     = DepthEstimator(use_midas=use_midas)

    # ── Public ───────────────────────────────────────────────────────────────

    def analyze(self, frame: np.ndarray) -> ObstacleAnalysis:
        """
        Run the full pipeline on one BGR frame.
        Returns ObstacleAnalysis with obstacle_mask + per-object passability.
        """
        h, w = frame.shape[:2]
        clearance_px = max(30, int(w * self._clearance_frac))

        # 1. Segmented detection
        dets = self._detect_segmented(frame)

        # 2. Depth / proximity map
        prox_map = self._depth_est.estimate(frame)

        # 3. Build per-object info + combined raw mask
        raw_mask = np.zeros((h, w), dtype=bool)
        objects: list[ObjectInfo] = []

        if len(dets) > 0 and dets.class_id is not None:
            for i in range(len(dets)):
                cid  = int(dets.class_id[i])
                conf_arr = dets.confidence if dets.confidence is not None else np.ones(len(dets))
                conf = float(conf_arr[i])
                bbox = tuple(dets.xyxy[i].astype(int))          # x1 y1 x2 y2
                x1, y1, x2, y2 = bbox

                # Per-object mask (use seg mask if available, else bbox)
                obj_mask = np.zeros((h, w), dtype=bool)
                if dets.mask is not None and i < len(dets.mask):
                    seg = dets.mask[i]
                    if seg.shape == (h, w):
                        obj_mask = seg.astype(bool)
                    else:
                        seg_r = cv2.resize(
                            seg.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                        obj_mask = seg_r
                    has_mask = True
                else:
                    obj_mask[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = True
                    has_mask = False

                raw_mask |= obj_mask

                # Proximity for this object
                proximity = self._depth_est.proximity_for_bbox(bbox, h, w)

                # Skip objects that are far away (below threshold)
                if proximity < self._prox_threshold:
                    continue

                # Gaps in 4 directions (frame boundary to object edge)
                gap_left  = x1
                gap_right = w - x2
                gap_above = y1
                gap_below = h - y2

                gaps = {
                    "left":  gap_left,
                    "right": gap_right,
                    "above": gap_above,
                    "below": gap_below,
                }
                best_dir = max(gaps, key=gaps.__getitem__)
                best_gap = gaps[best_dir]
                passable = best_gap >= clearance_px

                objects.append(ObjectInfo(
                    class_id           = cid,
                    label              = class_name(cid),
                    category           = get_category(cid),
                    bbox               = bbox,
                    has_mask           = has_mask,
                    proximity          = proximity,
                    gap_left           = gap_left,
                    gap_right          = gap_right,
                    gap_above          = gap_above,
                    gap_below          = gap_below,
                    passable           = passable,
                    best_dir           = best_dir if passable else "blocked",
                    best_gap_px        = best_gap,
                    estimated_height_m = COCO_TYPICAL_HEIGHT_M.get(cid, _DEFAULT_HEIGHT_M),
                ))

        # 4. Dilate raw_mask by drone clearance → occupancy input for A*
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (clearance_px, clearance_px)
        )
        dilated = cv2.dilate(raw_mask.astype(np.uint8), kernel) > 0

        n_blocked  = sum(1 for o in objects if not o.passable)
        n_passable = sum(1 for o in objects if o.passable)

        return ObstacleAnalysis(
            obstacle_mask = dilated,
            raw_mask      = raw_mask,
            proximity_map = prox_map,
            objects       = objects,
            frame_h       = h,
            frame_w       = w,
            clearance_px  = clearance_px,
            n_blocked     = n_blocked,
            n_passable    = n_passable,
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _detect_segmented(self, frame: np.ndarray) -> "sv.Detections":
        """Lazy-load yolov8n-seg.pt and run inference."""
        import supervision as sv
        from pathlib import Path
        MODEL_DIR = Path(__file__).parent.parent / "vision" / "models"

        if self._seg_model is None:
            from ultralytics import YOLO
            import torch
            seg_name = f"yolov8{self._model_size}-seg.pt"
            try:
                self._seg_model = YOLO(str(MODEL_DIR / seg_name))
            except Exception:
                self._seg_model = YOLO(seg_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  [ObstacleMapper] YOLOv8{self._model_size}-seg loaded on {device}")
            self._device = device

        results = self._seg_model(
            frame,
            conf    = 0.30,
            iou     = 0.45,
            device  = self._device,
            verbose = False,
        )
        return sv.Detections.from_ultralytics(results[0])


# ─── Visualisation helpers ────────────────────────────────────────────────────

# Colour per category (BGR)
_CAT_COLOUR = {
    "person":         (0,   220, 220),
    "vehicle":        (255, 220, 0  ),
    "animal":         (255, 0,   220),
    "infrastructure": (0,   165, 255),
    "object":         (200, 200, 200),
}


def draw_masks(
    frame: np.ndarray,
    analysis: ObstacleAnalysis,
    alpha: float = 0.40,
) -> np.ndarray:
    """
    Draw per-object segmentation masks as semi-transparent colour fills.
    Each category gets its own colour.
    """
    if len(analysis.objects) == 0:
        return frame

    overlay = frame.copy()
    for obj in analysis.objects:
        colour = _CAT_COLOUR.get(obj.category, (200,200,200))
        x1, y1, x2, y2 = obj.bbox
        # Fill bbox with category colour (mask already baked into raw_mask)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)

    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Draw labels
    for obj in analysis.objects:
        x1, y1, x2, y2 = obj.bbox
        colour = _CAT_COLOUR.get(obj.category, (200,200,200))
        status = f"PASS:{obj.best_dir}" if obj.passable else "BLOCK"
        label  = f"{obj.label} | {status} | prox:{obj.proximity:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cy = max(y1 - 4, lh + 4)
        cv2.rectangle(frame, (x1, cy - lh - 4), (x1 + lw + 4, cy + 2), colour, -1)
        cv2.putText(frame, label, (x1 + 2, cy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)

    return frame


def draw_proximity_heatmap(
    frame: np.ndarray,
    prox_map: np.ndarray,
    alpha: float = 0.30,
) -> np.ndarray:
    """
    Overlay a red-tinted heatmap showing object proximity.
    Bright red = very close, dark = far.
    """
    heat = (prox_map * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_HOT)
    return cv2.addWeighted(heat_bgr, alpha, frame, 1 - alpha, 0)


# ─── Singleton ────────────────────────────────────────────────────────────────

_mapper: ObstacleMapper | None = None


def get_mapper(model_size: str = "n") -> ObstacleMapper:
    global _mapper
    if _mapper is None:
        _mapper = ObstacleMapper(model_size=model_size)
    return _mapper
