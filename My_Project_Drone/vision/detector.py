"""
Phase 4.3 — YOLOv8 Scene Detector
Detects all outdoor-relevant COCO-80 objects in a drone camera frame.

COCO-80 coverage
----------------
  persons, vehicles (car/bus/truck/motorcycle/bicycle/boat),
  animals (bird/dog/cat/horse/cow/elephant/bear/sheep/zebra/giraffe),
  infrastructure (traffic light/fire hydrant/stop sign/bench)

NOT in COCO-80 (requires custom model or YOLO-World):
  buildings, doors, building entry/exit points, windows, fences
  TODO (Phase 4.2): Add YOLO-World open-vocab detector for
    "building entrance", "door", "gate", "window", "fence"
    without needing a custom labelled dataset.

Run standalone to verify:
    C:/Users/akhil/anaconda3/envs/LLM_GPU/python.exe vision/detector.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_CONF = 0.35
DEFAULT_IOU  = 0.45
MODEL_DIR    = Path(__file__).parent.parent / "vision" / "models"

# ─── COCO-80 class registry ───────────────────────────────────────────────────

COCO_CLASS_NAMES: dict[int, str] = {
    0: "person",
    1: "bicycle",    2: "car",        3: "motorcycle",  4: "airplane",
    5: "bus",        6: "train",      7: "truck",        8: "boat",
    9: "traffic light", 10: "fire hydrant", 11: "stop sign", 12: "parking meter",
    13: "bench",
    14: "bird",      15: "cat",       16: "dog",         17: "horse",
    18: "sheep",     19: "cow",       20: "elephant",    21: "bear",
    22: "zebra",     23: "giraffe",
    24: "backpack",  25: "umbrella",  26: "handbag",     27: "tie",
    28: "suitcase",  29: "frisbee",
    32: "sports ball", 35: "baseball bat",
    39: "bottle",    41: "cup",
    56: "chair",     57: "couch",     58: "potted plant", 60: "dining table",
    63: "laptop",    67: "cell phone",
    72: "refrigerator", 73: "book",  74: "clock",
}

# Category groupings used by scene classifier + video overlay
CATEGORY_MAP: dict[str, set[int]] = {
    "person":         {0},
    "vehicle":        {1, 2, 3, 4, 5, 6, 7, 8},
    "animal":         {14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
    "infrastructure": {9, 10, 11, 12, 13},
    "object":         set(),   # catch-all; filled in get_category()
}


def get_category(class_id: int) -> str:
    """Return the category name for a COCO class ID."""
    for cat, ids in CATEGORY_MAP.items():
        if cat == "object":
            continue
        if class_id in ids:
            return cat
    return "object"


def class_name(class_id: int) -> str:
    """Human-readable COCO class label."""
    return COCO_CLASS_NAMES.get(class_id, f"cls_{class_id}")


# ─── Detector ─────────────────────────────────────────────────────────────────

class SceneDetector:
    """
    YOLOv8 detector that returns ALL outdoor-relevant objects in the scene,
    not just persons.

    Parameters
    ----------
    model_size  : "n" | "s" | "m" | "l"
    conf        : confidence threshold
    iou         : NMS IOU threshold
    device      : "cuda" | "cpu"
    classes     : list of COCO class IDs to detect, or None for all classes
    """

    def __init__(
        self,
        model_size: str   = "m",
        conf:       float = DEFAULT_CONF,
        iou:        float = DEFAULT_IOU,
        device:     str   = "cuda",
        classes:    list[int] | None = None,
    ):
        from ultralytics import YOLO
        import torch

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        model_name        = f"yolov8{model_size}.pt"
        self._conf        = conf
        self._iou         = iou
        self._device      = device if torch.cuda.is_available() else "cpu"
        self._classes     = classes  # None = detect all 80 COCO classes
        self._model_size_char = model_size

        try:
            self._model = YOLO(str(MODEL_DIR / model_name))
        except Exception:
            self._model = YOLO(model_name)

        scope = "all classes" if classes is None else f"{len(classes)} classes"
        print(f"  [Detector] YOLOv8{model_size} loaded on {self._device} ({scope})")

    def detect(self, frame: np.ndarray) -> "sv.Detections":
        """
        Run inference on a single BGR frame.
        Returns supervision.Detections for all detected objects.
        class_id field indicates COCO class; use get_category() to group.
        """
        import supervision as sv

        results = self._model(
            frame,
            classes = self._classes,
            conf    = self._conf,
            iou     = self._iou,
            device  = self._device,
            verbose = False,
        )
        return sv.Detections.from_ultralytics(results[0])

    def detect_persons(self, frame: np.ndarray) -> "sv.Detections":
        """Convenience: returns only person detections (class_id == 0)."""
        import supervision as sv
        dets = self.detect(frame)
        if dets.class_id is None or len(dets) == 0:
            return sv.Detections.empty()
        return dets[dets.class_id == 0]

    def detect_segmented(self, frame: np.ndarray) -> "sv.Detections":
        """
        Run YOLOv8-seg inference for pixel-level object masks.
        Uses yolov8{size}-seg.pt (downloaded automatically on first call, ~7 MB for nano).
        Returns Detections with .mask field populated as (N, H, W) bool array.
        Used by ObstacleMapper for accurate gap measurement.
        """
        import supervision as sv

        if not hasattr(self, "_seg_model"):
            from ultralytics import YOLO
            seg_name = f"yolov8{self._model_size_char}-seg.pt"
            try:
                self._seg_model = YOLO(str(MODEL_DIR / seg_name))
            except Exception:
                self._seg_model = YOLO(seg_name)
            print(f"  [Detector] YOLOv8{self._model_size_char}-seg loaded on {self._device}")

        results = self._seg_model(
            frame,
            classes = self._classes,
            conf    = self._conf,
            iou     = self._iou,
            device  = self._device,
            verbose = False,
        )
        return sv.Detections.from_ultralytics(results[0])


# ─── Planned: segmentation support (Phase 4 obstacle mapper) ─────────────────
# TODO (Phase 4 obstacle_mapper): Add detect_segmented() using yolov8n-seg.pt.
# Segmentation gives pixel-level masks per object so obstacle_mapper can
# estimate real-world gap sizes (e.g. clearance under a car) and decide
# whether the drone can physically pass through or must route around.
# Model swap: YOLO("yolov8n-seg.pt") — same weights size, adds .masks field
# to sv.Detections. See vision/obstacle_mapper.py for full design.

# ─── Backward-compat alias ────────────────────────────────────────────────────
# PersonDetector kept so existing tests and agent_graph.py don't break.

class PersonDetector(SceneDetector):
    """Alias for SceneDetector kept for backward compatibility."""
    def __init__(self, model_size="m", conf=DEFAULT_CONF, iou=DEFAULT_IOU, device="cuda"):
        super().__init__(model_size=model_size, conf=conf, iou=iou,
                         device=device, classes=None)

    def detect(self, frame: np.ndarray) -> "sv.Detections":
        """Returns ALL scene detections (same as SceneDetector.detect)."""
        return super().detect(frame)


# ─── Singleton accessors ──────────────────────────────────────────────────────

_detector: SceneDetector | None = None


def get_detector(model_size: str = "m") -> SceneDetector:
    global _detector
    if _detector is None:
        _detector = SceneDetector(model_size=model_size)
    return _detector


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cv2
    from vision.camera_mock import CameraMock

    mock     = CameraMock(scene="distress")
    detector = SceneDetector(model_size="n")

    frame = mock.next_frame()
    dets  = detector.detect(frame)
    print(f"Frame shape     : {frame.shape}")
    print(f"Objects detected: {len(dets)}")
    for box, conf, cid in zip(dets.xyxy, dets.confidence, dets.class_id or []):
        x1, y1, x2, y2 = map(int, box)
        print(f"  {class_name(cid):20s} conf={conf:.2f}  "
              f"box=({x1},{y1},{x2},{y2})  cat={get_category(cid)}")

    import supervision as sv
    ann = sv.BoundingBoxAnnotator()
    lbl = sv.LabelAnnotator()
    labels = [f"{class_name(c)} {conf:.2f}"
              for c, conf in zip(dets.class_id or [], dets.confidence or [])]
    out = ann.annotate(frame.copy(), dets)
    out = lbl.annotate(out, dets, labels=labels)
    Path("docs").mkdir(exist_ok=True)
    cv2.imwrite("docs/detector_test.png", out)
    print("Saved -> docs/detector_test.png")
