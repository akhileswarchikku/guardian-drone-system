"""
Phase 4.3 — Vision Scene Classifier
Integrates PersonDetector + PersonTracker + NightMode into a single
classify() call that returns the SceneClassification Pydantic model
used by Agent 9 (scene_intelligence_node).

This replaces the LLM guess in Agent 9 with real visual evidence.

Usage
-----
    from vision.scene_classifier import VisionSceneClassifier

    clf = VisionSceneClassifier()
    result = clf.classify(frame, danger_score=91.0)
    # result is agents.state.SceneClassification

Run standalone:
    C:/Users/akhil/anaconda3/envs/LLM_GPU/python.exe vision/scene_classifier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from typing import Optional

import numpy as np

from agents.state import SceneClassification
from vision.night_mode import preprocess
from vision.detector import get_category, class_name, CATEGORY_MAP


# Thresholds for scene type + threat decision
_SUSPECT_THREAT_BOOST  = 1       # extra threat levels if suspect detected
_NIGHT_THREAT_BOOST    = 1       # extra threat level in night mode (reduced visibility)
_HIGH_PERSON_COUNT     = 4       # scene is "crowded" above this
_VEHICLE_THREAT_BOOST  = 1       # extra threat level if vehicles near victim


class VisionSceneClassifier:
    """
    Full vision pipeline: CLAHE → YOLOv8 detect → ByteTrack → SceneClassification.

    Parameters
    ----------
    model_size  : YOLOv8 model variant ("n" | "m")
    frame_rate  : expected FPS of incoming frames (for tracker tuning)
    """

    def __init__(self, model_size: str = "m", frame_rate: int = 10):
        from vision.detector import PersonDetector
        from vision.tracker  import PersonTracker

        self._detector = PersonDetector(model_size=model_size)
        self._tracker  = PersonTracker(frame_rate=frame_rate)
        self._frame_n  = 0

    def classify(
        self,
        frame:        np.ndarray,
        danger_score: float = 85.0,
        drone_dist_km: float = 0.0,
        battery_pct:  float = 100.0,
        mission_status: str = "arrived",
    ) -> SceneClassification:
        """
        Run the full vision pipeline on one frame.
        Returns SceneClassification matching the Agent 9 contract.
        """
        t0 = time.perf_counter()

        # 1. Night mode pre-processing
        processed, night_active = preprocess(frame)

        # 2. Full scene detection (all COCO outdoor classes)
        detections = self._detector.detect(processed)

        # 3. Count objects by category
        cat_counts: dict[str, int] = {"person": 0, "vehicle": 0,
                                       "animal": 0, "infrastructure": 0}
        if detections.class_id is not None:
            for cid in detections.class_id:
                cat = get_category(int(cid))
                if cat in cat_counts:
                    cat_counts[cat] += 1

        person_count  = cat_counts["person"]
        vehicle_count = cat_counts["vehicle"]
        animal_count  = cat_counts["animal"]

        # 4. Person tracking + role assignment (ByteTrack is person-specific)
        import supervision as sv
        if detections.class_id is not None and len(detections) > 0:
            person_dets = detections[detections.class_id == 0]
        else:
            person_dets = sv.Detections.empty()

        tracked = self._tracker.update(person_dets, processed)
        roles   = self._tracker.roles   # {track_id: "victim"|"suspect"|"bystander"}

        suspect_present = "suspect" in roles.values()
        victim_tracked  = "victim"  in roles.values()

        # 5. Scene type classification
        scene_type = _classify_scene(
            person_count, suspect_present, night_active, drone_dist_km,
            vehicle_count, animal_count,
        )

        # 6. Threat level (1–5)
        threat = _compute_threat(
            danger_score, person_count, suspect_present, night_active, vehicle_count
        )

        # 7. Build description
        elapsed_ms = (time.perf_counter() - t0) * 1000
        parts = [f"{person_count} person(s)"]
        if vehicle_count:
            parts.append(f"{vehicle_count} vehicle(s)")
        if animal_count:
            parts.append(f"{animal_count} animal(s)")
        parts.append(f"victim {'tracked' if victim_tracked else 'not confirmed'}")
        if suspect_present:
            parts.append("SUSPECT approaching victim")
        if night_active:
            parts.append("night mode active (CLAHE)")
        parts.append(f"inference {elapsed_ms:.1f}ms")
        description = ". ".join(parts) + "."

        # 8. Recommended action
        action = _recommend_action(scene_type, threat, suspect_present, drone_dist_km)

        self._frame_n += 1

        return SceneClassification(
            scene_type         = scene_type,
            threat_level       = threat,
            description        = description,
            recommended_action = action,
        )

    def reset_tracker(self) -> None:
        """Call when a new mission starts to clear track history."""
        self._tracker.reset()


# ─── Scene type logic ─────────────────────────────────────────────────────────

def _classify_scene(
    n_persons:  int,
    suspect:    bool,
    night:      bool,
    dist_km:    float,
    n_vehicles: int = 0,
    n_animals:  int = 0,
) -> str:
    if dist_km > 0.5:
        return "drone_en_route"
    if n_persons == 0 and n_vehicles == 0 and n_animals == 0:
        return "empty_area"
    if n_animals > 0 and n_persons == 0:
        scene = "wildlife_area"
    elif n_vehicles > 2 and n_persons > 0:
        scene = "roadside_area"
    elif n_persons > _HIGH_PERSON_COUNT:
        scene = "crowded_public"
    elif suspect:
        scene = "active_threat"
    else:
        scene = "isolated_area"
    return scene + ("_night" if night else "")


def _compute_threat(
    score:      float,
    n_persons:  int,
    suspect:    bool,
    night:      bool,
    n_vehicles: int = 0,
) -> int:
    base = 1
    if score >= 85:   base = 4
    elif score >= 70: base = 3
    else:             base = 2

    if suspect:          base += _SUSPECT_THREAT_BOOST
    if night:            base += _NIGHT_THREAT_BOOST
    if n_vehicles > 0:   base += _VEHICLE_THREAT_BOOST   # vehicles = harder to help victim
    if n_persons == 0:   base = max(1, base - 1)         # no persons = lower certainty

    return min(5, max(1, base))


def _recommend_action(
    scene_type: str,
    threat:     int,
    suspect:    bool,
    dist_km:    float,
) -> str:
    if dist_km > 0.3:
        return "Proceed to victim location at maximum speed."
    if "active_threat" in scene_type or suspect:
        return "Broadcast alert siren, illuminate scene, stream live to police control room."
    if threat >= 4:
        return "Hover at 8m altitude, maintain visual on victim, alert ground units."
    if "crowded" in scene_type:
        return "Orbit at 15m to maintain crowd overview, mark victim location on dashboard."
    return "Maintain visual contact at 10m, await ground unit confirmation."


# ─── Singleton accessor ───────────────────────────────────────────────────────

_clf: VisionSceneClassifier | None = None


def get_classifier(model_size: str = "m") -> VisionSceneClassifier:
    global _clf
    if _clf is None:
        _clf = VisionSceneClassifier(model_size=model_size)
    return _clf


# ─── Standalone demo ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cv2
    from vision.camera_mock import CameraMock

    print("\nGuardian Drone — Vision Scene Classifier Demo")
    print("=" * 52)

    clf = VisionSceneClassifier(model_size="n")  # nano for fast demo

    for scene_name in ("safe", "distress", "night"):
        print(f"\nScene: {scene_name.upper()}")
        print("-" * 40)
        mock = CameraMock(scene=scene_name)

        # Warm up tracker with 5 frames before reading result
        for _ in range(5):
            frame  = mock.next_frame()
            result = clf.classify(frame, danger_score=91.0 if scene_name != "safe" else 45.0)

        print(f"  scene_type  : {result.scene_type}")
        print(f"  threat_level: {result.threat_level}/5")
        print(f"  description : {result.description}")
        print(f"  action      : {result.recommended_action}")

        # Save annotated frame
        import supervision as sv
        from vision.detector import get_detector
        from vision.night_mode import preprocess as nm_preprocess

        frame_out, night = nm_preprocess(frame)
        dets = get_detector("n").detect(frame_out)
        ann  = sv.BoundingBoxAnnotator(thickness=2)
        lbl  = sv.LabelAnnotator()
        frame_ann = ann.annotate(frame_out.copy(), dets)
        labels = [f"person {c:.2f}" for c in (dets.confidence or [])]
        frame_ann = lbl.annotate(frame_ann, dets, labels=labels)

        out_path = f"docs/vision_{scene_name}.png"
        cv2.imwrite(out_path, frame_ann)
        print(f"  saved -> {out_path}")

    clf.reset_tracker()
    print("\nDemo complete.")
