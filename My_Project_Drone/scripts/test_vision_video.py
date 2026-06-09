"""
Phase 4 — Vision Pipeline Video Tester

Two modes
---------
  Default (--obstacle not set):
    YOLOv8 detect all COCO objects + ByteTrack person roles + scene classification

  Obstacle mode (--obstacle):
    YOLOv8-seg segmentation masks + depth proximity + obstacle passability
    + A* optimal path from drone position (bottom-centre) to victim (top-centre)
    Overlays:
      RED tint      — blocked zones (dilated obstacle mask)
      GREEN tint    — flyable corridor
      GREEN line    — A* optimal path
      Coloured fill — per-object mask (person/vehicle/animal/infra)
      Labels        — object name | PASS:left / BLOCK | proximity score
      HEATMAP       — depth heatmap (--depth flag)

Detected categories (all COCO-80)
-------------------
  person, vehicle (car/truck/bus/motorcycle/bicycle/boat),
  animal (bird/dog/cat/horse/cow/elephant/bear), infrastructure

NOT detected — needs YOLO-World (Phase 4.2):
  buildings, doors, entry/exit points, windows, fences

Usage
-----
  python scripts/test_vision_video.py path/to/video.mp4
  python scripts/test_vision_video.py path/to/video.mp4 --obstacle
  python scripts/test_vision_video.py path/to/video.mp4 --obstacle --depth
  python scripts/test_vision_video.py --webcam --obstacle
  python scripts/test_vision_video.py video.mp4 --obstacle --save
  python scripts/test_vision_video.py video.mp4 --model m   (accurate)

Keyboard controls
-----------------
  q / ESC  — quit
  space    — pause / resume
  s        — screenshot -> docs/snapshot_NNN.png
  o        — toggle obstacle overlay on/off at runtime

Run from project root:
  C:/Users/akhil/anaconda3/envs/LLM_GPU/python.exe scripts/test_vision_video.py <video>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import supervision as sv

from vision.night_mode       import preprocess
from vision.detector         import SceneDetector, get_category, class_name
from vision.tracker          import PersonTracker, ROLE_VICTIM, ROLE_SUSPECT, ROLE_BYSTANDER
from vision.scene_classifier import VisionSceneClassifier
from vision.obstacle_mapper  import ObstacleMapper, draw_masks, draw_proximity_heatmap
from vision.occupancy_grid   import (plan_path, assign_altitudes,
                                     draw_path, draw_obstacle_overlay, draw_clear_overlay)


# ─── Colour palette ───────────────────────────────────────────────────────────

ROLE_COLOUR = {
    ROLE_VICTIM:    (0,   255, 255),
    ROLE_SUSPECT:   (0,   0,   255),
    ROLE_BYSTANDER: (0,   255, 0  ),
}
CATEGORY_COLOUR = {
    "vehicle":        (255, 255, 0  ),
    "animal":         (255, 0,   255),
    "infrastructure": (0,   165, 255),
    "object":         (200, 200, 200),
}


# ─── Normal-mode detection overlay ───────────────────────────────────────────

def _draw_normal_detections(
    frame:   np.ndarray,
    dets:    "sv.Detections",
    tracked: "sv.Detections",
    roles:   dict[int, str],
) -> np.ndarray:
    person_track: dict[tuple, int] = {}
    if tracked.tracker_id is not None:
        for tid, box in zip(tracked.tracker_id, tracked.xyxy):
            person_track[tuple(box.astype(int))] = int(tid)

    if dets.class_id is None or len(dets) == 0:
        return frame

    conf_arr = dets.confidence if dets.confidence is not None else np.zeros(len(dets))
    for i, (box, conf) in enumerate(zip(dets.xyxy, conf_arr)):
        cid   = int(dets.class_id[i])
        x1, y1, x2, y2 = map(int, box)
        tid    = person_track.get(tuple(map(int, box)))

        if cid == 0:
            role   = roles.get(tid, ROLE_BYSTANDER) if tid is not None else ROLE_BYSTANDER
            colour = ROLE_COLOUR[role]
            label  = f"{role.upper()}" + (f" #{tid}" if tid is not None else "")
        else:
            colour = CATEGORY_COLOUR.get(get_category(cid), (200, 200, 200))
            label  = f"{class_name(cid)} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return frame


# ─── HUD ─────────────────────────────────────────────────────────────────────

def _draw_hud(
    frame:       np.ndarray,
    mode:        str,
    scene_type:  str,
    threat:      int,
    description: str,
    action:      str,
    cat_counts:  dict[str, int],
    path_status: str,
    night:       bool,
    fps:         float,
    frame_n:     int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 170), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.60, frame, 0.40, 0)

    col  = (255, 200, 80) if night else (255, 255, 255)
    nite = "NIGHT[CLAHE]" if night else "DAY"
    tbar = "|" * threat + "-" * (5 - threat)
    cnts = "  ".join(f"{k}:{v}" for k, v in cat_counts.items() if v > 0) or "none"

    cv2.putText(frame, f"Guardian Drone  [{nite}] [{mode}]   FPS:{fps:.1f}   F:{frame_n}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)
    cv2.putText(frame, f"Scene   : {scene_type}",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)
    cv2.putText(frame, f"Threat  : [{tbar}] {threat}/5",
                (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)
    cv2.putText(frame, f"Objects : {cnts}",
                (8, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)
    cv2.putText(frame, f"Path    : {path_status}",
                (8, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (0, 255, 100) if "CLEAR" in path_status else (0, 80, 255), 1)
    cv2.putText(frame, f"Desc    : {description[:80]}",
                (8, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)
    cv2.putText(frame, f"Action  : {action[:80]}",
                (8, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)

    # Colour legend (top-right)
    legend = [
        ("VICTIM",    ROLE_COLOUR[ROLE_VICTIM]),
        ("SUSPECT",   ROLE_COLOUR[ROLE_SUSPECT]),
        ("BYSTANDER", ROLE_COLOUR[ROLE_BYSTANDER]),
        ("vehicle",   CATEGORY_COLOUR["vehicle"]),
        ("animal",    CATEGORY_COLOUR["animal"]),
        ("infra",     CATEGORY_COLOUR["infrastructure"]),
        ("A* PATH",   (0, 230, 60)),
        ("BLOCKED",   (0, 0, 200)),
    ]
    for idx, (lbl, c) in enumerate(legend):
        lx = w - 130
        ly = 20 + idx * 20
        cv2.rectangle(frame, (lx, ly - 12), (lx + 14, ly), c, -1)
        cv2.putText(frame, lbl, (lx + 18, ly - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, c, 1)
    return frame


# ─── Main loop ────────────────────────────────────────────────────────────────

def run(
    source:       str | int,
    model_size:   str   = "n",
    danger_score: float = 88.0,
    obstacle_mode: bool = False,
    show_depth:   bool  = False,
    save:         bool  = False,
    no_display:   bool  = False,
) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    fps_native   = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mode_str = "OBSTACLE+PATH" if obstacle_mode else "DETECT+TRACK"
    print(f"\nGuardian Drone Vision Test")
    print(f"  Source       : {source}")
    print(f"  Resolution   : {fw}x{fh}  @{fps_native:.1f}fps  ({total_frames} frames)")
    print(f"  Model        : YOLOv8-{model_size}")
    print(f"  Mode         : {mode_str}")
    print(f"  Detected     : all COCO classes (person/vehicle/animal/infra)")
    print(f"  NOT detected : buildings/doors/entry-exit (Phase 4.2 YOLO-World)")
    print(f"\nLoading models...\n")

    detector   = SceneDetector(model_size=model_size)
    tracker    = PersonTracker(frame_rate=int(fps_native))
    classifier = VisionSceneClassifier(model_size=model_size,
                                       frame_rate=int(fps_native))
    mapper     = ObstacleMapper(model_size=model_size) if obstacle_mode else None

    writer = None
    if save:
        out_path = str(Path("docs") / "vision_output.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps_native, (fw, fh))
        print(f"  Saving to    : {out_path}")

    print("Press  q/ESC=quit   space=pause   s=screenshot   o=toggle obstacle\n")

    frame_n       = 0
    paused        = False
    snap_n        = 0
    obs_active    = obstacle_mode
    t_fps         = time.perf_counter()
    fps_disp      = 0.0
    display       = np.zeros((fh, fw, 3), dtype=np.uint8)

    last_scene    = "initialising"
    last_threat   = 1
    last_action   = ""
    last_desc     = ""
    last_counts: dict[str, int] = {}
    last_path_status = "N/A"

    while True:
        if not paused:
            ret, raw_frame = cap.read()
            if not ret:
                print("\n[END] Video finished.")
                break
            frame_n += 1

            # ── Night mode pre-process ───────────────────────────────────────
            processed, night_active = preprocess(raw_frame)

            # ── FPS ─────────────────────────────────────────────────────────
            if frame_n % 15 == 0:
                elapsed  = time.perf_counter() - t_fps
                fps_disp = 15.0 / max(elapsed, 1e-6)
                t_fps    = time.perf_counter()

            # ════════════════════════════════════════════════════════════════
            #  OBSTACLE MODE
            # ════════════════════════════════════════════════════════════════
            if obs_active and mapper is not None:
                # 1. Segment + passability analysis
                analysis = mapper.analyze(processed)

                # 2. A* 2D path:  start = bottom-centre,  goal = top-centre
                start = (fh - 1, fw // 2)
                goal  = (0,      fw // 2)
                waypoints_2d = plan_path(analysis.obstacle_mask, start, goal)

                # 3. Extend to 3D waypoints (row, col, alt_m)
                waypoints_3d = assign_altitudes(
                    waypoints_2d, analysis.objects, fh, fw,
                    default_alt_m=10.0, clearance_m=2.0,
                )

                # 4. Build path status string
                if not analysis.objects:
                    last_path_status = "CLEAR — no obstacles detected"
                elif waypoints_3d:
                    alts = [a for _, _, a in waypoints_3d]
                    last_path_status = (
                        f"CLEAR 3D PATH — {len(waypoints_3d)} pts  "
                        f"alt {min(alts):.0f}–{max(alts):.0f}m  "
                        f"[{analysis.n_blocked} blocked, {analysis.n_passable} passable]"
                    )
                else:
                    last_path_status = (
                        f"NO PATH — fully blocked  "
                        f"[{analysis.n_blocked} blocked objects]"
                    )

                # 4. Category counts from segmented detections
                cat_counts: dict[str, int] = {
                    "person":0,"vehicle":0,"animal":0,"infrastructure":0
                }
                for obj in analysis.objects:
                    if obj.category in cat_counts:
                        cat_counts[obj.category] += 1

                # 5. Scene classification (pass processed frame)
                scene_result = classifier.classify(
                    processed, danger_score=danger_score,
                    drone_dist_km=0.05, battery_pct=90.0,
                )
                last_scene  = scene_result.scene_type
                last_threat = scene_result.threat_level
                last_action = scene_result.recommended_action
                last_desc   = scene_result.description
                last_counts = cat_counts

                # 6. Annotate frame
                display = processed.copy()

                if show_depth:
                    display = draw_proximity_heatmap(display, analysis.proximity_map, alpha=0.25)

                display = draw_obstacle_overlay(display, analysis.obstacle_mask,
                                                colour=(30, 30, 200), alpha=0.30)
                display = draw_clear_overlay(display, analysis.obstacle_mask,
                                             colour=(0, 160, 0), alpha=0.12)
                display = draw_masks(display, analysis, alpha=0.35)

                if waypoints_3d:
                    # draw_path takes (row, col) — strip altitude for 2D overlay
                    display = draw_path(display, [(r, c) for r, c, _ in waypoints_3d])
                    # Annotate altitude at every ~20th waypoint along the path
                    for r, c, alt in waypoints_3d[::max(1, len(waypoints_3d)//8)]:
                        cv2.putText(display, f"{alt:.0f}m", (c + 5, r),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 60), 1)
                else:
                    h_, w_ = display.shape[:2]
                    cv2.putText(display, "NO CLEAR PATH — REROUTING",
                                (w_//2 - 160, h_//2),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)

            # ════════════════════════════════════════════════════════════════
            #  NORMAL DETECT + TRACK MODE
            # ════════════════════════════════════════════════════════════════
            else:
                dets = detector.detect(processed)

                cat_counts = {"person":0,"vehicle":0,"animal":0,"infrastructure":0}
                if dets.class_id is not None:
                    for cid in dets.class_id:
                        cat = get_category(int(cid))
                        if cat in cat_counts:
                            cat_counts[cat] += 1

                person_dets = (dets[dets.class_id == 0]
                               if dets.class_id is not None and len(dets) > 0
                               else sv.Detections.empty())
                tracked = tracker.update(person_dets, processed)

                scene_result = classifier.classify(
                    processed, danger_score=danger_score,
                    drone_dist_km=0.05, battery_pct=90.0,
                )
                last_scene  = scene_result.scene_type
                last_threat = scene_result.threat_level
                last_action = scene_result.recommended_action
                last_desc   = scene_result.description
                last_counts = cat_counts
                last_path_status = "N/A (use --obstacle for path planning)"

                display = processed.copy()
                display = _draw_normal_detections(display, dets, tracked, tracker.roles)

            # ── Console log every ~1 second ──────────────────────────────────
            if frame_n % max(1, int(fps_native)) == 0:
                print(
                    f"  F{frame_n:5d}  {last_scene:<22s}  threat={last_threat}/5  "
                    f"p={last_counts.get('person',0)} "
                    f"v={last_counts.get('vehicle',0)} "
                    f"a={last_counts.get('animal',0)}  "
                    + (f"path={last_path_status[:40]}" if obs_active else "")
                )

            display = _draw_hud(
                display, mode_str, last_scene, last_threat,
                last_desc, last_action, last_counts,
                last_path_status, night_active, fps_disp, frame_n,
            )

            if writer:
                writer.write(display)

        # ── Key handling ─────────────────────────────────────────────────────
        if not no_display:
            cv2.imshow("Guardian Drone Vision", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("\n[QUIT]")
                break
            elif key == ord(" "):
                paused = not paused
                print(f"  {'PAUSED' if paused else 'RESUMED'}")
            elif key == ord("s"):
                sp = f"docs/snapshot_{snap_n:03d}.png"
                cv2.imwrite(sp, display)
                snap_n += 1
                print(f"  Snapshot -> {sp}")
            elif key == ord("o"):
                obs_active = not obs_active
                mode_str   = "OBSTACLE+PATH" if obs_active else "DETECT+TRACK"
                print(f"  Obstacle mode: {'ON' if obs_active else 'OFF'}")
                if obs_active and mapper is None:
                    mapper = ObstacleMapper(model_size=model_size)
        else:
            if frame_n >= total_frames:
                break

    cap.release()
    if writer:
        writer.release()
        print(f"\nSaved -> {out_path}")
    if not no_display:
        cv2.destroyAllWindows()
    print(f"\nProcessed {frame_n} frames. Done.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Guardian Drone Vision Pipeline — Video Tester"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("video",    nargs="?",           help="Path to video file")
    group.add_argument("--webcam", action="store_true", help="Use webcam (index 0)")

    p.add_argument("--model",    default="n", choices=["n","s","m","l"],
                   help="YOLOv8 model size (n=nano fastest, m=medium best accuracy)")
    p.add_argument("--obstacle", action="store_true",
                   help="Enable obstacle mapper + A* path planning overlay")
    p.add_argument("--depth",    action="store_true",
                   help="Show depth proximity heatmap (requires --obstacle)")
    p.add_argument("--score",    type=float, default=88.0,
                   help="Fixed danger score 0-100 (default 88)")
    p.add_argument("--save",     action="store_true",
                   help="Save annotated output to docs/vision_output.mp4")
    p.add_argument("--no-display", action="store_true",
                   help="Headless mode — no cv2 window")
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    source = 0 if args.webcam else args.video
    run(
        source        = source,
        model_size    = args.model,
        danger_score  = args.score,
        obstacle_mode = args.obstacle,
        show_depth    = args.depth,
        save          = args.save,
        no_display    = args.no_display,
    )
