"""
Phase 4 — Synthetic Camera Mock
Generates BGR frames that simulate a drone camera feed,
so the vision pipeline can be tested without real hardware.

Scenes
------
  "safe"     — 0–1 person, stationary (bystander, no threat)
  "distress" — 2–3 persons; one stationary (victim), one approaching (suspect)
  "night"    — same as distress but dark (triggers CLAHE night mode)

Run standalone:
    C:/Users/akhil/anaconda3/envs/LLM_GPU/python.exe vision/camera_mock.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from dataclasses import dataclass, field

FRAME_W = 640
FRAME_H = 480
FPS     = 10   # mock runs at 10 fps


@dataclass
class _Person:
    cx: float
    cy: float
    w:  int
    h:  int
    vx: float = 0.0   # pixels per frame
    vy: float = 0.0

    def update(self, frame_w: int, frame_h: int) -> None:
        self.cx = float(np.clip(self.cx + self.vx, self.w // 2, frame_w - self.w // 2))
        self.cy = float(np.clip(self.cy + self.vy, self.h // 2, frame_h - self.h // 2))
        if self.cx in (self.w // 2, frame_w - self.w // 2):
            self.vx *= -1
        if self.cy in (self.h // 2, frame_h - self.h // 2):
            self.vy *= -1

    @property
    def box_xyxy(self) -> tuple[int, int, int, int]:
        x1 = int(self.cx - self.w / 2)
        y1 = int(self.cy - self.h / 2)
        return x1, y1, x1 + self.w, y1 + self.h


class CameraMock:
    """
    Generates synthetic drone-view frames.

    Parameters
    ----------
    scene   : "safe" | "distress" | "night"
    width   : frame width  (default 640)
    height  : frame height (default 480)
    seed    : random seed for reproducible sequences
    """

    def __init__(
        self,
        scene:  str = "distress",
        width:  int = FRAME_W,
        height: int = FRAME_H,
        seed:   int = 42,
    ):
        self.scene  = scene
        self.width  = width
        self.height = height
        self._rng   = np.random.default_rng(seed)
        self._frame_idx = 0
        self._persons: list[_Person] = []
        self._bg_brightness = 20 if scene == "night" else 80
        self._setup_scene()

    def _setup_scene(self) -> None:
        w, h = self.width, self.height
        if self.scene == "safe":
            # One bystander, stationary
            self._persons = [_Person(cx=w * 0.5, cy=h * 0.5, w=60, h=140)]

        elif self.scene in ("distress", "night"):
            # Victim: frame centre (drone hovers above victim GPS)
            victim  = _Person(cx=w * 0.5, cy=h * 0.5,  w=60, h=140)
            # Suspect: left side, moving toward victim at 4 px/frame
            suspect = _Person(cx=w * 0.2, cy=h * 0.5,  w=65, h=145,
                              vx=4.0, vy=0.0)
            # Bystander: right side, slow random movement
            bystander = _Person(cx=w * 0.8, cy=h * 0.4, w=55, h=130,
                                vx=0.5, vy=0.3)
            self._persons = [victim, suspect, bystander]

    def next_frame(self) -> np.ndarray:
        """Return a synthetic BGR frame and advance the simulation one step."""
        frame = np.full((self.height, self.width, 3),
                        self._bg_brightness, dtype=np.uint8)

        # Ground plane gradient
        for y in range(self.height):
            brightness = int(self._bg_brightness + (y / self.height) * 20)
            frame[y, :] = brightness

        for p in self._persons:
            p.update(self.width, self.height)
            x1, y1, x2, y2 = p.box_xyxy
            # Body
            cv2.rectangle(frame, (x1, y1), (x2, y2), (140, 100, 80), -1)
            # Head
            head_r = p.w // 4
            cv2.circle(frame, (int(p.cx), y1 - head_r), head_r, (200, 160, 120), -1)
            # Simple clothes variation
            cv2.rectangle(frame, (x1, y1 + p.h // 2), (x2, y2), (60, 60, 180), -1)

        # Add mild Gaussian noise for realism
        noise = self._rng.integers(-10, 10, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        self._frame_idx += 1
        return frame

    def frames(self, n: int):
        """Generator: yield n consecutive frames."""
        for _ in range(n):
            yield self.next_frame()

    @property
    def frame_count(self) -> int:
        return self._frame_idx


# ─── Standalone preview ───────────────────────────────────────────────────────

if __name__ == "__main__":
    for scene in ("safe", "distress", "night"):
        mock  = CameraMock(scene=scene)
        frame = mock.next_frame()
        path  = f"docs/camera_mock_{scene}.png"
        cv2.imwrite(path, frame)
        print(f"Saved {scene} frame -> {path}  shape={frame.shape}")
