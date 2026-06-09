"""
Phase 4 — Depth Estimator

Estimates per-pixel proximity (how close each pixel is to the drone camera).

Phase 4 (software):
  Tries MiDaS_small via torch.hub. Falls back to a vertical-gradient heuristic
  (bottom of frame = near, top = far) if MiDaS fails to load.

Phase 5 (hardware swap):
  Replace estimate() call with RealSense D435i depth stream.
  Interface stays the same: frame -> (H x W float32 [0,1]).
  0 = far/safe   1 = very close/high-priority obstacle

Install note (optional, for MiDaS):
  pip install timm   # required by MiDaS_small
"""
from __future__ import annotations

import numpy as np
import cv2


class DepthEstimator:
    """
    Unified depth interface for Phase 4 (MiDaS) and Phase 5 (RealSense).

    Parameters
    ----------
    device      : "cuda" | "cpu"
    use_midas   : attempt MiDaS load; if False or load fails, uses heuristic
    """

    def __init__(self, device: str = "cuda", use_midas: bool = True):
        import torch
        self._device   = device if torch.cuda.is_available() else "cpu"
        self._model    = None
        self._transform = None
        self._midas_ok  = False

        if use_midas:
            self._try_load_midas()

    # ── Public ───────────────────────────────────────────────────────────────

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns proximity map: H x W float32 in [0, 1].
        1 = very close (obstacle immediate priority)
        0 = far (safe to ignore for now)
        """
        if self._midas_ok:
            return self._midas(frame)
        return self._heuristic(frame)

    def proximity_for_bbox(
        self, bbox: tuple[int,int,int,int], frame_h: int, frame_w: int
    ) -> float:
        """
        Quick per-object proximity score from bounding box size alone.
        Larger box relative to frame = closer. Returns [0, 1].
        """
        x1, y1, x2, y2 = bbox
        obj_area   = max(1, (x2 - x1) * (y2 - y1))
        frame_area = max(1, frame_h * frame_w)
        return float(min(1.0, (obj_area / frame_area) * 6.0))

    @property
    def backend(self) -> str:
        return "MiDaS_small" if self._midas_ok else "heuristic"

    # ── MiDaS ────────────────────────────────────────────────────────────────

    def _try_load_midas(self) -> None:
        try:
            import torch
            self._model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small",
                trust_repo=True, verbose=False,
            )
            transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms",
                trust_repo=True, verbose=False,
            )
            self._transform = transforms.small_transform
            self._model.to(self._device).eval()
            self._midas_ok = True
            print(f"  [Depth] MiDaS_small on {self._device}")
        except Exception as exc:
            print(f"  [Depth] MiDaS unavailable ({exc.__class__.__name__}) "
                  f"-- using vertical-gradient heuristic. "
                  f"(pip install timm to enable MiDaS)")

    def _midas(self, frame: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = self._transform(rgb).to(self._device)

        with torch.no_grad():
            pred = self._model(inp)
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

        # MiDaS outputs inverse depth (larger = closer).  Normalize to [0,1].
        d_min, d_max = float(pred.min()), float(pred.max())
        if d_max > d_min:
            return ((pred - d_min) / (d_max - d_min)).astype(np.float32)
        return np.zeros(frame.shape[:2], dtype=np.float32)

    # ── Heuristic fallback ────────────────────────────────────────────────────

    def _heuristic(self, frame: np.ndarray) -> np.ndarray:
        """
        Simple vertical gradient: bottom row = proximity 1.0 (near ground),
        top row = proximity 0.0 (sky/far horizon).
        Reasonable default for a forward-facing drone camera.
        """
        h, w = frame.shape[:2]
        gradient = np.linspace(1.0, 0.0, h, dtype=np.float32)
        return np.tile(gradient[:, np.newaxis], (1, w))


# ─── Singleton ────────────────────────────────────────────────────────────────

_depth: DepthEstimator | None = None


def get_depth_estimator(use_midas: bool = True) -> DepthEstimator:
    global _depth
    if _depth is None:
        _depth = DepthEstimator(use_midas=use_midas)
    return _depth
