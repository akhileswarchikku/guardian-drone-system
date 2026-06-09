"""
Phase 4.4 — Night Vision Mode
CLAHE contrast enhancement + automatic low-light detection.
Full IR mode requires Intel RealSense D435i (hardware — Phase 5).
"""
from __future__ import annotations
import cv2
import numpy as np

# Lux threshold below which CLAHE kicks in (spec: 40 lux)
LOW_LIGHT_THRESHOLD = 40


def is_low_light(frame: np.ndarray, threshold: int = LOW_LIGHT_THRESHOLD) -> bool:
    """True when mean frame brightness is below threshold (approximates lux)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(gray.mean()) < threshold


def apply_clahe(frame: np.ndarray, clip_limit: float = 3.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a BGR frame.
    Operates on the L channel of LAB colour space to preserve hue.
    """
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_eq  = clahe.apply(l)
    enhanced = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def preprocess(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Auto-enhance if low-light; return (processed_frame, night_mode_active).
    Call this before running any detector.
    """
    if is_low_light(frame):
        return apply_clahe(frame), True
    return frame, False
