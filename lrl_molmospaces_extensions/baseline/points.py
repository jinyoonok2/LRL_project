"""Deterministic point prompts sampled from visible target segmentation."""

import numpy as np


def visible_point(mask: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Choose the visible pixel nearest the mask centroid; normalize x/y to 0–1."""
    pixels = np.argwhere(np.asarray(mask).squeeze() > 0)
    points = np.full((max_points, 2), np.nan, dtype=np.float32)
    count = np.array([0], dtype=np.int32)
    if len(pixels):
        center = pixels.mean(axis=0)
        y, x = pixels[np.argmin(np.sum((pixels - center) ** 2, axis=1))]
        height, width = np.asarray(mask).shape[:2]
        points[0] = (x / width, y / height)
        count[0] = 1
    return points, count
