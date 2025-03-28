import numpy as np

from enum import Enum


class Shape(Enum):
    rectangle = "rectangle"
    circle = "circle"


def order_points(pts: np.ndarray):
    """Orders points in [top-left, top-right, bottom-right, bottom-left] order"""
    # Initialize ordered coordinates
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest difference, bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect
