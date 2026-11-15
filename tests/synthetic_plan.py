"""Generate a deterministic raster plan; no reference drawings are needed."""
from pathlib import Path
import argparse

import cv2
import numpy as np

SCALE = 0.05  # meters per pixel: the 200 px reference represents 10 m
WALL_PX = np.array([[60.0, 60.0], [360.0, 60.0]])
DIAGONAL_PX = np.array([[420.0, 60.0], [540.0, 180.0]])
WALL_METERS = 15.0
LENGTH_TOLERANCE_M = 0.30  # 6 px total endpoint error; 2% of the wall length


def make_plan(path: Path) -> Path:
    image = np.full((480, 640, 3), 250, dtype=np.uint8)
    cv2.rectangle(image, (60, 60), (360, 260), (30, 30, 30), 3)
    cv2.line(image, (420, 60), (540, 180), (30, 30, 30), 3)
    # An 8 px raster gap becomes 12 px after thinning: a documented limitation.
    cv2.line(image, (100, 310), (196, 310), (30, 30, 30), 3)
    cv2.line(image, (204, 310), (300, 310), (30, 30, 30), 3)
    for y in (110, 140, 170):
        cv2.line(image, (110, y), (260, y), (160, 160, 160), 2)
    cv2.circle(image, (480, 300), 35, (100, 180, 100), -1)
    cv2.rectangle(image, (420, 380), (500, 420), (70, 150, 220), -1)
    # Known-length scale reference, kept separate from the plan.
    cv2.line(image, (60, 400), (260, 400), (30, 30, 30), 3)
    for x in (60, 260):
        cv2.line(image, (x, 393), (x, 407), (30, 30, 30), 2)
    image = cv2.GaussianBlur(image, (3, 3), 0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write synthetic plan: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(make_plan(args.output))
