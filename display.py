from slam_types import Laser
from utils import cartesian_coords

import cv2
import config
import numpy as np
from math import cos, sin


def to_display_space(p, pose=(.0, .0, .0)):
    # TODO: Is there a need for translation to global coords in this
    # function?
    x, y, th = pose

    R = np.array([
        [cos(th), -sin(th)],
        [sin(th), cos(th)]
    ])

    tp = (R @ np.array(p).T + np.array([x, y]).T +
          np.array([1, 5]).T) * config.meters_to_px_ratio

    return [int(tp[0]), int(tp[1])]


def display_raw_points(frame: np.array, X: np.array, laser: Laser):

    cv2.circle(frame, (int(config.meters_to_px_ratio),
                       config.frame_height // 2), 5, (0, 0, 255), cv2.FILLED)

    # for p in cartesian_coords(laser, (X[0], X[1], X[2])):
    for p in cartesian_coords(laser):
        cv2.circle(frame, to_display_space(p), 2, (0, 0, 0), cv2.FILLED)


def display_extracted_lines(frame: np.array, X: np.array, lines, z):

    for line in lines:
        m, b, associated, associatedCount = line

        p1 = to_display_space(associated[0], (X[0], X[1], X[2]))
        p2 = to_display_space(
            associated[associatedCount-1], (X[0], X[1], X[2]))
        cv2.line(frame, p1, p2, (0, 0, 255), 2)
