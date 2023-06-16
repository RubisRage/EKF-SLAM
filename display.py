from slam_types import Laser
from utils import cartesian_coords

import cv2
import config
import numpy as np
from math import cos, sin


def draw_lines(frame, lines, laser_points, laser, show_border=False,
               show_text=False):

    pose = (0., 0., 0.)

    display_raw_points(frame, pose, laser)
    display_mesh(frame)

    for i, line in enumerate(lines):
        i1, i2 = line

        p1 = to_display_space(laser_points[i1])
        p2 = to_display_space(laser_points[i2])

        color = (0, 0, 255)

        cv2.line(frame, p1, p2, color, 1)

        if show_border:
            cv2.circle(frame, p1, 3, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, p2, 3, (0, 255, 0), cv2.FILLED)

        if show_text:
            cv2.putText(frame, f"{i1}, {i2}",
                        (p2[0], p2[1] + (20 * (1 if i & 1 else -1))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


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


def display_mesh(frame: np.array):
    for y in range(0, config.frame_height, 100):
        cv2.line(frame, (0, y), (config.frame_height, y), (128, 128, 128), 1)

    for x in range(0, config.frame_width, 100):
        cv2.line(frame, (x, 0), (x, config.frame_width), (128, 128, 128), 1)
