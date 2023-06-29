from math import cos, sin

import cv2
import config
import numpy as np


def draw_lines(frame, lines, laser_points, show_border=False,
               show_text=False):

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


def to_display_space(p):
    A = np.array([
        [1, 0],
        [0, -1]
    ])

    tp = (A @ np.array(p).T + np.array([1, 5]).T) * config.meters_to_px_ratio

    return np.array([int(tp[0]), int(tp[1])])


def draw_points(frame: np.array, points, **kwargs):
    color = kwargs["color"] if "color" in kwargs else (0, 0, 0)
    label_color = kwargs["label_color"] if "label_color" in kwargs else (
        0, 0, 0)
    radius = kwargs["radius"] if "radius" in kwargs else 2
    labels = kwargs["labels"] if "labels" in kwargs else None
    label_offset = kwargs["label_offset"] if "label_offset" in kwargs else [
        0, -10]

    for p in points:
        cv2.circle(frame, to_display_space(p), radius, color, cv2.FILLED)

    if labels is not None:
        for p, label in zip(points, labels):
            cv2.putText(frame, f'{label}', to_display_space(p)
                        + label_offset, cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        label_color, 1)


def draw_robot(frame, pose, **kwargs):
    color = kwargs["color"] if "color" in kwargs else (0, 0, 0)
    radius = kwargs["radius"] if "radius" in kwargs else 2

    x, y, th = pose

    R = np.array([
        [cos(th), -sin(th)],
        [sin(th), cos(th)]
    ])

    location = np.array([x, y]).T

    p1 = R @ np.array([-0.05, -0.09]).T + location
    p2 = R @ np.array([-0.05, 0.09]).T + location
    p3 = R @ np.array([0.3, 0]).T + location

    cv2.line(frame, to_display_space(p1), to_display_space(p2), color, 1)
    cv2.line(frame, to_display_space(p1), to_display_space(p3), color, 1)
    cv2.line(frame, to_display_space(p2), to_display_space(p3), color, 1)
    cv2.circle(frame, to_display_space(location.T), radius, color, cv2.FILLED)


def draw_mesh(frame: np.array):
    for y in range(0, config.frame_height, 100):
        cv2.line(frame, (0, y), (config.frame_height, y), (0, 128, 0), 1)

    for x in range(0, config.frame_width, 100):
        cv2.line(frame, (x, 0), (x, config.frame_width), (0, 128, 0), 1)


def draw_corner(frame, corners):
    color = (255, 0, 255)

    for corner in corners:
        cv2.circle(frame, to_display_space(corner), 5, color, -1)
