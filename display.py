from math import cos, sin

import cv2
import config
import numpy as np


def draw_lines(frame, lines, laser_points, laser, show_border=False,
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


def to_display_space(p, pose=(.0, .0, .0)):
    x, y, th = pose

    R = np.array([
        [cos(th), -sin(th)],
        [sin(th), cos(th)]
    ])

    tp = (R @ np.array(p).T + np.array([x, y]).T +
          np.array([1, 5]).T) * config.meters_to_px_ratio

    return [int(tp[0]), int(tp[1])+100]


def draw_points(frame: np.array, points, color=(0, 0, 0)):

    cv2.circle(frame, to_display_space((0,0)), 4, (165,255,0), cv2.FILLED)

    for p in points:
        cv2.circle(frame, to_display_space(p), 2, color, cv2.FILLED)


def draw_mesh(frame: np.array):
    frame_height, frame_width = frame.shape[:2]
    for y in range(0, config.frame_height, int(frame_height/10)):
        cv2.line(frame, (0, y), (config.frame_width, y), (0, 128, 0), 1)

    for x in range(0, config.frame_width, int(frame_width/10)):
        cv2.line(frame, (x, 0), (x, config.frame_height), (0, 128, 0), 1)


def draw_corner(frame, Vcorner):
    color = (255, 0, 255)

    for corner in Vcorner:
        cv2.circle(frame, to_display_space(corner), 5, color, -1)

def draw_frame_border(frame, color=(0, 0, 0), thickness=4):
    height, width = frame.shape[:2]
    cv2.line(frame, (0, 0), (width, 0), color, thickness)  # Línea superior
    cv2.line(frame, (width, 0), (width, height), color, thickness)  # Línea derecha
    cv2.line(frame, (0, height), (width, height), color, thickness)  # Línea inferior
    cv2.line(frame, (0, 0), (0, height), color, thickness)  # Línea izquierda 