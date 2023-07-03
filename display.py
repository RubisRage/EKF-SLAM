from math import cos, sin
from itertools import zip_longest

import cv2
import config
import numpy as np


def draw_lines(frame, lines, laser_points, frame_config, show_border=False,
               show_text=False):

    for i, line in enumerate(lines):
        i1, i2 = line

        p1 = to_display_space(laser_points[i1], frame_config)
        p2 = to_display_space(laser_points[i2], frame_config)

        color = (0, 0, 255)

        cv2.line(frame, p1, p2, color, 1)

        if show_border:
            cv2.circle(frame, p1, 3, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, p2, 3, (0, 255, 0), cv2.FILLED)

        if show_text:
            cv2.putText(frame, f"{i1}, {i2}",
                        (p2[0], p2[1] + (20 * (1 if i & 1 else -1))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def to_display_space(p, frame_config):
    A = np.array([
        [1, 0],
        [0, -1]
    ])

    org = frame_config["origin"]

    tp = A @ np.array(p).T * frame_config["meters_to_px_ratio"] + org.T

    return np.array([int(tp[0]), int(tp[1])])


def draw_points(frame: np.array, points, frame_config, **kwargs):
    color = kwargs["color"] if "color" in kwargs else (0, 0, 0)
    label_color = kwargs["label_color"] if "label_color" in kwargs else (
        0, 0, 0)
    radius = kwargs["radius"] if "radius" in kwargs else 2
    labels = kwargs["labels"] if "labels" in kwargs else []
    label_offset = kwargs["label_offset"] if "label_offset" in kwargs else [
        0, -10]

    for p, label in zip_longest(points, labels):
        tp = to_display_space(p, frame_config)

        cv2.circle(frame, tp, radius, color, cv2.FILLED)

        if len(labels) != 0:
            cv2.putText(frame, f'{label}', tp + label_offset,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1)


def draw_robot(frame, pose, frame_config, **kwargs):
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

    cv2.line(frame, to_display_space(p1, frame_config),
             to_display_space(p2, frame_config), color, 1)
    cv2.line(frame, to_display_space(p1, frame_config),
             to_display_space(p3, frame_config), color, 1)
    cv2.line(frame, to_display_space(p2, frame_config),
             to_display_space(p3, frame_config), color, 1)
    cv2.circle(frame, to_display_space(location.T, frame_config), radius,
               color, cv2.FILLED)


def draw_mesh(frame: np.array, frame_config):
    frame_height = frame_config["height"]
    frame_width = frame_config["width"]
    meters_to_px_ratio = frame_config["meters_to_px_ratio"]

    for y in range(0, frame_height, int(meters_to_px_ratio)):
        cv2.line(frame, (0, y), (frame_width, y), (0, 128, 0), 1)

    for x in range(0, frame_width, int(meters_to_px_ratio)):
        cv2.line(frame, (x, 0), (x, frame_height), (0, 128, 0), 1)


def draw_corner(frame, corners):
    color = (255, 0, 255)

    for corner in corners:
        cv2.circle(frame, to_display_space(corner), 5, color, -1)


def build_frame(robot_pov, global_frame, map):
    scale_factor = .005
    border_width = int(robot_pov.shape[0] * scale_factor)

    frame = np.zeros((robot_pov.shape[0] + border_width,
                      robot_pov.shape[1] * 2 + border_width,
                      3))

    rp_height, rp_width, _ = robot_pov.shape

    frame[: rp_height, : rp_width] = robot_pov
    frame[: rp_height // 2, rp_width + border_width:] = global_frame
    frame[rp_height // 2 + border_width:, rp_width + border_width:] = map

    return frame


def build_global_frame(xtrue, X, laser_points, z, associatedLm, newLm):
    from utils import global_coords, cartesian_coords
    from models import observe_model

    frame_height = config.global_frame_config["height"]
    frame_width = config.global_frame_config["width"]

    frame = np.ones((frame_height, frame_width, 3)) * 255

    draw_mesh(frame, config.global_frame_config)

    # Draw robot
    draw_robot(frame, X[:3], config.global_frame_config, color=(0, 0, 255))
    draw_robot(frame, xtrue, config.global_frame_config, color=(255, 0, 0))

    # Draw laser data
    draw_points(frame, global_coords(
        laser_points, xtrue), config.global_frame_config)

    # Draw observations
    zGlobal = global_coords(cartesian_coords(z), X[:3])

    draw_points(frame, zGlobal, config.global_frame_config, color=(0, 0, 255),
                radius=4, labels=range(len(zGlobal)))

    # Draw associated landmarks
    associatedLmGlobal = global_coords(cartesian_coords(
        np.array(list(map(lambda lm: lm.z, associatedLm)), dtype=np.double)
    ), X[:3])

    ids = list(map(lambda lm: lm.id, associatedLm))

    draw_points(frame, associatedLmGlobal, config.global_frame_config,
                color=(255, 0, 255), radius=3, label_color=(255, 0, 255))

    predicted_lm = []

    for fid in range(3, X.shape[0] - 1, 2):
        predicted_lm.append(global_coords(cartesian_coords(
            [observe_model(X, fid)[0]]), X[:3])[0])

    # Predicted landmarks
    draw_points(frame, predicted_lm, config.global_frame_config,
                color=(0, 255, 0), radius=2, label_color=(0, 0, 255))

    # System landmarks (X)
    draw_points(frame, X[3:].reshape(((X.shape[0] - 3) // 2, 2)),
                config.global_frame_config, color=(255, 0, 0), radius=1)

    # Draw new landmarks
    draw_points(frame, global_coords(cartesian_coords(newLm), X[:3]),
                config.global_frame_config, color=(0, 255, 255), radius=3)

    return frame


def main():
    robot_pov = np.ones((500, 500, 3)) * 255
    global_frame = np.ones((250, 500, 3)) * 255
    map = np.ones((250, 500, 3)) * 255

    frame = build_frame(robot_pov, global_frame, map)

    cv2.imshow("Build frame test", frame)

    while cv2.waitKey(0) != ord('q'):
        pass


if __name__ == "__main__":
    main()
