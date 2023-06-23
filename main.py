#!/usr/bin/python
import config
import numpy as np
import cv2
import random

from ekf import predict, update, augment
from loader import loader
from associate import associate
from corner_extraction import find_corners
from utils import cartesian_coords, process_laser


def main():
    data_loader = loader(config.log)
    random.seed(0)

    X = config.X
    P = config.P
    Q = config.Q
    R = config.R
    dt = config.dt
    INNER_GATE = config.INNER_GATE
    OUTER_GATE = config.OUTER_GATE

    for controls, laser in data_loader:

        # STEP 1: Predict
        X, P = predict(X, P, controls, Q, dt)

        laser_points = process_laser(laser, X[:3])

        z = find_corners(X, laser_points, laser.data)

        lm, nLm = associate(X, P, z, R, INNER_GATE, OUTER_GATE)

        print(f"Associated: {len(lm)}, New: {len(nLm)}")
        display_helper(X, laser_points, z, lm, nLm)

        # STEP 2: Update
        X, P = update(X, P, lm, R)

        # STEP 3: Augment
        X, P = augment(X, P, nLm, R)

        # Draw map

    cv2.destroyAllWindows()


def display_helper(X, laser_points, corners, lm, nLm):
    from display import draw_points

    frame = np.ones((config.frame_height, config.frame_width, 3)) * 255

    # Draw robot
    draw_points(frame, [(0, 0)], color=(0, 0, 255))

    # Draw observed points
    draw_points(frame, laser_points)

    # Draw associated landmarks
    draw_points(frame, cartesian_coords(
        np.array(list(map(lambda lm: lm.z, lm)), dtype=np.double)), X[:3])

    cv2.imshow("Association test", frame)

    while cv2.waitKey() != ord(' '):
        pass


if __name__ == "__main__":
    main()
