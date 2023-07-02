#!/usr/bin/python
import config
import numpy as np
import cv2
import random

from ekf import predict, update, augment
from loader import loader
from associate import associate
from corner_extraction import find_corners
from utils import process_laser
from display import build_global_frame


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
    xtrue = np.zeros((3,))

    for controls, laser in data_loader:

        # STEP 1: Predict
        X, P = predict(X, P, controls, Q, dt)
        xtrue += [*controls]

        # TODO: Add process and observe noise
        laser_points = process_laser(laser)
        z = find_corners(X, laser_points)

        lm, nLm = associate(X, P, z, R, INNER_GATE, OUTER_GATE)

        print(f"Found: {len(z)}, Associated: {len(lm)}, New: {len(nLm)}")

        frame = build_global_frame(xtrue, X, laser_points, z, lm, nLm)
        cv2.imshow("EKF", frame)

        # Draw map
        key = cv2.waitKey(dt)

        if key == ord('s'):
            dt = 10 if dt == 0 else 0

        if key == ord('q'):
            break

        # STEP 2: Update
        X, P = update(X, P, lm, R)

        # STEP 3: Augment
        X, P = augment(X, P, nLm, R)


if __name__ == "__main__":
    main()
