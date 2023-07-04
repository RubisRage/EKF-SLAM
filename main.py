#!/usr/bin/python
import config
import numpy as np
import cv2

from ekf import predict, update, augment
from loader import loader
from associate import associate
from corner_extraction import find_corners
from utils import process_laser, cartesian_coords
from display import build_global_frame
from noise import add_control_noise, add_observe_noise


def main():
    data_loader = loader(config.log)

    # Random number generator
    rnd = np.random.RandomState(0)

    X = config.X
    P = config.P
    Q = config.Q
    R = config.R
    dt = config.dt
    INNER_GATE = config.INNER_GATE
    OUTER_GATE = config.OUTER_GATE
    xtrue = np.zeros((3,))
    dtsum = 0

    for i, (controls, laser) in enumerate(data_loader):
        print(i)

        # True controls / measurements
        xtrue += controls
        laser_points, laser_polar = process_laser(laser)

        # Noised controls / measurements
        noised_controls = add_control_noise(rnd, controls)
        noised_laser_points = cartesian_coords(
            add_observe_noise(rnd, laser_polar))

        # STEP 1: Predict
        X, P = predict(X, P, noised_controls, Q, dt)

        # Extract observations
        z = find_corners(X, noised_laser_points, laser_polar)
        lm, nLm = associate(X, P, z, R, INNER_GATE, OUTER_GATE)

        dtsum += 0.1

        if dtsum >= config.DT_OBSERVE:

            # STEP 2: Update
            X, P = update(X, P, lm, R)

            # STEP 3: Augment
            X, P = augment(X, P, nLm, R)

            dtsum = 0

        # Display
        frame = build_global_frame(xtrue, X, P, noised_laser_points, z, lm,
                                   nLm)
        cv2.imshow("EKF", frame)

        key = cv2.waitKey(dt)

        if key == ord('s'):
            dt = 10 if dt == 0 else 0

        if key == ord('q'):
            break


if __name__ == "__main__":
    main()
