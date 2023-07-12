#!/usr/bin/python
import config
import numpy as np
import cv2

from ekf import predict, update, augment
from loader import loader
from associate import associate
from corner_extraction import find_corners
from utils import process_laser, cartesian_coords
from display import build_global_frame, build_frame_test, build_local_frame
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
    xfalse = np.zeros((3,))

    for i, (controls, laser) in enumerate(data_loader):
        print("Iteration: ", i)

        # True controls / measurements
        xtrue += controls
        laser_points, laser_polar = process_laser(laser)

        # Noised controls / measurements
        noised_controls = add_control_noise(rnd, controls)
        noised_laser_polar = add_observe_noise(rnd, laser_polar)
        noised_laser_points = cartesian_coords(noised_laser_polar)

        xfalse += noised_controls

        # STEP 1: Predict
        X, P = predict(X, P, noised_controls, Q, dt)

        # Extract observations
        z = find_corners(X, noised_laser_points, noised_laser_polar)
        lm, nLm = associate(X, P, z, R, INNER_GATE, OUTER_GATE)

        frame_height = config.global_frame_config["height"]
        frame_width = config.global_frame_config["width"]
        frame = np.ones((frame_height, frame_width, 3)) * 255

        import display

        # System landmarks (X)
        display.draw_points(frame, X[3:].reshape(((X.shape[0] - 3) // 2, 2)),
                    config.global_frame_config, color=(0, 0, 255), radius=2, 
                    labels=list(range(3, X.shape[0]-1, 2)), label_offset=[-10, 10],
                    label_color=(255, 0, 0))

        # STEP 2: Update
        X, P = update(X, P, lm, R)

        # STEP 3: Augment
        X, P = augment(X, P, nLm, R)

        # Display
        build_global_frame(frame, xtrue, xfalse, X, P, noised_laser_points, z, lm,
                                   nLm)

        local_frame = build_local_frame(noised_laser_points, X)

        final_frame = build_frame_test(local_frame, frame)

        cv2.imshow("EKF", frame)

        key = cv2.waitKey(dt)

        if key == ord('s'):
            dt = 10 if dt == 0 else 0

        if key == ord('q'):
            break




if __name__ == "__main__":
    main()
