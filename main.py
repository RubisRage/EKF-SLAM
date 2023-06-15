#!/usr/bin/python
import config
import numpy as np
import cv2
import random

from ekf import predict, update, augment
from loader import loader
from associate import associate
from extract_landmarks import extract_landmarks
from display import display_raw_points, display_extracted_lines


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

        print(f"Controls: {controls}")

        print(f"Post predict: {X[: 3]}")

        lines, z = extract_landmarks(laser, X)

        frame = np.ones((config.frame_width, config.frame_height, 3)) * 255

        display_raw_points(frame, X, laser)
        display_extracted_lines(frame, X, lines, z)

        cv2.imshow("Grid map", frame)
        key = cv2.waitKey(0)

        if key == ord('q'):
            break

        lm, nLm = associate(X, P, z, R, INNER_GATE, OUTER_GATE)

        # STEP 2: Update
        X, P = update(X, P, lm, R)

        print(f"Post update: {X[: 3]}")

        # STEP 3: Augment
        X, P = augment(X, P, nLm, R)

        print(f"Post augment: {X[: 3]}")
        print("================")

        # Draw map

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
