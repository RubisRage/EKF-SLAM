#!/usr/bin/python
import config
import numpy as np
import matplotlib.pyplot as plt
import cv2

from ekf import predict, update, augment
from loader import loader
from associate import associate
from ransac import findLines
from display import display_raw_points, display_extracted_lines


def main():
    data_loader = loader(config.log)

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

        lines, z = findLines(laser, X)

        frame = np.ones((config.frame_width, config.frame_height, 3)) * 255

        display_raw_points(frame, X, laser)
        display_extracted_lines(frame, X, lines, z)

        """
        for line, lm in zip(lines, z):
        """

        cv2.imshow("Grid map", frame)
        key = cv2.waitKey(0)

        if key == ord('q'):
            break

        lm, nLm = associate(X, P, z, R, INNER_GATE, OUTER_GATE)

        # STEP 2: Update
        X, P = update(X, P, lm, R)

        # STEP 3: Augment
        X, P = augment(X, P, nLm, R)

        # Draw map

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
