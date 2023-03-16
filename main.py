#!/usr/bin/python

import numpy as np
from ekf import predict
from loader import loader
from utils import pi_to_pi

dt = 0.1  # seconds

# State matrix X
X = np.zeros((3,))

# Covariance matrix
P = np.zeros((3, 3))

# Prediction model noise
Q = 0.5


def main():
    global X, P

    data_loader = loader("medium_nd_5.log")

    debug_start = [0, 0, 0]

    for controls, laser in data_loader:

        # STEP 1: Predict
        X, P = predict(X, P, controls, Q, dt)
        debug_start[0] += controls[0]
        debug_start[1] += controls[1]
        debug_start[2] = pi_to_pi(debug_start[2] + controls[2])

        print("Predicted:", X)
        print("Expected:", debug_start)


        # STEP 2:
        # STEP 3:


if __name__ == "__main__":
    main()
