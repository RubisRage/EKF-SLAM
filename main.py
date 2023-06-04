#!/usr/bin/python

import config
from ekf import predict, update, augment
from loader import loader
from associate import associate


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

        # STEP 2:
        # lm = extract_landmarks()
        z = None
        lm, nLm = associate(X, P, z, R, INNER_GATE, OUTER_GATE)
        X, P = update(X, P, lm, R)

        # STEP 3:
        X, P = augment(X, P, nLm, R)


if __name__ == "__main__":
    main()
