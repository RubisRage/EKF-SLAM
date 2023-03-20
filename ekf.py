import numpy as np
from loader import Odom
from utils import pi_to_pi


def predict(
        X: np.array,     # System state
        P: np.array,     # Covariance matrix
        controls: Odom,  # control inputs: [ix, iy, ith]
        Q: float,        # Prediction model noise (Odometry accuracy)
        dt: float        # Timestep (s)
        ) -> tuple[np.array, np.array]:  # Returns new state(X), covariance(P)

    ix, iy, ith = controls

    # Predict state
    X[0] = X[0] + ix
    X[1] = X[1] + iy
    X[2] = pi_to_pi(X[2] + ith)

    # Jacobian of the prediction model
    A = np.array([
        [1, 0, -iy],
        [0, 1,  ix],
        [0, 0,   1]
    ])

    # Prediction model noise
    W = np.array([
        [ix],
        [iy],
        [ith]
    ])

    WQW = np.matmul(W * Q, W.T)

    # Update robot-robot covariance Prr = A * Prr * A.T + W * Q * W.T
    P[0:3, 0:3] = np.matmul(np.matmul(A, P[0:3, 0:3]), A.T) + WQW

    # Update robot-landmark covariance Pri = A * Pri, Pir = Pri.T
    # TODO: Check if it should be Pri = A * Pri or Pri = A.T * Pri
    for idx in range(3, P.shape[0], 2):
        P[idx:idx+2, 0:2] = np.matmul(A, P[idx:idx+2, 0:2])
        P[0:2, idx:idx+2] = P[idx:idx+2, 0:2].T

    return X, P


def update():
    pass


def augment():
    pass
