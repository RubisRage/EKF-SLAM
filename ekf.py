import numpy as np
from numpy.linalg import inv
from associate import AssociatedLandmark
from loader import Odom
from utils import pi_to_pi
from models import observe_model


def predict(
        X: np.array,     # System state
        P: np.array,     # Covariance matrix
        controls: Odom,  # Control inputs: [ix, iy, ith]
        Q: float,        # Prediction model noise (Odometry accuracy)
        dt: float        # Timestep (s)
) -> tuple[
    np.array,  # Predicted system state
    np.array   # Updated covariance matrix
]:

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

    WQW = W * Q @ W.T

    # Update robot-robot covariance Prr = A * Prr * A.T + W * Q * W.T
    # P[0:3, 0:3] = np.matmul(np.matmul(A, P[0:3, 0:3]), A.T) + WQW
    P[0:3, 0:3] = A @ P[0:3, 0:3] @ A.T + WQW

    # Update robot-landmark covariance Pri = A * Pri, Pir = Pri.T
    for idx in range(3, P.shape[0], 2):
        # P[0:2, idx:idx+2] = np.matmul(A, P[0:2, idx:idx+2])
        P[0:2, idx:idx+2] = A @ P[0:2, idx:idx+2]
        P[idx:idx+2, 0:2] = P[0:2, idx:idx+2].T

    return X, P


def update(
        X: np.array,                    # System state
        P: np.array,                    # Covariance matrix
        lms: list[AssociatedLandmark],  # Associated landmarks (Measurement)
        R                               # Measurement noise
) -> tuple[
    np.array,  # Updated system state
    np.array   # Updated covariance matrix
]:

    for lm in lms:
        zp, H = observe_model(X, lm.id)

        # Compute innovation
        v = lm.z - zp
        v[1] = pi_to_pi(v[1])

        # Compute Kalman gain: P * H' * (H * P * H.T + V * R * V.T)^-1
        VRV = v @ R @ v.T
        S = H @ P @ H.T + VRV
        K = P @ H.T @ inv(S)

        # Update state vector
        X = X + K @ v

        # Update covariance matrix
        Ip = np.identity(P.shape[0])
        P = (Ip - K @ H) @ P

    return X, P


def augment():
    pass
