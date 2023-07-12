from slam_types import Odom, AssociatedLandmark
from utils import pi_to_pi
from models import observe_model
from numpy.linalg import inv
from math import sin, cos

import config
import numpy as np


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
    W = np.array([[ix], [iy], [ith]])
    WQW = W * config.sigmaX @ W.T

    # TODO: Get WQW to be a 3x3 matrix instead of being a scalar

    # Update robot-robot covariance Prr = A * Prr * A' + W * Q * W'
    # P[0:3, 0:3] = np.matmul(np.matmul(A, P[0:3, 0:3]), A.T) + WQW
    P[0:3, 0:3] = A @ P[0:3, 0:3] @ A.T + WQW

    B = P.copy()

    # Update robot-landmark covariance Pri = A * Pri, Pir = Pri.T
    if X.shape[0] > 3:
        P[0:3, 3:] = A @ P[0:3, 3:]
        P[3:, 0:3] = P[0:3, 3:].T

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

        PHt = P @ H.T
        S = H @ PHt + R
        S = (S + S.T) * 0.5 # Make symmetric

        # Tikhonov Regulation 
        A_reg = S + config.tikhonov_factor * np.identity(S.shape[0])
        
        try:
            SChol = np.linalg.cholesky(S)
            SCholInv = inv(SChol.T)
            W1 = PHt @ SCholInv
            W = W1 @ SCholInv.T

            X = X + W @ v
            P = P - W1 @ W1.T
        except:
            print(f"Not positive definite, asssociation: {lm.id}")


    return X, P


def augment(
    X: np.array,  # System state
    P: np.array,  # Covariance matrix
    z: np.array,  # New landmarks
    R: np.array   # Observation noise
) -> tuple[
    np.array,  # Augmented system state
    np.array   # Augmented covariance matrix
]:

    for lm in z:
        r = lm[0]            # Range
        b = lm[1]            # Bearing
        s = sin(X[2] + b)
        c = cos(X[2] + b)

        # Augment system state
        X = np.append(X, [X[0] + r*c, X[1] + r*s])

        # Jacobian of the prediction model for landmarks
        Jxr = np.array([
            [1, 0, -r*s],
            [0, 1,  r*c]
        ])

        # Jacobian of the prediction model for range-bearing
        Jz = np.array([
            [c, -r*s],
            [s,  r*c]
        ])

        # Agument Covariance matrix
        N = P.shape[0]
        P = np.pad(P, ((0, 2), (0, 2)), mode="constant")

        # Landmark Covariance
        P[N:N+2, N:N+2] = Jxr @ P[0:3, 0:3] @ Jxr.T + Jz @ R @ Jz.T

        # Robot-landmark covariance
        P[N:N+2, 0:3] = Jxr @ P[0:3, 0:3]
        P[0:3, N:N+2] = P[N:N+2, 0:3].T

        if N > 3:
            P[N:N+2, 3:N] = Jxr @ P[0:3, 3:N]
            P[3:N, N:N+2] = P[N:N+2, 3:N].T

    return X, P
