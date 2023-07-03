from slam_types import Odom, AssociatedLandmark
from utils import pi_to_pi
from models import observe_model
from numpy.linalg import inv
from math import sin, cos

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
    W = np.array([
        [ix],
        [iy],
        [ith]
    ])

    WQW = W * Q @ W.T

    # Update robot-robot covariance Prr = A * Prr * A' + W * Q * W'
    # P[0:3, 0:3] = np.matmul(np.matmul(A, P[0:3, 0:3]), A.T) + WQW
    P[0:3, 0:3] = A @ P[0:3, 0:3] @ A.T + WQW

    # Update robot-landmark covariance Pri = A * Pri, Pir = Pri.T
    for idx in range(3, P.shape[0], 2):
        # P[0:2, idx:idx+2] = np.matmul(A, P[0:2, idx:idx+2])
        P[0:3, idx:idx+3] = A @ P[0:3, idx:idx+3]
        P[idx:idx+3, 0:3] = P[0:3, idx:idx+3].T

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

        """
        # Compute Kalman gain: P * H' * (H * P * H.T + V * R * V.T)^-1
        S = H @ P @ H.T + R

        K = P @ H.T @ inv(S)

        # Update state vector
        X = X + K @ v

        # Update covariance matrix
        Id = np.identity(P.shape[0])
        P = (Id - K @ H) @ P
        """

        PHt = P @ H.T
        S = H @ PHt + R
        S = (S+S.T) * 0.5
        SChol = np.linalg.cholesky(S)
        SCholInv = inv(SChol)
        W1 = PHt @ SCholInv
        W = W1 @ SCholInv.T

        X = X + W @ v
        P = P - W1 @ W1.T

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
        P[N:N+2, 0:2] = Jz @ P[0:2, 0:2]
        P[0:2, N:N+2] = P[N:N+2, 0:2].T

        # Landmark-landmark covariance
        # if len>3
        #   rnm= 4:len;
        #   P(rng,rnm)= Gv*P(1:3,rnm); % map to feature xcorr
        #   P(rnm,rng)= P(rng,rnm)';
        # end

        if N > 3:
            P[N:N+2, 3:N] = Jxr @ P[0:3, 3:N]
            P[3:N, N:N+2] = P[N:N+2, 3:N].T

    return X, P
