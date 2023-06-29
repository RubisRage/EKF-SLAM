from slam_types import AssociatedLandmark
from math import inf, log
from models import observe_model
from utils import pi_to_pi, cartesian_coords, distance
from numpy.linalg import det

import numpy as np


def associate(
        x: np.array,  # State matrix
        P: np.array,  # Covariance matrix
        z: np.array,  # Extracted landmarks
        R,            # Measurement noise
        innerGate: float,  # Association gate
        outerGate: float   # New landmark gate
    ) -> tuple[
                AssociatedLandmark,  # Associated landmarks (position and id)
                np.array             # New landmarks (position)
             ]:

    associatedLm = []
    newLm = []

    nRb = 3                     # Number of vehicle state variables
    nLm = (x.shape[0] - 3) // 2  # Number of already present features

    for lm in z:
        bestId = 0
        bestN = inf
        outer = inf

        # For each present feature
        for fid in range(nRb, nLm*2 + 2, 2):
            nis, nd = compute_association(x, P, lm, fid, R)

            if nis < innerGate and nd < bestN:
                bestN = nd
                bestId = fid
            elif nis < outer:
                outer = nis

        if bestId != 0:
            associatedLm.append(AssociatedLandmark(np.array(lm), bestId))
        elif outer > outerGate:
            newLm.append(lm)

    return associatedLm, newLm


def compute_association(
        X: np.array,  # State matrix
        P: np.array,  # Covariance matrix
        z: np.array,  # Extracted landmark (Measure)
        fid: int,     # Existing feature to compare with z
        R             # Measurement noise
) -> tuple[
    float,  # Malahalanobis distance
    float   # Normalised distance
]:

    zp, H = observe_model(X, fid)
    v = z - zp  # Innovation
    v[1] = pi_to_pi(v[1])

    # Innovation covariance: H * P * H' + R
    S = H @ P @ H.T + R

    # Normalised innovation squared: v' * S^-1 * v
    # TODO: Check this -> nis = v.T @ np.linalg.inv(S) @ v
    nis = v.T @ S @ v

    # Normalised distance: nis + ln(|S|)
    # nd = nis + log(det(S))
    nd = nis * 2.

    print(z, end=f' - {fid}: eucl(')
    print(distance(*cartesian_coords([z, zp], X[:3])), end='')
    print(f') - norm({nd})')

    return nis, nd
