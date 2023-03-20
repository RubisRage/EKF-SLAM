from math import inf
from collections import namedtuple
import numpy as np


AssociatedLandmark = namedtuple("AssociatedLandmark", "pos id")


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
    nLm = (x.shape[0] - 3) / 2  # Number of already present features

    for lm in z:
        bestId = 0
        bestN = inf
        outer = inf

        # For each present feature
        for fid in range(nRb, nLm):
            nis, nd = compute_association(x, P, lm, fid, R)

            if nis < innerGate and nd < bestN:
                bestN = nd
                bestId = fid
            elif nis < outer:
                outer = nis

        if bestId != 0:
            associatedLm.append(AssociatedLandmark(lm, bestId))
        elif outer > outerGate:
            newLm.append(lm)

    return associatedLm, newLm


def compute_association(
        x: np.array,  # State matrix
        P: np.array,  # Covariance matrix
        z: np.array,  # Extracted landmark (Measure)
        fid: int,     # Existing feature to compare with z
        R             # Measurement noise
    ) -> tuple[
            float,  # Malahalanobis distance
            float   # Normalised distance
         ]:

    # TODO: Implement this function

    pass
