from utils import pi_to_pi
from math import atan2, sqrt
import numpy as np


def observe_model(
    X: np.array,  # System state
    fid: int      # Feature id
) -> tuple[
    np.array,  # Predicted feature position [range, bearing]
    np.array   # Jacobian of the observation model
]:
    """
    Computes the expected position of a known landmark and that landmarks
    prediction Jacobian.
    """

    H = np.zeros((2, X.shape[0]))

    xdiff = X[fid] - X[0]
    ydiff = X[fid+1] - X[1]
    distance2 = xdiff**2 + ydiff**2
    distance = sqrt(distance2)
    xd = xdiff / distance
    yd = ydiff / distance
    xd2 = xdiff / distance2
    yd2 = ydiff / distance2

    z = np.array([
        distance,
        pi_to_pi(atan2(ydiff, xdiff) - X[2])
    ])

    H[:, 0:3] = [[-xd, -yd,   0],
                 [yd2, -xd2, -1]]

    H[:, fid:fid+2] = [[xd,    yd],
                       [-yd2, xd2]]

    return z, H
