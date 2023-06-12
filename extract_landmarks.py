from ransac import findLines
from slam_types import Laser
import numpy as np


def extract_landmarks(laser: Laser, X: np.array):
    lines, landmarks = findLines(laser, (X[0], X[1], X[2]))
    return landmarks
