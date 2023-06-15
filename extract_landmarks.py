from ransac import findLines
from slam_types import Laser
from utils import cartesian_coords
import numpy as np


def extract_landmarks(laser: Laser, X: np.array):

    laser_points = cartesian_coords(laser, (X[0], X[1], X[2]))

    lines, landmarks = findLines(laser_points, X)
    return lines, landmarks
