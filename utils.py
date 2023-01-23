import numpy as np
from loader import Pose
from math import sqrt, cos, sin

def cartesian_coords(laser, robotPose = Pose(0, 0, 0, 0, 0, 0, 0)):
    """
    Converts a set of laser measurements in 2d points 
    """
    _, start, _, step, laserdata = laser 

    points = np.zeros((laserdata.shape[0], 2))

    theta = start
    degreeResolution = step

    for i, r in enumerate(laserdata):
        points[i][0] = r * cos(theta + robotPose.th) 
        points[i][1] = r * -sin(theta + robotPose.th)
        theta += degreeResolution

    return points


def least_squares(points, len = None) -> tuple[float, float]:
    """
    Find least-squares line for point cloud
    y = Ap where p = [m, b] and A = [[x, 1]]
    """
    if len is None:
        len, _ = points.shape

    A = np.vstack([points[:len, 0], np.ones(len)]).T

    return np.linalg.lstsq(A, points[:len, 1], rcond=None)[0]


def distance_to_line(m, b, point) -> float:
    """
    Compute distance from point(x,y) to line y = mx + b
    """
    x, y = point

    return abs(-m*x + y - b) / sqrt(m**2 + 1)

def distance(a: tuple[int, int], b: tuple[int, int]):
    ax, ay = a
    bx, by = b

    vx = bx - ax
    vy = by - ay

    return sqrt(vx**2 + vy**2)