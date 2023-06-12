import numpy as np
from slam_types import Laser
from math import sqrt, cos, sin, pi


def range_bearing(point, robotPose: tuple[float, float, float]):
    rx, ry, rth = robotPose
    x, y = point

    r = distance((rx, ry), (x, y))
    b = pi_to_pi(np.arctan(y/x) - rth)

    return r, b


def cartesian_coords(laser: Laser, robotPose=(0., 0., 0.)):
    """
    Converts a set of laser measurements in 2d points
    """
    _, start, _, step, laserdata = laser

    x, y, th = robotPose

    points = np.zeros((laserdata.shape[0], 2))

    theta = start
    degreeResolution = step

    for i, r in enumerate(laserdata):
        points[i][0] = r * cos(theta + th) - x
        points[i][1] = r * -sin(theta + th) - y
        theta += degreeResolution

    return points


def least_squares(points) -> tuple[float, float]:
    """
    Find least-squares line for point cloud
    y = Ap where p = [m, b], A = [[x1, 1], [x2, 1], ...] and y = [y1, y2, ...]
    """
    len, _ = points.shape

    A = np.vstack([points[:len, 0], np.ones(len)]).T

    return np.linalg.lstsq(A, points[:len, 1], rcond=None)[0]


def distance_to_line(m, b, point) -> float:
    """
    Compute distance from point(x,y) to line f(x) = mx + b
    """
    x, y = point

    return abs(-m*x + y - b) / sqrt(m**2 + 1)


def distance(a: tuple[float, float], b: tuple[float, float]):
    ax, ay = a
    bx, by = b

    return sqrt((bx - ax)**2 + (by - ay)**2)


def pi_to_pi(angle):
    if angle > pi:
        return angle - 2*pi
    elif angle < -pi:
        return angle + 2*pi
    
    return angle
