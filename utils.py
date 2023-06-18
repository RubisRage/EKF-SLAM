import numpy as np
from slam_types import Laser
from math import sqrt, cos, sin, pi, inf


def angle_between_vectors(v1, v2):
    return np.arccos(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )


def range_bearing(cartesian_points, robotPose: tuple[float, float, float]):
    rx, ry, rth = robotPose

    polar_points = np.ndarray((len(cartesian_points), 2), dtype=np.double)

    for i, point in enumerate(cartesian_points):
        x, y = point

        polar_points[i][0] = distance((rx, ry), (x, y))
        polar_points[i][1] = pi_to_pi(np.arctan(y/x) - rth)

    return polar_points


def cartesian_coords(polar_points, robotPose=(0., 0., 0.)):
    rx, ry, rth = robotPose

    cartesian_points = np.ndarray((len(polar_points), 2), dtype=np.double)

    for i, p in enumerate(polar_points):
        r, b = p
        cartesian_points[i][0] = r * cos(b + rth) - rx
        cartesian_points[i][1] = r * -sin(b + rth) - ry

    return cartesian_points


def process_laser(laser: Laser, robotPose=(0., 0., 0.)):
    """
    Converts a set of laser measurements in 2d points in
    cartesian coordinates.
    """
    _, start, _, step, laserdata = laser

    x, y, th = robotPose
    points = []

    theta = start
    degreeResolution = step

    # TODO: Different indices for laserdata and points

    for r in laserdata:
        if r != inf:
            points.append((r * cos(theta + th) - x, r * -sin(theta + th) - y))

        theta += degreeResolution

    return np.array(points)


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


def intersection_two_lines(l1, l2, laser_points):
    x1, y1 = laser_points[l1[0]]
    x2, y2 = laser_points[l1[1]]
    x3, y3 = laser_points[l2[0]]
    x4, y4 = laser_points[l2[1]]

    denominator = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))

    # Parallel line, there is no intersection
    if denominator == 0:
        return None

    numerator_x = ((x1 * y2 - y1 * x2) * (x3 - x4)) - \
        ((x1 - x2) * (x3 * y4 - y3 * x4))
    numerator_y = ((x1 * y2 - y1 * x2) * (y3 - y4)) - \
        ((y1 - y2) * (x3 * y4 - y3 * x4))

    intersection_x = numerator_x / denominator
    intersection_y = numerator_y / denominator

    return intersection_x, intersection_y


def main():

    x, y = intersection_two_lines([(0, 4), (-1, 3)], [(0, 0), (-1, 1)])
    print(x, y)  # (2,2) Da correcto.


if __name__ == "__main__":
    main()
