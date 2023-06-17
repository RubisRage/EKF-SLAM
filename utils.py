import numpy as np
from slam_types import Laser
from math import sqrt, cos, sin, pi, inf


def angle_between_vectors(v1, v2):
    return np.arccos(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
           )


def range_bearing(point, robotPose: tuple[float, float, float]):
    rx, ry, rth = robotPose
    x, y = point

    r = distance((rx, ry), (x, y))
    b = pi_to_pi(np.arctan(y/x) - rth)

    return r, b


def cartesian_coords(laser: Laser, robotPose=(0., 0., 0.)):
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

    for i, r in enumerate(laserdata):
        r = laserdata[i]
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

    # Calcula los valores para la fórmula de intersección de rectas
    denominator = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    if denominator == 0:
        # Las rectas son paralelas, no hay intersección
        return None

    numerator_x = ((x1 * y2 - y1 * x2) * (x3 - x4)) - ((x1 - x2) * (x3 * y4 - y3 * x4))
    numerator_y = ((x1 * y2 - y1 * x2) * (y3 - y4)) - ((y1 - y2) * (x3 * y4 - y3 * x4))

    intersection_x = numerator_x / denominator
    intersection_y = numerator_y / denominator

    return intersection_x, intersection_y
#    a1, b1 = l1
#    a2, b2 = l2
#
#    x1 = laser_points[a1][0]
#    y1 = laser_points[a1][1]
#    x2 = laser_points[b1][0]
#    y2 = laser_points[b1][1]
#    x3 = laser_points[a2][0]
#    y3 = laser_points[a2][1]
#    x4 = laser_points[b2][0]
#    y4 = laser_points[b2][1]
#
#    #Pendiente
#    px1 = y2-y1
#    px2 = x2-x1
#    py1 = y4-y3
#    py2 = x4-x3
#    m1 = px1/px2
#    m2 = py1/py2
#
#    #Parallel
#    if m1==m2:
#        return None
#   # Construir el sistema de ecuaciones
#    # Construir el sistema de ecuaciones
#    sistema_ecuaciones = np.array([[x2 - x1, -(x4 - x3)], [y2 - y1, -(y4 - y3)]])
#    valores_independientes = np.array([x3 - x1, y3 - y1])
#    # Resolver el sistema de ecuaciones
#    try:
#        interseccion = np.linalg.solve(sistema_ecuaciones, valores_independientes)
#        return interseccion[0], interseccion[1]  # Retorna las coordenadas (x, y) de la intersección
#    except np.linalg.LinAlgError:
#        # Las líneas son coincidentes
#        return None 