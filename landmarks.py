#!/usr/bin/python

from loader import loader, Pose, Laser
import numpy as np
import random
from math import sin, cos, pi, sqrt
import matplotlib.pyplot as plt

MAX_TRIES = 100
SAMPLE_WINDOW = 10
MAX_SAMPLES = 5
TOLERANCE = 0.05
CONSENSUS = 40

class Landmark:

    def __init__(self) -> None:
        self.id = -1
        self.pos = np.zeros(2)
        self.m = -1
        self.b = -1

def least_squares(points):
    # Find least-squares line for point cloud
    # y = Ap where p = [m, b] and A = [[x, 1]]
    len, _ = points.shape
    A = np.vstack([points[:, 0], np.ones(len)]).T

    return np.linalg.lstsq(A, points[:, 1], rcond=None)[0]

def distance_to_line(m, b, point):
    x, y = point

    return abs(-m*x + y - b) / sqrt(m**2 + 1)

#ranges = [x, y] in world coordinates
def findLandmarks(ranges):
    noRanges, _ = ranges.shape
    noTries = 0

    lines = []

    notInLine = ranges.copy()

    while noTries < MAX_TRIES and len(notInLine) > CONSENSUS:

        randomSamples = np.full((MAX_SAMPLES, 2), 0.0, dtype=np.float64)
        centerIndex = random.randrange(
            SAMPLE_WINDOW // 2, 
            len(ranges) - SAMPLE_WINDOW // 2
        )

        randomSamples[0] = ranges[centerIndex]

        for i in range(1, MAX_SAMPLES):
            newPoint = False

            while not newPoint:
                sampleIndex = random.randrange(
                    centerIndex - SAMPLE_WINDOW // 2,
                    centerIndex + SAMPLE_WINDOW // 2
                )

                if ranges[sampleIndex] not in randomSamples:
                    newPoint = True

            randomSamples[i] = ranges[sampleIndex] 

        # Find fitting line
        m, b = least_squares(randomSamples)

        # Check how many points are near enough to the line to be considered
        # part of it.

        associated = np.zeros((noRanges, 2))
        associatedCount = 0
        notAssociated = np.zeros((noRanges, 2))
        notAssociatedCount = 0

        for p in notInLine:
            d = distance_to_line(m, b, p)

            if d < TOLERANCE:
                associated[associatedCount] = p
                associatedCount += 1
            else:
                notAssociated[notAssociatedCount] = p
                notAssociatedCount += 1

        if associatedCount >= CONSENSUS:
            notInLine = notAssociated[ : notAssociatedCount]

            lines.append((m, b, associated, associatedCount))

            noTries = 0
        else:
            noTries += 1

    #end

    if len(lines) == 0:
        raise Exception()

    return lines

#end 

def cartesian_coords(laser, pose = Pose(0, 0, 0, 0, 0, 0, 0)):
    """
    Converts a set of laser measurements in 2d points 
    """
    _, start, _, step, laserdata = laser 

    points = np.zeros((laserdata.shape[0], 2))

    theta = start
    degreeResolution = step

    for i, r in enumerate(laserdata):
        points[i][0] = r * cos(theta)
        points[i][1] = r * -sin(theta)
        theta += degreeResolution

    return points


def main():
    data_loader = loader("medium_nd_5.log")

    for pose, laser in data_loader:

        points = cartesian_coords(laser)

        x = points[:, 0]
        y = points[:, 1]

        plt.plot(x, y, 'o', markersize=4, label="data")
        plt.legend()
        plt.show()

        lines = findLandmarks(points)

        colors = "bgrcmykw"

        for i, l in enumerate(lines):
            m, b, a, c = l
            x, y = a[:c, 0], a[:c, 1]
            plt.plot(x, m*x + b, colors[i], label=f"line {i}")
            plt.plot(x, y, colors[i]+'*', markersize=5)

        plt.show()

if __name__ == "__main__":
    main()