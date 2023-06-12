#!/usr/bin/python

from utils import least_squares, distance_to_line, cartesian_coords
from slam_types import Laser
import numpy as np
import random

"""
RANSAC configuration parameters
"""
MAX_TRIES = 100
SAMPLE_WINDOW = 10
MAX_SAMPLES = 5
TOLERANCE = 0.05
CONSENSUS = 40


def findLines(laser: Laser, X: np.array):
    """
    RANSAC line landmark detector.
    """

    laser_points = cartesian_coords(laser, (X[0], X[1], X[2]))

    noRanges, _ = laser_points.shape
    noTries = 0

    lines = []

    notInLine = laser_points.copy()

    while noTries < MAX_TRIES and len(notInLine) > CONSENSUS:

        randomSamples = np.full((MAX_SAMPLES, 2), 0.0, dtype=np.float64)
        centerIndex = random.randrange(
            SAMPLE_WINDOW // 2,
            len(laser_points) - SAMPLE_WINDOW // 2
        )

        randomSamples[0] = laser_points[centerIndex]

        for i in range(1, MAX_SAMPLES):
            newPoint = False

            while not newPoint:
                sampleIndex = random.randrange(
                    centerIndex - SAMPLE_WINDOW // 2,
                    centerIndex + SAMPLE_WINDOW // 2
                )

                if laser_points[sampleIndex] not in randomSamples:
                    newPoint = True

            randomSamples[i] = laser_points[sampleIndex]

        # Fitting line to sampled points
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
            notInLine = notAssociated[: notAssociatedCount]

            m, b = least_squares(associated)

            # TODO: Dont include associated and associatedCount
            # currently needed for display
            lines.append((m, b, associated, associatedCount))

            noTries = 0
        else:
            noTries += 1

    # end

    if len(lines) == 0:
        raise Exception()

    return lines

# end
