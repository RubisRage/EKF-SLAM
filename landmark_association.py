from ransac import findLines
from loader import Laser, Pose
from utils import cartesian_coords, distance
from dataclasses import dataclass
from math import sqrt, atan, inf

STARTING_LIFE = 40

@dataclass
class Landmark:
    # Unique identifier
    id: int = -1

    # Landmark information
    x: float = -1.
    y: float = -1.
    range: float = -1.       # Distance to landmark
    bearing: float = -1.     # Direction to landmark

    # Line constants
    m: float = -1.
    b: float = -1.

    # Counter to check whether to discard this landmark
    life: int = -1

    # Number of times seen
    timesObserved: int = -1



"""
Associator policy configuration parameters
"""
MIN_OBSERVATIONS = 5 
MIN_DISTANCE = 5000

class Associator:

    def __init__(self) -> None:
        self.landmarkDB: list[Landmark] = []
        self.ID_INCREMENT = 0
        self.eskID_2_DBID: dict[int, int] = {}
        self.associatedLandmarks: list[tuple[Landmark, Landmark]] = []

    def extractLandmarks(self, robotPose: tuple[float, float, float], laser: Laser):
        self.associatedLandmarks.clear()

        points = cartesian_coords(laser, robotPose)

        lines = findLines(points, robotPose)

        for line in lines:
            lm = self.createLandmark(line[0], line[1], robotPose)
            if lm.id == -1:
                self.addToDB(lm)
            else:
                self.associatedLandmarks.append((lm, self.landmarkDB[lm.id]))


    def addToDB(self, lm: Landmark):
        lm.id = self.ID_INCREMENT
        self.ID_INCREMENT += 1

        self.landmarkDB.append(lm)


    def getAssociatedLandmarks(self):

        valid_associations = []

        for measured, associated in self.associatedLandmarks:
            if associated.timesObserved > MIN_OBSERVATIONS:
                valid_associations.append((measured, associated))
            else:
                associated.timesObserved += 1

        return self.valid_associations
        

    def createLandmark(self, m, b, robotPose):
        xr, yr, thr = robotPose

        m0 = -1.0 / m
        x = b / (m0 - m)
        y = (m0 * b) / (m0 - m)
        range = sqrt((x - xr)**2 + (y - yr)**2)
        bearing = atan((y - yr) / (x - xr))

        # Compute rangeError, bearingError ? GetLineLandmark

        lm = Landmark()

        lm.id = -1

        lm.x = x
        lm.y = y
        lm.range = range
        lm.bearing = bearing

        lm.m = m
        lm.b = b

        lm.life = STARTING_LIFE
        lm.timesObserved = 0

        self.associateLandmark(lm)

        return lm


    def associateLandmark(self, lm: Landmark):

        least_distance = inf # math.inf
        best_match_index = -1

        for index, candidate in enumerate(self.landmarkDB):
            candidate_pos = (candidate.x, candidate.y)
            lm_pos        = (lm.x, lm.y)

            d = distance(candidate_pos, lm_pos)

            if d < least_distance:
                best_match_index = index
                least_distance = d

        if least_distance > MIN_DISTANCE:
            lm.id = -1
        else:
            lm.id = self.landmarkDB[best_match_index].id

    def updateLandmark(self, lm: Landmark, matched: bool):

        if matched:
            self.landmarkDB[lm.id].timesObserved += 1
        else:
            self.addToDB(lm)
