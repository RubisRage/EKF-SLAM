from ransac import findLines
from loader import Laser, Pose
from utils import cartesian_coords, distance
from dataclasses import dataclass
from math import sqrt, atan, inf

STARTING_LIFE = 40

@dataclass
class Landmark:
    # Unique identifier
    id: int

    # Landmark information
    x: float 
    y: float
    range: float        # Distance to landmark
    bearing: float      # Direction to landmark

    # Line constants
    m: float
    b: float

    # Counter to check whether to discard this landmark
    life: int

    # Number of times seen
    timesObserved: int



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
        self.associatedLandmarks: list[tuple[Landmark, Landmark]]

    def extractLandmarks(self, robotPose: tuple[float, float, float], laser: Laser):
        self.measuredLandmarks.clear()

        robotPose = Pose(*robotPose, 0, 0, 0)
        points = cartesian_coords(laser, robotPose)

        lines = findLines(points, robotPose)

        for line in lines:
            lm = self.createLandmark(line[0], line[1], robotPose)
            if lm.id == -1:
                self.addToDB(lm)
            else:
                self.associatedLandmarks.append(lm)


    def addToDB(self, lm: Landmark):
        lm.id = self.ID_INCREMENT
        self.ID_INCREMENT += 1

        self.landmarkDB.append(lm)


    def getAssociatedLandmarks(self):
        return self.associatedLandmarks
        

    def createLandmark(self, m, b, robotPose):
        xr, yr, thr, _ = robotPose

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
