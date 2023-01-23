from landmarks import Landmark, findLines
from loader import Laser, Pose
from utils import cartesian_coords
from dataclasses import dataclass
from ekf import X
from math import sqrt, atan

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
MAX_DISTANCE = 5000

class Associator:

    def __init__(self) -> None:
        self.landmarkDB: list[Landmark] = []
        self.eskID_2_DBID: dict[int, int] = {}

    def extractLandmarks(self, laser: Laser):
        robotPose = Pose(*X[0:3], 0, 0, 0)
        points = cartesian_coords(laser, robotPose)

        lines = findLines(points, robotPose)

        for line in lines:
            lm = self.createLandmark(line[0], line[1], robotPose)

    
    def associateLandmark(self, lm: Landmark):
        for candidate in self.landmarkDB:
            pass


    def updateLandmarks(self):
        pass

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

        # Do associations? 


        return lm