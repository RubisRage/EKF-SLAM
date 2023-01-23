from landmarks import Landmark, findLines
from loader import Laser, Pose
from utils import cartesian_coords
from dataclasses import dataclass
from ekf import X

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


def createLandmark(m, b, robotPose):
    pass

"""
Associator policy configuration parameters
"""
MIN_OBSERVATIONS = 5 

class Associator:

    def __init__(self) -> None:
        self.landmarkDB: list[Landmark] = []
        self.eskID_2_DBID: dict[int, int] = {}

    def extractLandmarks(self, laser: Laser):
        robotPose = Pose(*X[0:3], 0, 0, 0)
        points = cartesian_coords(laser, robotPose)

        lines = findLines(points, robotPose)




    def updateLandmarks(self):
        pass

    def get_valid_landmarks(self):

        valid_landmarks = list(
            filter(
                lambda landmark: landmark.noObservations >= MIN_OBSERVATIONS,
                self.landmarkDB
            )
        )

        return valid_landmarks
