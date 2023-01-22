from landmarks import Landmark, findLandmarks

"""
Associator policy configuration parameters
"""
MIN_OBSERVATIONS = 5 

class Associator:

    def __init__(self) -> None:
        self.landmarkDB: list[Landmark] = []
        self.eskID_2_DBID: dict[int, int] = {}

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
