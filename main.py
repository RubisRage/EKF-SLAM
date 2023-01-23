#!/usr/bin/python

from loader import loader, Pose
from landmark_association import *
from ekf import (
    odometryUpdate,
    estimatedPose,
    reobservedLandmarkUpdate,
    includeNewLandmarks
)


# Landmark manager and associator 
associator = Associator()

def main():
    data_loader = loader("medium_nd_5.log")

    for controls, laser in data_loader:
        # STEP 1: Update odometry estimation
        odometryUpdate(controls)

        associator.extractLandmarks(estimatedPose(), laser)

        associated_landmarks = associator.getAssociatedLandmarks()

        for measured, associated in associated_landmarks: 
            # STEP 2: Update odometry estimation

            only_validate = associated.timesObserved < MIN_OBSERVATIONS

            matched = reobservedLandmarkUpdate(
                measured, 
                associated, 
                associator,
                only_validate
            )

            associator.updateLandmark(measured, matched)

        # STEP 3: Update odometry estimation
        includeNewLandmarks()


if __name__ == "__main__":
    main()