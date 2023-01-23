from loader import loader, Pose
from ekf import *
from landmark_association import *

# Landmark manager and associator 
associator = Associator()

def main():
    data_loader = loader("medium_nd_5.log")

    for controls, laser in data_loader:
        associator.extractLandmarks(laser)

        # STEP 1: Update odometry estimation
        odometryUpdate(controls)

        # STEP 2: Update odometry estimation
        reobservedLandmarksUpdate()

        # STEP 3: Update odometry estimation
        includeNewLandmarks()


if __name__ == "__main__":
    main()