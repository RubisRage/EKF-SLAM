#!/usr/bin/python
import config
import numpy as np
import cv2
import random

from ekf import predict, update, augment
from loader import loader
from associate import associate
from corner_extraction import find_corners
from utils import cartesian_coords, process_laser, global_coords


def main():
    data_loader = loader(config.log)
    random.seed(0)

    X = config.X
    P = config.P
    Q = config.Q
    R = config.R
    dt = config.dt
    INNER_GATE = config.INNER_GATE
    OUTER_GATE = config.OUTER_GATE
    xtrue = np.zeros((3,))

    for controls, laser in data_loader:

        # STEP 1: Predict
        X, P = predict(X, P, controls, Q, dt)
        xtrue += [*controls]

        laser_points = process_laser(laser)

        pose = X[:3] * [1, -1, 1]

        print("System: ", pose)
        print("Laser: ", laser.data[:5])
        print("Processed laser: ", global_coords(laser_points[:5], pose))

        z = find_corners(X, laser_points, laser.data)

        lm, nLm = associate(X, P, z, R, INNER_GATE, OUTER_GATE)

        # print(f"Associated: {len(lm)}, New: {len(nLm)}")
        display_helper(X, laser_points, lm, nLm)

        # Draw map
        while (key := cv2.waitKey()) != ord(' '):
            if key == ord('q'):
                cv2.destroyAllWindows()
                exit()

        # STEP 2: Update
        X, P = update(X, P, lm, R)

        # STEP 3: Augment
        X, P = augment(X, P, nLm, R)


def display_helper(X, laser_points, associatedLm, newLm):
    from display import draw_points, draw_mesh, draw_robot

    frame = np.ones((config.frame_height, config.frame_width, 3)) * 255

    draw_mesh(frame)

    pose = X[:3] * [1, -1, 1]

    # Draw robot
    # draw_points(frame, [pose[:2]], color=(0, 0, 255))
    draw_robot(frame, pose, color=(0, 0, 255))

    # Draw observed points
    draw_points(frame, global_coords(laser_points, pose))

    # Draw associated landmarks
    z = global_coords(cartesian_coords(
        np.array(list(map(lambda lm: lm.z, associatedLm)), dtype=np.double)
        ), pose)

    ids = list(map(lambda lm: lm.id, associatedLm))

    draw_points(frame, z, color=(255, 0, 255), radius=3, labels=ids)

    systemAssociatedLm = []

    for id in ids:
        from models import observe_model
        systemAssociatedLm.append(global_coords(cartesian_coords(
            [observe_model(X, id)[0]]), pose)[0])

    # Draw system associated landmarks
    # Predicted
    draw_points(frame, systemAssociatedLm, color=(
        0, 255, 0), radius=2)
    # Last seen
    draw_points(frame, X[3:].reshape(
        ((X.shape[0] - 3) // 2, 2)) * [1, -1], color=(255, 0, 0), radius=1)

    # Draw new landmarks
    draw_points(frame, global_coords(cartesian_coords(newLm), pose), color=(0, 255, 255), radius=3)


    cv2.imshow("Association test", frame)


if __name__ == "__main__":
    main()
