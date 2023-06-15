from utils import (distance, cartesian_coords,
                   angle_between_vectors, pi_to_pi, distance_to_line)
from math import pi
import numpy as np
from random import randrange


def line_segmentation(laser_points, laser_data):

    line_segment_start = 0
    first_phase_lines = []

    for i, p in enumerate(laser_points):

        d = None

        if i+1 < len(laser_points):
            d = distance(p, laser_points[i+1])
            dmax = 0.1 + 0.05*(laser_data[i] + laser_data[i+1])

        if d is None or d > dmax:
            first_phase_lines.append((line_segment_start, i))
            line_segment_start = i+1

    second_phase_lines = []
    alfa_max = pi_to_pi(30*pi / 180)

    for line in first_phase_lines:
        for j in range(line[0], line[1]):
            if j+3 < len(laser_points):

                v1 = np.array(laser_points[j]) - np.array(laser_points[j+1])
                v2 = np.array(laser_points[j+2]) - np.array(laser_points[j+3])

                alfa = angle_between_vectors(v1, v2)

                if alfa < alfa_max:
                    second_phase_lines.append((j, j+3))

    third_phase_lines = []

    for i, line in enumerate(second_phase_lines):
        x1, y1 = laser_points[line[0]]
        x2, y2 = laser_points[line[1]]

        lbd_max = 0.1 * distance((x1, y1), (x2, y2))

        m = (y2-y1) / (x2-x1)
        b = y1 - m * x1

        line_segment_start = line[0]

        for n in range(line[0]+1, line[1]):
            lbd = distance_to_line(m, b, (x1, y1))

            if lbd < lbd_max:
                third_phase_lines.append((line_segment_start, n))
                line_segment_start = n

    return third_phase_lines


def main():
    import cv2
    import config
    import loader
    import display

    data_loader = loader.loader(config.log)

    _, laser = next(data_loader)

    frame = np.ones((config.frame_height, config.frame_width, 3)) * 255

    pose = (0., 0., 0.)

    display.display_raw_points(frame, pose, laser)

    laser_points = cartesian_coords(laser, pose)

    lines = line_segmentation(laser_points, laser.data)

    for line in lines:
        i1, i2 = line

        p1 = display.to_display_space(laser_points[i1])
        p2 = display.to_display_space(laser_points[i2])

        # color = (randrange(0, 255), randrange(0, 255), randrange(0, 255))
        color = (0, 0, 255)

        cv2.line(frame, p1, p2, color, 1)

    cv2.imshow("Corner extraction test", frame)

    while cv2.waitKey(0) != ord('q'):
        pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
