from utils import (distance, cartesian_coords,
                   angle_between_vectors, pi_to_pi, distance_to_line)
from math import pi
import numpy as np
import cv2
import config


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

    # TODO: Review repeated line segments. Ej: [[1, 2], [1,2] ...]
    # TIA: Last index of i-th line segment may coincide with first index of
    # i+1-th line segment (d = 0).

    third_phase_lines = []

    for i, line in enumerate(second_phase_lines):
        x1, y1 = laser_points[line[0]]
        x2, y2 = laser_points[line[1]]

        lbd_max = 0.1 * distance((x1, y1), (x2, y2))

        m = (y2-y1) / (x2-x1)
        b = y1 - m * x1

        line_segment_start = line[0]

        for n in range(line[0]+1, line[1]):
            x, y = laser_points[n]

            lbd = distance_to_line(m, b, (x, y))

            if lbd < lbd_max:
                third_phase_lines.append([line_segment_start, n])
                line_segment_start = n

    return third_phase_lines


def line_merging(lines, laser_points, laser):

    dmax = 1.5  # meters
    alfa_max = pi_to_pi(5*pi/180)  # radians

    n = 0
    merged_lines = [lines[0]]

    for i in range(1, len(lines)):
        last_point = laser_points[merged_lines[n][1]]
        first_point = laser_points[lines[i][0]]

        d = distance(last_point, first_point)

        v1 = np.array(laser_points[merged_lines[n][1]]) - \
            np.array(laser_points[merged_lines[n][0]])

        v2 = np.array(laser_points[lines[i][1]]) - \
            np.array(laser_points[lines[i][0]])

        alfa = angle_between_vectors(v1, v2)

        # draw_lines([merged_lines[n], lines[i]], laser_points, laser)

        if d < dmax and alfa < alfa_max:
            merged_lines[n][1] = lines[i][1]
        else:
            merged_lines.append(lines[i])
            n += 1

    return list(filter(lambda line: line[0] != line[1]-1, merged_lines))


def draw_lines(lines, laser_points, laser, show_border=False, show_text=False):
    import display

    pose = (0., 0., 0.)
    frame = np.ones((config.frame_height, config.frame_width, 3)) * 255

    draw_mesh(frame)

    display.display_raw_points(frame, pose, laser)

    for i, line in enumerate(lines):
        i1, i2 = line

        p1 = display.to_display_space(laser_points[i1])
        p2 = display.to_display_space(laser_points[i2])

        # color = (randrange(0, 255), randrange(0, 255), randrange(0, 255))
        color = (0, 0, 255)

        cv2.line(frame, p1, p2, color, 1)

        if show_border:
            cv2.circle(frame, p1, 3, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, p2, 3, (0, 255, 0), cv2.FILLED)

        if show_text:
            cv2.putText(frame, f"{i1}, {i2}", (p2[0], p2[1] + (20 * (1 if i & 1 else -1))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        n = i

    cv2.imshow("Corner extraction test", frame)

    while cv2.waitKey(0) != ord(' '):
        pass

    del frame


def draw_mesh(frame):
    meter = int(config.meters_to_px_ratio)
    horizontal_lines = config.frame_height // meter
    vertical_lines = config.frame_width // meter

    color = (150, 150, 150)

    for i in range(horizontal_lines):
        cv2.line(frame, (0, i*meter), (config.frame_width, i*meter), color, 3)

    for i in range(vertical_lines):
        cv2.line(frame, (i*meter, 0), (i*meter, config.frame_height), color, 3)


def main():
    import loader

    data_loader = loader.loader(config.log)

    for _, laser in data_loader:

        laser_points = cartesian_coords(laser)

        lines = line_segmentation(laser_points, laser.data)
        lines = line_merging(lines, laser_points, laser)

        draw_lines(lines, laser_points, laser)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
