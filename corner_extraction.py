from utils import (distance, angle_between_vectors, distance_to_line,
                   intersection_two_lines, range_bearing, process_laser)
import numpy as np
import config


def line_segmentation(laser_points, laser_data):

    line_segment_start = 0
    first_phase_lines = []

    base_distance = config.lseg_base_distance
    distance_scale_factor = config.lseg_distance_scale_factor

    for i, p in enumerate(laser_points):

        d = None

        if i+1 < len(laser_points):
            d = distance(p, laser_points[i+1])
            dmax = base_distance + \
                distance_scale_factor*(laser_data[i] + laser_data[i+1])

        if d is None or d > dmax:
            first_phase_lines.append((line_segment_start, i))
            line_segment_start = i+1

    second_phase_lines = []
    alfa_max = config.lseg_alfa_max

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
    lbd_scale_factor = config.lseg_lbd_scale_factor

    for i, line in enumerate(second_phase_lines):
        x1, y1 = laser_points[line[0]]
        x2, y2 = laser_points[line[1]]

        lbd_max = lbd_scale_factor * distance((x1, y1), (x2, y2))

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


def line_merging(lines, laser_points):

    dmax = config.lmerg_max_distance
    alfa_max = config.lmerg_max_angle

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

        if d < dmax and alfa < alfa_max:
            merged_lines[n][1] = lines[i][1]
        else:
            merged_lines.append(lines[i])
            n += 1

    return list(filter(lambda line: abs(line[0] - line[1]) > 1, merged_lines))


def corner_extraction(lines: list, laser_points):
    Vcorners = []
    dmin = config.cext_dmin
    max_distance_to_corner = config.cext_max_distance_to_corner
    alfa_min = config.cext_alfa_min
    alfa_max = config.cext_alfa_max

    for i in range(len(lines)-1):
        j = i+1
        d = distance(laser_points[lines[i][1]], laser_points[lines[j][0]])
        v1 = np.array(laser_points[lines[j][1]]) - \
            np.array(laser_points[lines[j][0]])

        v2 = np.array(laser_points[lines[i][1]]) - \
            np.array(laser_points[lines[i][0]])

        alfa = angle_between_vectors(v1, v2)

        if d < dmin and alfa_min < alfa < alfa_max:
            x, y = intersection_two_lines(lines[i], lines[j], laser_points)
            dp1 = distance(laser_points[lines[i][1]], (x, y))
            dp2 = distance(laser_points[lines[j][0]], (x, y))
            if dp1 < max_distance_to_corner and dp2 < max_distance_to_corner:
                Vcorners.append((x, y))

    return Vcorners


def find_corners(X, laser_points, laser_data):
    segmented_lines = line_segmentation(laser_points, laser_data)
    merged_lines = line_merging(segmented_lines, laser_points)
    corners = corner_extraction(merged_lines, laser_points)

    return range_bearing(corners, X[:3])


def main():
    import loader
    import display
    import cv2
    import config

    data_loader = loader.loader(config.log)

    for _, laser in data_loader:
        laser_points = process_laser(laser)
        frame = np.ones((config.frame_height, config.frame_width, 3)) * 255

        lines = line_segmentation(laser_points, laser.data)
        lines = line_merging(lines, laser_points)
        corners = corner_extraction(lines, laser_points)

        display.draw_mesh(frame)
        display.draw_points(frame, laser_points)
        display.draw_lines(frame, lines, laser_points, show_border=True)
        display.draw_corner(frame, corners)

        cv2.imshow("Corner extraction test", frame)

        key = cv2.waitKey(0)

        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
