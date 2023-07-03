from utils import (distance, angle_between_vectors, distance_to_line,
                   intersection_two_lines, range_bearing, process_laser,
                   cartesian_coords)
from noise import add_observe_noise
import numpy as np
import config


def line_segmentation(laser_points, laser_polar):

    line_segment_start = 0
    first_phase_lines = []

    base_distance = config.lseg_base_distance
    distance_scale_factor = config.lseg_distance_scale_factor

    for i in range(len(laser_points)-1):
        p1 = laser_polar[i]
        p2 = laser_polar[i+1]

        d = distance(laser_points[i], laser_points[i+1])
        dmax = base_distance + distance_scale_factor*(p1[0] + p2[0])

        if d > dmax:
            first_phase_lines.append((line_segment_start, i))
            line_segment_start = i+1

    if line_segment_start != len(laser_points) - 1:
        first_phase_lines.append((line_segment_start, len(laser_points)-1))

    second_phase_lines = []
    alfa_max = config.lseg_alfa_max

    for line in first_phase_lines:
        for j in range(line[0], line[1]):
            if j+3 < line[1]:
                v1 = np.array(laser_points[j]) - np.array(laser_points[j+1])
                v2 = np.array(laser_points[j+2]) - np.array(laser_points[j+3])

                alfa = angle_between_vectors(v1, v2)

                if alfa < alfa_max:
                    second_phase_lines.append((j, j+3))

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

    return merged_lines


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


def find_corners(X, laser_points, laser_polar):
    segmented_lines = line_segmentation(laser_points, laser_polar)
    merged_lines = line_merging(segmented_lines, laser_points)
    corners = corner_extraction(merged_lines, laser_points)

    return range_bearing(corners)


def main():
    import loader
    import cv2
    import config
    import display
    from utils import global_coords

    data_loader = loader.loader(config.log)
    xtrue = np.zeros((3,))
    dt = config.dt

    rnd = np.random.RandomState(0)

    # TODO: Check duplicate corner extraction

    for i, (controls, laser) in enumerate(data_loader):
        xtrue += controls

        height = config.global_frame_config["height"]
        width = config.global_frame_config["height"]

        frame = np.ones((height, width, 3)) * 255

        laser_points, laser_polar = process_laser(laser)
        noised_laser_polar = add_observe_noise(rnd, laser_polar)
        noised_laser_points = cartesian_coords(noised_laser_polar)

        first_lines = line_segmentation(noised_laser_points, noised_laser_polar)
        second_lines = line_merging(first_lines, noised_laser_points)
        corners = corner_extraction(second_lines, noised_laser_points)

        laser_points = global_coords(noised_laser_points, xtrue)

        display.draw_mesh(frame, config.global_frame_config)
        display.draw_robot(frame, xtrue, config.global_frame_config)
        display.draw_points(frame, laser_points, config.global_frame_config)
        display.draw_lines(frame, second_lines, laser_points,
                           config.global_frame_config)
        display.draw_points(frame, global_coords(corners, xtrue),
                            config.global_frame_config, color=(255, 0, 0),
                            radius=3, labels=range(len(corners)))

        cv2.imshow("Corner extraction test", frame)

        key = cv2.waitKey(dt)

        if key == ord('s'):
            dt = 10 if dt == 0 else 0

        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
