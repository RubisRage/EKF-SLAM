from utils import distance, cartesian_coords


def line_segmentation(laser_points, laser_data):

    line_segment_start = 0
    lines = []

    for i, p in enumerate(laser_points):

        d = None

        if i+1 < len(laser_points):
            d = distance(p, laser_points[i+1])
            dmax = 0.3 + 0.05*(laser_data[i] + laser_data[i+1])

        if d is None or d > dmax:
            lines.append((line_segment_start, i))
            line_segment_start = i+1

    return lines


def main():
    import cv2
    import config
    import numpy as np
    import loader
    import display

    data_loader = loader.loader(config.log)

    _, laser = next(data_loader)

    frame = np.ones((config.frame_height, config.frame_width, 3)) * 255

    pose = (0., 0., 0.)

    display.display_raw_points(frame, pose, laser)

    laser_points = cartesian_coords(laser, pose)

    for line in line_segmentation(laser_points, laser.data):
        i1, i2 = line

        p1 = display.to_display_space(laser_points[i1])
        p2 = display.to_display_space(laser_points[i2])

        cv2.line(frame, p1, p2, (0, 0, 255), 1)

    cv2.imshow("Corner extraction test", frame)

    while cv2.waitKey(0) != ord('q'):
        pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
