import numpy as np
from collections import namedtuple

#Odom = namedtuple("Odom", "ix, iy, ith, vx, vy, vth")
Odom = namedtuple("Odom", "ix, iy, ith")
Pose = namedtuple("Pose", "timestamp x y th vx vy vth")
Laser = namedtuple("Laser", "timestamp start end step data")


def loader(filename) -> tuple[Odom, Laser]:
    position2d_current = None
    position2d_last = None
    laser = None

    with open(filename, "r") as f:

        line = f.readline()

        while len(line) > 0:
            data_list = line.split()
            line = f.readline()

            if data_list[0] == "##":
                continue

            timestamp = float(data_list[0])

            if timestamp == 0.0:
                continue

            data_list = data_list[3:]

            match data_list:
                case ["position2d", *_, x, y, th, vx, vy, vth, _]:
                    position2d_last = position2d_current
                    position2d_current = Pose(
                        timestamp,
                        float(x),
                        float(y),
                        float(th),
                        float(vx),
                        float(vy),
                        float(vth)
                    )
                case ["laser", _, _, _, _, start, end, step, max, count, *data]:
                    data = filter(
                        lambda v: v != 0 and v <= float(max),
                        map(lambda v: float(v), data)
                    )

                    np_data = np.array(list(data), dtype=np.float64)

                    laser = Laser(
                        timestamp,
                        float(start),
                        float(end),
                        float(step),
                        np_data
                    )

            if (position2d_current is not None
                    and position2d_last is not None
                    and laser is not None):

                odom = Odom(
                    position2d_current.x - position2d_last.x,
                    position2d_current.y - position2d_last.y,
                    position2d_current.th - position2d_last.th,
                  # position2d_current.vx,
                  # position2d_current.vy,
                  # position2d_current.vth
                )

                if position2d_current.timestamp == laser.timestamp:
                    yield (odom, laser)

                    laser = None


def main():
    data_loader = loader("medium_nd_5.log")

    for odom, laser in data_loader:
        print(f"{odom=}", f"{laser.timestamp}", sep="\n")


if __name__ == "__main__":
    main()
