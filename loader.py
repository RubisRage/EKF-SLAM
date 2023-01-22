import numpy as np
from collections import namedtuple

Pose = namedtuple("Pose", "timestamp x y th vx vy vth")
Laser = namedtuple("Laser", "timestamp start end step data")

def loader(filename) -> tuple[Pose, Laser]:

    position2d = None
    laser = None

    with open(filename, "r") as f:
        
        line = f.readline()

        while len(line) > 0:
            data_list = line.split()
            line = f.readline()

            if data_list[0] == "##": continue

            timestamp = float(data_list[0])

            if timestamp == 0.0: continue

            data_list = data_list[3:]

            match data_list:
                case ["position2d", *_, x, y, th, vx, vy, vth, _]:
                    position2d = Pose(
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
                        lambda v: v!=0 and v <= float(max), 
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

            if position2d is not None and laser is not None:
                if position2d.timestamp == laser.timestamp:
                    yield (position2d, laser) 

                position2d = None
                laser = None
