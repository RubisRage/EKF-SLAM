from collections import namedtuple

Odom = namedtuple("Odom", "ix, iy, ith")
AssociatedLandmark = namedtuple("AssociatedLandmark", "z id")
Laser = namedtuple("Laser", "timestamp start end step data")
