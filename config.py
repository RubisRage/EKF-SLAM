import numpy as np
from math import pi
from utils import pi_to_pi

log = "medium_nd_5.log"
dt = 0.1  # seconds

# FRAME CONFIG ======================

frame_width = 600
frame_height = 350
meters_to_px_ratio = frame_width / 10

# ===================================
mesh_x = int(frame_width / 10)
mesh_y = int(frame_height / 10)
# ===================================

# CORNER EXTRACTION CONFIG ==========

# Line segmentation
lseg_base_distance = 0.1            # meters
lseg_distance_scale_factor = 0.05
lseg_alfa_max = 30*pi / 180         # radians
lseg_lbd_scale_factor = 0.1

# Line merging
lmerg_max_distance = 1.5              # meters
lmerg_max_angle = pi_to_pi(5*pi/180)  # radians

# Corner extraction
cext_alfa_min = (90-15) * pi / 180    # radians
cext_alfa_max = (90+15) * pi / 180    # radians
cext_dmin = 1                         # meters
cext_max_distance_to_corner = 1       # meters

# ===================================

# SLAM CONFIG =======================

# State matrix X
X = np.zeros((3,))

# Covariance matrix
P = np.zeros((3, 3))

# Prediction model noise
# sigmaV = 0.3        # m/s
# sigmaG = 3.*pi/180  # radians
# Q = np.array([
# [sigmaV**2, 0],
# [0, sigmaG**2]
# ])

Q = 0.3

# Observe model noise
sigmaR = 0.1        # meters
sigmaB = 1.*pi/180  # radians

R = np.array([
    [sigmaR**2, 0],
    [0, sigmaB**2]
])

# Association gates
INNER_GATE = 4.0
OUTER_GATE = 25.0

# ===================================
