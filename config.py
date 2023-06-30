import numpy as np
from math import pi

log = "logs/log-5.log"
dt = 0.1  # seconds

# FRAME CONFIG ======================

# Global frame
global_frame_config = {
        "width": 1000,
        "height": 1000,
}

global_frame_config["meters_to_px_ratio"] = global_frame_config["width"] / 25
global_frame_config["origin"] = np.array([
    global_frame_config["width"] / 2,
    global_frame_config["height"] / 2
])

# ===================================

# CORNER EXTRACTION CONFIG ==========

# Line segmentation
lseg_base_distance = 0.05           # meters
lseg_distance_scale_factor = 0.10
lseg_alfa_max = 10*pi / 180         # radians
lseg_lbd_scale_factor = 0.1

# Line merging
lmerg_max_distance = 1.5              # meters
lmerg_max_angle = 5*pi/180            # radians

# Corner extraction
cext_alfa_min = (90-12) * pi / 180    # radians
cext_alfa_max = (90+12) * pi / 180    # radians
cext_dmin = 1                         # meters
cext_max_distance_to_corner = 1       # meters

# ===================================

# SLAM CONFIG =======================

# State matrix X
X = np.zeros((3,))

# Covariance matrix
P = np.zeros((3, 3))

# Prediction model noise
Q = 0.3

# Observe model noise
sigmaR = 0.1        # meters
sigmaB = 1.*pi/180  # radians

R = np.array([
    [sigmaR**2, 0],
    [0, sigmaB**2]
])

# Association gates
INNER_GATE = 3.0
OUTER_GATE = 25.0

# ===================================
