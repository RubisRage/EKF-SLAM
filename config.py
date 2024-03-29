import numpy as np
from math import pi

log = "./logs/hall-1.log"
dt = 0  # milliseconds, 0 for step-by-step execution

# FRAME CONFIG ======================

# Global frame
global_frame_config = {
        "width": 1000,
        "height": 1000,
}

global_frame_config["meters_to_px_ratio"] = global_frame_config["width"] / 15
global_frame_config["origin"] = np.array([
    global_frame_config["width"] * 0.1,
    global_frame_config["height"] * 0.5
])

# Local frame
local_frame_config = {
    "width": 1000,
    "height": 600,
}

local_frame_config["meters_to_px_ratio"] = global_frame_config["width"] / 20
local_frame_config["origin"] = np.array([
    local_frame_config["width"] * 0.4,
    local_frame_config["height"] * 0.5
])


# ===================================

# CORNER EXTRACTION CONFIG ==========

# Line segmentation
lseg_base_distance = 0.05           # meters
lseg_distance_scale_factor = 0.02
lseg_alfa_max = 5*pi / 180         # radians
lseg_lbd_scale_factor = 0.1

# Line merging
lmerg_max_distance = .1               # meters
lmerg_max_angle = 5*pi/180            # radians

# Corner extraction
cext_alfa_min = (90-12) * pi / 180    # radians
cext_alfa_max = (90+12) * pi / 180    # radians
cext_dmin = 0.5                       # meters
cext_max_distance_to_corner = 0.5    # meters

# ===================================

# SLAM CONFIG =======================

ADD_NOISE = True
DT_OBSERVE = 8*0.025

# State matrix X
X = np.zeros((3,))

# Covariance matrix
P = np.zeros((3, 3))

# Prediction model noise
sigmaX = 0.003
sigmaY = 0.003
sigmaT = 0.003 

Q = np.array([
    [sigmaX**2, 0, 0],
    [0, sigmaY**2, 0],
    [0, 0, sigmaT**2]
])

# Observe model noise
sigmaR = 0.001       # meters
sigmaB = 0.01*pi/180  # radians

R = np.array([
    [sigmaR**2, 0],
    [0, sigmaB**2]
])

# Association gates
INNER_GATE = 2500
OUTER_GATE = 200000

# Tikhonov Regulation factor
tikhonov_factor = 0.0

# ===================================
