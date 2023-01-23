import numpy as np
from collections import namedtuple
from landmark_association import Associator, Landmark
from math import sqrt, atan

Odom = namedtuple("Odom", "ix, iy, ith, vx, vy, vth")

"""
System state vector, contains estimations for robot and landmarks locations.
Initially only robot information.

X = | xr  | Robot x
    | yr  | Robot y
    | thr | Robot orientation, theta
    | x1  | Landsmark 1 x
    | y1  | Landsmark 1 y
    | x2  | Landsmark 2 x
    | y2  | Landsmark 2 y
      ... 
"""
X = np.array([19.0, 7.5, 0])

"""
Covariance matrix. Contains the robot-landmark, landmark-landmark covariance.
"""
P = np.zeros((3,3))

# Initial position uncertainty. Low since we now the robots initial position.
for i in range(3): P[i][i] = 0.1 

"""
Kalman gain matrix. First column range gain, second column bearing range.
"""
K = np.zeros((3, 2))

VALIDATION_CONSTANT = 0.5

def estimatedPose():
    return X[0:3]

def odometryUpdate(odom: Odom):
    """
    EKF first step. 
    """
    ix, iy, ith, _, _, _ = odom

    # Update position based on odometry
    X[0] += ix
    X[1] += iy
    X[2] += ith

    # Update Jacobian of the prediction model, A
    A = np.array([
        [ 1, 0, -iy ],
        [ 0, 1,  ix ],
        [ 0, 0,  1  ]
    ])

    # Update the process noise matrix, Q
    gaussian_sample = 0.5
    Q = gaussian_sample * np.array({
        [ ix**2,  ix*iy,  ix*ith ],
        [ iy*ix,  iy**2,  iy*ith ],
        [ ith*ix, ith*iy, ith**2 ]
    })
    
    # Update covariance matrix (Robot covariance and robot-landmark covariance)

    # Prr = A * Prr * A + Q
    P[0:3, 0:3] = np.matmul(np.matmul(A, P[0:3, 0:3]), A) + Q 

    # Pri = A * Pri; Pir = T(Pri)
    for i in range(3, P.shape[0], 2):
        P[i:i+2, 0:3] = np.matmul(A, P[i:i+2, 0:3])
        P[0:3, i:i+2] = P[i:i+2, 0:3].T


def reobservedLandmarkUpdate(
        measured: Landmark, 
        associated, 
        associator: Associator,
        only_validate: bool
    ) -> bool:
    """
    EKF second step. Acts as validation gate for landmark association as well.
    """

    x = X[0]
    y = X[1]
    th = X[2]

    lx, ly = associated.x, associated.y

    r = sqrt((lx - x)**2 + (ly - y)**2)
    r2 = r**2

    robot_H = np.array([
        [   (x-lx)/r  , (y-ly)/r    ,  0 ],
        [ (ly - y)/r2 , (lx - x)/r2 , -1 ],
    ])

    landmark_H = -1*robot_H[0:2, 0:2]
    
    noLandmarks = (P.shape[0] - 3) // 2

    H = np.zeros((2, 2 * noLandmarks))

    H[0:2, 0:3] = robot_H
    # TODO: Fill H with corresponding landmark_H

    ekf_id = associator.getEFK_ID(associated.id)


    c = 0.01

    R = np.array([
        [ r*c , 0 ],
        [  0  , 1 ]
    ])

    V = np.identity(2)

    # Inovation covariance for landmark i: Si = (H*P*T(H) + V*R*T(V))
    S = np.matmul(np.matmul(H, P), H.T) + np.matmul(np.matmul(V, R), V.T)

    # TODO: Check gate condition
    z = np.array([
        measured.x,
        measured.y
    ])

    h = np.array([
        sqrt((associated.x - x)**2 + (associated.y - y)**2),
        atan((associated.y - y) / (associated.x - x)) - th
    ])

    innovation = z - h

    validation_gate = np.matmul(np.matmul(innovation.T, np.linalg.inv(S)), innovation)

    if validation_gate > VALIDATION_CONSTANT:
        return False

    if only_validate: return True

    # Kalman gain: K = P * t(H) * S^-1
    K = np.matmul(np.matmul(P, H.T), np.linalg.inv(S))

    # Find z and h in order to do (z - h)
    X = X + np.matmul(K, ())

    return True

def includeNewLandmarks(associator: Associator):
    pass
