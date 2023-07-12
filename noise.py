import config


def add_control_noise(rnd, controls):
    noised_controls = controls.copy()

    if config.ADD_NOISE:
        noised_controls[0] += rnd.normal(0, 1, 1) * config.sigmaX
        noised_controls[1] += rnd.normal(0, 1, 1) * config.sigmaY
        noised_controls[2] += rnd.normal(0, 1, 1) * config.sigmaT

    return noised_controls


def add_observe_noise(rnd, laser_polar):
    noised_laser_polar = laser_polar.copy()

    if config.ADD_NOISE: 
        n_measurements = noised_laser_polar.shape[0]

        noised_laser_polar[:, 0] += rnd.normal(0, 1, n_measurements)*config.sigmaR
        noised_laser_polar[:, 1] += rnd.normal(0, 1, n_measurements)*config.sigmaB

    return noised_laser_polar
