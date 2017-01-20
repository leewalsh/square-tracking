from __future__ import division

import numpy as np

def make_trajectory(DR, DT, v0, size=100):
    theta = make_theta(DR, size=size)
    normal = make_normal(theta)
    dbody = np.random.normal(loc=(v0, 0),
                             scale=sqrt(2*DT),
                             size=normal.shape)


def body_to_lab(body, theta=None, normal=None):
    if normal is None:
        normal = make_normal(theta)


def make_theta(DR=None, std=None, size=100, w0=0):
    if std is None:
        if DR is None:
            raise ValueError("DR or std required")
        std = sqrt(2*DR)
    dtheta = np.random.normal(loc=w0, scale=std, size=size)
    return np.cumsum(dtheta, axis=-1)

def make_normal(theta):
    return np.stack([np.cos(theta), np.sin(theta)], axis=0)

