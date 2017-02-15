from __future__ import division

import numpy as np
from math import sqrt
import helpy

class cached_property(object):
    """
    A property that is only computed once per instance and then replaces itself
    with an ordinary attribute. Deleting the attribute resets the property.
    Source: github.com/pydanny/cached-property/blob/1.3.0/cached_property.py
    """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class SimTrack(object):
    """A simulated particle track"""

    def __init__(self, DR, DT, v0, w0=0, size=100, fps=1):
        self.DR = DR
        self.DT = DT
        self.v0 = v0
        self.w0 = w0
        self.size = size
        self.fps = fps

    @cached_property
    def eta(self):
        """eta, translational noise (lab frame)"""
        return helpy.rotate(self.v, self.n, angle=True)


    @cached_property
    def v(self):
        """v, translational velocity (body frame)"""
        return np.random.normal(loc=[[self.v0], [0]],
                                scale=sqrt(2*self.DT),
                                size=(2, self.size))


    @cached_property
    def xi(self):
        """xi, rotational noise"""
        return np.random.normal(loc=self.w0,
                                scale=sqrt(2*self.DR),
                                size=self.size)


    @cached_property
    def o(self):
        """o, orientation of particle"""
        return np.cumsum(self.xi, axis=-1)


    @cached_property
    def n(self):
        """n, normal vector of particle"""
        cos = np.cos(self.o)
        sin = np.sin(self.o)
        return np.array([cos, sin])


    @cached_property
    def r(self):
        """r, position of particle (lab frame)"""
        return np.cumsum(self.eta, axis=-1)


    @cached_property
    def x(self):
        """x, x-position of particle (lab frame)"""
        return self.r[0]


    @cached_property
    def y(self):
        """y, y-position of particle (lab frame)"""
        return self.r[1]

