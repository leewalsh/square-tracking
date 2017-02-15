from __future__ import division

import numpy as np
from math import sqrt
import helpy


class SimTrack(object):
    """A simulated particle track"""

    def __init__(self, DR, DT, v0, w0=0, size=100, fps=1):
        self.DR = DR
        self.DT = DT
        self.v0 = v0
        self.w0 = w0
        self.size = size
        self.fps = fps

        self._eta, self._v, self._xi = None, None, None
        self._theta, self._n = None, None
        self._r, self._x, self._y = None, None, None


    @property
    def eta(self):
        """eta, translational noise (lab frame)"""
        if self._eta is None:
            self._eta = helpy.rotate(self.v, self.n)
        return self._eta


    @property
    def v(self):
        """v, translational velocity (body frame)"""
        if self._v is None:
            self._v = np.random.normal(loc=[[self.v0], [0]],
                                       scale=sqrt(2*self.DT),
                                       size=(2, self.size))
        return self._v


    @property
    def xi(self):
        """xi, rotational noise"""
        if self._xi is None:
            self._xi = np.random.normal(loc=self.w0,
                                        scale=sqrt(2*self.DR),
                                        size=self.size)
        return self._xi


    @property
    def theta(self):
        """theta, orientation of particle"""
        if self._theta is None:
            self._theta = np.cumsum(self.xi, axis=-1)
        return self._theta


    @property
    def n(self):
        """n, normal vector of particle"""
        if self._n is None:
            cos = np.cos(self.theta)
            sin = np.sin(self.theta)
            self._n = np.array([cos, sin])
        return self._n


    @property
    def r(self):
        """r, position of particle (lab frame)"""
        if self._r is None:
            self._r = np.cumsum(self.eta, axis=-1)
        return self._r


    @property
    def x(self):
        """x, x-position of particle (lab frame)"""
        if self._x is None:
            self._x, self._y = self.r
        return self._x


    @property
    def y(self):
        """y, y-position of particle (lab frame)"""
        if self._y is None:
            self._x, self._y = self.r
        return self._y



