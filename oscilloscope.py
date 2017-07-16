"""Shaker_Startup_Tektronix_1012B_Printout"""
from __future__ import division

import numpy as np

import helpy

def try_number(v):
    """convert string to float or int, or leave as string"""
    num = set('-0123456789')
    vset = set(v)
    if num > vset:
        return int(v)
    elif num | set('.eE') > vset:
        return float(v)
    else:
        return v


class Oscilloscope(object):
    """A class to load and analyze readouts from a Tektronix 1012B data file"""

    # some constants related to data format
    CHANNEL_WIDTH = 6
    META_COLUMNS = slice(0, 3)
    DATA_COLUMNS = (3, 4)
    DELIMITER = '\t'
    NEWLINES = '\r\n'

    @classmethod
    def parse(cls, datafile):
        """load and parse a data file from a Tektronix 1012B Oscilloscope"""
        with open(datafile, 'r') as f:
            lines = [l.rstrip(cls.NEWLINES).split(cls.DELIMITER)
                     for l in f.readlines()]
        if lines[0][0] != 'Record Length' or int(lines[0][1]) != len(lines):
            raise ValueError('Oscilloscope.parse: bad format in ' + datafile)
        columns = zip(*lines)
        nchannels = len(columns) // cls.CHANNEL_WIDTH
        columns = [columns[c*cls.CHANNEL_WIDTH:(c+1)*cls.CHANNEL_WIDTH]
                   for c in xrange(nchannels)]

        # meta is tuple (with len nchannels) of dicts
        meta = tuple({k: (try_number(v), u) if u else try_number(v)
                      for k, v, u in filter(''.join, zip(*c[cls.META_COLUMNS]))}
                     for c in columns)

        #data is array with shape (nchannels, ndim=2, len(lines))
        data = np.array([[columns[c][d] for d in cls.DATA_COLUMNS]
                         for c in xrange(nchannels)], float)

        return nchannels, data, meta


    def __init__(self, datafile, channel=slice(None)):
        """load datafile and create oscilloscope instance"""
        self.datafile = datafile
        self.nchannels, datas, metas = self.parse(self.datafile)

        if isinstance(channel, int):
            self.channel = channel
            self.nchannels = 1
        elif self.nchannels == 1:
            self.channel = 0

        if self.nchannels == 1:
            self.meta = metas[self.channel]
        else:
            self.meta = helpy.transpose_list_of_dicts(
                metas, missing=None, collapse=True)

        if self.nchannels == 1:
            self.t, self.v = datas[self.channel]
        else:
            self.t = datas[channel, 0]
            if np.allclose(self.t[0], self.t[1:],
                           atol=self.meta['Sample Interval'][0]/2):
                self.t = self.t[0]
            self.v = datas[:, 1]

        self._check_clipping()


    def _check_clipping(self, threshold=5.0):
        bounds = self.v.T.min(0), self.v.T.max(0)
        offset = np.array(self.meta['Vertical Offset'])
        scale = np.array(self.meta['Vertical Scale'])
        bounds = np.abs(bounds - offset)/scale
        if np.any(bounds >= threshold):
            msg = 'Oscilloscope data may be clipped in file {}\n{}'.format
            raise ValueError(msg(self.datafile, bounds))


    def plot(self, ax, channel=None, **kwargs):
        """plot the channels, kwargs passed to matplotlib"""
        lines = ax.plot(self.t, self.v[channel].T, **kwargs)

