"""Shaker_Startup_Tektronix_1012B_Printout"""
from __future__ import division

import numpy as np

class Oscilloscope(object):
    """A class to load and analyze readouts from a Tektronix 1012B data file"""

    # some constants related to data format
    channel_width = 6
    meta_columns = slice(0, 3)
    data_columns = (3, 4)
    delimiter = '\t'
    newlines = '\r\n'

    @classmethod
    def parse(cls, datafile):
        with open('100Hz_100mV.txt', 'r') as f:
            lines = [l.rstrip(cls.newlines).split(cls.delimiter)
                     for l in f.readlines()]
        columns = zip(*lines)
        nchannels = len(columns) // cls.channel_width
        columns = [columns[i:i+cls.channel_width] for i in xrange(nchannels)]

        usecols = [col + ch*cls.channel_width
                   for ch in xrange(nchannels) for col in cls.data_columns]
        data = np.genfromtxt(datafile, usecols=usecols, delimiter=cls.delimiter)
        data.reshape(-1, nchannels, 2).T

    def __init__(self, datafile):
        self.datafile = datafile
        self.data = self.parse(self.datafile)
