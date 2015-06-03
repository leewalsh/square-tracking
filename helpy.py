#!/usr/bin/env python
# encoding: utf-8

from itertools import izip
import numpy as np

def splitter(data, frame=None, method=None, ret_dict=False):
    """ Splits a dataset into subarrays with unique frame value
        `data` : the dataset (will be split along first axis)
        `frame`: the values to group by. Uses `data['f']` if `None`
        `method`: 'diff' or 'unique'
            diff is faster*, but
            unique returns the `frame` value

        returns a list of subarrays of `data` split by unique values of `frame`
        if `method` is 'unique', returns tuples of (f, sect)
        if `ret_dict` is True, returns a dict of the form { f : section } is desired
        *if method is 'diff', assumes frame is sorted and not missing values

        examples:

            for f, fdata in splitter(data, method='unique'):
                do stuff

            for fdata in splitter(data):
                do stuff

            fsets = splitter(data, method='unique', ret_dict=True)
            fset = fsets[f]
    """
    if frame is None:
        frame = data['f']
    if method is None:
        method = 'unique' if ret_dict else 'diff'
    if method.lower().startswith('d'):
        sects = np.split(data, np.diff(frame).nonzero()[0] + 1)
        if ret_dict:
            return dict(enumerate(sects))
        else:
            return sects
    elif method.lower().startswith('u'):
        u, i = np.unique(frame, return_index=True)
        sects = np.split(data, i[1:])
        if ret_dict:
            return dict(izip(u, sects))
        else:
            return izip(u, sects)

def load_MSD(fullprefix, pos=True, ang=True):
    """ Given `fullprefix`
        Returns `msds`, `msdids`, `msads`, `msadids`, `dtau`, `dt0`
    """
    ret = ()
    if pos:
        msdnpz = np.load(fullprefix+'_MSD.npz')
        ret += msdnpz['msds'], msdnpz['msdids']
        dtau = msdnpz['dtau'][()]
        dt0 = msdnpz['dt0'][()]
    if ang:
        msadnpz = np.load(fullprefix+'_MSAD.npz')
        ret += msadnpz['msds'], msadnpz['msdids']
        if pos:
            assert dtau == msadnpz['dtau'][()]\
                and dt0 == msadnpz['dt0'][()],\
                   'dt mismatch'
        else:
            dtau = msadnpz['dtau'][()]
            dt0 = msadnpz['dt0'][()]
    ret += dtau, dt0
    print 'loading MSDs for', fullprefix
    return ret

def load_data(fullprefix, ret_odata=True, ret_cdata=False):
    """ Given `fullprefix`
        returns: `data`, `trackids`,[ `odata`,] `omask`,[ `cdata`]
    """
    ret = ()

    datanpz = np.load(fullprefix+'_TRACKS.npz')
    ret += datanpz['data'], datanpz['trackids']

    odatanpz = np.load(fullprefix+'_ORIENTATION.npz')
    if ret_odata:
        ret += (odatanpz['odata'],)
    ret += (odatanpz['omask'],)

    if ret_cdata:
        cdatanpz = np.load(fullprefix+'_CORNER_POSITIONS.npz')
        ret += (cdatanpz['data'],)

    print 'loaded data for', fullprefix
    return ret

def bool_input(question):
    "Returns True or False from yes/no user-input question"
    answer = raw_input(question)
    return answer.lower().startswith('y') or answer.lower().startswith('t')

def farange(start,stop,factor):
    start_power = np.log(start)/np.log(factor)
    stop_power = np.log(stop)/np.log(factor)
    return factor**np.arange(start_power,stop_power, dtype=type(factor))

# Pixel-Physical Unit Conversions
# Physical measurements
R_inch = 4.0           # as machined
R_mm   = R_inch * 25.4
S_measured = np.array([4,3,6,7,9,1,9,0,0,4,7,5,3,6,2,6,0,8,8,4,3,4,0,-1,0,1,7,7,5,7])*1e-4 + .309
S_inch = S_measured.mean()
S_mm = S_inch * 25.4
R_S = R_inch / S_inch

# Still (D5000)
R_slr = 2459 / 2
S_slr_m = np.array([3.72, 2.28, 4.34, 3.94, 2.84, 4.23, 4.87, 4.73, 3.77]) + 90 # don't use this, just use R_slr/R_S

# Video (Phantom)
R_vid = 585.5 / 2
S_vid_m = 22 #ish


# What we'll use:
R = R_S         # radius in particle units
S_vid = R_vid/R # particle in video pixels
S_slr = R_slr/R # particle in still pixels
A_slr = S_slr**2 # particle area in still pixels
A_vid = S_vid**2 # particle area in still pixels

pi = np.pi
# N = max number of particles (πR^2)/S^2 where S = 1
Nb = lambda margin: pi * (R - margin)**2
N = Nb(0)

bobby = 10
