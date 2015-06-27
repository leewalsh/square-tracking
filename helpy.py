#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
from itertools import izip

import numpy as np

import orientation as orient

def splitter(data, frame=None, method=None, ret_dict=False, noncontiguous=False):
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

            for trackid, trackset in splitter(data, data['lab'], noncontiguous=True)
            tracksets = splitter(data, data['lab'], noncontiguous=True, ret_dict=True)
            trackset = tracksets[trackid]
    """
    if frame is None:
        frame = data['f']
    if method is None:
        method = 'unique' if ret_dict or noncontiguous else 'diff'
    if method.lower().startswith('d'):
        sects = np.split(data, np.diff(frame).nonzero()[0] + 1)
        if ret_dict:
            return dict(enumerate(sects))
        else:
            return sects
    elif method.lower().startswith('u'):
        u, i = np.unique(frame, return_index=True)
        if noncontiguous:
            # no nicer way to do this:
            sects = [ data[np.where(frame==fi)] for fi in u ]
        else:
            sects = np.split(data, i[1:])
        if ret_dict:
            return dict(izip(u, sects))
        else:
            return izip(u, sects)

def pad_uneven(lst, fill=0, return_mask=False, dtype=None):
    """ take uneven list of lists
        return new 2d array with shorter lists padded with fill value"""
    if dtype is None:
        dtype = np.result_type(fill, lst[0][0])
    shape = len(lst), max(map(len, lst))
    result = np.zeros(shape, dtype) if fill==0 else np.full(shape, fill, dtype)
    if return_mask:
        mask = np.zeros(shape, bool)
    for i, row in enumerate(lst):
        result[i, :len(row)] = row
        if return_mask:
            mask[i, :len(row)] = True
    return (result, mask) if return_mask else result

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

    #print 'loaded data for', fullprefix
    return ret

def circle_three_points(*xs):
    """ With three points, calculate circle
        e.g., see paulbourke.net/geometry/circlesphere
        returns center, radius as (xo, yo), r
    """
    xs = np.squeeze(xs)
    if xs.shape == (3, 2): xs = xs.T
    (x1, x2, x3), (y1, y2, y3) = xs

    ma = (y2-y1)/(x2-x1)
    mb = (y3-y2)/(x3-x2)
    xo = ma*mb*(y1-y3) + mb*(x1+x2) - ma*(x2+x3)
    xo /= 2*(mb-ma)
    yo = (y1+y2)/2 - (xo - (x1+x2)/2)/ma
    c = np.array([xo, yo])         # center
    r = np.hypot(*(c - [x1, y1]))  # radius

    return c, r

def circle_click(im):
    """ saves points as they are clicked
        once three points have been saved, calculate the center and
        radius of the circle pass through them all. Draw it and save it.
    """

    print """
    To use:
        when image is shown, click three non-co-linear points along the
        perimeter.  neither should be vertically nor horizontally aligned
        (gives divide by zero) when three points have been clicked, a circle
        should appear. Then close the figure to allow the script to continue.
    """

    import matplotlib
    from matplotlib import pyplot as plt
    if isinstance(im, str):
        im = plt.imread(im)
    xs = []
    ys = []
    if False:   # Using Qt and skimage.ImageViewer
        fig = ImageViewer(im)
        ax = fig.canvas.figure.add_subplot(111)
    else:       # Using matplotlib
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(im)

    def circle_click_connector(click):
        #print 'you clicked', click.xdata, '\b,', click.ydata
        xs.append(click.xdata)
        ys.append(click.ydata)
        if len(xs) == 3:
            # With three points, calculate circle
            print 'got three points'
            global c, r # can't access connector function's returned value
            c, r = circle_three_points(xs, ys)
            cpatch = matplotlib.patches.Circle(c, radius=r, color='g', fill=False)
            ax.add_patch(cpatch)
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', circle_click_connector)
    plt.show()
    return c, r

def load_tracksets(data, trackids, odata=None, omask=True, min_length=10):
    """ Returns a dict of slices into data based on trackid
    """
    if omask is not True:
        trackids = trackids[omask]
    longtracks = np.where(np.bincount(trackids+1)[1:] >= min_length)[0]
    tracksets  = { track: data[(data['lab']==track) & omask]
                   for track in longtracks }
    if odata is not None:
        otracksets = { track: orient.track_orient(
               odata[(data['lab']==track) & omask]['orient'], onetrack=True)
                   for track in longtracks }
        return tracksets, otracksets
    else:
        return tracksets

def loadall(fullprefix, ret_msd=True):
    """ returns data, tracksets, odata, otracksets,
         + (msds, msdids, msads, msadids, dtau, dt0) if ret_msd
    """
    data, trackids, odata, omask = \
            load_data(fullprefix, ret_odata=True, ret_cdata=False)
    fsets = splitter(data, ret_dict=True)
    fosets = splitter(odata[omask], data['f'], ret_dict=True)
    tracksets, otracksets = load_tracksets(data, trackids, odata, omask)
    if ret_msd:
        return (data, tracksets, odata, otracksets) + load_MSD(fullprefix, True, True)
    else:
        return data, tracksets, odata, otracksets

def bool_input(question):
    "Returns True or False from yes/no user-input question"
    answer = raw_input(question)
    return answer.lower().startswith('y') or answer.lower().startswith('t')

def farange(start,stop,factor):
    start_power = np.log(start)/np.log(factor)
    stop_power = np.log(stop)/np.log(factor)
    return factor**np.arange(start_power,stop_power, dtype=type(factor))

def loglog_slope(x, y, smooth=0):
    dx = 0.5*(x[1:] + x[:-1])
    dy = np.diff(np.log(y)) / np.diff(np.log(x))

    if smooth:
        from scipy.ndimage import gaussian_filter1d
        dy = gaussian_filter1d(dy, smooth, mode='reflect')
    return dx, dy

def dist(a, b):
    """ The 2d distance between two arrays of shape (N, 2) or just (2,)
    """
    return np.hypot(*(a - b).T)

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
