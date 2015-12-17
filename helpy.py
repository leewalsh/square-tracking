#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

from itertools import izip
from math import log
import sys, os, ntpath
from time import strftime

import numpy as np

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

            for trackid, trackset in splitter(data, data['t'], noncontiguous=True)
            tracksets = splitter(data, data['t'], noncontiguous=True, ret_dict=True)
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
        return new 2d array with shorter lists padded with fill value
    """
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

def str_union(a, b):
    if a==b:
        return a
    elif len(a)==len(b):
        l = [ac if ac==bc else '?' for ac, bc in izip(a, b)]
        return ''.join(l)
    else:
        raise ValueError, "can only compare equal length strings"
        i = 1
        while a.startswith(b[:1]):
            i += 1
        for i in itertools.count(1):
            a.startswith(b[:i])

def eval_string(s, hashable=False):
    s = s.strip()
    first = s[0]
    if '\\' in s:
        s = ntpath.normpath(s)
    if first==s[-1] and first in '\'\"':
        return s[1:-1]
    nums = '-0123456789'
    issub = set(s).issubset
    if issub(set(nums + 'L')):
        return int(s)
    if issub(set(nums + '.+eE')) and not hashable:
        return float(s)
    return s

def load_meta(prefix):
    suffix = '_META.txt'
    path = prefix if prefix.endswith(suffix) else prefix+suffix
    with open(path, 'r') as f:
        lines = f.readlines()
    return dict((eval_string(i) for i in l.split(':', 1)) for l in lines)

def save_meta(prefix, meta_dict=None, **meta_kw):
    try:
        meta = load_meta(prefix)
    except IOError:
        meta = {}
    if meta_dict:
        meta.update(meta_dict)
    meta.update(meta_kw)
    lines = map('{0[0]!r:18}: {0[1]!r}\n'.format, meta.iteritems())
    suffix = '_META.txt'
    path = prefix if prefix.endswith(suffix) else prefix+suffix
    with open(path, 'w') as f:
        f.writelines(lines)

def save_log_entry(prefix, entries, mode='a'):
    timestamp = strftime('%Y-%m-%d %H:%M:%S %Z ')
    suffix = '_LOG.txt'
    path = prefix if prefix.endswith(suffix) else prefix+suffix
    if entries=='argv':
        entries = [' '.join(sys.argv)]
    elif isinstance(entries, basestring):
        entries = filter(None, entries.split('\n'))
    entries = [timestamp + (e if e.endswith('\n') else e+'\n') for e in entries]
    with open(path, mode) as f:
        f.writelines(entries)

def load_data(fullprefix, choices='tracks orientation', verbose=False):
    """ Load data from an npz file

        Given `fullprefix`, returns data arrays from a choice of:
            tracks, orientation, position, corner
    """
    choices = [c[0].lower() for c in choices.replace(',',' ').split()]

    name = {'t': 'tracks', 'o': 'orientation',
            'p': 'positions', 'c': 'corner_positions'}

    npzs = {}
    data = {}
    for c in choices:
        datapath = fullprefix+'_'+name[c].upper()+'.npz'
        try:
            npzs[c] = np.load(datapath)
        except IOError as e:
            print e
            print ("Found no {} npz file. Please run ".format(name[c]) +
                ("`tracks -l` to convert a {0}.txt to {0}.npz".format(name[c].upper()) +
                 "file, or rerun `positions` on your tiffs" if c in 'pc' else
                 "`tracks -{}` to generate a {}.npz file".format(c, name[c].upper())))
            raise
        else:
            if verbose:
                print "Loaded {} data from {}".format(name[c], datapath)
            data[c] = npzs[c][c*(c=='o')+'data']
    if 't' in choices and 'trackids' in npzs['t'].files:
        # separate trackids means this is an old-style TRACKS.npz, convert it:
        orient = npzs['o']['odata']['orient'] if 'o' in npzs else np.nan
        data['t'] = initialize_tdata(data['t'], npzs['t']['trackids'], orient)
    ret = [data[c] for c in choices]
    return ret if len(ret)>1 else ret[0]

def load_MSD(fullprefix, pos=True, ang=True):
    """ Loads ms(a)ds from an MS(A)D.npz file

        Given `fullprefix`, and choice of position and angular

        Returns [`msds`, `msdids`,] [`msads`, `msadids`,] `dtau`, `dt0`
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

def load_tracksets(data, trackids=None, min_length=10, verbose=False,
        run_remove_dupes=False, run_fill_gaps=False, run_track_orient=False):
    """ Returns a dict of slices into data based on trackid
    """
    if trackids is None:
        # copy actually speeds it up by a factor of two
        trackids = data['t'].copy()
    elif not trackids.flags.owndata:
        # copy in case called as ...(data, data['t'])
        trackids = trackids.copy()
    lengths = np.bincount(trackids+1)[1:]
    if min_length > 1:
        lengths = lengths >= min_length
    longtracks = np.where(lengths)[0]
    tracksets = {track: data[trackids==track] for track in longtracks}
    if run_remove_dupes:
        from tracks import remove_duplicates
        remove_duplicates(tracksets=tracksets, inplace=True, verbose=verbose)
    if run_fill_gaps:
        from tracks import fill_gaps
        fill_gaps(tracksets=tracksets, inplace=True, verbose=verbose)
    if run_track_orient:
        from orientation import track_orient
        for track in tracksets:
            tracksets[track]['o'] = track_orient(tracksets[track]['o'])
    return tracksets

def loadall(fullprefix, ret_msd=True, ret_fsets=False):
    """ returns data, tracksets, odata, otracksets,
         + (msds, msdids, msads, msadids, dtau, dt0) if ret_msd
    """
    print "warning, this may not return what you want"
    data = load_data(fullprefix, 'tracks')
    tracksets = load_tracksets(data, odata, omask)
    ret = data, tracksets, odata, otracksets
    if ret_msd:
        ret += load_MSD(fullprefix, True, True)
    if ret_fsets:
        fsets = splitter(data, ret_dict=True)
        fosets = splitter(odata[omask], data['f'][omask], ret_dict=True)
        ret += (fsets, fosets)
    return ret

def fields_view(arr, fields):
    """ by [HYRY](https://stackoverflow.com/users/772649/hyry)
        at http://stackoverflow.com/a/21819324/"""
    dtype2 = np.dtype({name:arr.dtype.fields[name] for name in fields})
    return np.ndarray(arr.shape, dtype2, arr, 0, arr.strides)

def quick_field_view(arr, field, careful=True):
    dt, off = arr.dtype.fields[field]
    out = np.ndarray(arr.shape, dt, arr, off, arr.strides)
    if careful:
        i = arr.dtype.names.index(field)
        a, o = arr.item(-1)[i], out[-1]
        if dt.shape is ():
            assert a==o or np.isnan(a) and np.isnan(o)
        else:
            eq = a==o
            assert np.all(eq) or np.all(eq[np.isfinite(a*o)])
    return out

def consecutive_fields_view(arr, fields, careful=True):
    shape, j = arr.shape, len(fields)
    df = arr.dtype.fields
    dt, offset = df[fields[0]]
    strides = arr.strides
    if j>1:
        shape += (j,)
        strides += (dt.itemsize,)
    out = np.ndarray(shape, dt, arr, offset, strides)
    if careful:
        names = arr.dtype.names
        i = names.index(fields[0])
        assert tuple(fields)==names[i:i+j], 'fields not consecutive'
        assert all([df[f][0]==dt for f in fields[1:]]), 'fields not same type'
        l, r = arr.item(-1)[i:i+j], out[-1]
        assert all([a==o or np.isnan(a) and np.isnan(o) for a,o in izip(l,r)]),\
            "last row mismatch\narr: {}\nout: {}".format(l, r)
    return out

track_dtype = np.dtype({'names': 'id f  t  x  y  o'.split(),
                      'formats': 'u4 u2 i4 f4 f4 f4'.split()})
pos_dtype = np.dtype({  'names': 'f  x  y  lab ecc area id'.split(),
                      'formats': 'i4 f8 f8 i4  f8  i4   i4'.split()})

def initialize_tdata(pdata, trackids=-1, orientations=np.nan):
    if pdata.dtype==track_dtype:
        data = pdata
    else:
        data = np.empty(len(pdata), dtype=track_dtype)
        for field in pdata.dtype.names:
            if field in track_dtype.names:
                data[field] = pdata[field]
    if trackids is not None:
        data['t'] = trackids
    if orientations is not None:
        data['o'] = orientations
    return data

def dtype_info(dtype='all'):
    if dtype=='all':
        [dtype_info(s+b) for s in 'ui' for b in '1248']
        return
    dt = np.dtype(dtype)
    bits = 8*dt.itemsize
    if dt.kind=='f':
        print np.finfo(dt)
        return
    if dt.kind=='u':
        mn = 0
        mx = 2**bits - 1
    elif dt.kind=='i':
        mn = -2**(bits-1)
        mx = 2**(bits-1) - 1
    print "{:6} ({}{}) min: {:20}, max: {:20}".format(
                dt.name, dt.kind, dt.itemsize, mn, mx)

def txt_to_npz(datapath, verbose=False, compress=True):
    """ Reads raw txt positions data into a numpy array and saves to an npz file

        `datapath` is the path to the output file from finding particles
        it must end with "results.txt" or "POSITIONS.txt", depending on its
        source, and its structure is assumed to match a certain pattern
    """
    if not os.path.exists(datapath):
        if os.path.exists(datapath+'.gz'):
            datapath += '.gz'
        elif os.path.exists(datapath[:-3]):
            datapath = datapath[:-3]
        else:
            raise IOError, 'File {} not found'.format(datapath)
    if verbose:
        print "loading positions data from", datapath,
    if datapath.endswith('results.txt'):
        shapeinfo = False
        # imagej output (called *_results.txt)
        dtargs = {  'usecols' : [0,2,3,5],
                    'names'   : "id,x,y,f",
                    'dtype'   : [int,float,float,int]} \
            if not shapeinfo else \
                 {  'usecols' : [0,1,2,3,4,5,6],
                    'names'   : "id,area,mean,x,y,circ,f",
                    'dtype'   : [int,float,float,float,float,float,int]}
        data = np.genfromtxt(datapath, skip_header=1, **dtargs)
        data['id'] -= 1 # data from imagej is 1-indexed
    elif 'POSITIONS.txt' in datapath:
        # positions.py output (called *_POSITIONS.txt[.gz])
        from numpy.lib.recfunctions import append_fields
        # successfully tested the following reduced datatype sizes on
        # Squares/diffusion/orientational/n464_CORNER_POSITIONS.npz
        # with no real loss in float precision nor cutoff in ints
        # names:   'f  x  y  lab ecc area id'
        # formats: 'u2 f4 f4 i4  f2  u2   u4'
        data = np.genfromtxt(datapath, skip_header=1,
                             names="f,x,y,lab,ecc,area",
                             dtype="u2,f4,f4,i4,f2,u2")
        ids = np.arange(len(data), dtype='u4')
        data = append_fields(data, 'id', ids, usemask=False)
    else:
        raise ValueError, ("is {} from imagej or positions.py?".format(datapath.rsplit('/')[-1]) +
                "Please rename it to end with _results.txt[.gz] or _POSITIONS.txt[.gz]")
    if verbose: print '...',
    outpath = datapath[:datapath.rfind('txt')] + 'npz'
    (np.savez, np.savez_compressed)[compress](outpath, data=data)
    if verbose: print 'and saved to', outpath
    return

def compress_existing_npz(path, overwrite=False, test=False):
    orig = np.load(path)
    arrs = {n: orig[n] for n in orig.files}
    out = path[:-4]+'compressed'*(not overwrite)
    np.savez_compressed(out, **arrs)
    if test:
        comp = np.load(out+'.npz')
        assert comp.files==orig.files
        assert all([orig[n]==comp[n] for n in comp.files])
        print 'Success!'

def merge_data(data, savename=None, do_orient=True):
    """ returns (and optionally saves) new `data` array merged from list or
        tuple of individual `data` arrays or path prefixes.

        parameters
        ----------
        data : list of arrays, prefixes, or single prefix with wildcards
        savename : path prefix at which to save the merged data,
            saved as "<savename>_MERGED_<TRACKS|ORIENTATION>.npz"
        do_orient : True or False, whether to merge the orientation data as
            well. default is True

        returns
        -------
        data : always returned, the main merged data array
        trackids : returned if paths are given
        odata : returned if do_orient
        omask : returned if do_orient

        if orientational data is to be merged, then a list of filenames or
        prefixes must be given instead of data objects.

        only data is returned if array objects are given.
    """
    if isinstance(data, str):
        if '*' in data or '?' in data:
            from glob import glob
            suf = '_TRACKS.npz'
            data = [ s[:-len(suf)] for s in glob(data+suf) ]
        elif data.endsith('.npz'):
            raise ValueError, "please give only the prefix"
        else:
            raise ValueError, "only one file given"

    if isinstance(data[0], str):
        if data[0].endswith('.npz'):
            raise ValueError, "please only give the prefix"
        data = zip(*map(lambda d: load_data(d, ret_odata=do_orient), data))
    else:
        data = (data,)

    track_increment = 0
    for i, datum in enumerate(data[0]):
        goodtracks = datum['t'] >= 0
        datum['t'][goodtracks] += track_increment
        if len(data) > 1:
            data[1][i][goodtracks] += track_increment # trackids
        track_increment = datum['t'].max() + 1

    merged = map(np.concatenate, data)

    if savename:
        fulldir = os.path.abspath(os.path.dirname(savename))
        if not os.path.exists(fulldir):
            print "Creating new directory", fulldir
            os.makedirs(fulldir)
        np.savez_compressed(savename+'_MERGED_TRACKS.npz',
                            **dict(zip(['data', 'trackids'], merged)))
        if do_orient and len(data) > 2:
            np.savez_compressed(savename+'_MERGED_ORIENTATION.npz',
                                odata=merged[2], omask=merged[3])
    return merged

def bool_input(question, default=None):
    "Returns True or False from yes/no user-input question"
    if question and question[-1] not in ' \n\t':
        question += ' '
    answer = raw_input(question).strip().lower()
    if answer=='':
        if default is None:
            return bool_input(question, default)
        else:
            return default
    return answer.startswith('y') or answer.startswith('t')

def farange(start, stop, factor):
    start_power = log(start, factor)
    stop_power = log(stop, factor)
    dt = np.result_type(start, stop, factor)
    return factor**np.arange(start_power, stop_power, dtype=dt)

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
    r = ((xo - x1)**2 + (yo - y1)**2)**0.5

    return xo, yo, r

def find_first_frame(paths, ext='.tif', err=None, load=False):
    from glob import glob
    join = lambda p: os.path.join(*p)
    patterns = [join(paths)+"*", join(paths)+"/*"]
    paths.insert(-1, '..')
    patterns.extend([join(paths)+"*", join(paths)+"/*"])
    for pattern in patterns:
        bgimage = glob(pattern+ext)
        if bgimage:
            bgimage = bgimage[0]
            break
    else:
        if isinstance(err, basestring):
            bgimage = raw_input(err or 'Please give path to tif:\n')
        else:
            return err
    if load:
        from scipy.ndimage import imread
        return imread(bgimage)
    else:
        return bgimage

def circle_click(im):
    """ saves points as they are clicked
        once three points have been saved, calculate the center and
        radius of the circle pass through them all. Draw it and save it.
    """
    import matplotlib
    if matplotlib.is_interactive():
        raise RuntimeError, "Cannot do circle_click when in interactive/pylab mode"
    print """
    To use:
        when image is shown, click three non-co-linear points along the
        perimeter.  neither should be vertically nor horizontally aligned
        (gives divide by zero) when three points have been clicked, a circle
        should appear. Then close the figure to allow the script to continue.
    """
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
            global xo, yo, r # can't access connector function's returned value
            xo, yo, r = circle_three_points(xs, ys)
            cpatch = matplotlib.patches.Circle([xo, yo], r,
                        linewidth=3, color='g', fill=False)
            ax.add_patch(cpatch)
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', circle_click_connector)
    plt.show()
    return xo, yo, r

def der(f, dx=None, x=None, xwidth=None, iwidth=None, order=1):
    """ Take a finite derivative of f(x) using convolution with the derivative
        of a gaussian kernel.  For any convolution:
            (f * g)' = f * g' = g * f'
        so we start with f and g', and return g and f', a smoothed derivative.

        parameters
        ----------
        f : an array to differentiate
        xwidth or iwidth : smoothing width (sigma) for gaussian.
            use iwidth for index units, (simple array index width)
            use xwidth for the physical units of x (x array is required)
            use 0 for no smoothing. Gives an array shorter by 1.
        x or dx : required for normalization
            if x is provided, dx = np.diff(x)
            otherwise, a scalar dx is presumed
            if dx=1, use a simple finite difference with np.diff
            if dx>1, convolves with the derivative of a gaussian, sigma=dx

        returns
        -------
        df_dx : the derivative of f with respect to x
    """
    nf = len(f)

    if dx is None and x is None:
        dx = 1
        nx = 1
    elif dx is None:
        nx = len(x)
        dx = x.copy()
        dx[:-1] = dx[1:] - dx[:-1]
        assert dx[:-1].min() > 1e-6, ("Non-increasing independent variable "
                                      "(min step {})".format(dx[:-1].min()))
        dx[-1] = dx[-2]
        dx **= order

    if xwidth is None and iwidth is None:
        if x is None:
            iwidth = 1
        else:
            xwidth = 1
    if iwidth is None:
        iwidth = xwidth / dx

    if iwidth == 0:
        if order == 1:
            df = f.copy()
            df[:-1] = df[1:] - df[:-1]
            df[-1] = df[-2]
        else:
            df = np.diff(f, n=order)
            beg, end = order//2, (order+1)//2
            df = np.concatenate([[df[0]]*beg, df, [df[-1]]*end])

    #elif iwidth < .5:
    #   raise ValueError("width of {} too small for reliable "
    #                    "results".format(iwidth))
    else:
        from scipy.ndimage import gaussian_filter1d
        df = gaussian_filter1d(f, iwidth, order=order)

    newnf = len(df)
    assert nf==newnf, "df was len {}, now len {}".format(nf, newnf)
    newnx = len(np.atleast_1d(dx))
    assert nx==newnx, "dx was len {}, now len {}".format(nx, newnx)
    return df/dx**order

from string import Formatter
class SciFormatter(Formatter):
    def format_field(self, value, format_spec):
        if format_spec.endswith('T'):
            s = format(value, format_spec[:-1])
            if 'e' in s:
                s = s.replace('e', r'\times10^{') + '}'
        elif format_spec.endswith('t'):
            s = format(value, format_spec[:-1]).replace('e', '*10**')
        else:
            s = format(value, format_spec)
        return s

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
