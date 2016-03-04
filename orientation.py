# coding: utf-8

from __future__ import division

import itertools as it
from math import sqrt
import numpy as np
from PIL import Image as Im

import helpy
import correlation as corr

if __name__=='__main__':
    HOST = helpy.gethost()
    if HOST=='foppl':
        locdir = '/home/lawalsh/Granular/Squares/orientation/'
    elif HOST=='rock':
        locdir = '/Users/leewalsh/Physics/Squares/orientation/'
    else:
        print "computer not defined"
        print "where are you working?"

pi = np.pi
twopi = 2*pi

def field_rename(a, old, new):
    a.dtype.names = [ fn if fn != old else new for fn in a.dtype.names ]

def get_fft(ifile=None,location=None):
    """ get the fft of an image
        FFT information from:
        http://stackoverflow.com/questions/2652415/fft-and-array-to-image-image-to-array-conversion
    """
    if ifile is None:
        ifile= "n20_bw_dots/n20_b_w_dots_0010.tif"
    if location is None:
        x = 424.66; y = 155.56; area = 125
        #x = 302.95; y = 221.87; area = 145
        #x = 386.27; y = 263.62; area = 141
        #x = 35.39; y = 305.92; area = 154
        location = x,y
    wdth = int(24 * sqrt(2))
    hght = wdth
    cropbox = map(int,(x - wdth/2., y - hght/2.,\
            x + wdth/2., y + hght/2.))
    i = Im.open(ifile)
    i = i.crop(cropbox)
    i.show()
    a = np.asarray(i)
    aa = np.gradient(a)
    b = np.fft.fft2(a)
    j = Im.fromarray(abs(b))
    ii = Im.fromarray(aa)
    return b

def get_orientation(b):
    """ get orientation from `b`, the fft of an image """
    p = []
    for (mi,m) in enumerate(b):
        for (ni, n) in enumerate(m):
            ang = np.arctan2(mi - hght/2, ni - wdth/2)
            p.append([ang,abs(n)])
    p = np.asarray(p)
    p[:,0] = p[:,0] + pi
    slices = 45
    slicewidth = 2*pi/slices
    s = []
    for sl in range(slices):
        si = np.nonzero(abs(p[:,0] - sl*slicewidth) < slicewidth)
        sm = np.average(p[si,1])
        s.append([sl*slicewidth,sm])
    s = np.asarray(s)
    return s, p

def find_corner(particle, corners, nc, rc, drc=0, ang=None, dang=None,
                rank_by='rc', tree=None, do_average=True):
    """find the corner dot(s) closest to distance rc from center dot

    Parameters
    ----------
    particle:   is particle position as [x,y] array of shape (2,)
    corners:    is shape (N, 2) array of positions of corner dots as x, y pairs
    tree:       a KDTree of the `corners`, if available
    nc:         number of corner dots
    rc:         is the expected distance to corner from particle position
    drc:        delta rc is the tolerance on rc, defaults to sqrt(rc)
    ang:        angular separation between corners (if nc > 1)
    dang:       tolerance for ang (if None, ang is ignored if nfound == nc, but
                is uses to choose best nc of nfound if nfound > nc)
    do_average: whether to average the nc corners to one value for return

    Returns
    -------
    pcorner:    (mean) position(s) (x,y) of corner that belongs to particle
    porient:    (mean) particle orientation(s) (% 2pi)
    cdisp:      (mean) vector(s) (x,y) from particle center to corner(s)
    """

    if drc <= 0:
        drc = sqrt(rc)

    # displacements from center to corners
    if tree:
        # if kdtree is available, only consider nearby corners
        icnear = tree.query_ball_point(particle, rc + drc)
        corners = corners[icnear]

    cdisps = corners - particle
    cdists = np.hypot(*cdisps.T)
    cdiffs = np.abs(cdists - rc)
    legal = np.where(cdiffs < drc)[0]
    nfound = len(legal)
    if nfound < nc:
        # too few, skip.
        return (None,)*3

    pcorner = corners[legal]
    cdisp = cdisps[legal]
    cdist = cdists[legal]
    cdiffs = cdiffs[legal]

    if ang and nc > 1:
        corients = np.arctan2(cdisp[:, 1], cdisp[:, 0])
        pairs = np.array(list(it.combinations(xrange(nfound), 2)))
        cangles = corr.dtheta(corients[pairs])
        dcangles = np.abs(cangles - ang)
        legal = pairs[np.where(dcangles < dang)]
        nfound = len(legal)*2
    else:
        legal = slice(None)
    if nfound < nc:
        # too few, skip.
        return (None,)*3
    elif nfound > nc and rank_by == 'ang':
        if nc == 2:
            # too many found, keep the two with the best angular separation
            ibest = dcangles.argmin()
            best = dcangles[ibest]
            if dang and best > dang:
                return (None,)*3
            legal = pairs[ibest]
        elif nc == 3:
            best = dcangles.argsort()[:2]
            if dang and np.any(dcangles[best] > dang):
                return (None,)*3
            legal = np.unique(pairs[best])
            if len(legal) > nc:
                # not trivial to pick best 3 if the two best separation angles
                # don't share a corner
                raise NotImplementedError("Ask me later")
    elif nfound > nc and rank_by == 'rc':
        # too many, keep only the nc closest to rc away
        legal = legal[np.argmin(cdiffs[legal].sum(-1))]
    else:
        legal = legal[0]

    pcorner = pcorner[legal]
    cdisp = cdisp[legal]
    cdist = cdist[legal]

    if do_average and nc > 1:
        # average the angle by finding angle of mean vector displacement
        # keep the corner displacements (as amplitude) and positions
        meandisp = (cdisp/cdist[...,None]).mean(0)
        porient = np.arctan2(*meandisp[::-1]) % twopi
    else:
        porient = np.arctan2(cdisp[...,1], cdisp[...,0]) % twopi

    return pcorner, porient, cdist

# TODO: use p.map() to find corners in parallel
# try splitting by frame first, use views for each frame
# or just pass a tuple of (datum, cdata[f==f]) to get_angle()

def get_angle((datum, cdata)):
    corner = find_corner(
                np.asarray((datum['x'],datum['y'])),
                np.column_stack((cdata['x'][cdata['f']==datum['f']],
                                 cdata['y'][cdata['f']==datum['f']])))
    dt = np.dtype([('corner',float,(2,)),('orient',float),('cdisp',float,(2,))])
    return np.array([corner], dtype=dt)

def get_angles_map(data, cdata, nthreads=None):
    """ get_angles(data, cdata, nthreads=None)

        arguments:
            data    - data array with 'x' and 'y' fields for particle centers
            cdata   - data array wity 'x' and 'y' fields for corners
            (these arrays need not have the same length,
                but both must have 'f' field for the image frame)
            nthreads - number of processing threads (cpu cores) to use
                None uses all cores of machine (8 for foppl, 2 for rock)

        returns:
            odata   - array with fields:
                'orient' for orientation of particles
                'corner' for particle corner (with 'x' and 'y' sub-fields)
            (odata has the same shape as data)
    """
    field_rename(data,'s','f')
    field_rename(cdata,'s','f')

    from multiprocessing import Pool
    if nthreads is None or nthreads > 8:
        nthreads = 8 if HOST=='foppl' else 2
    elif nthreads > 2 and HOST=='rock':
        nthreads = 2
    print "on {}, using {} threads".format(HOST,nthreads)
    pool = Pool(nthreads)

    datums = [ (datum,cdata[cdata['f']==datum['f']])
                for datum in data ]
    odatalist = pool.map(get_angle, datums)
    odata = np.vstack(odatalist)
    return odata

def get_angles_loop(pdata, cdata, pfsets, cfsets, cftrees, nc, rc, drc=None,
                    ang=None, dang=None, do_average=True, verbose=False):
    """find the orientations of particles given center and corner positions

    Parameters
    ----------
    pdata:      data array with 'x' and 'y' fields for particle centers
    cdata:      data array wity 'x' and 'y' fields for corners
        pdata and cdata arrays need not have the same length,
        but both must have 'f' field for the image frame)
    pfsets:     slices into pdata, by frame
    cfsets:     slices into cdata, by frame
    cftrees:    dict of KDTrees for corners, by frame
        the following arguments are passed to find_corner:
    nc:         number of corner dots
    rc:         expected distance between center and corner dot
    drc:        tolerance for rc, defaults to sqrt(rc)
    ang:        angular separation between corners (if nc > 1)
    dang:       tolerance for ang (if None, ang is ignored if nfound == nc, but
                is uses to choose best nc of nfound if nfound > nc)
    do_average: whether to average the nc corners to one value for return

    Returns
    -------
    odata:  structured array, same shape as pdata, with fields:
            'corner' for particle corner (with 'x' and 'y' sub-fields)
            'orient' for orientation of particles
            'cdisp' for the corner - center displacement
    """
    if do_average or nc == 1:
        dt = [('corner', float, (nc, 2)),
              ('orient', float),
              ('cdisp', float, (nc,))]
    elif nc > 1:
        dt = [('corner', float, (nc, 2,)),
              ('orient', float, (nc,)),
              ('cdisp', float, (nc,))]
    if ang > pi:
        ang = np.radians(ang)
        if dang:
            dang = np.radians(dang)
    odata = np.full(len(pdata), np.nan, dtype=dt)
    odata_corner = odata['corner']
    odata_orient = odata['orient']
    odata_cdisp = odata['cdisp']
    full_ids = pdata['id']
    id_ok = full_ids[0]==0 and np.all(np.diff(full_ids)==1)
    print_freq = len(pfsets)//(100 if verbose>1 else 5) + 1
    print 'seeking orientations'
    for f in pfsets:
        if verbose and not f % print_freq:
            print f,
        fpdata = pfsets[f]
        fcdata = cfsets[f]
        cftree = cftrees[f]
        positions = helpy.consecutive_fields_view(fpdata, 'xy')
        cpositions = helpy.consecutive_fields_view(fcdata, 'xy')
        fp_ids = helpy.quick_field_view(fpdata, 'id')
        for fp_id, posi in it.izip(fp_ids, positions):
            #TODO could probably be sped up by looping through the output of
            #     ptree.query_ball_tree(ctree)
            corner, orient, disp = \
                find_corner(posi, cpositions, nc=nc, rc=rc, drc=drc, ang=ang,
                            dang=dang, tree=cftree, do_average=do_average)
            if orient is None: continue
            full_id = fp_id if id_ok else np.searchsorted(full_ids, fp_id)
            odata_corner[full_id] = corner
            odata_orient[full_id] = orient
            odata_cdisp[full_id] = disp

    if do_average or nc == 1:
        mask = np.isfinite(odata['orient'])
    elif nc > 1:
        mask = np.all(np.isfinite(odata['orient']), axis=1)

    return odata, mask

def plot_orient_hist(odata, figtitle=''):
    if HOST!='rock':
        print 'computer must be rock'
        return False
    import matplotlib.pyplot as pl

    pl.figure()
    pl.hist(odata['orient'][np.isfinite(odata['orient'])], bins=90)
    pl.title('orientation histogram' if figtitle is '' else figtitle)
    return True

def plot_orient_quiver(data, odata, mask=None, imfile='', fps=1, savename='', figsize=None):
    """ plot_orient_quiver(data, odata, mask=None, imfile='')
    """
    import matplotlib.pyplot as pl
    import matplotlib.colors as mcolors
    import matplotlib.colorbar as mcolorbar
    pl.figure(tight_layout=False, figsize=figsize)
    if imfile is not None:
        bgimage = Im.open(extdir+prefix+'_0001.tif' if imfile is '' else imfile)
        pl.imshow(bgimage, cmap=pl.cm.gray, origin='upper')
    #pl.quiver(X, Y, U, V, **kw)
    if mask is None:
        try:
            mask = np.all(np.isfinite(odata['orient']), axis=1)
        except ValueError:
            mask = np.isfinite(odata['orient'])

    n = odata.shape[-1] if odata.ndim > 1 else 1
    ndex = np.repeat(np.arange(mask.sum()), n)
    nz = mcolors.Normalize()
    nz.autoscale(data['f'][mask]/fps)
    qq = pl.quiver(
            data['y'][mask][ndex], data['x'][mask][ndex],
            odata['cdisp'][mask][...,1].flatten(), -odata['cdisp'][mask][...,0].flatten(),
            color=pl.cm.jet(nz(data['f'][mask]/fps)),
            scale=1, scale_units='xy')
    #pl.title(', '.join(imfile.split('/')[-1].split('_')[:-1]) if imfile else '')
    cax,_ = mcolorbar.make_axes(pl.gca())
    cb = mcolorbar.ColorbarBase(cax, cmap=pl.cm.jet, norm=nz)
    cb.set_label('time '+('(s)'if fps > 1 else '(frame)'))
    if savename:
        print "saving to", savename
        pl.savefig(savename)
    pl.show()
    return qq, cb

def track_orient(orients, omask=None, cutoff=pi, inplace=True):
    """ tracks branch cut crossings for orientation data
        assumes that dtheta << cutoff for each frame
    """
    if omask is None:
        omask = np.isfinite(orients)
    if not omask.any():
        # all nan, return as is
        return orients
    if not inplace:
        orients = orients.copy()
    tracked = orients[omask]
    deltas = np.diff(tracked)
    crossings = (np.abs(deltas) > cutoff)*np.sign(deltas)
    tracked[1:] -= twopi*crossings.cumsum()
    orients[omask] = tracked
    return orients

def plot_orient_time(data, odata, tracks, omask=None, delta=False, fps=1, save='', singletracks=False):
    import matplotlib.pyplot as pl
    if omask is None:
        omask = np.isfinite(odata['orient'])
    goodtracks = np.unique(tracks[omask])
    if goodtracks[0] == -1: goodtracks = goodtracks[1:]
    if singletracks:
        if singletracks is True:
            goodtracks = list(goodtracks)[:4]
        elif isinstance(singletracks, list):
            goodtracks = singletracks
    print 'tracks used are', goodtracks
    #tmask = np.in1d(tracks, goodtracks)
    pl.figure(figsize=(6,5))
    colors = ['red','green','blue','cyan','black','magenta','yellow']
    for goodtrack in goodtracks:
        tmask = tracks == goodtrack
        fullmask = omask & tmask
        if np.count_nonzero(fullmask) < 1:
            continue
        plotrange = slice(None, 600 if singletracks is True else None)
        if delta:
            c = colors[goodtrack%7]
            pl.plot(data['f'][fullmask][plotrange]/fps,
                    odata['orient'][fullmask][plotrange],
                    c=c,label="Track {}".format(goodtrack))
            pl.plot(data['f'][fullmask][plotrange][0 if singletracks else 1:]/fps,
                    np.diff(odata['orient'][fullmask])[plotrange],
                    'o', c=c,label='delta {}'.format(goodtrack))
        else:
            pl.plot(data['f'][fullmask][plotrange]/fps,
                    track_orient(odata['orient'][fullmask])[plotrange],
                    '--', label='tracked {}'.format(goodtrack))
    if delta:
        for n in np.arange(-2 if delta else 0,2.5,0.5):
            pl.plot(np.ones_like(odata['orient'][fullmask])[plotrange]*n*pi,'k--')
    if len(goodtracks) < 10:
        pl.legend()
    #pl.title('Orientation over time')#\ninitial orientation = 0')
    pl.xlabel('Time ({})'.format('s' if fps > 1 else 'frame'), fontsize='x-large')
    pl.ylabel('orientation', fontsize='x-large')
    pl.xlim(0, data['f'].max()/fps)
    if save:
        pl.savefig(save)
    pl.show()

def plot_orient_location(data,odata,tracks):
    import matplotlib.pyplot as pl
    import correlation as corr

    omask = np.isfinite(odata['orient'])
    goodtracks = np.array([78,95,191,203,322])

    ss = 22.

    pl.figure()
    for goodtrack in goodtracks:
        tmask = tracks == goodtrack
        fullmask = np.all(np.asarray(zip(omask,tmask)),axis=1)
        loc_start = (data['x'][fullmask][0],data['y'][fullmask][0])
        orient_start = odata['orient'][fullmask][0]
        sc = pl.scatter(
                (odata['orient'][fullmask] - orient_start + pi) % twopi,
                np.asarray(map(corr.get_norm,
                    zip([loc_start]*fullmask.sum(),
                        zip(data['x'][fullmask],data['y'][fullmask]))
                    ))/ss,
                #marker='*',
                label = 'track {}'.format(goodtrack),
                color = pl.cm.jet(1.*goodtrack/max(tracks)))
                #color = pl.cm.jet(1.*data['f'][fullmask]/1260.))
        print "track",goodtrack
    pl.legend()
    pl.show()
    return True

