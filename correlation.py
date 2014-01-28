#!/usr/bin/env python

from __future__ import division

from operator import itemgetter
from itertools import combinations, chain

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import Voronoi
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert


from socket import gethostname
hostname = gethostname()
if 'foppl' in hostname:
    locdir = '/home/lawalsh/Granular/Squares/spatial_diffusion/'
elif 'rock' in hostname:
    computer = 'rock'
    import matplotlib.pyplot as pl
    import matplotlib.cm as cm
    locdir = '/Users/leewalsh/Physics/Squares/spatial_diffusion/'
else:
    print "computer not defined"
    print "where are you working?"

ss = 92   # side length of square in pixels
rr = 1255 # radius of disk in pixels
x0, y0 = 1375, 2020 # center of disk within image, in pixels

pi = np.pi
tau = 2*pi

def count_in_ring(positions,center,r,dr=1):
    """ count_in_ring(positions,center,r,dr)
        return number of particles in a ring of
            centered at center,
            radius r,
            thickness dr (defaults to 1.0),
        normalized by the area of the ring
    """
    count = 0
    for position in positions:
        if r - dr < norm(np.asarray(position)-np.asarray(center)) < r + dr :
            count += 1
        else: continue

    ring_area = tau * r * dr
    return count / ring_area

def pair_corr_hist(positions, dr=ss, dmax=None, rmax=None, nbins=None, boundary=0, do_error=False):
    """ pair_corr_hist(positions):
        the pair correlation function g(r)
        calculated using a histogram of distances between particle pairs
    """
    pmax, pmin = positions.max(0), positions.min(0)
    center = (pmax + pmin)/2   #TODO accuracy of this is critical
    #radius = np.mean(pmax - pmin)/2
    d = np.hypot(*(positions - center).T)
    r = cdist(positions, positions) # faster than squareform(pdist(positions)) wtf
    radius = np.maximum(r.max()/2, d.max())#TODO accuracy of this is critical  #TODO add ss/2?
    if rmax is None:
        rmax = 2*radius # this will have terrible statistics
    if nbins is None:
        nbins = rmax/dr
    if dmax is None:
        dmax = radius - boundary
    ind = np.triu_indices(len(positions), 1)
    # for weighting, use areas of the annulus, which is:
    #   number * arclength * dr = N alpha r dr
    #   where alpha = 2 arccos( (r2 + d2 - R2) / 2 r d )
    cosalpha = 0.5 * (r*r + d*d - radius*radius) / (r * d)
    alpha = 2 * np.arccos(np.clip(cosalpha, -1, None))
    dmask = d <= dmax
    w = np.where(dmask, np.reciprocal(alpha*r*dr), 0)
    w = 0.5*(w + w.T)
    assert np.all(np.isfinite(w[ind]))
    n = np.count_nonzero(dmask) # number of 'inner' particles
    #n = 0.5*(1 + sqrt(1 + 8*np.count_nonzero(w[ind]))) # effective N from number of pairs
    #n = len(w) # total number of particles
    w *= 2/n
    assert np.allclose(positions.shape[0], [len(r), len(d), len(w), len(positions)])
    ret = np.histogram(r[ind], bins=nbins, range=(0, rmax), weights=w[ind])
    if do_error:
        return ret, np.histogram(r[ind], bins=nbins, range=(0, rmax))
    else:
        return ret

def get_positions(data, frame, pid=None):
    """ get_positions(data,frame)
        
        Takes:
            data: structured array of data
            frame: int or list of ints of frame number

        Returns:
            list of tuples (x,y) of positions of all particles in those frames
    """
    fmask = np.in1d(data['f'], frame) if np.iterable(frame) else data['f']==frame
    if pid is not None:
        fiddata = data[fmask & (data['id']==pid)]
        return np.array(fiddata['x'], fiddata['y'])
    return np.column_stack((data['x'][fmask], data['y'][fmask]))

def avg_hists(gs, rgs):
    """ avg_hists(gs,rgs)
        takes:
            gs: an array of g(r) for several frames
            rgs: their associated r values
        returns:
            g_avg: the average of gs over frames
            dg_avg: their std dev / sqrt(length)
            rg: r for the avgs (just uses rgs[0] for now) 
    """
    assert np.all([np.allclose(rgs[i], rgs[j])
        for i in xrange(rgs.shape[0]) for j in xrange(rgs.shape[0])])
    rg = rgs[0]
    g_avg = gs.mean(0)
    dg_avg = gs.std(0)/np.sqrt(gs.shape[0])
    return g_avg, dg_avg, rg

def build_gs(data, framestep=1, dr=None, dmax=None, rmax=None, boundary=0, do_error=False):
    """ build_gs(data, framestep=10)
        calculates and builds g(r) for each (framestep) frames
        Takes:
            data: the structued array of data
            framestep=10: how many frames to skip
        Returns:
            gs: an array of g(r) for several frames
            rgs: their associated r values
    """
    frames = np.arange(data['f'].min(), data['f'].max()+1, framestep)
    dr = ss*(.1 if dr is None else dr)
    #if rmax is None:
        #rmax = rr - ss*3
    #elif rmax:
        #rmax = rr - ss*rmax
    nbins = rmax/dr if rmax and dr else None
    gs = rgs = egs = ergs = None
    for nf, frame in enumerate(frames):
        positions = get_positions(data, frame)
        g, rg = pair_corr_hist(positions, dr=dr, dmax=dmax, rmax=rmax, nbins=nbins,
                               boundary=boundary, do_error=do_error)
        if do_error:
            (g, rg), (eg, erg) = g, rg
            erg = erg[1:]
        rg = rg[1:]
        if gs is None:
            nbins = g.size
            gs = np.zeros((frames.size, nbins))
            rgs = gs.copy()
            if do_error:
                egs = np.zeros((frames.size, nbins))
                ergs = gs.copy()
        gs[nf,:len(g)]  = g
        rgs[nf,:len(g)] = rg
        if do_error:
            egs[nf, :len(eg)] = eg
            ergs[nf, :len(eg)] = erg
    return (gs, rgs), (egs, ergs) if do_error else (gs, rgs)

def global_particle_orientational(orientations, m=4, ret_complex=True):
    """ global_particle_orientational(orientations, m=4)
        Returns the global m-fold particle orientational order parameter

                1   N    i m theta
        Phi  = --- SUM e          j
           m    N  j=1
    """
    np.mod(orientations, tau/4, orientations)
    phi = np.exp(m*orientations*1j).mean()
    err = phi.std(ddof=1)/np.sqrt(phi.size)
    return (phi, err) if ret_complex else (np.abs(phi), err)

def dtheta(i, j=None, m=4, sign=False):
    """ given two angles or one array (N,2) of pairs
        returns the _smallest angle between them, modulo m
        if sign is True, retuns a negative angle for i<j, else abs
    """
    ma = 2*np.pi/m
    if j is not None:
        diff = i - j
    elif i.shape[1]==2:
        diff = np.subtract(*i.T)
    diff = (diff + ma/2)%ma - ma/2
    return diff if sign else np.abs(diff)

def orient_corr(positions, orientations, m=4, dr=ss, rmax=10*ss, nbins=None):
    """ orient_corr():
        the orientational correlation function g_m(r)
        given by mean(phi(0)*phi(r))
    """
    center = (positions.max(0) + positions.min(0))/2
    loc_mask = np.hypot(*(positions - center).T) < rmax
    distances = pdist(positions[loc_mask])
    pairs = orientations[loc_mask][np.column_stack(np.triu_indices(loc_mask.sum(), 1))]
    diffs = np.cos(m*dtheta(pairs))
    return distances, diffs

def get_neighbors(v, p, pm=None, ret_pairs=False):
    """ give neighbors in voronoi tessellation v of point id p
        if already calculated, pm is point mask
    """
    if pm is None:
        pm = v.ridge_points == p
    else:
        pm = pm[p]
    pm = np.any(pm, 1)
    pairs = v.ridge_points[pm]
    return pairs if ret_pairs else pairs[pairs != p]

def local_particle_orientational(orientations, vor, m=4, ret_complex=True):
    """ local m-fold particle orientational order parameter
        THIS IS WRONG :-( but unnecessary :-/

        phi(r_i) = mean(exp(i*m*(theta_i -theta_j)))
    """
    phi = np.empty(orientations.shape, complex)
    for p in xrange(orientations.size):
        pairs = get_neighbors(vor, p, ret_pairs=True)
        phi[p] = np.exp(1j*m*dtheta(*orientations[pairs.T])).mean()
    return phi

def get_id(data,position,frames=None,tolerance=10e-5):
    """ take a particle's `position' (x,y)
        optionally limit search to one or more `frames'

        return that particle's id
    """
    if frames is not None:
        if np.iterable(frames):
            data = data[np.in1d(data['f'], frames)]
        else:
            data = data[data['f']==frames]
    xmatch = data[abs(data['x']-position[0])<tolerance]
    return xmatch['id'][abs(xmatch['y']-position[1])<tolerance]

def get_norm((posi,posj)):
    return norm(np.asarray(posj) - np.asarray(posi))

def get_angle(posi,posj):
    vecij = np.asarray(posi) - np.asarray(posj)
    dx = vecij[0]
    dy = vecij[1]
    return np.arctan2(dy,dx) % tau

def merge_data(data,ndata,**kw):
    if data is None:
        if ndata is not None:
            print "Returning ndata"
            return ndata
        elif ndata is None:
            print "Returning None"
            return None
    elif ndata is None:
        print "Returning data"
        return data
    else:
        return merge_arrays([data,ndata], flatten=True, **kw)


def add_neighbors(data, nn=6, n_dist=None, delauney=None, ss=22):
    """ add_neighbors(data)
        takes data structured array, adds field of neighbors
        which is a list of nearest neighbors
        returns new array with neighbors field
    """
    from multiprocessing import Pool
    if 'n' in data.dtype.names:
        print "It's your lucky day, neighbors have already been added to this data"
        return data
    if 's' in data.dtype.names:
        fieldnames = np.array(data.dtype.names)
        fieldnames[fieldnames == 's'] = 'f'
        data.dtype.names = tuple(fieldnames)
    if nn is not None:
        n_dist = None
    elif n_dist is True:
        print "hm haven't figured that out yet"
        n_dist = ss*np.sqrt(2)
    elif n_dist is None and delauney is not True:
        print "hm haven't figured that out yet"
        n_dist = ss*np.sqrt(2)
    elif delauney is True:
        print "hm haven't figured that out yet"
        return data
    nsdtype = [('nid',int),('norm',float),('angle',float)]
    ndata = np.zeros(len(data), dtype=[('n',nsdtype,(nn,))] )
    nthreads = 2
    #TODO p = Pool(nthreads)
    framestep = 50 # large for testing purposes.
    frames = np.arange(min(data['f']),max(data['f']),framestep)
    #def f(frame,data):
    for frame in frames:
        positions = get_positions(data,frame)
        ineighbors = []
        for posi in positions:
            idi = get_id(data,posi,frame)
            ineighbors = [ (
                        posj,           #to become get_id(data,posj,frame),
                        get_norm((posi,posj)),
                        (posi,posj)     #to become get_angle(posi,posj)
                        ) for posj in positions ]
            ineighbors.sort(key=itemgetter(1))      # sort by element 1 of tuple (norm)
            ineighbors = ineighbors[1:nn+1]         # the first neighbor is posi itself
            ineighbors = [ (get_id(data,nposj,frame),nnorm,get_angle(*npos)) 
                    for (nposj,nnorm,npos) in ineighbors]
            ndata['n'][data['id']==idi] = ineighbors
    return ndata

def get_gdata(locdir,ns):
    return dict([
            ('n'+str(n), np.load(locdir+'n'+str(n)+'_GR.npz'))
            for n in ns])

def find_gpeaks(ns,locdir,binmax=258):
    """ find_gpeaks(ns,locdir,binmax)
        finds peaks and valleys in g(r) curve
        takes:
            ns, list of densities to analyse
            locdir, local directory for data
            binmax, the max bin number, hopefully temporary problem
        returns:
            peaks,  list of [list of peaks and list of valleys]
                    in format given by peakdetect.py
    """
    import peakdetect as pk
    #ns = np.array([8,16,32,64,128,192,256,320,336,352,368,384,400,416,432,448])
    binmax = 258
    gdata = get_gdata(locdir,ns)
    peaks  = {}
    maxima = {}
    minima = {}
    for k in gdata:
        extrema = pk.peakdetect(
                gdata[k]['g'][:binmax]/22.0, gdata[k]['rg'][:binmax]/22.,
                lookahead=2.,delta=.0001)
        peaks[k] = extrema
        maxima[k] = np.asarray(extrema[0])
        minima[k] = np.asarray(extrema[1])
    return peaks

def plot_gpeaks(peaks,gdata,pksonly=False,hhbinmax=258):
    """ plot_gpeaks(peaks,gdata,binmax)
        plots locations and/or heights of peaks and/or valleys in g(r)
        takes:
            peaks,  list of peaks from output of find_gpeaks()
            gdata,  g(r) arrays, loaded from get_gdata()
            binmax, the max bin number, hopefully temporary problem
        side affects:
            creates a figure and plots things
        returns:
            nothing
    """
    if computer is 'foppl':
        print "cant do this on foppl"
        return
    pl.figure()
    for k in peaks:
        try:
            pl.plot(gdata[k]['rg'][:binmax]/22.0,gdata[k]['g'][:binmax]/22.0,',-',label=k)
            #pl.scatter(*np.asarray(peaks[k][0]).T,
            #        marker='o', label=k, c = cm.jet((int(k[1:])-200)*255/300))
            #pl.scatter(*np.asarray(peaks[k][1]).T,marker='x',label=k)  # minima

            if pksonly is False:
                pks = np.asarray(peaks[k][0]).T # gets just maxima
            elif pksonly is True:
                pks = np.asarray(peaks[k]).T    # if peaks is already just maxima
            try:
                pkpos = pks[0]
            except:
                print "pks has wrong shape for k=",k
                print pks.shape
                continue
            #pl.scatter(int(k[1:])*np.ones_like(pkpos),pkpos,marker='*',label=k)  # maxima
        except:
            print "failed for ",k
            continue
    pl.legend()

def apply_hilbert(a, sig=None, full=False):
    """ Attempts to apply hilbert transform to a signal about a mean.
        First, smooth the signal, then subtract the smoothed signal.
        Apply hilbert to the residual, and add the smoothed signal back in.
    """
    assert a.ndim == 1, "Only works for 1d arrays"
    if sig is None:
        sig = a.size/10.
    if sig:
        a_smoothed = gaussian_filter(a, sig, mode='reflect')
    else:
        a_smoothed = a.mean()
    h = hilbert(a - a_smoothed)
    if full:
        return h, a_smoothed
    else:
        return np.abs(h) + a_smoothed

def gpeak_decay(peaks,f,pksonly=False):
    """ gpeak_decay(peaks,f)
    fits curve to the peaks in g(r)
    takes:
        peaks,  list of peak/valley positions and heights
        f,      the function for the curve, right now either:
                    exp_decay or powerlaw

    returns:
        popt, a tuple of parameters for f
        pcov, their covariances
    """
    if computer is 'foppl':
        print "cant do this on foppl"
        return
    from scipy.optimize import curve_fit
    if pksonly is False:
        maxima = dict([ (k, np.asarray(peaks[k][0])) for k in peaks])
        minima = dict([ (k, np.asarray(peaks[k][1])) for k in peaks])
    elif pksonly is True:
        maxima = peaks
    popt = {}
    pcov = {}
    pl.figure()
    for k in peaks:
        maximak = maxima[k].T
        print "k: f,maximak"
        print k,f,maximak
        if len(maxima[k]) > 1:
            popt[k],pcov[k] = curve_fit(f,maximak[0],maximak[1])
            fitrange = np.arange(min(maximak[0]),max(maximak[0]),.05)
            pl.plot(fitrange,f(fitrange,*popt[k]),'--',label='fit '+k)
        else:
            print "maximak empty:",maximak
    return popt,pcov

def exp_decay(s, sig=1., a=1., c=0):
    """ exp_decay(s,sigma,c,a)
        exponential decay function for fitting

        Args:
            s,  independent variable
        Params:
            sigma,  decay constant
            a,  prefactor
            c,  constant offset

        Returns:
            exp value at s
    """
    return c + a*np.exp(-s/sig)

def powerlaw(t, b=1., a=1., c=0):
    """ powerlaw(t,b,c,a)
        power law function for fitting

        Args:
            t,  independent variable
        Params:
            b,  exponent (power)
            a,  prefactor
            c,  constant offset
        Returns:
            power law value at t
    """
    return c + a * np.power(t, -b)

def log_decay(t, a=1, l=1., c=0.):
    return c - a*np.log(t/l)
    
def domyfits():
    if computer is 'foppl':
        print "cant do this on foppl"
        return
    for k in fixedpeaks:
        pl.figure()
        pl.plot(gdata[k]['rg'][:binmax]/22.0,gdata[k]['g'][:binmax]/22.0,',',label=k)
        pl.scatter(*np.asarray(fixedpeaks[k]).T,marker='o')
        pexps[k],cexp = curve_fit(corr.exp_decay,*np.array(fixedpeaks[k]).T,p0=(3,.0005,.0001))
        ppows[k],cpow = curve_fit(corr.powerlaw,*np.array(fixedpeaks[k]).T,p0=(-.5,.0005,.0001))
        xs = np.arange(0.8,10.4,0.2)
        pl.plot(xs,exp_decay(xs,*pexps[k]),label='exp_decay')
        pl.plot(xs,powerlaw(xs,*ppows[k]),label='powerlaw')
    return pexps,ppows

def delta_distro(n,histlim):
    """ delta_distro(n, histlim)
        returns a delta distribution of angles for n-fold order
    """
    return list(np.arange(0,1,1./n)*tau)*histlim

def domyhists(nbins=180, ang_type='relative',boundaries=True,ns=None,nn=None):
    """ ang_type can be 'relative', 'delta', or 'absolute'

        if boundaries is True: boundaries are included, else: excluded
            boundary width is assumed to be 0.3*r
            if boundaries is false, data must be loaded to find positions
    """
    if computer is 'foppl':
        print "cant do this on foppl"
        return
    if ns is None:
        ns = np.arange(320,464,16)
    if not np.iterable(ns):
        ns = [ns]
    if nn is None:
        nn = 6
    for n in ns:
        print 'n=',n
        prefix = 'n'+str(n)
        ndatanpz = np.load(locdir+prefix+'_NEIGHBORS.npz')
        ndata = ndatanpz['ndata']
        histlim = nn*len(ndata)/nbins/2
        if ang_type is not 'delta':
            histlim /= 2
        if boundaries is False:
            histlim /= 6
            datanpz = np.load(locdir+prefix+'_TRACKS.npz')
            #alldata = merge_data(datanpz['data'],ndata)
            alldata = datanpz['data']
            x0 = np.mean([max(alldata['x']),min(alldata['x'])])
            y0 = np.mean([max(alldata['y']),min(alldata['y'])])
            r0 = 0.5*np.mean([
                max(alldata['x']) - min(alldata['x']),
                max(alldata['y']) - min(alldata['y'])])
            center = (x0,y0)
            is_bulk = np.asarray(
                    map(get_norm,zip(zip(alldata['x'],alldata['y']),list(center)*len(alldata)))
                        < r0*0.7,
                        dtype=bool)
            if len(is_bulk) == len(ndata):
                ndata = ndata[is_bulk]
                #ndata = alldata.view(dtype = [('n',alldata['n'].dtype,(nn,))])
            else:
                print "length mismatch"
        # remove data without neighbor info:
        ndata = ndata[np.any(ndata['n']['nid'],axis=1)]
        if ang_type is 'relative':
            allangles = np.array([
                    (ndata['n']['angle'][:,i] - ndata['n']['angle'][:,0])%tau
                    for i in np.arange(nn) ])
        elif ang_type is 'delta':
            allangles = []
            for i in range(len(ndata['n'])):
                ineighbors = ndata['n'][i][:nn]
                ineighbors.sort(order='angle')
                allangles.append(list([
                        (ineighbors['angle'][i] - ineighbors['angle'][i-1])%tau
                        for i in range(len(ineighbors))
                        ]))
        elif ang_type is 'absolute':
            allangles = ndata['n']['angle']
        else:
            print "uknown ang_type:",ang_type
            continue
        allangles = np.asarray(allangles).flatten() % tau

        pl.figure(figsize=(12,9))
        for nfold in [8,6,4]:
            pl.hist(delta_distro(nfold,histlim), bins = nbins,label="%d-fold"%nfold)
        pl.hist(allangles, bins = nbins,label=ang_type+' theta')
        pl.ylim([0,histlim])
        pl.xlim([0, pi if ang_type is 'delta' else tau])
        pl.title("%s, %s theta, %d neighbors, boundaries %scluded"%\
                (prefix,ang_type,nn,"in" if boundaries else "ex"))
        pl.legend()
        pl.savefig("%s%s_ang_%s_%d%s_hist.png"%\
                    (locdir,prefix,ang_type,nn,'' if boundaries else '_nobndry'))

def domyneighbors(prefix):
    tracksnpz = np.load(locdir+prefix+"_TRACKS.npz")
    data = tracksnpz['data']
    ndata = add_neighbors(data)
    np.savez(locdir+prefix+'_NEIGHBORS.npz',ndata=ndata)



if __name__ == '__main__':

    prefix = 'n400'

    ss = 92#22  # side length of square in pixels
    rmax = ss*10.

    try:
        datapath = locdir+prefix+"_GR.npz"
        print "loading data from",datapath
        grnpz = np.load(datapath)
        g  = grnpz['g']
        dg = grnpz['dg']
        rg = grnpz['rg']
    except:
        print "NPZ file not found for n =",prefix[1:]
        datapath = locdir+prefix+'_results.txt'
        print "loading data from",datapath
        data = np.genfromtxt(datapath,
                skip_header = 1,
                usecols = [0,2,3,5],
                names   = "id,x,y,f",
                dtype   = [int,float,float,int])
        data['id'] -= 1 # data from imagej is 1-indexed
        print "\t...loaded"
        print "loading positions"
        gs, rgs = build_gs(data)
        print "\t...gs,rgs built"
        print "averaging over all frames..."
        g, dg, rg = avg_hists(gs, rgs)
        print "\t...averaged"
        print "saving data..."
        np.savez(locdir+prefix+"_GR",
                g  = np.asarray(g),
                dg = np.asarray(dg),
                rg = np.asarray(rg))
        print "\t...saved"

    binmax = len(rg[rg<rmax])
    #pl.figure()
    pl.plot(1.*rg[:binmax]/ss,g[:binmax],'.-',label=prefix)
    #pl.title("g[r],%s,dr%d"%(prefix,ss/2))
    pl.legend()
    #pl.show()
