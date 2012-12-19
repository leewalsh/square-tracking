#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm
from numpy.lib.recfunctions import append_fields,merge_arrays
from operator import itemgetter

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

    ring_area = 2 * np.pi * r * dr
    return count / ring_area

def pair_corr(positions, dr=22, rmax=220):
    """ pair_corr(positions)

        the pair correlation function g(r)

        takes a list of positions of particles in one 2-d frame
        dr is step size in r for function g(r)
            (units are those implied in coords of positions)
    """
    ss = 22  # side length of square in pixels
    rr = 300 # radius of disk in pixels
    rs = np.arange(ss,rmax,dr)
    g  = np.zeros(np.shape(rs))
    dg = np.zeros(np.shape(rs))
    rg = np.zeros(np.shape(rs))
    for ir,r in enumerate(rs):
        # Three ways to do same thing: list comp, map/lambda, loop
        #gr = [count_in_ring(positions,position,r) for position in positions]
        #gr = map( lambda x,y=r,p=positions: count_in_ring(p,x,y), positions )
        gr = []
        for position in positions:
            if norm(np.asarray(position)-np.asarray((rr,rr))) < rr-rmax:
                gr.append(count_in_ring(positions,position,r))
        if np.asarray(gr).any():
            g[ir]  = np.mean(gr)
            dg[ir] = np.std(gr)
            rg[ir] = r
        else: print "none for r =",r
    return g,dg,rg

def pair_corr_hist(positions, dr=22,rmax=220,nbins=None):
    """ pair_corr_hist(positions):
        the pair correlation function g(r)
        calculated using a histogram of distances between particle pairs
    """
    ss = 22  # side length of square in pixels
    rr = 300 # radius of disk in pixels
    nbins = ss*rmax/dr if nbins is None else nbins
    distances = []
    for pos1 in positions:
        if norm(np.asarray(pos1)-np.asarray((rr,rr))) < rr-rmax:
            distances.append(
                    [norm(np.asarray(pos2) - np.asarray(pos1)) for pos2 in positions]
                    )
    distances = np.asarray(distances)
    distances = distances[np.nonzero(distances)]
    return np.histogram(distances
            , bins = nbins
            , weights = 1/(np.pi*np.asarray(distances)*dr) # normalize by pi*r*dr
            )

def get_positions(data,frame,pid=None):
    """ get_positions(data,frame)
        
        Takes:
            data: structured array of data
            frame: int or list of ints of frame number

        Returns:
            list of tuples (x,y) of positions of all particles in those frames
    """
    if pid is not None:
        fdata = data[data['f']==frame]
        fiddata = fdata[data['id']==pid]
        return (fiddata['x'],fiddata['y'])
    if np.iterable(frame):
        return zip(data['x'][data['f'] in frame],data['y'][data['f'] in frame])
    else:
        return zip(data['x'][data['f']==frame],data['y'][data['f']==frame])

def avg_hists(gs,rgs):
    """ avg_hists(gs,rgs)
        takes:
            gs: an array of g(r) for several frames
            rgs: their associated r values
        returns:
            g_avg: the average of gs over frames
            dg_avg: their std dev
            rg: r for the avgs (just uses rgs[0] for now) 
    """
    #TODO: use better rg here, not just rgs[0]
    rg = rgs[0]
    g_avg = [ np.mean(gs[:,ir]) for ir,r in enumerate(rg) ]
    dg_avg = [ np.std(gs[:,ir]) for ir,r in enumerate(rg) ]
    return g_avg, dg_avg, rg

def build_gs(data,prefix,framestep=10):
    """ build_gs(data,prefix,framestep=10)
        calculates and builds g(r) for each (framestep) frames
        Takes:
            data: the structued array of data
            prefix: which n to use, in string form 'n416', e.g.
            framestep=10: how many frames to skip
        Returns:
            gs: an array of g(r) for several frames
            rgs: their associated r values
    """

    frames = np.arange(min(data['f']),max(data['f']),framestep)
    ss = 22
    dr = ss/2
    rmax = ss*10
    nbins  = ss*rmax/dr
    gs = np.array([ np.zeros(nbins) for frame in frames ])
    rgs = np.copy(gs)
    print "gs initiated with shape (nbins,nframes)",np.shape(gs)
    for nf,frame in enumerate(frames):
        #print "\t appending for frame",frame
        positions = get_positions(data,frame)
        #g,dg,rg = pair_corr(positions)
        g,rg = pair_corr_hist(positions
                ,dr=dr,rmax=rmax,nbins=nbins)
        rg = rg[1:]

        gs[nf,:len(g)]  = g
        rgs[nf,:len(g)] = rg

    return gs,rgs

def get_id(data,position,frames=None):
    """ take a particle's `position' (x,y)
        optionally limit search to one or more `frames'

        return that particle's id
    """
    if frames is not None:
        if np.iterable(frames):
            data = data[data['f'] in frames]
        else:
            data = data[data['f']==frames]
    xmatch = data[data['x']==position[0]]
    return xmatch['id'][xmatch['y']==position[1]]

def get_norm(posi,posj):
    return norm(np.asarray(posj) - np.asarray(posi))

def get_angle(posi,posj):
    vecij = np.asarray(posi) - np.asarray(posj)
    dx = vecij[0]
    dy = vecij[1]
    return np.arctan2(dy,dx)

def add_neighbors(data, nn=6, n_dist=None, delauney=None):
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
    ss = 22
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
    framestep = 500 # large for testing purposes.
    frames = np.arange(min(data['f']),max(data['f']),framestep)
    nsdtype = [('nid',int),('norm',float),('angle',float)]
    neighbors = np.empty(len(data),dtype=[('n',nsdtype,(nn,))])
    data = merge_arrays([data,neighbors],'n', flatten=True)
    nthreads = 2
    #TODO p = Pool(nthreads)
    #def f(frame,data):
    for frame in frames:
        positions = get_positions(data,frame)
        ineighbors = []
        for posi in positions:
            idi = get_id(data,posi,frame)
            ineighbors = [ (
                        posj,        #get_id(data,posj,frame),
                        get_norm(posi,posj),
                        (posi,posj) #placeholder for get_angle(posi,posj)
                        ) for posj in positions ]
            ineighbors.sort(key=itemgetter(1))      # sort by element 1 of tuple (norm)
            ineighbors = ineighbors[1:nn+1]         # the first neighbor is posi
            ineighbors = [ (get_id(data,nposj,frame),nnorm,get_angle(*npos)) 
                    for (nposj,nnorm,npos) in ineighbors]
            data['n'][data['id']==idi] = ineighbors
            #TODO data['n'][data['id']==data['n']]
    return data

def get_gdata(locdir,ns):
    return dict([
            ('n'+str(n), np.load(locdir+'n'+str(n)+'_GR.npz'))
            for n in ns])

def find_gpeaks(ns,locdir,binmax):
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
    import matplotlib.pyplot as pl
    import matplotlib.cm as cm
    #locdir = '/Users/leewalsh/Physics/Squares/spatial_diffusion/'  #rock
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

def plot_gpeaks(peaks,gdata,binmax):
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
    import matplotlib.pyplot as pl
    pl.figure()
    for k in peaks:
        try:
            #pl.plot(gdata[k]['rg'][:binmax]/22.0,gdata[k]['g'][:binmax]/22.0,',-',label=k)
            #pl.scatter(*np.asarray(peaks[k][0]).T,
            #        marker='o', label=k, c = cm.jet((int(k[1:])-200)*255/300))
            #pl.scatter(*np.asarray(peaks[k][1]).T,marker='x',label=k)  # minima
            pks = np.asarray(peaks[k][0]).T
            try:
                pkpos = pks[1]
            except:
                print "pks has wrong shape for k=",k
                print pks.shape
                continue
            pl.scatter(int(k[1:])*np.ones_like(pkpos),pkpos,marker='*',label=k)  # maxima
        except:
            print "failed for ",k
            continue
    pl.legend()

def gpeak_decay(peaks,f):
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
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as pl
    maxima = dict([ (k, np.asarray(peaks[k][0])) for k in peaks])
    minima = dict([ (k, np.asarray(peaks[k][1])) for k in peaks])
    popt = {}
    pcov = {}
    pl.figure()
    for k in peaks:
        maximak = maxima[k].T
        print "k: f,maximak"
        print k,f,maximak
        if len(maxima[k]) > 1:
            try:
                popt[k],pcov[k] = curve_fit(f,maximak[0],maximak[1])
                pl.plot(maximak[0],f(maximak[0],*popt[k]),'--',label='fit '+k)
            except:
                print "error fitting for",k
        else:
            print "maximak empty:",maximak
    return popt,pcov

def exp_decay(s,sigma,c,a):
    """ exp_decay(s,sigma,c,a)
        exponential decay function for fitting

        Args:
            s,  independent variable
        Params:
            sigma,  decay constant
            c,  constant offset
            a,  prefactor

        Returns:
            exp value at s
    """
    return c + a*np.exp(-s/sigma)

def powerlaw(t,b,c,a):
    """ powerlaw(t,b,c,a)
        power law function for fitting

        Args:
            t,  independent variable
        Params:
            b,  exponent (power)
            c,  constant offset
            a,  prefactor
        Returns:
            power law value at t
    """
    # to allow fits for b and/or c,
    # then add them as args to function and delete them below.
    #b = 1
    #c = 0
    return c + a * t**b

def domyfits():
    for k in fixedpeaks:
        figure()
        plot(gdata[k]['rg'][:binmax]/22.0,gdata[k]['g'][:binmax]/22.0,',',label=k)
        scatter(*np.asarray(fixedpeaks[k]).T,marker='o')
        pexps[k],cexp = curve_fit(corr.exp_decay,*np.array(fixedpeaks[k]).T,p0=(3,.0001,.0005))
        ppows[k],cpow = curve_fit(corr.powerlaw,*np.array(fixedpeaks[k]).T,p0=(-.5,.0001,.0005))
        xs = arange(0.8,10.4,0.2)
        plot(xs,corr.exp_decay(xs,*pexps[k]),label='exp_decay')
        plot(xs,corr.powerlaw(xs,*ppows[k]),label='powerlaw')

    

if __name__ == '__main__':
    import matplotlib.pyplot as pl

    locdir = '/Users/leewalsh/Physics/Squares/spatial_diffusion/'  #rock
    prefix = 'n400'

    ss = 22  # side length of square in pixels
    rmax = ss*10

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
        gs,rgs = build_gs(data,prefix)
        print "\t...gs,rgs built"
        print "averaging over all frames..."
        g,dg,rg = avg_hists(gs,rgs)
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
