#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm
from numpy.lib import recfunctions as rec

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
        if r - dr < norm(np.array(position)-np.array(center)) < r + dr :
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
            if norm(np.array(position)-np.array((rr,rr))) < rr-rmax:
                gr.append(count_in_ring(positions,position,r))
        if np.array(gr).any():
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
        if norm(np.array(pos1)-np.array((rr,rr))) < rr-rmax:
            distances.append(
                    [norm(np.array(pos2) - np.array(pos1)) for pos2 in positions]
                    )
    distances = np.array(distances)
    distances = distances[np.nonzero(distances)]
    return np.histogram(distances
            , bins = nbins
            , weights = 1/(np.pi*np.array(distances)*dr) # normalize by pi*r*dr
            )

def get_positions(data,frame):
    if np.iterable(frame):
        return zip(data['x'][data['f'] in frame],data['y'][data['f'] in frame])
    else:
        return zip(data['x'][data['f']==frame],data['y'][data['f']==frame])

def avg_hists(gs,rgs):
    rg = rgs[0]
    g_avg = [ np.mean(gs[:,ir]) for ir,r in enumerate(rg) ]
    dg_avg = [ np.std(gs[:,ir]) for ir,r in enumerate(rg) ]
    return g_avg, dg_avg, rg

def build_gs(data,prefix,framestep=10):
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

def add_neighbors(data,n_dist=None,delauney=None):
    """ add_neighbors(data)
        takes data structured array, adds field of neighbors
        which is a list of nearest neighbors
        returns new array with neighbors field
    """
    if 'n' in data.dtype.names:
        print "It's your lucky day, neighbors have already been added to this data"
        return data
    ss = 22
    nn = 8 # max number nearest neighbors
    if n_dist is True:
        n_dist = ss*np.sqrt(2)
    elif n_dist is None and delauney is not True:
        n_dist = ss*np.sqrt(2)
    elif delauney is True:
        print "hm haven't figured that out yet"
        return data
    framestep = 10 # for testing purposes.  Use 1 otherwise
    frames = np.arange(min(data['f']),max(data['f']),framestep)
    neighbors = -np.ones(len(data),dtype=str(nn)+'int32')
    data = rec.append_fields(data,'n',neighbors)
    for nf,frame in frames:
        positions = get_positions(data,frame)
        distances = []
        for pos0 in positions:
            id0 = get_id(data,pos0,frame)
            if np.all(data['n'][data['id']==id0] == -1):
                ns = []
                for pos1 in positions:
                    distance = norm(np.array(pos0) - np.array(pos1))
                    if distance < n_dist:
                        ns.append(get_id(data,pos1,frame))
                ns.sort()
                if len(ns) <= nn:
                    data['n'][data['id']==id0][:len(nn)] = ns
                else:
                    print "too many nearest neighbors\n\tusing closest",nn
                    data['n'][data['id']==id0][:] = ns[:nn]
            else:
                continue
    return data

def build_angles(data):
    if 'n' not in data.dtype.names:
        print "Oh no, neighest neighbors haven't been calculated!"
        print "Just wait, I'll do it for you"
        return build_angles(add_neighbors(data))
    elif 'n' in data.dtype.names:

        return



if __name__ == '__main__':
    import matplotlib.pyplot as pl

    locdir = '/Users/leewalsh/Physics/Squares/spatial_diffusion/'  #rock
    prefix = 'n400'
    datapath = locdir+prefix+'_results.txt'

    ss = 22  # side length of square in pixels
    rmax = ss*10

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

    binmax = len(rg[rg<rmax])
    pl.figure()
    pl.plot(1.*rg[:binmax]/ss,g[:binmax],'.-',label=prefix)
    pl.title("g[r],%s,dr%d"%(prefix,ss/2))
    pl.show()
