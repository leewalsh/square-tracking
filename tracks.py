#!/usr/bin/env python

import numpy as np
from PIL import Image as Im
import sys

from socket import gethostname
hostname = gethostname()
if 'rock' in hostname:
    computer = 'rock'
    locdir = '/Users/leewalsh/Physics/Squares/lowdensity/'
    extdir = locdir#'/Volumes/bhavari/Squares/lighting/still/'
elif 'foppl' in hostname:
    computer = 'foppl'
    locdir = '/home/lawalsh/Granular/Squares/diffusion/'
    extdir = '/media/bhavari/Squares/diffusion/still/'
    import matplotlib
    matplotlib.use("agg")
else:
    print "computer not defined"
    print "where are you working?"

from matplotlib import pyplot as pl
from matplotlib import cm as cm

is_main = __name__=='__main__'

verbose = True

if is_main:
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('prefix')
    args = parser.parse_args()

    prefix = args.prefix#'n32_100mv_50hz'
    print 'using prefix', prefix
    dotfix = ''#_CORNER'
    if dotfix:
        print 'using dotfix', dotfix

    loaddata   = False   # Create and save structured array from data txt file?

    findtracks = False   # Connect the dots and save in 'trackids' field of data
    plottracks = False   # plot their tracks

    formula = 'pos_ang'
    findcorr = True      # Calculate the corr
    loadcorr = False      # load previoius corr from npz file
    plotcorr = False      # plot the corr


    if plottracks:
        bgimage = Im.open(extdir+prefix+'_0001.tif') # for bkground in plot
    if loaddata:
        datapath = locdir+prefix+dotfix+'_POSITIONS.txt'

def find_closest(thisdot, trackids, n=1, maxdist=25., giveup=1000):
    """ recursive function to find nearest dot in previous frame.
        looks further back until it finds the nearest particle
        returns the trackid for that nearest dot, else returns new trackid"""
    frame = thisdot['f']
    if frame < n:  # at (or recursed back to) the first frame
        newtrackid = max(trackids) + 1
        if verbose:
            print "New track:", newtrackid
            print '\tframe:', frame,'n:', n,'dot:', thisdot['id']
        return newtrackid
    else:
        oldframe = data[data['f']==frame-n]
        dists = (thisdot['x']-oldframe['x'])**2 + (thisdot['y']-oldframe['y'])**2
        closest = oldframe[np.argmin(dists)]
        if min(dists) < maxdist:
            return trackids[closest['id']]
        elif n < giveup:
            return find_closest(thisdot, trackids, n=n+1, maxdist=maxdist, giveup=giveup)
        else: # give up after giveup frames
            if verbose:
                print "Recursed {} times, giving up. frame = {} ".format(n, frame)
            newtrackid = max(trackids) + 1
            if verbose:
                print "New track:", newtrackid
                print '\tframe:', frame,'n:', n,'dot:', thisdot['id']
            return newtrackid

# Tracking
def load_data(datapath):
    print "loading data from", datapath
    if  datapath.endswith('results.txt'):
        shapeinfo = False
        # imagej output (called *_results.txt)
        dtargs = {  'usecols' : [0,2,3,5],
                    'names'   : "id,x,y,f",
                    'dtype'   : [int,float,float,int]} \
            if not shapeinfo else \
                 {  'usecols' : [0,1,2,3,4,5,6],
                    'names'   : "id,area,mean,x,y,circ,f",
                    'dtype'   : [int,float,float,float,float,float,int]}
        data = np.genfromtxt(datapath, skip_header = 1,**dtargs)
        data['id'] -= 1 # data from imagej is 1-indexed
    elif datapath.endswith('POSITIONS.txt'):
        from numpy.lib.recfunctions import append_fields
        # positions.py output (called *_POSITIONS.txt)
        data = np.genfromtxt(datapath,
                skip_header = 1,
                names = "f,x,y,lab,ecc,area",
                dtype = [int,float,float,int,float,int])
        data = append_fields(data, 'id', np.arange(data.shape[0]), usemask=False)
    else:
        print "is {} from imagej or positions.py?".format(datapath.split('/')[-1])
        print "Please rename it to end with _results.txt or _POSITIONS.txt"
    return data

if is_main and loaddata:
    data = load_data(datapath)
    print "\t...loaded"


def find_tracks(data, giveup = 1000):
    sys.setrecursionlimit(2*giveup)

    trackids = -np.ones(data.shape, dtype=int)

    print "seeking tracks"
    for i in range(len(data)):
        trackids[i] = find_closest(data[i], trackids)

    # save the data record array and the trackids array
    print "saving track data"
    np.savez(locdir+prefix+dotfix+"_TRACKS",
            data = data,
            trackids = trackids)

    return trackids

if is_main:
    if findtracks:
        trackids = find_tracks(data)
    elif loaddata:
        print "saving data only (no tracks)"
        np.savez(locdir+prefix+dotfix+"_POSITIONS",
                data = data)
        print '\t...saved'
    else: 
        # assume existing tracks.npz
        print "loading tracks from npz files"
        tracksnpz = np.load(locdir+prefix+"_TRACKS.npz")
        data = tracksnpz['data']
        trackids = tracksnpz['trackids']
        cdatanpz = np.load(locdir+prefix+'_CORNER_POSITIONS.npz')
        cdata = cdatanpz['data']
        print "\t...loaded"
    try:
        odatanpz = np.load(locdir+prefix+'_ORIENTATION.npz')
        odata = odatanpz['odata']
        omask = odatanpz['omask']
    except IOError:
        print "calculating orientation data"
        from orientation import get_angles_loop
        odata, omask = get_angles_loop(data, cdata)
        np.savez(locdir+prefix+'_ORIENTATION.npz',
                odata=odata,
                omask=omask)
        print '\t...saved'


# Plotting tracks:
def plot_tracks(data, trackids, bgimage=None):
    pl.figure()
    pl.scatter( data['y'], data['x'],
            c=np.array(trackids)%12, marker='o')
    pl.imshow(bgimage, cmap=cm.gray, origin='upper')
    pl.title(prefix)
    print "saving tracks image"
    pl.savefig(locdir+prefix+"_tracks.png", dpi=180)
    pl.show()

if is_main and plottracks and computer is 'rock':
    try:
        bgimage = Im.open(extdir+prefix+'_0001.tif') # for bkground in plot
    except IOError:
        bgimage = Im.open(locdir+prefix+'_0001.tif') # for bkground in plot
    plot_tracks(data, trackids, bgimage)

# Mean Squared Displacement
# dx^2 (tau) = < ( x_i(t0 + tau) - x_i(t0) )^2 >
#              <  averaged over t0, then i   >

def farange(start, stop, factor):
    start_power = np.log(start)/np.log(factor)
    stop_power = np.log(stop)/np.log(factor)
    return factor**np.arange(start_power, stop_power)

from orientation import track_orient
def track_corr(track, dt0, dtau, data, trackids, odata=None, omask=None, formula='', mod_2pi=False, stack=False):
    """ track_corr(track, dt0, dtau, odata, omask)
        finds the track corr, as function of tau, averaged over t0, for one track (worldline)
    """
    tcorr = []
    formula = formula.lower()
    pos = formula.count('pos') or formula.count('tr')
    ang = formula.count('ang') or formula.count('ori')
    mask = (trackids==track) & omask if ang else trackids==track
    if mask.sum() <= 1:
        if verbose:
            print 'for track {}, mask size is {}, skipping this track'.format(track,mask.sum())
        return None
    trackdots = data[mask]
    trackodata = (odata[mask]['orient'] if mod_2pi
            else track_orient(data, odata, track, trackids, omask)) if ang else None
    trackend   = trackdots['f'][-1]
    trackbegin = trackdots['f'][0]
    tracklen = trackend - trackbegin + 1
    if verbose:
        print "tracklen =", tracklen
        print "\t from %d to %d"%(trackbegin, trackend)
    if isinstance(dtau, float):
        taus = farange(dt0, tracklen, dtau)
    elif isinstance(dtau, int):
        if dtau > 0:
            taus = xrange(dtau, tracklen, dtau)
        elif dtau < 0:
            taus = xrange(-tracklen/dtau*dtau, tracklen, -dtau)
    if stack:
        tau = dtau
        tcorr = t0avg(trackdots, tracklen, tau, trackodata, dt0, formula=formula, mod_2pi=mod_2pi, stack=True) - \
                t0avg(trackdots, tracklen, tau, trackodata, dt0, formula=formula, mod_2pi=mod_2pi, stack=False)
    else:
        for tau in taus:  # for tau in T, by factor dtau
            avg = t0avg(trackdots, tracklen, tau, trackodata, dt0, formula=formula, mod_2pi=mod_2pi, stack=stack)
            if avg > 0 and np.isfinite(avg):
                tcorr.append([tau, avg])
    if verbose:
        print "\t...actually", len(tcorr)
    return tcorr

def t0avg(trackdots, tracklen, tau, trackodata, dt0, formula='', mod_2pi=False, stack=False):
    """ t0avg() averages over all t0, for given track, given tau """
    totsq = 0.0
    nt0s = 0.0
    allsq = []
    formula = formula.lower()
    pos = formula.count('pos') or formula.count('tr')
    ang = formula.count('ang') or formula.count('ori')

    if ang and pos:
        dt = 10 # constant step size in frames for derivative
        tau, dt = dt, tau #swap values
    else:
        dt = 0

    for t0 in np.arange(max(1,-tau), tracklen-max(0,tau)-1+dt, dt0): # for t0 in (T - tau - 1), by dt0 stepsize
        if pos:
            try:
                olddot = trackdots[trackdots['f']==t0][0]
                newdot = trackdots[trackdots['f']==t0+tau][0]
            except IndexError:
                continue
            sqdisp = (newdot['x'] - olddot['x'])**2 \
                   + (newdot['y'] - olddot['y'])**2

        if ang:
            try:
                oldorient = trackodata[trackdots['f']==t0+dt][0]
                neworient = trackodata[trackdots['f']==t0+dt+tau][0]
            except IndexError:
                continue
            if mod_2pi:
                odisp = (neworient - oldorient)%(2*np.pi)
                if odisp > np.pi:
                    odisp -= 2*np.pi
            else:
                odisp = neworient - oldorient
            sqodisp = odisp**2

        if pos and not ang:
            quantity = sqdisp
        elif ang and not pos:
            quantity = sqodisp
        elif ang and pos:
            quantity = sqdisp * sqodisp

        nt0s += 1
        if stack:
            allsq.append(quantity)
        else:
            totsq += quantity
    return np.asarray(allsq) if stack else totsq/nt0s if nt0s else None

def find_corr(formula, dt0, dtau, data, trackids, odata, omask, tracks=None, mod_2pi=False, stack=False):
    """ Calculates the correlation given by formula"""
    print "Begin calculating correlation:", formula
    corr = []
    if tracks is None:
        tracks = set(trackids)
    for trackid in tracks:
        if verbose:
            print "calculating corr for track", trackid
        tcorr = track_corr(trackid, dt0, dtau, data, trackids, odata, omask, formula, mod_2pi, stack=stack)
        if tcorr is not None:
            corr.append(tcorr)

    corr = np.asarray(corr)
    print "saving corr data",formula
    if formula.count('pos'):
        suffix = 'ATC' if formula.count('ang') else 'MSD'
    else:
        suffix = 'MSAD'
    np.savez(locdir+prefix+'_'+'stacked'*stack+suffix,
            dt0  = np.asarray(dt0),
            dtau = np.asarray(dtau),
            **{suffix.lower()+'s' : corr})
    print "\t...saved"
    return corr

if is_main and findcorr:
    dt0  = 10 # small for better statistics, larger for faster calc
    dtau = 0# int for stepwise, float for factorwise
    corr = find_corr(formula, dt0, dtau, data, trackids, odata, omask,stack=True)
            
elif is_main and loadcorr:
    print "loading corr ({}) data from npz files".format(formula)
    if formula.count('pos'):
        suffix = 'ATC' if formula.count('ang') else 'MSD'
    else:
        suffix = 'MSAD'
    corrnpz = np.load(locdir+prefix+'_'+suffix+".npz")
    corr = corrnpz[suffix]
    if corrnpz['dt0']:
        dt0  = corrnpz['dt0'][()] # [()] gets element from 0D array
        dtau = corrnpz['dtau'][()]
    else:
        dt0  = 10 # here's assuming...
        dtau = 10 #  should be true for all from before dt* was saved
    print "\t...loaded"

def corr_hist(formula, data, corrs):
    h = pl.hist(np.concatenate(corrs), bins=1000)
    pl.xlim(0,10000)
    return h


def plot_corr(formula, data, corrs, dtau, dt0,
              tnormalize=False, prefix='', show_tracks=True, plfunc=pl.semilogx):
    """ Plots the corr"""
    nframes = max(data['f'])
    if isinstance(dtau, float):
        taus = farange(dt0, nframes, dtau)
    elif isinstance(dtau, int):
        if dtau > 0:
            taus = np.arange(dtau, nframes, dtau)
        elif dtau < 0:
            taus = np.arange(-nframes/dtau*dtau, nframes, -dtau)
    corr = np.transpose([taus, np.zeros_like(taus)]).astype(float)

    plf = pl.semilogy if dtau < 0 else pl.loglog
    pl.figure()
    added = np.zeros(len(corr), float)

    if formula.count('pos'):
        suffix = 'ATC' if formula.count('ang') else 'MSD'
        corrunit = '' if formula.count('ang') else 'particle area'
    else:
        suffix = 'MSAD'
        corrunit = 'radian'+r'$^2$'

    for tcorr in corrs:
        if len(tcorr) > 0:
            tcorr = np.asarray(tcorr)
            if 'pos' in formula:
                tcorr[:,1] /= 22**2 # convert to unit "particle area"
            if tnormalize:
                tcorrt, tcorrd = zip(*tcorr)
                tcorrt = np.asarray(tcorrt)
                tcorrd = np.asarray(tcorrd)
                if show_tracks:
                    plfunc(tcorrt, tcorrd/tcorrt**tnormalize)
            elif show_tracks:
                plf(*zip(*tcorr))
            lim = min(len(corr), len(tcorr))
            corr[:lim,1] += np.array(tcorr)[:lim,1]
            added[:lim] += 1.
    #assert not np.any(corr[:,1]==0), "no tcorr for some value of tau!"
    #TODO FIX THIS!  don't just divide these by one: -- why not?
    added[added==0] = 1
    corr[:,1] /= added
    if tnormalize:
        plfunc(corr[:,0],corr[:,1]/corr[:,0]**tnormalize,
                'ko', label="<{}>/Time{}".format(formula,
                    "^{}".format(tnormalize) if tnormalize != 1 else ''))
        if dtau > 0:
            plfunc(taus, corr[0,1]*taus**(1-tnormalize)/dtau,
                    'k-',label="ref slope = 1",lw=4)
            plfunc(taus, (1.0*taus)**(-tnormalize),
                    'k--', label="One "+corrunit, lw=2)
        pl.ylim([0,1.5*np.max(corr[:,1]/corr[:,0]**tnormalize)])
    else:
        plf(corr[:,0],corr[:,1],'ko',label='<'+formula+'>')
        if dtau > 0:
            plf(taus, corr[0,1]*taus/dtau,
                    'k-',label="ref slope = 1",lw=4)
            plf(taus, np.ones_like(taus),
                    'k--', label="One "+corrunit)
    if dtau<0:
        pl.xlim([-500,500])
    pl.legend(loc=2 if tnormalize else 4)
    pl.title(prefix+' <'+formula+'>\ndt0=%d dtau=%d'%(dt0, dtau))
    pl.xlabel('Time tau (Image frames)')
    pl.ylabel('Squared'+(' Angular ' if 'radian' in corrunit else ' ')+'Displacement ('+corrunit+')')
    pl.savefig(locdir+prefix+"_MSD.png", dpi=180)
    pl.show()

if is_main and plotcorr and computer is 'rock':
    print 'plotting now!'
    plot_corr(data, corr)

