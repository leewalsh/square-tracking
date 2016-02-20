from __future__ import division
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from collections import defaultdict

from tracks import mean_msd, diff_const

### SETTINGS ###
Ang = 'A'           # 'A' if angular, '' otherwise, converts between MSD and MSAD
msd_file = 'load'   #raw_input('load, save, or anything else to do neither: ').lower()
plotD = True        #raw_input('Plot D? y/[n]').startswith('y')

#ns = np.array([  8,  16,  32,  64, 128, 192, 256, 320, 336, 352, 368, 384,
#               400, 416, 432, 448])
ns = np.array([ 16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192, 208,
               224, 240, 256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336,
               344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432, 440,
               448, 456, 464])
if Ang:
    nmask = ns<=328
else:
    nmask = ns<=368
ns = ns[nmask]
# Physical measurements
R_inch = 4.0           # as machined
S_measured = np.array([4,3,6,7,9,1,9,0,0,4,7,5,3,6,2,6,0,8,8,4,3,4,0,-1,0,1,7,7,5,7])*1e-4 + .309
S_inch = S_measured.mean()

# Digital measurements
R_pix = 585.5 / 2
#S_pix = 22 #ish

# What we'll use:
R = R_inch / S_inch # radius in particle units
S = R_pix/R         # particle side length in pixels
A = 1 if Ang else S**2 # particle area
N = np.pi * R**2    # maximum number of particles

# Time
fps = 120 #150
#nframes = 3000
locdir = '/Users/leewalsh/Physics/Squares/diffusion/orientational/'  #rock

def powerlaw(t, d, b=1, c=0):
    return c + d * np.power(t, b)

def diff_const(taus, msd, tau_start=None, tau_end=None, msd_err=None, fit=True, nargs=1):
    if tau_start == 'min': tau_start = np.argmin(msd/taus)
    if tau_end  ==  'min': tau_end  =  np.argmin(msd/taus)
    start = np.searchsorted(taus, tau_start) if tau_start else None
    end = np.searchsorted(taus, tau_end) if tau_end else None
    if fit:
        return curve_fit(powerlaw, taus[start:end], msd[start:end],
                         [1]*nargs, msd_err[start:end], True)
    else:
        mt = msd[start:end]/taus[start:end]
        w = 1/msd_err[start:end] if msd_err is not None else None
        d  = np.average(mt, weights=w)
        dd = np.average((mt - d)**2, weights=w)
        return d, dd

dmin, dmax = ns.min()/N, ns.max()/N
col = lambda dens: cm.jet((dens - dmin)/(dmax - dmin))

ds, bs, dds, dbs = [], [], [], []
dsd = 0**2 # uncert in squared disp (in pixels)
pl.figure(figsize=(6,5))

if Ang:
    kill_flats = defaultdict(lambda: 0, {48: 5, 344: 0.03})
    kill_jumps = defaultdict(lambda: 1,
                {16: 10, 32: 5, 48: 1, 64: 2, 80: 1, 96: 1, 344: .5, 448: 0.1,
                 456: 0.1, 464: 0.1})
else:
    kill_flats = defaultdict(lambda: 0.5, # units are pixels
                    { 16: 1, 32: 3, 48: 1, 64: 0.5, 80: 1, 96: 0.2, 112: 0.7,
                     128: 0.6, 144: 1, 160: 10, 176: 1, 192: 0.4, 336: 1,
                     360: 0.1, 368: .6, 400: 0.2, 464: 0.2 })
    kill_jumps = defaultdict(lambda: 100, {})
if msd_file=='load':
    msd_loadname = locdir + "MS{}DS.npz".format(Ang)
    print "loading all averaged MS{}DS from {}".format(Ang, msd_loadname)
    MSDS = np.load(msd_loadname)
else:
    print "Will average all MS{}DS".format(Ang)
    MSDS = {}
for n in ns:
    if str(n) in (MSDS.files if hasattr(MSDS, 'files') else MSDS.keys()):
        try:
            taus, msd, msd_err = MSDS[str(n)]
        except ValueError:
            taus, msd = MSDS[str(n)]
    else:
        prefix = 'n{:03d}'.format(n)
        print "loading from", prefix+"_MS"+Ang+"D.npz"
        msdnpz = np.load(locdir+prefix+"_MS"+Ang+"D.npz")
        msds = msdnpz['msds']
        if 'dt0' in msdnpz.keys():
            dt0  = msdnpz['dt0'][()] # [()] gets element from 0D array
            dtau = msdnpz['dtau'][()]
        else:
            print "assuming dt0 = dtau = 10" #  should be true for all from before dt* was saved
            dt0  = 10
            dtau = 10
        print "averaging track MS"+Ang+"Ds"
        nframes = max([np.array(msd)[:,0].max() for msd in msds]) + 1
        taus = np.arange(dtau, nframes, dtau)
        taus, msd, msd_err = mean_msd(msds, taus, errorbars=True,
                                      kill_flats=kill_flats[n],
                                      kill_jumps=kill_jumps[n])
        MSDS[str(n)] = np.row_stack([taus, msd, msd_err])

    # plot them
    #pl.loglog(taus/fps, (msd-dsd)/A, c = col(n/N))
    pl.loglog(taus/fps, (msd-dsd)/A/taus*fps, c=col(n/N))
    #pl.semilogy(taus/fps, (msd-dsd)/A/taus*fps, c=col(n/N))

    # find coeffient of diffusion
    if plotD:
        tau_start = None if Ang else 10
        tau_end   = 2    if Ang else None
        d, dd = diff_const(taus/fps, msd/A, tau_start, tau_end, msd_err/A, fit=True)
        if len(np.atleast_1d(d)) > 1:
            d, b = d
            dd, db = np.diag(dd)
            bs.append(b)
            dbs.append(db)
        else:
            b = 1
        ds.append(d)
        dds.append(dd)

        pl.loglog(taus/fps, powerlaw(taus/fps, d, b)/taus*fps, '--', lw=.5, c=col(n/N))

if msd_file=='save':
    #msd_savename = locdir + "n{:03d}_MS{}D.npz".format(n, Ang)
    msd_savename = locdir + 'MS'+Ang+'DS'
    print 'saving MS{}DS to {}'.format(Ang, msd_savename)
    np.savez(msd_savename, **MSDS)
#pl.loglog(taus/fps, 10*taus/fps/dtau,
#          'k-', label="ref slope = 1")
try:
    dtau
except NameError:
    print 'hi!'
    dtau = dt0 = 10
# plot the first one, times three, to show 'slope=1' guideline
#pl.loglog(taus/fps, 3*powerlaw(taus/fps, ds[0]), 'k--')#, lw=.5, c=col(n/N))
#pl.legend(loc=4)
#pl.title("MS"+Ang+"D v N, uncert in sq disp = "+str(dsd))
pl.xlabel(r'Time ({})'.format('s' if fps > 1 else 'image frames'), fontsize='x-large')
pl.ylabel('Squared {}Displacement ({})'.format('Angular ' if Ang else '',
                                                'particle area' if fps > 1 else '$pixels^2$'),
          fontsize='large')
pl.xlim(dtau/fps, taus[-1]/fps)
savename = locdir + "MS"+Ang+"DvN.pdf"
# = locdir+"MS"+Ang+"DvN_dt0=%d_dtau=%d.png"%(dt0,dtau)
# = locdir + "MS"+Ang+"DvN_dsd=%d.png"%dsd
save_in = raw_input('save figure at {}? y/suffix/[n]'.format(savename))
if save_in == 'y':
    pl.savefig(savename)
elif save_in:
    pl.savefig(savename[:-4] + '_' + save_in + savename[-4:])

if plotD:
    ds, dds = map(np.squeeze, (ds, dds))
    pl.figure(figsize=(6,5))
    if bs:
        pl.errorbar(ns/N, bs, np.sqrt(dbs), fmt='.', label='b')
        if Ang: bs_ang, dbs_ang = bs, dbs
        else:   bs_tr, dbs_tr = bs, dbs
    pl.errorbar(ns/N, ds, np.sqrt(dds)*A/fps, fmt='.', label='D')
    #pl.title("Constant of diffusion vs. density")
    pl.xlabel(r"Density $\rho$", fontsize='x-large')
    unit = '(radians ^ 2' if Ang else '(particle area'
    pl.ylabel("$D$ "+(unit+' / s)' if fps > 1 else r"$pix^2/frame$"), fontsize='x-large')
    pl.ylim(0,None if Ang else 1)#np.max(ds)*1.2)
    #pl.gca().set_yscale('log')
    pl.legend(loc='best')
    pl.savefig(locdir+"D"+Ang+"vN.pdf")#"DvN_dt0=%d_dtau=%d.png"%(dt0,dtau))
    if Ang: ds_ang, dds_ang = ds, dds
    else:   ds_tr, dds_tr = ds, dds

pl.show()
