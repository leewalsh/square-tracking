from __future__ import division
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as pl
import matplotlib.cm as cm

from tracks import mean_msd

#ns = np.array([8,16,32,64,128,192,256,320,336,352,368,384,400,416,432,448])
ns = np.array([ 16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192, 208,
               224, 240, 256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336,
               344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432, 440,
               448, 456, 464])

# Physical measurements
R_inch = 4.0           # as machined
S_measured = np.array([4,3,6,7,9,1,9,0,0,4,7,5,3,6,2,6,0,8,8,4,3,4,0,-1,0,1,7,7,5,7])*1e-4 + .309
S_inch = S_measured.mean()

# Digital measurements
R_pix = 585.5 / 2
#S_pix = 22 #ish

# What we'll use:
R = R_inch / S_inch # radius in particle units
S = R_pix/R # particle side length in pixels
A = S**2
N = np.pi * R**2

# Time
fps = 120 #150
#nframes = 3000
#locdir = '/Users/leewalsh/Physics/Squares/spatial_diffusion/'  #rock
locdir = '/Users/leewalsh/Physics/Squares/diffusion/orientational/'  #rock
Ang = '' # 'A' if angular, '' otherwise, converts between MSD and MSAD
A = 1 if Ang else S**2
msd_file = raw_input('load, save, or anything else to do neither: ').lower()

def powerlaw(t, d):
    # to allow fits for b and/or c,
    # then add them as args to function and delete them below.
    b = 1
    c = 0
    return c + d * t**b

def diff_const(taus, msd):
    msd_p = np.convolve
    # find plateau
    # fit to constant (just find its mean)
    popt, pcov = curve_fit(powerlaw, taus, msd)
    #print "popt, pcov = {}, {}".format(popt, pcov)
    return popt[0], pcov[0,0]

dmin, dmax = ns.min()/N, ns.max()/N
col = lambda dens: cm.jet((dens - dmin)/(dmax - dmin))

ds = []
dds = []
dsd = 0**2 # uncert in squared disp
pl.figure(figsize=(6,5))
if msd_file=='load':
    #MSDS = np.load(locdir + 'MS'+Ang+'DS.npz')
    #msd_loadname = locdir + "n{:03d}_MS{}D.npz".format(n, Ang)
    msd_loadname = locdir + "MS{}DS.npz".format(Ang)
    print "loading all averaged MS{}DS from {}".format(Ang, msd_loadname)
    #MSDS = { n: np.load(msd_loadname) for n in ns }
    MSDS = np.load(msd_loadname)
else:
    print "Will average all MS{}DS".format(Ang)
    MSDS = {}
for n in ns:
    if str(n) in (MSDS.files if hasattr(MSDS, 'files') else MSDS.keys()):
        print "using dict saved for ", n
        taus, msd = MSDS[str(n)].T
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
        msd = mean_msd(msds, taus)
        MSDS[str(n)] = msd

    pl.loglog(taus/fps, (msd-dsd)/A, c = col(n/N))#, label="%.1f%s" % (n/5.2,'%'))
    pl.loglog(taus/fps, (msd-dsd)/A/taus, c=col(n/N), lw=2)
    d, dd = diff_const(taus/fps, msd/A)
    ds.append(d)
    dds.append(dd)

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
pl.loglog(taus/fps, 3*powerlaw(taus/fps, ds[0]), 'k--')#, lw=.5, c=col(n/N))
#pl.legend(loc=4)
#pl.title("MS"+Ang+"D v N, uncert in sq disp = "+str(dsd))
pl.xlabel(r'Time ({})'.format('s' if fps > 1 else 'image frames'), fontsize='x-large')
pl.ylabel('Squared {}Displacement ({})'.format('Angular ' if Ang else '',
                                                'particle area' if fps > 1 else '$pixels^2$'),
          fontsize='large')
pl.xlim(dtau/fps, nframes/fps)
savename = locdir + "MS"+Ang+"DvN.pdf"
# = locdir+"MS"+Ang+"DvN_dt0=%d_dtau=%d.png"%(dt0,dtau)
# = locdir + "MS"+Ang+"DvN_dsd=%d.png"%dsd
if raw_input('save figure at {}? y/[n]'.format(savename)).startswith('y'):
    pl.savefig(savename)

plotD = raw_input('Plot D? y/[n]').startswith('y')
if plotD:
    pl.figure(figsize=(6,5))
    #pl.plot(ns/N, ds, 'o')
    pl.errorbar(ns/N, ds, np.sqrt(np.sqrt(dds))*A/fps, fmt='.') #really?
    #pl.title("Constant of diffusion vs. density")
    pl.xlabel(r"Density $\rho$", fontsize='x-large')
    pl.ylabel("$D$ "+('(particle area / s)' if fps > 1 else r"$pix^2/frame$"), fontsize='x-large')
    pl.ylim(0,np.max(ds)*1.2)
    pl.savefig(locdir+"DvN.pdf")#"DvN_dt0=%d_dtau=%d.png"%(dt0,dtau))

#pl.show()
