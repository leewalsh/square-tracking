from __future__ import division
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as pl
import matplotlib.cm as cm

ns = np.array([8,16,32,64,128,192,256,320,336,352,368,384,400,416,432,448])
nframes = 3000
S = 22 # side length in pixels
A = S**2
N = 520
fps = 150
locdir = '/Users/leewalsh/Physics/Squares/spatial_diffusion/'  #rock

def powerlaw(t,d):
    # to allow fits for b and/or c,
    # then add them as args to function and delete them below.
    b = 1
    c = 0
    return c + d * t ** b

def diff_const(msd):
    popt,pcov = curve_fit(powerlaw,*msd.transpose())
    #print "popt,pcov =",popt,',',pcov
    return popt[0], pcov[0,0]

dmin, dmax = ns.min()/N, ns.max()/N
col = lambda dens: cm.jet((dens - dmin)/(dmax - dmin))

ds = []
dds = []
dsd = 0**2 # uncert in squared disp
pl.figure(figsize=(6,5))
MSDS = np.load(locdir + 'MSDS.npz')
#MSDS = {}
for n in ns:
    if str(n) in MSDS.keys():
        print "using dict saved for ", n
        msd = MSDS[str(n)]
        pl.loglog(msd[:,0]/fps, (msd[:,1]-dsd)/A,
                  c=col(n/N), lw=2)
        d,dd = diff_const(msd/np.array([fps, A]))
        ds.append(d)
        dds.append(dd)
        continue
    prefix = 'n'+str(n)
    print "loading from", prefix+"_MSD.npz"
    msdnpz = np.load(locdir+prefix+"_MSD.npz")
    msds = np.array(msdnpz['msds'])
    if 'dt0' in msdnpz.keys():
        dt0  = msdnpz['dt0'][()] # [()] gets element from 0D array
        dtau = msdnpz['dtau'][()]
    else:
        print "assuming dt0 = dtau = 10"
        #  should be true for all from before dt* was saved
        dt0  = 10
        dtau = 10
    #nframes = max(np.array(msds[0])[:,0])
    taus = np.arange(dtau,nframes,dtau)
    msd = [taus,np.zeros_like(taus)]
    #msd = [np.arange(dtau,nframes,dtau),np.zeros(-(-nframes/dtau) - 1)]
    msd = np.transpose(msd)
    added = 0
    print "averaging track msds"
    for tmsd in msds:
        tmsd = np.array(tmsd)
        if tmsd[:,1].any():
            #pl.loglog(zip(*tmsd)[0],zip(*tmsd)[1])
            if len(tmsd)>=len(msd):
            #    print "adding a tmsd for n = ",n
            #    print "\ttmsd: ",len(tmsd)," msd: ",len(msd)
                added += 1.0
                msd[:,1] += tmsd[:len(msd),1]
            #else:
            #    print "skipping a tmsd for n = ",n
            #    print "\ttmsd: ",len(tmsd)," msd: ",len(msd)
                #for tmsdrow in tmsd:
                #    print 'tmsdrow',tmsdrow
                #    print 'msd[(tmsdrow[0]==msd[:,0])[0],1]',msd[(tmsdrow[0]==msd[:,0])[0],1]
                #    print 'tmsdrow[1]',tmsdrow[1]
                #    #msd[(tmsdrow[0]==msd[:,0])[0],1] += tmsdrow[1]

    if added:
        msd[:,1] /= added
        MSDS[str(n)] = msd
        pl.loglog(msd[:,0]/fps, (msd[:,1]-dsd)/A,
                c = col(n/N)#, label="%.1f%s" % (n/5.2,'%')
                )
        d,dd = diff_const(msd/np.array([fps, A]))
        ds.append(d)
        dds.append(dd)
    else:
        print "no msd for n =", n
        ds.append(np.nan)
        dds.append(np.nan)

#ds = np.asarray(ds)
#dds = np.asarray(dds)
#pl.loglog(
#        np.arange(dtau,nframes,dtau)/fps,
#        10*np.arange(dtau,nframes,dtau)/fps/dtau,
#        'k-',label="ref slope = 1")
try:
    dtau
except NameError:
    dtau = dt0 = 10
for n, d in zip(ns, ds):
    pl.loglog(np.arange(dtau, nframes, dtau)/fps,
              3*powerlaw(np.arange(dtau, nframes, dtau)/fps, d), 'k--')
              #lw=.5, c=col(n/N))
    break # just plot the first one, times three, to show 'slope=1' guideline
#pl.legend(loc=4)
#pl.title("MSD v N, uncert in sq disp = "+str(dsd))
pl.xlabel(r'Time ({})'.format('s' if fps > 1 else 'image frames'), fontsize='x-large')
pl.ylabel('Squared Displacement ({})'.format('particle area' if fps > 1 else '$pixels^2$'), fontsize='x-large')
pl.xlim(dtau/fps, nframes/fps)
#savename = locdir+"MSDvN_dt0=%d_dtau=%d.png"%(dt0,dtau)
#savename = locdir + "MSDvN_dsd=%d.png"%dsd
savename = locdir + "MSDvN.pdf"
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
    #pl.savefig(locdir+"DvN_dt0=%d_dtau=%d.png"%(dt0,dtau))
    pl.ylim(0,np.max(ds)*1.2)
    pl.savefig(locdir+"DvN.pdf")

#pl.show()
