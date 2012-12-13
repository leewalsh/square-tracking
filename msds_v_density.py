import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as pl
import matplotlib.cm as cm

ns = np.array([8,16,32,64,128,192,256,320,336,352,368,384,400,416,432,448])
nframes = 3000

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
    return popt[0], pcov[0]

ds = []
dsd = 0**2 # uncert in squared disp
pl.figure()
for n in ns:
    prefix = 'n'+str(n)
    print "loading ",prefix
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
        pl.loglog(msd[:,0],msd[:,1]-dsd,
                marker = 'o',linestyle = '',
                c = cm.jet(np.nonzero(ns==n)[0][0]*255/len(ns)),
                label="%.1f%s" % (n/5.1,'%')
                )
        d,dd = diff_const(msd)
        ds.append(d)
    else:
        print "no msd for n = ",n

ds = np.array(ds)
#pl.loglog(
#        np.arange(dtau,nframes,dtau),
#        10*np.arange(dtau,nframes,dtau)/dtau,
#        'k-',label="ref slope = 1")
for n,d in enumerate(ds):
    pl.loglog(
            np.arange(dtau,nframes,dtau),
            powerlaw(np.arange(dtau,nframes,dtau),d),
            marker = ',',c = cm.jet(n*255/len(ds))
            #,label="d = "+str(d)+"n = "+str(ns[n])
            )
pl.legend(loc=4)
pl.title("MSD v N, uncert in sq disp = "+str(dsd))
pl.xlabel('Time tau (Image frames)')
pl.ylabel('Squared Displacement ('+r'$pixels^2$'+')')
#pl.savefig(locdir+"MSDvN_dt0=%d_dtau=%d.png"%(dt0,dtau))
pl.savefig(locdir+"MSDvN_dsd=%d.png"%dsd)

plotD = False
if plotD:
    pl.figure()
    pl.plot(ns/5.06,ds,'o')
    pl.title("Constant of diffusion vs. density")
    pl.xlabel("Packing fraction")
    pl.ylabel("Diffusion constant: \n"+r"$pix^2/frame$")
    pl.savefig(locdir+"DvN_dt0=%d_dtau=%d.png"%(dt0,dtau))

pl.show()
