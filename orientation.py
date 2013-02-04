import matplotlib.pyplot as pl
import matplotlib.cm as cm
import numpy as np
#from scipy.stats import nanmean
from PIL import Image as Im

def get_fft(ifile=None,location=None):
    # FFT information from:
    # http://stackoverflow.com/questions/2652415/fft-and-array-to-image-image-to-array-conversion
    if ifile is None:
        ifile= "n20_bw_dots/n20_b_w_dots_0010.tif"
    if location is None:
        x = 424.66; y = 155.56; area = 125
        #x = 302.95; y = 221.87; area = 145
        #x = 386.27; y = 263.62; area = 141
        #x = 35.39; y = 305.92; area = 154
        location = x,y
    wdth = int(24 * np.sqrt(2))
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
    if do_plots:
        ii.show()

    return b

def get_orientation(b):
    p = []
    for (mi,m) in enumerate(b):
        for (ni, n) in enumerate(m):
            ang = np.arctan2(mi - hght/2, ni - wdth/2)
            p.append([ang,abs(n)])
    p = np.asarray(p)
    p[:,0] = p[:,0] + np.pi
    slices = 45
    slicewidth = 2*np.pi/slices
    s = []
    for sl in range(slices):
        si = np.nonzero(abs(p[:,0] - sl*slicewidth) < slicewidth)
        sm = np.average(p[si,1])
        s.append([sl*slicewidth,sm])
    s = np.asarray(s)
    if do_plots:
        pl.figure()
        #pl.plot(p[:,0],p[:,1],'.',label='p')
        #pl.plot(s[:,0],s[:,1],'o',label='s')
        pl.plot(p[:,0]%(np.pi/2),p[:,1],'.',label='p')
        pl.plot(s[:,0]%(np.pi/2),s[:,1],'o',label='s')
        pl.legend()
        pl.show()
    return s, p

def find_corner(particle,corners,rc=11,drc=2):
    """ find_corner(particle,corners,rc=11,drc=2)

        looks in the given frame for the corner-marking dot closest to (and in
        appropriate range of) the particle

        arguments:
            particle - is particle position as (x,y) tuple
            corners  - is zipped list of positions of corner dots
                        as (x,y) vector tuples
            rc       - is the expected distance to corner from particle position
            drc      - delta r_c is the tolerance on rc

        returns:
            pcorner - position (x,y) of corner that belongs to particle
            porient - particle orientation (% 2pi)
            cdisp   - vector (x,y) from particle center to corner
    """
    from numpy.linalg import norm

    cdisps = np.array([
        (corner[0]-particle[0], corner[1]-particle[1])
        for corner in corners])
    cdists = np.array(map(norm, cdisps))
    legal = np.array(abs(cdists-rc) < drc, dtype=bool)

    if legal.sum() == 1:
        pcorner = tuple(np.asarray(corners)[legal].flatten())
        cdisp = cdisps[legal].flatten()
        #print "Bingo: one corner found at {}".format(cdists[legal])
    elif legal.sum() > 1:
        #print "Too many ({}) legal corners for this particle".format(legal.sum())
        legal = np.argmin(abs(cdists-rc))
        #print "Using closest to rc = {} pixels at {}".format(rc, cdists[legal])
        pcorner = corners[legal]
        cdisp = cdisps[legal].flatten()
    else:
        #print "No legal corners for this particle"
        return None, None, None

    porient = np.arctan2(cdisp[1],cdisp[0]) % (2*np.pi)

    return pcorner,porient,cdisp

def get_angles(data,cdata):#,framestep=100):
    """ get_angles(data,cdata)
        
        arguments:
            data    - data array with 'x' and 'y' fields for particle centers
            cdata   - data array wity 'x' and 'y' fields for corners
            (these arrays need not have the same length,
                but both must have 'f' field for the image frame)
            
        returns:
            odata   - array with fields:
                'orient' for orientation of particles
                'corner' for particle corner (with 'x' and 'y' sub-fields)
            (odata has the same shape as data)
    """
    from correlation import get_id
    if 's' in data.dtype.names:
        fieldnames = np.array(data.dtype.names)
        fieldnames[fieldnames == 's'] = 'f'
        data.dtype.names = tuple(fieldnames)
    if 's' in cdata.dtype.names:
        fieldnames = np.array(cdata.dtype.names)
        fieldnames[fieldnames == 's'] = 'f'
        cdata.dtype.names = tuple(fieldnames)
    #frames = np.arange(min(data['f']),max(data['f']),framestep)
    #postype = np.dtype()
    odata = np.zeros(len(data),dtype=[('corner',float,(2,)),('orient',float),('cdisp',float,(2,))])
    orients = []
    frame=0
    for datum in data:
        if frame != datum['f']:
            frame=datum['f']
            print 'frame',frame
        posi = (datum['x'],datum['y'])
        iid = get_id(data,posi,datum['f'])
        icorner, iorient, idisp = find_corner( posi,
                zip(cdata['x'][cdata['f']==datum['f']],
                    cdata['y'][cdata['f']==datum['f']])
                )
        imask = data['id']==iid
        odata['corner'][imask] = icorner
        odata['orient'][imask] = iorient
        odata['cdisp'][imask] = idisp
    return odata

def plot_orient_hist(odata,figtitle=''):

    pl.figure()
    pl.hist(odata['orient'][np.isfinite(odata['orient'])], bins=90)
    pl.title('orientation histogram' if figtitle is '' else figtitle)


def plot_orient_map(data,odata,imfile='',mask=None):
    import matplotlib.colors as mcolors
    import matplotlib.colorbar as mcolorbar
    pl.figure()
    if imfile is not None:
        bgimage = Im.open(extdir+prefix+'_0001.tif' if imfile is '' else imfile)
        pl.imshow(bgimage, cmap=cm.gray, origin='lower')
    #pl.quiver(X, Y, U, V, **kw)
    omask = np.isfinite(odata['orient'])
    if mask is None:
        mask=omask
    nz = mcolors.Normalize()
    nz.autoscale(data['f'][mask])
    pl.quiver(data['x'][mask], data['y'][mask],
            odata['cdisp'][mask][:,0], odata['cdisp'][mask][:,1],
            color=cm.jet(nz(data['f'][mask])),
            scale=400.)
    cax,_ = mcolorbar.make_axes(pl.gca())
    cb = mcolorbar.ColorbarBase(cax, cmap=cm.jet, norm=nz)
    cb.set_label('time')

def plot_orient_time(data,odata,tracks):
    omask = np.isfinite(odata['orient'])
    goodtracks = np.array([78,95,191,203,322])
    #tmask = np.in1d(tracks,goodtracks)
    pl.figure()
    for goodtrack in goodtracks:
        tmask = tracks == goodtrack
        fullmask = np.all(np.asarray(zip(omask,tmask)),axis=1)
        pl.plot(data['f'][fullmask],
                (odata['orient'][fullmask]
                 - odata['orient'][fullmask][np.argmin(data['f'][fullmask])]
                 + np.pi)%(2*np.pi),
                label="Track {}".format(goodtrack))
                #color=cm.jet(1.*tracks[fullmask]/max(tracks)))
                #color=cm.jet(1.*goodtrack/max(tracks)))
    for n in np.arange(0.5,2.0,0.5):
        pl.plot([0,1260],n*np.pi*np.array([1,1]),'k--')
    pl.legend()
    pl.title('Orientation over time\ninitial orientation = 0')
    pl.xlabel('frame (150fps)')
    pl.ylabel('orientation')

def plot_orient_location(data,odata,tracks):
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
        pl.scatter(
                (odata['orient'][fullmask] - orient_start + np.pi) % (2*np.pi),
                np.asarray(map(corr.get_norm,
                    zip([loc_start]*fullmask.sum(),
                        zip(data['x'][fullmask],data['y'][fullmask]))
                    ))/ss,
                #marker='*',
                label = 'track {}'.format(goodtrack),
                color = cm.jet(1.*goodtrack/max(tracks)))
                #color = cm.jet(1.*data['f'][fullmask]/1260.))
        print "track",goodtrack
    pl.legend()



