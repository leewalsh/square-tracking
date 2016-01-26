import numpy as np
import PIL.Image as image

import helpy

HOST = helpy.gethost()

if HOST not in ['foppl', 'rock']:
    print "computer undefined"
    print "where are you working?"

def im_mean(imf):
    im = image.open(imf)
    im = np.array(im)
    im = im / 255.
    
    return np.mean(im)

def get_means(fs, prefix='', nthreads=None):
    from multiprocessing import Pool
    if nthreads is None or nthreads > 8:
        nthreads = 8 if HOST is 'foppl' else 2
    elif nthreads > 2 and HOST is 'rock':
        nthreads = 2
    print "on {}, using {} threads".format(HOST,nthreads)
    pool = Pool(nthreads)

    if prefix is not '':
        fs = [prefix + f for f in fs]

    return pool.map(im_mean, fs)

def plot_means(means,fps=150.,**kwargs):
    if HOST is 'foppl':
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as pl

    #if HOST is 'rock':
        #pl.figure()

    pl.plot(np.arange(len(means))/fps, means,**kwargs)
    pl.ylim(0,1.)

    if HOST is 'rock':
        pl.show()
    elif HOST is 'foppl':
        pl.savefig(fs[0][:-4]+'_means.pdf')



