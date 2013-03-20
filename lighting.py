import numpy as np
import PIL.Image as image

from socket import gethostname
hostname = gethostname()
if 'foppl' in hostname:
    computer = 'foppl'
elif 'rock' in hostname:
    computer = 'rock'
else:
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
        nthreads = 8 if computer is 'foppl' else 2
    elif nthreads > 2 and computer is 'rock':
        nthreads = 2
    print "on {}, using {} threads".format(computer,nthreads)
    pool = Pool(nthreads)

    if prefix is not '':
        fs = [prefix + f for f in fs]

    return pool.map(im_mean, fs)

def plot_means(means,fps=150.,**kwargs):
    if computer is 'foppl':
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as pl

    #if computer is 'rock':
        #pl.figure()

    pl.plot(np.arange(len(means))/fps, means,**kwargs)
    pl.ylim(0,1.)

    if computer is 'rock':
        pl.show()
    elif computer is 'foppl':
        pl.savefig(fs[0][:-4]+'_means.pdf')



