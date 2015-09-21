''' This script plots a histogram of the orientation data for a one or several 
data sets. The histogram is saved in the format prefix.plothist.pdf

Run from the folder containing the positions file.

Sarah Schlossberg
August 2015
'''

from __future__ import division
from sys import path
path.append('C:\\Users\\Sarah\\Desktop\\tracking\\square-tracking')
import helpy
import os
import numpy as np
from matplotlib import pyplot as plt
import math

trackcount = 0
prefix = raw_input('Prefix without trial number: ')
histv = []
histbins = 100
sets = raw_input('How many sets? ')
sets = int(sets)
particle = raw_input('Particle type: ')

if sets > 1: 
    for setnum in range(1, sets+1):      # changed to 7 for testing
        spfprefix = prefix + str(setnum)
        print spfprefix
        os.chdir('..\\' + spfprefix + '_d')
        data, trackids, odata, omask = helpy.load_data(spfprefix)
        data = data[omask]
        odata = odata[omask]
        print odata
        
        for track in range(data['lab'].max()):
            tdata = odata[data['lab']==track]
            vx = helpy.der(tdata['orient'], iwidth=3)
            if len(vx) > 0:   
                histv = np.concatenate((histv, vx), axis=1)
                trackcount = trackcount + 1
            
elif sets == 1:
    spfprefix = prefix
    data, trackids, odata, omask = helpy.load_data(spfprefix)
    data = data[omask]
    odata = odata[omask]
    
    for track in range(data['lab'].max()):
        tdata = odata[data['lab']==track]
        vx = helpy.der(tdata['orient'], iwidth=3)
        if len(vx) > 0:   
            histv = np.concatenate((histv, vx), axis=1)
            trackcount = trackcount + 1
            
histv *= 2.4
mean = np.mean(histv)
variance = np.var(histv)
D_R = 0.5*variance
SE = np.sqrt(variance) / np.sqrt(len(histv))

fig = plt.figure()
plt.hist(histv, int(histbins), log=True, color=['blue'], label=['Mean = {:.5f} \n $D_R$ = {:.5f} \n Standard error = {:.5f}'.format(mean, D_R, SE)])
plt.legend(loc='center left')
plt.xlabel('Velocity step size in rad/frame')
plt.ylabel('Frequency')
plt.title("{} orientation tracks of {} ({})".format(trackcount, prefix, particle))
        
print 'Saving plot to C:Users\Sarah\Dropbox\\0SquareTrackingData\\{}.plothistorient'.format(spfprefix)
fig.savefig(spfprefix+'.plothistorient.pdf')
plt.show() 