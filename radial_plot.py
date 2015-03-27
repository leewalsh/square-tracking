#!/usr/bin/env python
import sys
import numpy as np
from matplotlib import pyplot as plt

if len(sys.argv) < 3:
    print("Format: radial_plot.py [file base] [psi/densities/r/speed/msd] "
          "[avg]? [log]?")
    sys.exit(0)

extra_params = sys.argv[3:]
avg = 'avg' in extra_params
avg_test = 'avg_test' in extra_params
log = 'log' in extra_params
normalize_options = ['r']#, 'speed', 'msd']
normalize = sys.argv[2] in normalize_options

if avg_test:
    test_n = int(sys.argv[4])
    avg = True
all_s_plots = []
TRIALS = {"aligned": 3,
          "inward": 3,
          "outward": 3,
          "random": 4}
fnames = [sys.argv[1] + ("_trial{0}".format(i+1) if i > 0 else "")
          for i in range(TRIALS[sys.argv[1][4:]] if avg else 1)]
stat = 'radial_{0}'.format(sys.argv[2])
fig = plt.figure()
ax = plt.subplot(111)
fig = plt.gcf()
fig.canvas.set_window_title(sys.argv[1])

for k, fname in enumerate(fnames):
    try:
        data = np.load(fname + "_DATA.npz")[stat]
    except KeyError:
        raise Exception("Invalid radial statistic")

    valencies = len(data[0])
    plots = [[x[i] if x[i]>=0. else None for x in data] for i in range(valencies)]
    s_plots = []
    for plot in plots:
        s_plots.append(plot/plot[0] if normalize else np.asarray(plot))
    all_s_plots.append(s_plots)
    if avg_test:
        l = s_plots[test_n-1]
        if log:
            l = np.log(l)
        ax.plot(l, label="Trial {0}".format(k + 1))

s_plots = [[x[i] for x in all_s_plots] for i in range(len(all_s_plots[0]))]
n_frames = min([len(x) for x in s_plots[0]])
avg_s_plots = [sum([y[:n_frames] for y in x])/len(x) for x in s_plots]

extra = False #(stat in ('radial_r', 'radial_speed'))

if not avg_test:
    for i, plot in enumerate(avg_s_plots):
        l = plot
        if log:
            l = np.log(l)
        ax.plot(l, label="Valency {0}".format(i if extra else i+1))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.xlabel("Time (frames)")
ax.legend(loc='center left', bbox_to_anchor=(1,.5))
name = 'R'+stat[1:].replace('_', ' ')
plt.title("{0} vs time".format(name))
if log:
    name = 'Log [{0}]'.format(name)
if normalize:
    name += " (normalized)"
plt.ylabel(name)
#plt.show()
plt.savefig(sys.argv[1]+"_"+sys.argv[2]+".png")
