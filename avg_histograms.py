#!/usr/bin/env python
import sys
import numpy as np
from matplotlib import pyplot as plt

if len(sys.argv) < 3:
    print("Format: avg_histograms.py [file base] [psi/densities] [log]?")
    sys.exit(0)

ORIENTATIONS = [("aligned", 3), ("inward", 3), ("outward", 3), ("random", 4)]

fname, stat = sys.argv[1], sys.argv[2]

MEDIAN = True
if MEDIAN:
    avg_func = lambda x: sorted(x)[len(x) / 2]
else:
    avg_func = lambda x: sum(x) / len(x)

for orient, extra_trials in ORIENTATIONS:
    y_list = []
    for i in range(extra_trials):
        if i==0:
            a = fname + "_{0}_DATA.npz".format(orient)
        else:
            a = fname + "_{0}_trial{1}_DATA.npz".format(orient, i+1)
        data = np.load(a)
        try:
            data = data[stat]
        except KeyError:
            raise Exception("Invalid statistic")
        y = [avg_func(x) for x in data]
        y_list.append(np.asarray(y))
    shortest = min([len(x) for x in y_list])
    y_list = [x[:shortest] for x in y_list]
    l = sum(y_list) / len(y_list)
    if 'log' in sys.argv:
        plt.plot(np.log(sum(y_list) / len(y_list)), label=orient)
    else:
        plt.plot(sum(y_list) / len(y_list), label=orient)

plt.title("{0} crystal".format(fname))
plt.xlabel("Time (frames)")
name = stat.replace("_", " ")
if 'log' in sys.argv:
    plt.ylabel('Log [{0}]'.format(name))
else:
    plt.ylabel(name[0].upper() + name[1:])
plt.legend(loc='upper right')
plt.savefig(fname + "_" + stat + ".png")
plt.show()
