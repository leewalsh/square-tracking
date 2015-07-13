#!/usr/bin/env python

from matplotlib import pyplot as plt
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Please specify an argument")
    sys.exit(0)
try:
    fname = ("CORNER_" if sys.argv[2].lower() == 'c' else "") + "POSITIONS"
except IndexError:
    fname = "POSITIONS"
data = np.genfromtxt(fname, dtype='i,f,f,i,f,i', names=True, skip_header=3)
if sys.argv[1].lower() == 'ecc':
    plt.hist(data["Eccen"], bins=35)
    plt.title("Eccentricities")
elif sys.argv[1].lower() == 'area':
    plt.hist(data["Area"], bins=35)
    plt.title("Area")
else:
    print("Unknown argument")
plt.show()
