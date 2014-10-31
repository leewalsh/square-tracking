from matplotlib import pyplot as plt
import numpy as np
import sys

fname = sys.argv[2] if len(sys.argv) > 2 else "POSITIONS"
data = np.genfromtxt(fname, dtype='i,f,f,i,f,i', names=True)
if sys.argv[1].lower() == 'ecc':
    plt.hist(data["Eccen"], bins=35)
    plt.title("Eccentricities")
elif sys.argv[1].lower() == 'area':
    plt.hist(data["Area"], bins=35)
    plt.title("Area")
else:
    print("Unknown argument")
plt.show()
