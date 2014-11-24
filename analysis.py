import numpy as np
import sys
import math
import os
from matplotlib import pyplot as plt

def analyze(input_file, output_file, label):
    particles = np.genfromtxt(input_file, dtype='i,f,f,i,f,i', names=True,
                          skip_header=3)
    variances = []

    if os.path.isfile(output_file):
        variances = np.load(output_file)
        plt.plot(variances, label=label)
        return
    for frame in range(particles[-1][0] + 1):
        print('Frame {0}'.format(frame))
        frame_particles = [row for row in particles if row[0] == frame]
        positions = [(row[1], row[2]) for row in frame_particles]
        center = (sum(p[0] for p in positions) / len(positions),
                  sum(p[1] for p in positions) / len(positions))
        distances = [math.sqrt((p[0] - center[0])**2 + (p[1] - center[1])**2)
                     for p in positions]
        avg_distance = sum(distances) / len(distances)
        variance = sum([(r - avg_distance)**2 for r in distances])
        variances.append(variance)
    np.save(output_file, variances)
    plt.plot(variances, label=label)

if len(sys.argv) > 1:
    input_files = [("POSITIONS_{0}.txt".format(x), "ANALYSIS_{0}.npy".format(x),
                    x) for x in sys.argv[1:]]
else:
    input_files = [("POSITIONS", "ANALYSIS.npy", "Data")]
for f, out_f, label in input_files:
    analyze(f, out_f, label)
plt.legend(loc='upper left')
plt.show()
