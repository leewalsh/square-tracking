#!/usr/bin/env python
import sys
from matplotlib import pyplot as plt
import numpy as np
import pygame
from pygame.locals import QUIT

'''
Graphs a plot of histograms over time for a given statistic
(either psi or density). For example, if the statistic is
psi, at any horizontal position t (corresponding to time t in frames)
there will be a vertical strip with colors corresponding to the
frequency at which particles at time t have psi value within the
corresponding range. There is also a black curve corresponding to
either the mean psi value or the median density value. Append
"save" or "s" at the end of sys.argv to save the resultant plot
as PNG.

Example usage:
graph_histograms.py densities s
'''

def plot(data, width, height, bins=100, contrast=1,
         colormap='rainbow', avg='median'):
    '''
    data: 2-dimensional list of points at each frame
    Plots a graph of the histogram of 'data' over time
    Expects data to be in range [0, 1]
    If avg is not None, draw an average curve (mean or median).
    See http://matplotlib.org/examples/color/colormaps_reference.html
    for colormap examples.
    '''
    nframes = len(data)
    dx = width / nframes
    dy = 1. / bins
    cmap = plt.get_cmap(colormap)
    screen = pygame.display.set_mode((dx * nframes, height))
    screen.fill([x*255 for x in cmap(0)])

    for t in range(nframes):
        for b in range(bins):
            y = height * (1. - float(b + 1) / bins)
            data_min = float(b) / bins
            data_max = data_min + dy
            in_range = [x for x in data[t] if x > data_min and x < data_max]
            freq = float(len(in_range)) / len(data[t])
            color = [x * 255 for x in cmap(freq * contrast)]
            pygame.draw.rect(screen, color, (t * dx, y, dx, dy * height), 0)

    if avg is not None:
        if avg == 'mean':
            avg_func = lambda x: sum(x) / len(x)
        elif avg == 'median':
            avg_func = lambda x: sorted(x)[len(x) / 2]
        else: # ??
            raise Exception("avg must be either mean or median")
        points = [(t * dx, (1 - avg_func(x)) * height)
                  for t, x in enumerate(data)]
        pygame.draw.lines(screen, (0, 0, 0), False, points, 4)
    return screen

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Format: graph_histograms.py [file base] "
              "[psi/densities]"
              "[save(s)]?")
        sys.exit(0)

    pygame.init()
    font = pygame.font.Font(None, 50)
    fname, stat = sys.argv[1], sys.argv[2]
    pygame.display.set_caption("Plot of histogram of {0} over time".format(stat))
    data = np.load(fname + "_DATA.npz")
    width, height = (800, 800)
    if stat == 'rpsi':
        screen = plot(data['radial_psi'], width, height, contrast=3)
    elif stat == 'rdensities':
        screen = plot(data['radial_densities'], width, height, contrast=3)
    elif stat == 'psi':
        screen = plot(data['psi'], width, height, contrast=5, avg='mean')
    elif stat == 'densities':
        screen = plot(data['densities'], width, height, contrast=3, avg='median')
    title = font.render(fname.replace('_', ' '), 1, (255,255,255))
    screen.blit(title, (330, 20))
    pygame.display.flip()

    if len(sys.argv) > 3 and sys.argv[3] in ('s', 'save'):
        pygame.image.save(screen, fname + "_{0}.png".format(stat))
    else:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
