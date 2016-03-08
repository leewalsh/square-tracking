#!/usr/bin/env python
from __future__ import division

import sys

import numpy as np
from matplotlib import pyplot as plt
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


def plot(data, size, bins=100, contrast=1,
         colormap='rainbow', avg='median'):
    '''
    data: 2-dimensional list of points at each frame
    Plots a graph of the histogram of 'data' over time
    Expects data to be in range [0, 1]
    If avg is not None, draw an average curve (mean or median).
    See http://matplotlib.org/examples/color/colormaps_reference.html
    for colormap examples.
    '''
    width, height = size
    nframes = min(len(data), width)
    data = data[:nframes]
    dx = width // nframes
    dy = 1 / bins
    cmap = plt.get_cmap(colormap)
    screen = pygame.display.set_mode((dx * nframes, height))
    screen.fill([x*255 for x in cmap(0)])

    for t in range(nframes):
        for b in range(bins):
            y = height * (1 - (b + 1) / bins)
            bmin = b / bins
            bmax = bmin + dy
            in_range = [x for x in data[t] if bmin < x < bmax]
            freq = len(in_range) / len(data[t])
            color = [x * 255 for x in cmap(freq * contrast)]
            pygame.draw.rect(screen, color, (t * dx, y, dx, dy * height), 0)

    if avg is not None:
        avg_func = {'mean': np.mean, 'median': np.median}[avg]
        points = [(t * dx, (1 - avg_func(x)) * height)
                  for t, x in enumerate(data)]
        pygame.draw.lines(screen, (0, 0, 0), False, points, 4)
    return screen

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Format: graph_histograms.py [file base] "
              "[psi/densities] [save(s)]?")
        sys.exit(0)

    pygame.init()
    font = pygame.font.Font(None, 50)
    fname, stat = sys.argv[1:3]
    pygame.display.set_caption("Plot of histogram of {} over time".format(stat))
    data = np.load(fname + "_DATA.npz")
    size = (800, 800)
    if stat == 'rpsi':
        screen = plot(data['radial_psi'], size, contrast=3)
    elif stat == 'rdensities':
        screen = plot(data['radial_densities'], size, contrast=3)
    elif stat == 'psi':
        screen = plot(data['psi'], size, contrast=5, avg='mean')
    elif stat == 'densities':
        screen = plot(data['densities'], size, contrast=3, avg='median')
    title = font.render(fname.replace('_', ' '), 1, (255,)*3)
    screen.blit(title, (330, 20))
    pygame.display.flip()

    if len(sys.argv) > 3 and sys.argv[3] in ('s', 'save'):
        pygame.image.save(screen,"{}_{}.png".format(fname, stat))
    else:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
