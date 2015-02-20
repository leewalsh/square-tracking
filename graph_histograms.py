#!/usr/bin/env python
import sys
from matplotlib import pyplot as plt
import numpy as np
import pygame
from pygame.locals import QUIT

def plot(screen, data, width, height, bins=100, contrast=1, colormap='rainbow'):
    '''
    data: 2-dimensional list of points at each frame
    Plots a graph of the histogram of 'data' over time
    Expects data to be in range [0, 1]
    See http://matplotlib.org/examples/color/colormaps_reference.html
    for colormap examples.
    '''
    nframes = len(data)
    dx = width / nframes
    dy = 1. / bins
    cmap = plt.get_cmap(colormap)
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

if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[2].lower() not in (
            'p', 'd', 'psi', 'densities'):
        print("Format: graph_histograms.py [file base] [psi(p)/densities(d)]")
        sys.exit(0)

    SIZE = WIDTH, HEIGHT = (608, 600)
    pygame.init()
    screen = pygame.display.set_mode(SIZE)
    pygame.display.set_caption("Plot of histograms over time")

    fname, stat = sys.argv[1], sys.argv[2]
    data = np.load(fname + "_DATA.npz")
    if stat[0] == 'p':
        plot(screen, data['psi'], WIDTH, HEIGHT, contrast=5)
    else:
        plot(screen, data['densities'], WIDTH, HEIGHT, contrast=3)
    pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
