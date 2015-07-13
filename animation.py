#!/usr/bin/env python

import pygame
from pygame.locals import *
import numpy
import sys
import numpy as np
from random import random

if len(sys.argv) < 2:
    print("Please specify a filename")
    sys.exit(0)

DX = 0#300
DY = 0#300
MSD_TEST = False
L = 10
k = 1

pygame.init()
screen = pygame.display.set_mode((608, 600))
pygame.display.set_caption("Particle movement animation")

if sys.argv[1].endswith('.txt'):
    positions = numpy.loadtxt(sys.argv[1], dtype='i,f,f,i,f,i')
else:
    positions = numpy.load(sys.argv[1] + '_TRACKS.npz')['data']
    corner_positions = numpy.load(sys.argv[1] + '_CORNER_TRACKS.npz')['data']

frames = []
frame = []
cur_frame = 0
for row in positions:
    if row[0] > cur_frame:
        frames.append(frame)
        cur_frame = row[0]
        frame = []
    frame.append((row[1], row[2], row[3]))
frames.append(frame)

def find_COM(frame):
    return (sum([p[0] for p in frame]) / len(frame),
            sum([p[1] for p in frame]) / len(frame))

COM = find_COM(frames[0])
dists = [((row[1]-COM[0])**2 + (row[2]-COM[1])**2, row[3])
         for row in positions if row[0]==0]
sorted_dists = sorted(dists)
n = 0
i = 0
valencies = {}
while True:
    n += 1
    box_size = 8 * (n - 1) if n > 1 else 1
    for j in range(box_size):
        valencies[sorted_dists[i][1]] = n
        i += 1
        if i >= len(frames[0]):
            break
    else:
        continue
    break

max_valency = max(valencies.values())

running = True
simulate = False
speed = 1.
paused = False
if len(sys.argv) > 2:
    speed = float(sys.argv[2])
font = pygame.font.Font(None, 36)
font2 = pygame.font.Font(None, 15)
COLORS = {}
COLORS2 = {}
'''for ID in valencies:
    s = valencies[ID]
    COLORS[ID] = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),
                  (255, 255, 0), (255, 0, 255), (0, 0, 0)][s-1]
    COLORS2[ID] = np.asarray(COLORS[ID]) / 2.'''

for i in range(10000):
    COLORS[i]=(0,0,0)
    COLORS2[i]=(0,0,0)

tau = 1

t = 0
while running and t < len(frames):
    draw = simulate and not paused
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
            break
        elif event.type == pygame.MOUSEBUTTONUP:
            simulate = True
            pos = pygame.mouse.get_pos()
            print(pos)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_LEFT and paused and t > 0:
                t -= 1
                draw = True
            elif event.key == pygame.K_RIGHT and paused and t < len(frames) - 1:
                t += 1
                draw = True
    if not draw:
        continue
    screen.fill((255, 255, 255))
    text = font.render("Time: {0}".format(t), 1, (10, 10, 10))
    screen.blit(text, text.get_rect())
    for x, y, ID in frames[t]:
        pygame.draw.rect(screen, COLORS[ID], (k*x-L-DX, k*y-L-DY,k*L,k*L), 0)

    if MSD_TEST and t+tau<len(frames):
        for x, y, ID in frames[t+tau]:
            for x0, y0, ID0 in frames[t]:
                if ID==ID0:
                    break
            else:
                print("ID {0} has a disconnect from frames {1} to {2}!".
                      format(ID, t, t+tau))
                continue
            pygame.draw.rect(screen, COLORS2[ID], (k*x-L-DX,k*y-L-DY,2*L,2*L), 0)
            pygame.draw.line(screen, (0,0,0), (k*x-DX,k*y-DY),(k*x0-DX,k*y0-DY))
            dist = (x0-x)**2 + (y0-y)**2
            lbl = font2.render("%.1f"%(dist**.5), 1, (0, 0, 0))
            screen.blit(lbl, (k*x-DX,k*y-DY+10))
    pygame.display.flip()
    if not paused:
        t += 1
    pygame.time.delay(int(10 / speed))





