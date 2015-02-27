#!/usr/bin/env python

import pygame
from pygame.locals import *
import numpy
import sys
from random import random

if len(sys.argv) < 2:
    print("Please specify a filename")
    sys.exit(0)

pygame.init()
screen = pygame.display.set_mode((608, 600))
pygame.display.set_caption("Particle movement simulation")

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

running = True
simulate = False
t = 0
speed = 1.
paused = False
if len(sys.argv) > 2:
    speed = float(sys.argv[2])
font = pygame.font.Font(None, 36)
COLORS = [(random() * 255, random() * 255, random() * 255)
          for i in range(len(frames[0]))]

while running and t < len(frames):
    draw = simulate and not paused
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
            break
        elif event.type == pygame.MOUSEBUTTONUP:
            simulate = True
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
        pygame.draw.rect(screen, COLORS[ID], (x-5, y-5, 10, 10), 0)
    pygame.display.flip()
    if not paused:
        t += 1
    pygame.time.delay(int(10 / speed))
