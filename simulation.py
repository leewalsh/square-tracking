import pygame
from pygame.locals import *
import numpy

pygame.init()
screen = pygame.display.set_mode((608, 600))
pygame.display.set_caption("Particle movement simulation")

positions = numpy.loadtxt("5x5_inward.txt", skiprows=4, dtype='i,f,f,i,f,i')
frames = []
frame = []
cur_frame = 0
for row in positions:
    if row[0] > cur_frame:
        frames.append(frame)
        cur_frame = row[0]
        frame = []
    frame.append((row[1], row[2]))
frames.append(frame)

running = True
t = 0
while running and t < len(frames):
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
            break
    screen.fill((255, 255, 255))
    for pos in frames[t]:
        pygame.draw.rect(screen, (0, 0, 0), (pos[0]-5, pos[1]-5,10,10), 0)
    pygame.display.flip()
    t += 1
    pygame.time.delay(10)
