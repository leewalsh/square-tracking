import sys
import Tkinter as TK
from PIL import Image, ImageTk
import math

'''
Program to detect possible overlap between particles.
Adjust resize_factor in make_square to change the tolerance
for detecting overlap.
'''


def get_length(positions):
    """ usage in previous file:
    positions = sorted(zip(particles['X'], particles['Y']))
    print(get_length(positions))
    """
    N = int(round(math.sqrt(len(positions))))
    if N*N != len(positions):
        raise ValueError("Cannot calculate square length with a non-square # of particles")

    lengths_sum = 0
    for i in range(N):
        col = positions[i * N: (i + 1) * N]
        lengths_sum += (col[-1][1] - col[0][1]) / (N - 1)

    return lengths_sum / N


def intersects(square1, square2):
    '''xvals = [square1[0][0], square1[1][0], square1[2][0], square1[3][0]]
    right1 = max(xvals)
    left1 = min(xvals)
    yvals = [square1[0][1], square1[1][1], square1[2][1], square1[3][1]]
    top1 = min(yvals)
    bottom1 = max(yvals)
    xvals = [square2[0][0], square2[1][0], square2[2][0], square2[3][0]]
    right2 = max(xvals)
    left2 = min(xvals)
    yvals = [square2[0][1], square2[1][1], square2[2][1], square2[3][1]]
    top2 = min(yvals)
    bottom2 = max(yvals)
    return right1 >= left2 and left1 <= right2 and top1 <= bottom2 and bottom1 >= top2
    '''
    from shapely.geometry import Polygon
    return Polygon(square1).intersects(Polygon(square2))

def make_square(x, y, theta, side, resize_factor=0.9):
    diag = side / math.sqrt(2) * resize_factor # resize_factor allows for some tolerance
    corners = [(x + diag * math.cos(phi), y + diag * math.sin(phi)) \
               for phi in [theta - math.pi/4 * k for k in (1, 3, 5, 7)]]
    return corners + [(x, y)]

def angle(v):
    return math.atan2(v[1], v[0])

particles, corners = helpy.load_data(prefix, 'p c')

if len(sys.argv) > 1:
    if sys.argv[1].upper() == 'T':
        side = get_length([(row['X'], row['Y']) for row in particles if row['Frame'] == 0])
    else:
        side = float(sys.argv[1])
else:
    side = 16.5
print('Side length: {0}'.format(side))

def find_intersections(positions, corner_positions, i):
    squares = []
    for x, y in positions:
        corners = sorted(((cx-x)**2+(cy-y)**2, cx, cy) for cx, cy in corner_positions)
        c1, c2 = corners[:2]
        if c2[0] < 150:
            r1 = (c1[1] - x, c1[2] - y)
            r2 = (c2[1] - x, c2[2] - y)
            ravg = ((r1[0] + r2[0]) / 2, (r1[1] + r2[1]) / 2)
            dtheta = abs(angle(r1) - angle(r2)) * 180. / math.pi
            dtheta = min(dtheta, 360 - dtheta) # deal with 270 degree case
            if dtheta > 70 and dtheta < 110:
                squares.append(make_square(x, y, angle(ravg), side))

    intersections = []
    for square in squares:
        for square2 in squares:
            if square is not square2 and intersects(square, square2):
                print("INTERSECTION: {0} and {1}".format(square[4], square2[4]))
                intersections.append(square)

    if len(sys.argv) > 2 and intersections:
        root = TK.Tk()
        root.wm_title("Frame {0}".format(i))
        photo = ImageTk.PhotoImage(Image.open("%s%.4d.tif"%(sys.argv[2],i)))
        can = TK.Canvas(root, width=photo.width(), height=photo.height())
        can.create_image((0, 0), image=photo, anchor=TK.NW)
        for c1, c2, c3, c4, center in squares:
            can.create_polygon(c1[::-1] + c2[::-1] + c3[::-1] + c4[::-1], fill='red')
        for c1, c2, c3, c4, c5  in intersections:
            can.create_polygon(c1[::-1] + c2[::-1] + c3[::-1] + c4[::-1], fill='green')
        can.pack()
        root.mainloop()

for i in range(particles['Frame'][-1]):
    particle_data = [(row['X'], row['Y']) for row in particles if row[0] == i]
    corner_data = [(row['X'], row['Y']) for row in corners if row[0] == i]
    find_intersections(particle_data, corner_data, i + 1)
