''' author: samuel tenka
    change: 2019-03-25 
    create: 2019-03-25 
    descrp: render SGD diagrams   
'''

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import circle, circle_perimeter_aa, line_aa


def draw_circle_aa(img, row, col, rad, color=(0.0, 0.0, 0.0)):
    expanded_color = np.expand_dims(np.expand_dims(np.array(color), 0), 0)
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    img[rr, cc, :] = np.expand_dims(val, 2)
    img[rr, cc, :] = img[rr, cc, :] * (1.0 - expanded_color)
    img[rr, cc, :] = 1.0 - img[rr, cc, :]
    rr, cc = circle(row, col, rad)
    img[rr, cc, :] = expanded_color

def draw_arc_aa(img, row, col_a, col_b, curve):
    cent_r = int(row + abs(col_a - col_b)/curve)
    cent_c = int((col_a+col_b)/2.0)
    radius = int(((cent_r-row)**2 + (cent_c-col_a)**2)**0.5)

    old_r, old_c = row, col_a
    for c in list(np.arange(col_a, col_b, 5.0)) + [col_b]:
        r = cent_r - (radius**2 - (c-cent_c)**2)**0.5 
        rr, cc, val = line_aa(old_r, old_c, int(r), int(c))
        img[rr, cc, :] = np.minimum(img[rr, cc, :], 1.0 - np.expand_dims(val, 2))
        old_r, old_c = int(r), int(c)

black = (0.0, 0.0, 0.0)
red   = (0.8, 0.2, 0.2)
green = (0.2, 0.8, 0.2)
blue  = (0.2, 0.2, 0.8)
gold  = (0.9, 0.7, 0.0)

def draw(colors, arcs, filename): 
    baseline = 65
    width = 40 + (len(colors)-1)*80 
    
    img = np.ones((80, width, 3), dtype=np.float32)
    counts = {}
    for i,j in arcs:
        if (i,j) not in counts:
            counts[(i,j)] = 0.0 
        draw_arc_aa(img, baseline, 20+80*i, 20+80*j, 2.0**counts[(i,j)]*abs(i-j))
        counts[(i,j)] += 1.0
    for i,color in enumerate(colors):
        draw_circle_aa(img, baseline, 20+80*i, 12, color)

    plt.imsave(filename, img)

draw(colors = [red, red],
     arcs = [(0, 1)],
     filename = 'rr.png')
draw(colors = [red, green],
     arcs = [(0, 1)],
     filename = 'rg.png')
draw(colors = [red, red, green],
     arcs = [(0, 2), (1, 2)],
     filename = 'rrg.png')
draw(colors = [red, green, blue],
     arcs = [(0, 1), (1, 2)],
     filename = 'rgb.png')
draw(colors = [red, green, blue],
     arcs = [(0, 1), (1, 2), (1, 2)],
     filename = 'rggb.png')
draw(colors = [red, green, green, blue],
     arcs = [(0, 1), (1, 3), (2, 3)],
     filename = 'rggb-a.png')
draw(colors = [red, green, green, blue],
     arcs = [(0, 2), (1, 3), (2, 3)],
     filename = 'rggb-b.png')
draw(colors = [red, green, blue],
     arcs = [(0, 1), (0, 1), (1, 2)],
     filename = 'rrgb.png')
draw(colors = [red, red, green, blue],
     arcs = [(0, 2), (1, 2), (2, 3)],
     filename = 'rrgb-a.png')




draw(colors = [red],
     arcs = [],
     filename = 'sgd-0.png')

draw(colors = [red, green],
     arcs = [(0, 1)],
     filename = 'sgd-1.png')

draw(colors = [red, green, blue],
     arcs = [(0, 1), (1, 2)],
     filename = 'sgd-2a.png')

draw(colors = [red, green, blue],
     arcs = [(0, 2), (1, 2)],
     filename = 'sgd-2b.png')

draw(colors = [red, green],
     arcs = [(0, 1), (0, 1)],
     filename = 'sgd-2c.png')

draw(colors = [red, green, blue, gold],
     arcs = [(0, 1), (1, 2), (2, 3)], 
     filename = 'sgd-3a.png')

draw(colors = [red, green, blue, gold],
     arcs = [(0, 2), (1, 2), (2, 3)], 
     filename = 'sgd-3b.png')

draw(colors = [red, green, blue, gold],
     arcs = [(0, 3), (1, 2), (2, 3)], 
     filename = 'sgd-3c.png')

draw(colors = [red, green, blue, gold],
     arcs = [(0, 1), (1, 3), (2, 3)], 
     filename = 'sgd-3d.png')

draw(colors = [red, green, blue, gold],
     arcs = [(0, 2), (1, 3), (2, 3)], 
     filename = 'sgd-3e.png')

draw(colors = [red, green, blue, gold],
     arcs = [(0, 3), (1, 3), (2, 3)], 
     filename = 'sgd-3f.png')

draw(colors = [red, green, blue],
     arcs = [(0, 1), (1, 2), (1, 2)], 
     filename = 'sgd-3g.png')

draw(colors = [red, green, blue],
     arcs = [(0, 1), (0, 1), (1, 2)], 
     filename = 'sgd-3h.png')

draw(colors = [red, green, blue],
     arcs = [(0, 2), (1, 2), (1, 2)], 
     filename = 'sgd-3i.png')

draw(colors = [red, green, blue],
     arcs = [(0, 2), (0, 2), (1, 2)], 
     filename = 'sgd-3j.png')

draw(colors = [red, green],
     arcs = [(0, 1), (0, 1), (0, 1)], 
     filename = 'sgd-3k.png')
