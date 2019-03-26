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
    for c in np.arange(col_a, col_b, 0.049*(col_b-col_a)):
        r = cent_r - (radius**2 - (c-cent_c)**2)**0.5 
        rr, cc, val = line_aa(old_r, old_c, int(r), int(c))
        img[rr, cc, :] = 1.0 - np.expand_dims(val, 2)
        old_r, old_c = int(r), int(c)

black = (0.0, 0.0, 0.0)
red   = (0.8, 0.2, 0.2)
green = (0.2, 0.8, 0.2)
blue  = (0.2, 0.2, 0.8)

def draw(colors, arcs, filename): 
    baseline = 40
    width = 20 + (len(colors)-1)*40 
    
    img = np.ones((50, width, 3), dtype=np.float32)
    for i,j in arcs:
        draw_arc_aa(img, baseline, 10+40*i, 10+40*j, 4.0)
    for i,color in enumerate(colors):
        draw_circle_aa(img, baseline, 10+40*i, 6, color)

    plt.imsave(filename, img)

draw(colors = [red, green, black, red],
     arcs = [(0, 2), (0, 1), (1,3), (0,3)],
     filename = 'hi.png')


