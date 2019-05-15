''' author: samuel tenka
    change: 2019-03-25 
    create: 2019-03-25 
    descrp: render SGD diagrams   
'''

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import circle, circle_perimeter_aa, line_aa


def draw_disk_aa(img, row, col, rad, color=(0.0, 0.0, 0.0)):
    expanded_color = np.expand_dims(np.expand_dims(np.array(color), 0), 0)
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    img[rr, cc, :] = np.expand_dims(val, 2)
    img[rr, cc, :] = img[rr, cc, :] * (1.0 - expanded_color)
    img[rr, cc, :] = 0.95 - img[rr, cc, :]
    rr, cc = circle(row, col, rad)
    img[rr, cc, :] = expanded_color

def draw_line_aa(img, row_s, col_s, row_e, col_e, color=(0.0, 0.0, 0.0)):
    expanded_color = np.expand_dims(np.expand_dims(np.array(color), 0), 0)
    rr, cc, val = line_aa(row_s, col_s, row_e, col_e)
    img[rr, cc, :] = np.expand_dims(val, 2)
    img[rr, cc, :] = img[rr, cc, :] * expanded_color

def draw_arc_aa(img, row, col_a, col_b, curve):
    cent_r = int(row + abs(col_a - col_b)/curve)
    cent_c = int((col_a+col_b)/2.0)
    radius = int(((cent_r-row)**2 + (cent_c-col_a)**2)**0.5)

    old_r, old_c = row, col_a
    for c in list(np.arange(col_a, col_b, 2.5)) + [col_b]:
        r = cent_r - (radius**2 - (c-cent_c)**2)**0.5 
        rr, cc, val = line_aa(old_r, old_c, int(r), int(c))
        img[rr, cc, :] = np.minimum(img[rr, cc, :], 1.0 - np.expand_dims(val, 2))
        old_r, old_c = int(r), int(c)

def draw_blob_aa(img, row, col_a, col_b, curve, thick = 16):
    cent_r = int(row - abs(col_a - col_b)/curve)
    cent_c = int((col_a+col_b)/2.0)
    radius = int(((cent_r-row)**2 + (cent_c-col_a)**2)**0.5)

    for c in list(np.arange(col_a, col_b, 0.5)) + [col_b]:
        r = cent_r + (radius**2 - (c-cent_c)**2)**0.5 
        rr, cc = circle(r, c, thick)
        img[rr, cc, :] = np.maximum(0.1, 0.995 * img[rr, cc, :])


black = (0.0, 0.0, 0.0)
red   = (0.8, 0.2, 0.2)
green = (0.2, 0.8, 0.2)
blue  = (0.2, 0.2, 0.8)
gold  = (0.9, 0.7, 0.0)
colors = [red, green, blue, gold]

RADIN = 8 
RADOUT = 12

def get_all_partitions(nb_nodes):
    if nb_nodes==0: return [[]]
    old_partitions = get_all_partitions(nb_nodes-1)
    new_partitions = []
    new_node = nb_nodes-1
    for pp in old_partitions:
        new_partitions.append(pp + [(new_node,)]) # singleton
        for i in range(len(pp)): # nonsingleton
            new_partitions.append([p if j!=i else p+(new_node,) for j,p in enumerate(pp)])
    return new_partitions

def get_all_trees(nb_nodes):
    if nb_nodes <= 1: return [[]]
    old_trees = get_all_trees(nb_nodes-1)
    new_trees = []
    for gg in old_trees:
        ggg = [(s+1, e+1) for (s,e) in gg] 
        for j in range(1, nb_nodes): 
            new_trees.append([(0,j)] + ggg)
    return new_trees

def draw(parts, arcs, filename): 
    assert len(parts) <= len(colors)

    nb_nodes = sum(len(p) for p in parts)
    height = 40 + (nb_nodes-1)*40
    width = 60 + (nb_nodes-1)*80
    baseline = height//2

    img = np.ones((height, width, 3), dtype=np.float32)
    for p,color in zip(parts, colors):
        for s,e in zip(p, p[1:]):
            draw_blob_aa(img, baseline, 30+80*s, 30+80*e, abs(s-e))
        for i in p:
            R, C = baseline, 30+80*i
            draw_disk_aa(img, R, C, RADIN, color)
    for i,j in arcs:
        draw_arc_aa(img, baseline, 30+80*i, 30+80*j, abs(i-j))

    plt.imsave(filename, img)

for nb_nodes in [1, 2, 3, 4]:
    for pp in get_all_partitions(nb_nodes):
        for gg in get_all_trees(nb_nodes):
            nm = '(%s)(%s)' % ('-'.join(''.join(str(s) for s in p) for p in pp), '-'.join(''.join(str(s) for s in g) for g in gg))
            print(nm)
            draw(parts=pp, arcs=gg, filename='%s.png'%nm)

