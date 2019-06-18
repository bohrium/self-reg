''' author: samtenka
    change: 2019-06-17
    create: 2010-02-14
    descrp: Compare and plot descent losses as dependent on learning rate.
            Valid plotting modes are OUT-GD, OUT-SGD, OUT-DIFF, GEN-GD, GEN-SGD.
            To run, type:
                python vis.py experdata.txt gradstats.txt OUT-DIFF out-diff.png
            The   experdata.txt   gives   a filename storing descent trajectory summaries;
            the   gradstats.txt   gives   a filename storing gradient statistic estimates;
            the        OUT-DIFF   gives   a plotting mode
            the    out-diff.png   gives   a filename to write to 
'''

from matplotlib import pyplot as plt
import numpy as np
from optimlogs import OptimKey
import sys 

red  ='#cc4444'
green='#44cc44'
blue ='#4444cc'

with open('ol.data') as f:
    ol = eval(f.read())

def plot_fill(x, y, s, color, label, z=1.96):
    ''' plot variance (s^2) around mean (y) via 2D shading around a curve '''
    plt.plot(x, y, color=color, alpha=0.5)
    plt.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([y-z*s, (y+z*s)[::-1]]),
        facecolor=color, alpha=0.5, label=label
    )

def plot_bars(x, y, s, color, label, z=1.96, bar_width=1.0/50): 
    ''' plot variance (s^2) around mean (y) via I-bars around a scatter plot '''
    e = bar_width * (max(x)-min(x))
    for (xx, yy, ss) in zip(x, y, s):
        # middle, top, and bottom stroke of I, respectively:
        plt.plot([xx  , xx  ], [yy-z*ss, yy+z*ss], color=color)
        plt.plot([xx-e, xx+e], [yy-z*ss, yy-z*ss], color=color)
        plt.plot([xx-e, xx+e], [yy+z*ss, yy+z*ss], color=color)
    # connect to the figure legend:
    plt.plot([xx, xx], [yy-z*ss, yy+z*ss], color=color, label=label)

    #--------------------------------------------------------------------------#
    #               2.1 plot curves                                            #
    #--------------------------------------------------------------------------#
X, Y, S = [], [], []
for okey in ol:
    X.append(okey.T)
    Y.append(ol[okey]['mean'])
    S.append(ol[okey]['stdv']/ol[okey]['nb_samples']**0.5)

plot_bars(
    X,
    Y,
    S,
    color=blue,
    label='experiment'
)

    #--------------------------------------------------------------------------#
    #               2.2 label and save figures                                 #
    #--------------------------------------------------------------------------#

plt.xlabel('T')
plt.ylabel('loss')
plt.legend(loc='best')
plt.savefig('plot.png')
