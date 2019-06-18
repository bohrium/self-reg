''' author: samtenka
    change: 2019-06-18
    create: 2019-02-14
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
from predictor import sgd_test_linear, sgd_test_quadratic
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
    X.append(okey.eta)
    Y.append(ol[okey]['mean'])
    S.append(ol[okey]['stdv']/ol[okey]['nb_samples']**0.5)
X = np.array(X)
Y = np.array(Y)
S = np.array(S)

plot_bars(
    X,
    Y,
    S,
    color=blue,
    label='experiment'
)

Y, S = sgd_test_linear(eta=X, T=100) 
plot_fill(X, Y, S, color=red, label='linear')
Y, S = sgd_test_quadratic(eta=X, T=100) 
plot_fill(X, Y, S, color=green, label='quadratic')

    #--------------------------------------------------------------------------#
    #               2.2 label and save figures                                 #
    #--------------------------------------------------------------------------#

plt.xlabel('eta')
plt.ylabel('loss')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.legend(loc='best')
plt.xticks(np.arange(0.00, 0.005, 0.001))
plt.yticks(np.arange(4, 9, 1))
plt.savefig('plot.png', pad_inches=0, bbox_inches='tight')
