''' author: samtenka
    change: 2019-06-10
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
import re
import sys 

assert(len(sys.argv) in [4, 6])
EXPERDATA_FILENM = sys.argv[1] 
GRADSTATS_FILENM = sys.argv[2]
MODE = sys.argv[3]
STATISTIC = MODE.split('-')[0].lower()
OPTIMIZER = MODE.split('-')[1].lower()
DEFAULT_FILENM= '%s.png'%MODE.lower()
PLOT_FILENM = sys.argv[4] if len(sys.argv)==6 else DEFAULT_FILENM 
MAX_LR = float(sys.argv[5]) if len(sys.argv)==6 else float('inf') 

################################################################################
#           0. READ PRECOMPUTED DATA                                           #
################################################################################

################################################################################
#           1. COMPUTE CURVES                                                  #
################################################################################

    #--------------------------------------------------------------------------#
    #               1.0 compute predictions based on gradient statistics       #
    #--------------------------------------------------------------------------#

    #--------------------------------------------------------------------------#
    #               1.1 process experimental data                              #
    #--------------------------------------------------------------------------#

################################################################################
#           2. PLOT CURVES                                                     #
################################################################################

    #--------------------------------------------------------------------------#
    #               2.0 define stylization of error bars                       #
    #--------------------------------------------------------------------------#

red  ='#cc4444'
green='#44cc44'
blue ='#4444cc'

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

plot_fill(X, E_lin, S_lin, red,   '1st order prediction') if E_lin is not None else None 
plot_fill(X, E_qua, S_qua, green, '2nd order prediction') if E_qua is not None else None 
plot_bars(X_unsorted, Y, S, blue, 'experiment')

    #--------------------------------------------------------------------------#
    #               2.2 label and save figures                                 #
    #--------------------------------------------------------------------------#

plt.xlabel('learning rate')
plt.ylabel('loss')
plt.ylim(*modes[MODE]['ylims'])
plt.title('%s   vs   Learning Rate' % modes[MODE]['title'])
plt.legend(loc='best')
plt.savefig(PLOT_FILENM)
