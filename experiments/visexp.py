''' author: samtenka
    change: 2019-02-20
    create: 2010-02-14
    descrp: Compare and plot losses for GD vs SGD and for out-of-sample vs in-sample --- as dependent on learning rate.
            To run, type:
                python vis.py experdata_gauss.txt gradstats_gauss.txt OUT-DIFF plots/out-diff.png 0.025
            The   experdata_gauss.txt   gives   a filename storing experimental results;
            the   gradstats_gauss.txt   gives   a filename storing estimates of gradient statistics;
            the              OUT-DIFF   gives   a plotting mode in {OUT-GD, OUT-SGD, OUT-DIFF, GEN-GD, GEN-SGD};
            the    plots/out-diff.png   gives   a filename to write to (defaults to lowercase version of plotting mode);
            the                 0.025   gives   a maximum learning rate beyond which not to plot (defaults to infinity).
'''

from matplotlib import pyplot as plt
import numpy as np
import re
import sys 


N = T = 10

modes = {
    'OUT-GD':  {'ylims':( 0.40 , 0.80 ), 'title':'GD Test Loss'}, 
    'OUT-SGD': {'ylims':( 0.40 , 0.80 ), 'title':'SGD Test Loss'},
    'OUT-DIFF':{'ylims':(-0.10 , 0.10 ), 'title':'Test-Time Benefit of Stochasticity'},
    'GEN-GD':  {'ylims':(-0.05 , 0.25 ), 'title':'GD Generalization Gap'},
    'GEN-SGD': {'ylims':(-0.05 , 0.25 ), 'title':'SGD Generalization Gap'},
#
#    'OUT-GD':  {'ylims':( 0.40 , 0.80 ), 'title':'GD Test Loss'}, 
#    'OUT-SGD': {'ylims':( 0.40 , 0.80 ), 'title':'SGD Test Loss'},
#    'OUT-DIFF':{'ylims':(-0.04 , 0.02 ), 'title':'Test-Time Benefit of Stochasticity'},
#    'GEN-GD':  {'ylims':(-0.02 , 0.09 ), 'title':'GD Generalization Gap'},
#    'GEN-SGD': {'ylims':(-0.02 , 0.09 ), 'title':'SGD Generalization Gap'},
##    'OUT-GD':  {'ylims':( 0.20 , 0.70 ), 'title':'GD Test Loss'}, 
##    'OUT-SGD': {'ylims':( 0.20 , 0.70 ), 'title':'SGD Test Loss'},
##    'OUT-DIFF':{'ylims':(-0.01 , 0.05 ), 'title':'Test-Time Benefit of Stochasticity'},
##    'GEN-GD':  {'ylims':(-0.01 , 0.2  ), 'title':'GD Generalization Gap'},
##    'GEN-SGD': {'ylims':(-0.01 , 0.2  ), 'title':'SGD Generalization Gap'},
#    'OUT-GD':  {'ylims':( 0.4  , 0.7 ), 'title':'GD Test Loss'}, 
#    'OUT-SGD': {'ylims':( 0.4  , 0.7 ), 'title':'SGD Test Loss'},
#    'OUT-DIFF':{'ylims':(-0.02 , 0.04), 'title':'Test-Time Benefit of Stochasticity'},
#    'GEN-GD':  {'ylims':(-0.02 , 0.10), 'title':'GD Generalization Gap'},
#    'GEN-SGD': {'ylims':(-0.02 , 0.10), 'title':'SGD Generalization Gap'},
##    'OUT-GD':  {'ylims':( 0.4  , 0.7 ), 'title':'GD Test Loss'}, 
##    'OUT-SGD': {'ylims':( 0.4  , 0.7 ), 'title':'SGD Test Loss'},
##    'OUT-DIFF':{'ylims':(-0.02 , 0.04), 'title':'Test-Time Benefit of Stochasticity'},
##    'GEN-GD':  {'ylims':(-0.05 , 0.05), 'title':'GD Generalization Gap'},
##    'GEN-SGD': {'ylims':(-0.05 , 0.05), 'title':'SGD Generalization Gap'},
#    'OUT-GD':  {'ylims':( 0.45 , 0.7 ), 'title':'GD Test Loss'}, 
#    'OUT-SGD': {'ylims':( 0.45 , 0.7 ), 'title':'SGD Test Loss'},
#    'OUT-DIFF':{'ylims':(-0.02 , 0.04), 'title':'Test-Time Benefit of Stochasticity'},
#    'GEN-GD':  {'ylims':(-0.05 , 0.6 ), 'title':'GD Generalization Gap'},
#    'GEN-SGD': {'ylims':(-0.05 , 0.6 ), 'title':'SGD Generalization Gap'},
##    'OUT-GD':  {'ylims':( 5.5 , 8.0 ), 'title':'GD Test Loss'}, 
##    'OUT-SGD': {'ylims':( 5.5 , 8.0 ), 'title':'SGD Test Loss'},
##    'OUT-DIFF':{'ylims':(-0.05, 0.30), 'title':'Test-Time Benefit of Stochasticity'},
##    'GEN-GD':  {'ylims':(-0.20, 4.80), 'title':'GD Generalization Gap'},
##    'GEN-SGD': {'ylims':(-0.20, 4.80), 'title':'SGD Generalization Gap'},
#    'OUT-GD':  {'ylims':( 0.6 , 2.1 ), 'title':'GD Test Loss'}, 
#    'OUT-SGD': {'ylims':( 0.6 , 2.1 ), 'title':'SGD Test Loss'},
#    'OUT-DIFF':{'ylims':(-0.01, 0.12), 'title':'Test-Time Benefit of Stochasticity'},
#    'GEN-GD':  {'ylims':(-0.05, 0.35), 'title':'GD Generalization Gap'},
#    'GEN-SGD': {'ylims':(-0.05, 0.35), 'title':'SGD Generalization Gap'},
}

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

    #--------------------------------------------------------------------------#
    #               0.0 read gradient statistics                               #
    #--------------------------------------------------------------------------#

with open(GRADSTATS_FILENM) as f:
    lines = f.read().split('\n')
    MTRIALS, _ = tuple(float(x) for x in lines[0].split() if x.isnumeric())
    split_line = lambda l: l.replace(',', ' ').split() 
    is_number = lambda s: s.replace('.','').replace('-','').isnumeric() 
    SEN, INT, UNC, PAS, AUD, PER, SER = tuple(float(x)               for x in split_line(lines[1]) if is_number(x))
    concentration = (1.0/MTRIALS**0.5) + (1.0/5000.0**0.5) 
    SEN_,INT_,UNC_,PAS_,AUD_,PER_,SER_ = tuple(float(x)*concentration for x in split_line(lines[2]) if is_number(x))

    #--------------------------------------------------------------------------#
    #               0.1 read experimental data                                 #
    #--------------------------------------------------------------------------#

X = []
Y_out = []
Y_ins = []
S_out = []
S_ins = []

with open(EXPERDATA_FILENM) as f:
    text = f.read() 
    paragraphs = filter(None, text.split('\n'))
    for p in paragraphs: 
        opt = re.search(r'OPT=(\S+)', p).group(1)
        if opt != OPTIMIZER: continue
        metric = re.search(r'METRIC=(\S+):', p).group(1)
        if metric not in ['OL', 'IL']: continue
        learning_rate = float(re.search(r'LEARNING_RATE=(\S+)', p).group(1))
        if learning_rate > MAX_LR: continue

        nb_trials = float(re.search(r'NB_TRIALS=(\S+)', p).group(1))
        mean = float(re.search(r':\s+(\S+)', p).group(1))
        stddev = float(re.search(r':\s+\S+\s+(\S+)', p).group(1))

        if metric == 'OL':
            X.append(learning_rate)
            Y_out.append(mean)
            concentration = (1.0/nb_trials**0.5)
            #if opt!='diff':
            #    concentration += (1.0/1000.0**0.5) 
            S_out.append(stddev * concentration)
        elif metric == 'IL':
            Y_ins.append(mean)
            concentration = (1.0/nb_trials**0.5)
            #if opt!='diff':
            #    concentration += (1.0/5000.0**0.5) 
            S_ins.append(stddev * concentration)

################################################################################
#           1. COMPUTE CURVES                                                  #
################################################################################

X = np.array(X)
X_unsorted = X[:]
X = np.array(sorted(X))

    #--------------------------------------------------------------------------#
    #               1.0 compute predictions based on gradient statistics       #
    #--------------------------------------------------------------------------#

''' For reference:
        A. mean loss                                    {()}                                    SENTIMENT
        B. trace of square gradient                     {(a)}{(a)}                              INTENSITY
        C. trace of covariance of gradients             {(a)(a)} - {(a)}{(a)}                   UNCERTAINTY
        D. 1st order increase of (B) along gradient     2{(a)}{(ab)}{(b)}                       PASSION
        E. 1st order increase of (C) along gradient     2{(a)}{(ab)(b)} - 2{(a)}{(ab)}{(b)}     AUDACITY
        F. trace of hessian times square gradient       {(ab)}{(a)}{(b)}                        PASSION/2
        G. trace of hessian times covariance            {(ab)}{(a)(b)} - {(ab)}{(a)}{(b)}       PERIL 
'''

E_qua = E_lin = None
if MODE == 'OUT-GD':
    E_qua =         SEN  - X*T*INT  + X*X*( (T*(T-1)/2.0)*(0.75*PAS + 0.5*AUD  /N + 0.5*PER /N) + (T)*(0.25*PAS  + 0.5*PER /N))
    S_qua =        (SEN_ + X*T*INT_ + X*X*( (T*(T-1)/2.0)*(0.75*PAS_+ 0.5*AUD_ /N + 0.5*PER_/N) + (T)*(0.25*PAS_ + 0.5*PER_/N)))

    A = SEN
    B = -T*INT
    C = 2.0 * ( (T*(T-1)/2.0)*(0.75*PAS + 0.5*AUD  /N + 0.5*PER /N) + (T)*(0.25*PAS  + 0.5*PER /N))
    b = -C/B
    a = -B/b
    c = A-a
    E_qua = np.exp(-b*X)*a + c 

    E_lin =         SEN  - X*T*INT 
    S_lin =         SEN_ + X*T*INT_
elif MODE == 'OUT-SGD':
    E_qua =         SEN  - X*T*INT  + X*X*( (T*(T-1)/2.0)*(0.75*PAS                           ) + (T)*(0.25*PAS  + 0.5*PER ))
    S_qua =        (SEN_ + X*T*INT_ + X*X*( (T*(T-1)/2.0)*(0.75*PAS_                          ) + (T)*(0.25*PAS_ + 0.5*PER_)))

    A = SEN
    B = -T*INT
    C = 2.0 * ( (T*(T-1)/2.0)*(0.75*PAS                           ) + (T)*(0.25*PAS  + 0.5*PER ))
    b = -C/B
    a = -B/b
    c = A-a
    E_qua = np.exp(-b*X)*a + c 

    E_lin =         SEN  - X*T*INT 
    S_lin =         SEN_ + X*T*INT_
elif MODE == 'OUT-DIFF':
    E_qua =                           X*X*( (T*(T-1)/2.0)*(           0.5*AUD  /N + 0.5*PER /N) + (T)*(-0.5*PER  + 0.5*PER /N))
    S_qua =        (                  X*X*( (T*(T-1)/2.0)*(           0.5*AUD_ /N + 0.5*PER_/N) + (T)*( 0.5*PER_ - 0.5*PER_/N)))

    A = SEN
    B = -T*INT
    C = 2.0 * ( (T*(T-1)/2.0)*(0.75*PAS + 0.5*AUD  /N + 0.5*PER /N) + (T)*(0.25*PAS  + 0.5*PER /N))
    b = -C/B
    a = -B/b
    c = A-a
    E_qua_gd = np.exp(-b*X)*a + c 

    A = SEN
    B = -T*INT
    C = 2.0 * ( (T*(T-1)/2.0)*(0.75*PAS                           ) + (T)*(0.25*PAS  + 0.5*PER ))
    b = -C/B
    a = -B/b
    c = A-a
    E_qua_sgd = np.exp(-b*X)*a + c 

    E_qua = E_qua_gd - E_qua_sgd

    E_lin =                X*0.0
    S_lin =                X*0.0
elif MODE == 'GEN-GD':
    E_lin =                X*T*UNC /N       
    S_lin =        (SEN_+  X*T*UNC_/N)  # sen_ due to incomplete cancellation 'tween finite insamples and outsamples
elif MODE == 'GEN-SGD':
    E_qua =                X*T*UNC /N        - X*X*( (T*(T-1)/2.0)*(PAS  + PER ) + (T)*(0.5*SER ))/N
    S_qua =         SEN_+  X*T*UNC_/N        + X*X*( (T*(T-1)/2.0)*(PAS_ + PER_) + (T)*(0.5*SER_))/N
    E_lin =                X*T*UNC /N       
    S_lin =        (SEN_+  X*T*UNC_/N)  # sen_ due to incomplete cancellation 'tween finite insamples and outsamples

    #--------------------------------------------------------------------------#
    #               1.1 process experimental data                              #
    #--------------------------------------------------------------------------#

if STATISTIC == 'gen':
    Y = np.array(Y_out) - np.array(Y_ins)
    S = np.array(S_out) + np.array(S_ins)
else:
    Y = np.array(Y_out)
    S = np.array(S_out)



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
    plt.plot(x, y, color=color, alpha=0.5)
    plt.fill(np.concatenate([x, x[::-1]]), np.concatenate([y-z*s, (y+z*s)[::-1]]), facecolor=color, alpha=0.5, label=label)

def plot_bars(x, y, s, color, label, z=1.96, bar_width=1.0/50): 
    e = bar_width * (max(x)-min(x))
    for (xx, yy, ss) in zip(x, y, s):
        plt.plot([xx, xx], [yy-z*ss, yy+z*ss],      color=color)
        plt.plot([xx-e, xx+e], [yy-z*ss, yy-z*ss], color=color)
        plt.plot([xx-e, xx+e], [yy+z*ss, yy+z*ss], color=color)
    plt.plot([xx, xx], [yy-z*ss, yy+z*ss], color=color, label=label)

    #--------------------------------------------------------------------------#
    #               2.1 plot curves                                            #
    #--------------------------------------------------------------------------#

plot_fill(X, E_lin, S_lin, red,   '1st order prediction') if E_lin is not None else None 
plot_fill(X, E_qua, S_qua, green, '2nd order prediction') if E_qua is not None else None 
plot_bars(X_unsorted, Y, S, blue, 'experiment')

    #--------------------------------------------------------------------------#
    #               2.2 label and write                                        #
    #--------------------------------------------------------------------------#

plt.xlabel('learning rate')
plt.ylabel('loss')
plt.ylim(*modes[MODE]['ylims'])
plt.title('%s --- vs --- Learning Rate' % modes[MODE]['title'])
plt.legend(loc='best')
plt.savefig(PLOT_FILENM)
