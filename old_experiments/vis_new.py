''' author: samtenka
    change: 2019-02-20
    create: 2010-02-14
    descrp: Compare and plot losses for GD vs SGD and for out-of-sample vs in-sample --- as dependent on learning rate.
            To run, type:
                python vis.py experdata_gauss_new.txt gradstats_gauss_new.txt OUT-DIFF plots/out-diff.png 0.025
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
    'OUT-GD':  {'ylims':(  0.00, 2.00 ), 'title':'GD Test Loss'}, 
    'OUT-SGD': {'ylims':(  0.00, 2.00 ), 'title':'SGD Test Loss'},
    'OUT-DIFF':{'ylims':( -0.15, 0.15 ), 'title':'Test-Time Benefit of Stochasticity'},
    'GEN-GD':  {'ylims':( -0.05, 0.15 ), 'title':'GD Generalization Gap'},
    'GEN-SGD': {'ylims':( -0.05, 0.15 ), 'title':'SGD Generalization Gap'},
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
    gradstat_means = {}
    gradstat_sdevs = {}
    for ln in filter(None, lines[1:]):
        name, mean, sdev = ln.split() 
        gradstat_means[name] = float(mean)
        gradstat_sdevs[name] = float(sdev) * 1.96 / MTRIALS**0.5

print(gradstat_means)

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

def evaluate(string): 
    for s in gradstat_means.keys():
        string = string.replace('(%s)'%s, 'gradstat_means["%s"]'%s) 
    return eval(string)
def svaluate(string): 
    for s in gradstat_means.keys():
        string = string.replace('(%s)'%s, 'gradstat_sdevs["%s"]'%s) 
    return eval(string)
def choose(T, n):
    assert (n==int(n)) and (0<=n)
    p = 1.0
    while n != 0:
        p *= T
        p /= float(n)
        n -= 1
        T -= 1
    return p

## GEN-SGD:
#E_lin = X*(
#    choose(T, 1) * evaluate('(1.0/N) * ((AA) - (A_A))')
#)
#E_qua = E_lin - X*X*(
#    choose(T, 2) * evaluate('(2.0/N)*((AAb_B) - (A_Ab_B)) + (1.0/N)*((AB_Ab)-(A_Ab_B))') +
#    choose(T, 1) * evaluate('(1.0/(2*N))*((AAbB) - (AB_Ab))')
#)
#E_cub=None
#
#S_lin = X*(
#    choose(T, 1) * svaluate('(1.0/N) * ((AA) + (A_A))')
#)
#S_qua = S_lin + X*X*(
#    choose(T, 2) * svaluate('(2.0/N)*((AAb_B) + (A_Ab_B)) + (1.0/N)*((AB_Ab) + (A_Ab_B))') +
#    choose(T, 1) * svaluate('(1.0/(2*N))*((AAbB) + (AB_Ab))')
#)
#S_cub=None

##OUT-SGD:
E_lin = evaluate('(Loss) - X*T*(A_A)')
E_qua = E_lin + X*X*(
    choose(T, 2) * evaluate('(3.0/2)*(A_Ab_B)') +
    choose(T, 1) * evaluate('(1.0/2)*(AB_Ab)')
)
E_cub = E_qua - X*X*X*(
    choose(T, 3) * evaluate('(5.0/2)*(A_Ab_Bc_C) + (2.0/3)*(A_Abc_B_C)') +
    choose(T, 2) * evaluate('(ABc_Ab_C) + (2.0/3)*(AB_Abc_C) + (AC_Ab_Bc)') +
    choose(T, 1) * evaluate('(1.0/6)*(ABC_Abc)') 
)

S_lin = svaluate('(Loss) + X*T*(A_A)')
S_qua = S_lin + X*X*(
    choose(T, 2) * svaluate('(3.0/2)*(A_Ab_B)') +
    choose(T, 1) * svaluate('(1.0/2)*(AB_Ab)')
)
S_cub = S_qua + X*X*X*(
    choose(T, 3) * svaluate('(5.0/2)*(A_Ab_Bc_C) + (2.0/3)*(A_Abc_B_C)') +
    choose(T, 2) * svaluate('(ABc_Ab_C) + (2.0/3)*(AB_Abc_C) + (AC_Ab_Bc)') +
    choose(T, 1) * svaluate('(1.0/6) * (ABC_Abc)') 
)

a = +evaluate('(Loss)')
b = -evaluate('T*(A_A)')
c = +2.0*( 
    choose(T, 2) * evaluate('(3.0/2)*(A_Ab_B)') +
    choose(T, 1) * evaluate('(1.0/2)*(AB_Ab)')
)
d = -6.0*(
    choose(T, 3) * evaluate('(5.0/2)*(A_Ab_Bc_C) + (2.0/3)*(A_Abc_B_C)') +
    choose(T, 2) * evaluate('(ABc_Ab_C) + (2.0/3)*(AB_Abc_C) + (AC_Ab_Bc)') +
    choose(T, 1) * evaluate('(1.0/6)*(ABC_Abc)') 
)

A = np.log(a)
B = b/a
C = (c-B*B)/a
D = (d-3*B*C+B*B*B)/a

E_lin = np.exp(A + B*X)
E_qua = np.exp(A + B*X + C*X*X/2)
E_cub = np.exp(A + B*X + C*X*X/2 + D*X*X*X/6)



################################################################################
#           2. PLOT CURVES                                                     #
################################################################################

if STATISTIC == 'gen':
    Y = np.array(Y_out) - np.array(Y_ins)
    S = np.array(S_out) + np.array(S_ins)
else:
    Y = np.array(Y_out)
    S = np.array(S_out)



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
plot_fill(X, E_cub, S_cub, blue, '3rd order prediction') if E_cub is not None else None 
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
