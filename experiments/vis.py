from matplotlib import pyplot as plt
import numpy as np
import re

'''
    A. mean loss                                    {()}                                    SENTIMENT
    B. trace of square gradient                     {(a)}{(a)}                              INTENSITY
    C. trace of covariance of gradients             {(a)(a)} - {(a)}{(a)}                   UNCERTAINTY
    D. 1st order increase of (B) along gradient     2{(a)}{(ab)}{(b)}                       PASSION
    E. 1st order increase of (C) along gradient     2{(a)}{(ab)(b)} - 2{(a)}{(ab)}{(b)}     AUDACITY
    F. trace of hessian times square gradient       {(ab)}{(a)}{(b)}                        PASSION/2
    G. trace of hessian times covariance            {(ab)}{(a)(b)} - {(ab)}{(a)}{(b)}       PERIL 
'''

# choices: OUT_GD, OUT_SGD, OUT_DIFF, GEN_GD, GEN_SGD
PLOT = 'OUT_DIFF'

X = []
Y = []
Y_in = []
S = []
S_in = []

with open('results_gauss.txt') as f:
    text = f.read() 
    paragraphs = filter(None, text.split('\n'))
    for p in paragraphs: 
        opt = re.search(r'OPT=(\S+)', p).group(1)
        if opt != {'OUT_SGD':'sgd', 'OUT_GD':'gd', 'OUT_DIFF':'diff', 'GEN_GD':'gd', 'GEN_SGD':'sgd'}[PLOT]: continue
        metric = re.search(r'METRIC=(\S+):', p).group(1)
        if metric == 'OL':
            learning_rate = float(re.search(r'LEARNING_RATE=(\S+)', p).group(1))
            #if learning_rate > 0.025: continue
            nb_trials = float(re.search(r'NB_TRIALS=(\S+)', p).group(1))
            mean = float(re.search(r':\s+(\S+)', p).group(1))
            stddev = float(re.search(r':\s+\S+\s+(\S+)', p).group(1))

            X.append(learning_rate)
            Y.append(mean)
            S.append(1.96 * stddev / np.sqrt(nb_trials))
        elif metric == 'IL':
            learning_rate = float(re.search(r'LEARNING_RATE=(\S+)', p).group(1))
            #if learning_rate > 0.025: continue
            nb_trials = float(re.search(r'NB_TRIALS=(\S+)', p).group(1))
            mean = float(re.search(r':\s+(\S+)', p).group(1))
            stddev = float(re.search(r':\s+\S+\s+(\S+)', p).group(1))

            Y_in.append(mean)
            S_in.append(1.96 * stddev / np.sqrt(nb_trials))


with open('ests_gauss.txt') as f:
    lines = f.read().split('\n')
    MTRIALS, _ = tuple(float(x) for x in lines[0].split() if x.isnumeric())
    SEN, INT, UNC, PAS, AUD, PER = tuple(float(x) for x in lines[1].replace(',', ' ').split() if x.replace('.','').replace('-','').isnumeric())
    SEN_,INT_,UNC_,PAS_,AUD_,PER_= tuple(float(x)/MTRIALS**0.5 for x in lines[2].replace(',', ' ').split() if x.replace('.','').replace('-','').isnumeric())

red='#cc4444'
green='#44cc44'
blue='#4444cc'

X = np.array(X)
X_ = X[:]


# PLOT theoretical predictions:
X = np.array(sorted(X))
N = T = 10
E = E_lin = None
if PLOT == 'OUT_GD':
    E  =         SEN  - X*T*INT  + X*X*( (T*(T-1)/2.0)*(0.75*PAS + 0.5*AUD  /N + 0.5*PER /N) + (T)*(0.25*PAS  + 0.5*PER /N))
    E_lin=       SEN  - X*T*INT
    S_ = 1.96 * (SEN_ - X*T*INT_ + X*X*( (T*(T-1)/2.0)*(0.75*PAS_+ 0.5*AUD_ /N + 0.5*PER_/N) + (T)*(0.25*PAS_ + 0.5*PER_/N)))
elif PLOT == 'OUT_DIFF':
    E  =                           X*X*( (T*(T-1)/2.0)*(           0.5*AUD  /N + 0.5*PER /N) + (T)*(-0.5*PER  + 0.5*PER /N))
    E_lin = 0.0*X 
    S_ = 1.96 * (                  X*X*( (T*(T-1)/2.0)*(           0.5*AUD_ /N + 0.5*PER_/N) + (T)*( 0.5*PER_ - 0.5*PER_/N)))
elif PLOT == 'OUT_SGD':
    E  =         SEN  - X*T*INT  + X*X*( (T*(T-1)/2.0)*(0.75*PAS                           ) + (T)*(0.25*PAS  + 0.5*PER ))
    E_lin=       SEN  - X*T*INT 
    S_ = 1.96 * (SEN_ - X*T*INT_ + X*X*( (T*(T-1)/2.0)*(0.75*PAS_                          ) + (T)*(0.25*PAS_ + 0.5*PER_)))
elif PLOT == 'GEN_GD':
    E_lin=      X*T*UNC/N       
    S_ = 1.96 * X*T*UNC_/N
elif PLOT == 'GEN_SGD':
    E  =        X*T*UNC/N        - X*X*( (T*(T-1)/2.0)*(0.5*PAS  + PER  + PAS  + AUD       ) + (T)*(0.0))/N
    E_lin=      X*T*UNC/N       
    S_ = 1.96 * X*T*UNC_/N
else:
    assert False 

if E is not None:
    plt.fill(np.concatenate([X, X[::-1]]), np.concatenate([E-S_, (E+S_)[::-1]]), facecolor=green, label='2nd order prediction')
if E_lin is not None:
    plt.plot(X, E_lin, color=red, label='1st order prediction')

# PLOT experimental results:
if PLOT in ['GEN_GD', 'GEN_SGD']:
    Y = np.array(Y) - np.array(Y_in)
    S = np.array(S) + np.array(S_in)
else:
    Y = np.array(Y)
    S = np.array(S)
e = (max(X_)-min(X_))/50.0
for (x,y,s) in zip(X_, Y, S):
    plt.plot([x, x], [y-s, y+s], color=blue)
    plt.plot([x-e, x+e], [y-s, y-s], color=blue)
    plt.plot([x-e, x+e], [y+s, y+s], color=blue)
plt.plot([x, x], [y-s, y+s], color=blue, label='experiment')




plt.xlabel('learning rate')
plt.ylabel('loss')
if PLOT == 'OUT_DIFF':
                                plt.ylim(-0.01, 0.12)
                                plt.title('Test-Time Benefit of Stochasticity --- vs --- Learning Rate')
elif PLOT == 'OUT_SGD':
                                plt.ylim(0.6, 2.1)
                                plt.title('SGD Test Loss --- vs --- Learning Rate')
elif PLOT == 'OUT_GD':
                                plt.ylim(0.6, 2.1)
                                plt.title('GD Test Loss --- vs --- Learning Rate')
elif PLOT == 'GEN_GD':
                                plt.ylim(-0.05, 0.35)
                                plt.title('GD Generalization Gap --- vs --- Learning Rate')
elif PLOT == 'GEN_SGD':
                                plt.ylim(-0.05, 0.35)
                                plt.title('SGD Generalization Gap --- vs --- Learning Rate')


plt.legend(loc='best')


plt.savefig('plots/%s.png' % PLOT)
