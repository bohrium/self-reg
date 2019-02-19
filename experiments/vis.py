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

X = []
Y = []
S = []

with open('results_shallow.txt') as f:
    text = f.read() 
    paragraphs = filter(None, text.split('\n'))
    for p in paragraphs: 
        opt = re.search(r'OPT=(\S+)', p).group(1)
        if opt != 'diff': continue
        metric = re.search(r'METRIC=(\S+):', p).group(1)
        if metric != 'OL': continue
        learning_rate = float(re.search(r'LEARNING_RATE=(\S+)', p).group(1))
        #if learning_rate >= 0.005: continue
        nb_trials = float(re.search(r'NB_TRIALS=(\S+)', p).group(1))
        mean = float(re.search(r':\s+(\S+)', p).group(1))
        stddev = float(re.search(r':\s+\S+\s+(\S+)', p).group(1))

        X.append(learning_rate)
        Y.append(mean)
        S.append(1.96 * stddev / np.sqrt(nb_trials))

with open('ests_shallow.txt') as f:
    lines = f.read().split('\n')
    MTRIALS, _ = tuple(float(x) for x in lines[0].split() if x.isnumeric())
    SEN, INT, UNC, PAS, AUD, PER = tuple(float(x) for x in lines[1].replace(',', ' ').split() if x.replace('.','').replace('-','').isnumeric())
    SEN_,INT_,UNC_,PAS_,AUD_,PER_= tuple(float(x) for x in lines[2].replace(',', ' ').split() if x.replace('.','').replace('-','').isnumeric())

red='#cc4444'
green='#44cc44'
blue='#4444cc'

X = np.array(X)

# PLOT theoretical predictions:
N = T = 10
E = SEN - X*T*INT + 0.5*X*X*( (T*(T-1)/4.0)*( PAS + PAS/2.0 ) + (T/2.0)*( PAS/2.0 + PER ) )
E -= SEN - X*T*INT + 0.5*X*X*( (T*(T-1)/4.0)*( PAS + AUD/N + PAS/2.0 + PER/N ) + (T/2.0)*( PAS/2.0 + PER/N ) )
#S_ = 1.96 * (SEN_ + X*T*INT_ + 0.5*X*X*( (T*(T-1)/4.0)*( PAS_ + PAS_/2.0 ) + (T/2.0)*( PAS_/2.0 + PER_ ) )) / MTRIALS**0.5
S_ = 1.96 * (SEN_ + X*T*INT_ + 0.5*X*X*( (T*(T-1)/4.0)*( PAS_ + AUD_/N + PAS_/2.0 + PER_/N ) + (T/2.0)*( PAS_/2.0 + PER_/N ) )) / MTRIALS**0.5
plt.fill(np.concatenate([X, X[::-1]]), np.concatenate([E-S_, (E+S_)[::-1]]), facecolor=green, alpha=0.5)

# PLOT experimental results:
Y = np.array(Y)
S = np.array(S)
e = 1e-5
for (x,y,s) in zip(X, Y, S):
    plt.plot([x, x], [y-s, y+s], color=blue)
    plt.plot([x-e, x+e], [y-s, y-s], color=blue)
    plt.plot([x-e, x+e], [y+s, y+s], color=blue)

plt.xlabel('learning rate')
plt.ylabel('out-of-sample cross entropy')
plt.title('Testtime Benefit of Stochasticity --- vs --- Learning Rate')
plt.savefig('hi.png')
