from matplotlib import pyplot as plt
import numpy as np
import re

X = []
Y = []
S = []

with open('results.txt') as f:
    text = f.read() 
    paragraphs = filter(None, text.split('\n'))
    for p in paragraphs: 
        opt = re.search(r'OPT=(\S+)', p).group(1)
        if opt != 'diff': continue
        metric = re.search(r'METRIC=(\S+):', p).group(1)
        if metric != 'OL': continue
        learning_rate = float(re.search(r'LEARNING_RATE=(\S+)', p).group(1))
        if learning_rate >= 0.00011: continue
        nb_trials = float(re.search(r'NB_TRIALS=(\S+)', p).group(1))
        mean = float(re.search(r':\s+(\S+)', p).group(1))
        stddev = float(re.search(r':\s+\S+\s+(\S+)', p).group(1))

        X.append(learning_rate)
        Y.append(mean)
        S.append(1.96 * stddev / np.sqrt(nb_trials))

red='#cc4444'
green='#44cc44'
blue='#4444cc'
X = np.array(X)
Y = np.array(Y)
S = np.array(S)
plt.plot(X, np.zeros(len(X)), color=red)
plt.fill(np.concatenate([X, X[::-1]]), np.concatenate([Y-S, (Y+S)[::-1]]), facecolor=blue, alpha=0.5)
plt.scatter(X, Y, color=blue, marker='x')

AUD, PER = 0.000071, 0.000127
E = 0.25*X*X * ((64*(64-1))/2.0 * (1.0/64) * (AUD+PER) + 64*(1.0/64 - 1) * PER)
plt.plot(X, E, color=green)

plt.xlabel('learning rate')
plt.ylabel('out-of-sample cross entropy')
plt.title('Testtime Benefit of Stochasticity --- vs --- Learning Rate')
plt.savefig('hi.png')
