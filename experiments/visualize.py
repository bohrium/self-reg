''' author: samtenka
    change: 2019-08-17
    create: 2019-02-14
    descrp: Compare and plot descent losses as dependent on learning rate.
            Valid plotting modes are
                test-gd,  test-sgd,  test-gdc,  test-diff,  test-all
                train-gd, train-sgd, train-gdc, train-diff, train-all
                gen-gd,   gen-sgd,   gen-gdc,   gen-diff,   gen-all
            To run, type:
                python visualize.py new-data/ol-lenet-00.data new-data/gs-lenet-00.data test-sgd test-sgd-lenet-00.png
            The   optimlogs.data   gives   a filename storing descent trajectory summaries;
            the   gradstats.data   gives   a filename storing gradient statistic estimates;
            the   test-DIFF         gives   a plotting mode
            the   out-diff.png     gives   a filename to write to 
'''

from matplotlib import pyplot as plt
import numpy as np
from predictor import sgd_test_taylor, sgd_gen, sgd_test_multiepoch, sgd_test_multiepoch_diff_e2h2, sgd_test_exponential
from optimlogs import OptimKey
import sys 

assert len(sys.argv)==1+4, '`visualize.py` needs 4 command line arguments'
OPTIMLOGS_FILENM, GRADSTATS_FILENM, MODE, IMG_FILENM = sys.argv[1:] 

with open(GRADSTATS_FILENM) as f:
    gradstats = eval(f.read())

def get_optimlogs(optimlogs_filenm, metric, optimizer, beta):
    with open(optimlogs_filenm) as f:
        ol = eval(f.read())

    X, Y, S = [], [], []
    for okey in ol:
        if okey.optimizer != optimizer: continue
        if okey.metric != metric: continue
        if okey.beta != beta: continue
        X.append(okey.eta)
        Y.append(ol[okey]['mean'])
        S.append(ol[okey]['stdv']/ol[okey]['nb_samples']**0.5)
    X = np.array(X)
    Y = np.array(Y)
    S = np.array(S)

    return (X,Y,S), okey 
        
    #--------------------------------------------------------------------------#
    #               2.1 plotting primitives                                    #
    #--------------------------------------------------------------------------#

red    ='#cc4444'
yellow ='#aaaa44'
green  ='#44cc44'
cyan   ='#44aaaa'
blue   ='#4444cc'
magenta='#aa44aa'

def prime_plot():
    plt.clf()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

def finish_plot(title, xlabel, ylabel, img_filenm):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.savefig(img_filenm, pad_inches=0.05, bbox_inches='tight')

def plot_fill(x, y, s, color, label, z=1.96):
    ''' plot variance (s^2) around mean (y) via 2D shading around a curve '''
    plt.plot(x, y, color=color, alpha=0.5)
    plt.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([y-z*s, (y+z*s)[::-1]]),
        facecolor=color, alpha=0.5, label=label
    )

def plot_bars(x, y, s, color, label, z=1.96, bar_width=1.0/50): 
    ''' plot variance (s^2) around mean (y) via S-bars around a scatter plot '''
    e = bar_width * (max(x)-min(x))
    for (xx, yy, ss) in zip(x, y, s):
        # middle, top, and bottom stroke of I, respectively:
        plt.plot([xx,   xx  ], [yy-z*ss, yy+z*ss], color=color)
        plt.plot([xx-e, xx+0], [yy-z*ss, yy-z*ss], color=color)
        plt.plot([xx-0, xx+e], [yy+z*ss, yy+z*ss], color=color)
    # connect to the figure legend:
    plt.plot([xx, xx], [yy-z*ss, yy+z*ss], color=color, label=label)

def interpolate(x):
    return np.arange(0.00, 1.01, 0.01)*(max(x)-min(x)) + min(x)

    #--------------------------------------------------------------------------#
    #               2.1 plot curves                                            #
    #--------------------------------------------------------------------------#

metric, optimizer = MODE.split('-') 

def plot_GEN():
    prime_plot()

    (X, Y, S), okey = get_optimlogs(OPTIMLOGS_FILENM, metric, optimizer, beta=0.0) 
    X, Y, S = (np.array([0.0]+list(nparr)) for nparr in (X,Y,S))
    plot_bars(X, Y, S, color=blue, label='experiment')

    Y, S = sgd_gen(gradstats, eta=X, T=okey.T, degree=1) 
    plot_fill(X, Y, S, color=red, label='theory (deg 1 poly)')

    Y, S = sgd_gen(gradstats, eta=X, T=okey.T, degree=2) 
    plot_fill(X, Y, S, color=yellow, label='theory (deg 2 poly)')



    finish_plot(
        title='Prediction of SGD \n(gen loss after {} steps on mnist-10 lenet)'.format(
            okey.T
        ), xlabel='learning rate', ylabel='gen loss', img_filenm=IMG_FILENM
    )


def plot_SGD():
    prime_plot()

    (X, Y, S), okey = get_optimlogs(OPTIMLOGS_FILENM, metric, optimizer, beta=0.0) 
    plot_bars(X, Y, S, color=blue, label='experiment')
    
    X = interpolate(X)

    Y, S = sgd_test_taylor(gradstats, eta=X, T=okey.T, degree=1) 
    plot_fill(X, Y, S, color=red, label='theory (deg 1 poly)')
    
    Y, S = sgd_test_taylor(gradstats, eta=X, T=okey.T, degree=2) 
    plot_fill(X, Y, S, color=yellow, label='theory (deg 2 poly)')
    #
    Y, S = sgd_test_taylor(gradstats, eta=X, T=okey.T, degree=3) 
    plot_fill(X, Y, S, color=green, label='theory (deg 3 poly)')

    #    Y, S = sgd_test_taylor(gradstats, eta=X, T=okey.T, degree=1) 
    #    plot_fill(X, Y, S, color=red, label='theory (deg 1 ode)')
    #
    #Y, S = sgd_test_exponential(gradstats, eta=X, T=okey.T, degree=2)
    #plot_fill(X, Y, S, color=yellow, label='theory (deg 2 ode)')
    #
    #Y, S = sgd_test_exponential(gradstats, eta=X, T=okey.T, degree=3)
    #plot_fill(X, Y, S, color=green, label='theory (deg 3 ode)')

    finish_plot(
        #title='Prediction of SGD \n(test loss after 100 steps on mnist-10 logistic)'.format(
        title='Prediction of SGD \n(test loss after {} steps on mnist-10 lenet)'.format(
            okey.T
        ), xlabel='learning rate', ylabel='test loss', img_filenm=IMG_FILENM
    )


def plot_OPT(): 
    prime_plot()

    #for opt, beta, color in [('sgd', 0.0, cyan), ('gd', 0.0, blue)]:#, ('gdc', 1.00, magenta)]:
    #for opt, beta, color in [('sgd', 0.0, cyan), ('gdc', 1.00, magenta)]:
    #for opt, beta, color in [('sgd', 0.0, cyan), ('gd', 0.0, blue), ('gdc', 1.00, magenta)]:
    for opt, beta, color in [('diffc', 1.0, cyan), ('diff', 0.0, magenta)]:#[('sgd', 0.0, cyan), ('gd', 0.0, blue), ('gdc', 1.0, magenta)]:
        (X, Y, S), okey = get_optimlogs(OPTIMLOGS_FILENM, metric, opt, beta) 
        plot_bars(X, Y, S, color=color, label=opt)

    finish_plot(
        title='Comparison of Optimizers \n({} after {} steps on mnist-10 lenet)'.format(
            metric,
            okey.T
        ), xlabel='learning rate', ylabel=metric, img_filenm=IMG_FILENM
    )

def plot_BETA_SCAN(): 
    prime_plot()

    #for beta, color in [(10**-3.0, green), (10**-2.5, cyan), (10**-2.0, blue), (10**-1.5, magenta), (10**-1.0, red)]:
    #for beta, color in [(0.25, green), (0.5, cyan), (1.0, blue), (2.0, magenta), (4.0, red)]:
    #for beta, color in [(0.25, green), (0.5, cyan), (1.0, blue)]:#, (2.0, magenta), (4.0, red)]:
    #for beta, color in [(0.0, green), (0.25, cyan)]:
    for beta, color in [(0.0, green), (0.25, cyan), (0.5, blue), (1.0, magenta), (2.0, red), (4.0, yellow)]:
        (X, Y, S), okey = get_optimlogs(OPTIMLOGS_FILENM, metric, 'sgdc', beta) 
        plot_bars(X, Y, S, color=color, label='sgdc {:.2e}'.format(beta))

    finish_plot(
        title='Comparison of Optimizers \n({} after {} steps per epoch for 10 epochs on mnist-10 lenet)'.format(
            metric,
            okey.T
        ), xlabel='learning rate', ylabel=metric, img_filenm=IMG_FILENM
    )

def plot_EPOCH(): 
    prime_plot()

    #for opt, beta, color in [('sgd.e2', 0.0, cyan), ('sgd.h2', 0.0, magenta)]:
    for opt, beta, color in [('diff.e2.h2', 0.0, yellow)]:
        (X, Y, S), okey = get_optimlogs(OPTIMLOGS_FILENM, metric, opt, beta) 
        plot_bars(X, Y, S, color=color, label=opt)

    X = interpolate(X)

    #Y, S = sgd_test_taylor(gradstats, eta=2*X, T=okey.T, degree=2) 
    #Y, S = sgd_test_multiepoch(gradstats, eta=2*X, T=okey.T, degree=2, E=1) 
    #Y_, S_ = sgd_test_multiepoch(gradstats, eta=X, T=okey.T, degree=2, E=2) 
    #Y, S = Y_ - Y, S_ + S
    Y, S = sgd_test_multiepoch_diff_e2h2(gradstats, eta=X, T=okey.T, degree=2, E=2) 
    plot_fill(X, Y, S, color=green, label='theory (deg 2 poly)')

    finish_plot(
        title='Comparison of Optimizers \n({} after {} steps on mnist-10 lenet)'.format(
            metric,
            okey.T
        ), xlabel='learning rate', ylabel=metric, img_filenm=IMG_FILENM
    )


#plot_GEN()
#plot_EPOCH()
#plot_SGD()
#plot_OPT()
plot_BETA_SCAN()

