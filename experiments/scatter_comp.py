''' author: samtenka
    change: 2019-08-21
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
from predictor import sgd_gd_diff, sgd_test_taylor, sgd_gen, sgd_test_multiepoch, sgd_test_multiepoch_exponential, sgd_test_multiepoch_diff_e2h2, sgd_test_exponential
from optimlogs import OptimKey
from mnist_landscapes import MnistLeNet 
from utils import prod
import sys 

def get_grad_stats(gradstats_filenm):
    with open(gradstats_filenm) as f:
        gradstats = eval(f.read())
    return gradstats

def get_optimlogs(optimlogs_filenm, metric, optimizer, beta, T, eta):
    with open(optimlogs_filenm) as f:
        ol = eval(f.read())

    X, Y, S = [], [], []
    for okey in ol:
        if okey.optimizer != optimizer: continue
        if okey.metric != metric: continue
        if okey.beta != beta: continue
        if abs(okey.eta-eta) > eta*1e-6: continue
        if okey.T != T: continue
        print(okey)
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
gray   ='#888888'

def prime_plot():
    plt.clf()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

def finish_plot(title, xlabel, ylabel, img_filenm):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    handles, labels = plt.gca().get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.gca().legend(handles, labels, loc='upper left')

    #plt.legend(loc='best')
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
        # middle, top, and bottom stroke of S, respectively:
        plt.plot([xx,   xx  ], [yy-z*ss, yy+z*ss], color=color)
        plt.plot([xx-e, xx+0], [yy-z*ss, yy-z*ss], color=color)
        plt.plot([xx-0, xx+e], [yy+z*ss, yy+z*ss], color=color)
    # connect to the figure legend:
    plt.plot([xx, xx], [yy-z*ss, yy+z*ss], color=color, label=label)

def plot_scatter(x, y, color, label, z=1.96, ang=0.0, bar_width=1.0/50, sides=3, op=0.0): 
    ''' plot plus-signs around a scatter plot '''
    e = bar_width * ((max(x)-min(x)) * (max(y)-min(y)))**0.5
    for (xx, yy) in zip(x, y):
        # vert and hori stroke of +, respectively:
        for s in range(sides):
            a = ang + 2*np.pi * float(s)/sides
            plt.plot(
                [xx + e*np.sin(a) + e*op*np.cos(a), xx - e*np.sin(a) + e*op*np.cos(a)],
                [yy - e*np.cos(a) + e*op*np.sin(a), yy + e*np.cos(a) + e*op*np.sin(a)],
            color=color)
        #plt.plot([xx - e*np.cos(ang), xx + e*np.cos(ang)], [yy - e*np.sin(ang), yy + e*np.sin(ang)], color=color)
    # connect to the figure legend:
    plt.plot([xx,   xx  ], [yy   , yy   ], color=color, label=label)

def interpolate(x):
    return np.arange(0.00, 1.01, 0.01)*(max(x)-min(x)) + min(x)

    #--------------------------------------------------------------------------#
    #               2.1 plot curves                                            #
    #--------------------------------------------------------------------------#

def get_ranks(nparr):
    # thanks to stackoverflow.com/questions/5284646
    temp = nparr.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(nparr))
    return ranks

def plot_test_batchmatch(img_filenm, T=100):
    prime_plot()

    oldsA = []
    oldsB = []
    for idxs, ETAS, olds in [([0,1,2,3], [0.05, 0.10, 0.15, 0.20], oldsA), ([4,5,6,7], [0.04, 0.08, 0.12, 0.16, 0.20], oldsB)]:
        for ETA in ETAS:
            expers_diffc = [] 
            expers_diff  = [] 
            degr2s = []
            for idx in idxs:
                gradstats_filenm = 'new-data/gs-lenet-{:02d}.data'.format(idx) 
                optimlogs_filenm = 'new-data/ol-lenet-covreg-long-small-2n-{:02d}.data'.format(idx)

                (X, Y, S), okey = get_optimlogs(optimlogs_filenm, metric='test', optimizer='diffc',T=T, beta=1.0, eta=ETA) 
                expers_diffc.append(Y[0])
                (X, Y, S), okey = get_optimlogs(optimlogs_filenm, metric='test', optimizer='diff', T=T, beta=0.0,eta=ETA) 
                expers_diff.append(Y[0])

                gradstats = get_grad_stats(gradstats_filenm) 
                Y, S = sgd_gd_diff(gradstats, eta=X, T=okey.T, degree=2, N=okey.N) 
                degr2s.append(-Y[0])
             
            expers_diffc = np.array(expers_diffc)
            expers_diff = np.array(expers_diff)
            degr2s = np.array(degr2s)

            olds.append((expers_diffc, expers_diff, degr2s))

    ITER = [([0,1,2,3], [0.05, 0.10, 0.15, 0.20], oldsA), ([4,5,6,7], [0.04, 0.08, 0.12, 0.16, 0.20], oldsB)]
    for idxs, ETAS, olds in ITER:
        for idx in idxs:
            diffc = np.array([olds[i][0][idx-min(idxs)] for i in range(len(ETAS))])
            diff  = np.array([olds[i][1][idx-min(idxs)] for i in range(len(ETAS))])
            degr2 = np.array([olds[i][2][idx-min(idxs)] for i in range(len(ETAS))])

            plt.plot(0.0*degr2, diff, label='no change' if idx==4 else None, color=blue)
            plot_scatter(diffc, diff, label=None, color=cyan)
            plot_fill(diffc, diff, 0.0*diff, label='gd with batchmatch' if idx==4 else None, color=cyan)

            plt.plot(diff, diff, label='identity' if idx==4 else None, color=red)
            plot_scatter(degr2, diff, label=None, color=magenta)
            plot_fill(degr2, diff, 0.0*diff, label='gd' if idx==4 else None, color=magenta)

            if min(degr2)==min(olds_[i][2][ii-min(idxs_)] for idxs_, ETAS_, olds_ in ITER for i in range(len(olds_)) for ii in idxs_):
                for dgr, dif, eta in zip(degr2, diff, ETAS):
                    plt.text(dgr-0.005, dif+0.005, '\u03B7 = {}'.format(eta), color='gray')

    plt.axis('equal')

    finish_plot(
        title='TestLoss Prediction \n(after 1 epoch on {} points)'.format(
            T
        ), ylabel='actual difference from sgd', xlabel='predicted difference from sgd', img_filenm=img_filenm
    )




def plot_test_conv(img_filenm, T=100, ETAS=[0.002, 0.006, 0.010, 0.014, 0.018, 0.022, 0.026, 0.030, 0.038, 0.046, 0.054, 0.058, 0.066, 0.074, 0.082]):#, 0.10]):#, 0.12]):#, #0.14, 0.16, 0.18, 0.20]):
    prime_plot()

    olds = []
    for ETA in ETAS:
        expers = [] 
        degr0s = []
        degr1s = []
        degr2s = []
        degr3s = []
        for idx in [0, 1, 2, 3]:
            gradstats_filenm2 = 'from-om/gs-lenet-converged-{:02d}.ord2.data'.format(idx) 
            gradstats_filenm3 = 'from-om/gs-lenet-converged-{:02d}.ord3.data'.format(idx) 

            optimlogs_filenm, BETA = (
                ('from-om/ol-lenet-converged-smalleretaT-fine-{:02d}.data'.format(idx), None)
                if ETA < 0.016 else
                ('from-om/ol-lenet-converged-smallerishetaT-fine-{:02d}.data'.format(idx), None)
                if ETA < 0.028 else
                ('from-om/ol-lenet-converged-smallerishishetaT-fine-{:02d}.data'.format(idx), None)
                if ETA < 0.056 else
                ('from-om/ol-lenet-converged-smallerishishishetaT-fine-{:02d}.data'.format(idx), None)
                if ETA < 0.09 else
                ('from-om/ol-lenet-converged-bigetaT-fine-{:02d}.data'.format(idx), None)
            )

            (X, Y, S), okey = get_optimlogs(optimlogs_filenm, metric='test', optimizer='sgd', beta=None, T=T, eta=ETA)
            assert abs(X[0]-ETA) <= ETA*1e-6

            gradstats2 = get_grad_stats(gradstats_filenm2) 
            gradstats3 = get_grad_stats(gradstats_filenm3) 
            gradstats = {k:(gradstats2[k] if gradstats2[k]['mean'] is not None else gradstats3[k]) for k in gradstats2}
            expers.append(Y[0]- gradstats['()(0)']['mean'])
            degr0s.append(gradstats['()(0)']['mean'])
            degr1s.append(sgd_test_taylor(gradstats, eta=X, T=T, degree=1)[0][0]- gradstats['()(0)']['mean']) 
            degr2s.append(sgd_test_exponential(gradstats, eta=X, T=T, degree=2)[0][0]- gradstats['()(0)']['mean'])
            degr3s.append(sgd_test_exponential(gradstats, eta=X, T=T, degree=3)[0][0]- gradstats['()(0)']['mean'])
         
        expers = np.array(expers)
        degr0s = np.array(degr0s)
        degr1s = np.array(degr1s)
        degr2s = np.array(degr2s)
        degr3s = np.array(degr3s)

        olds.append((expers, degr0s, degr1s, degr2s, degr3s))
        expers = interpolate(expers)

    for idx in [0,1,2,3]:
        e = np.array([olds[i][0][idx-1] for i in range(len(ETAS))]) - olds[0][0][idx-1]
        offsetx, offsety = ((idx%2)*2-1)*0.003 - np.mean(e), ((idx//2)*2-1)*0.003 - np.mean(e)
        for color, deg in [(red, 1),(yellow, 2),(green, 3)]:
            plot_fill(
                np.array([olds[i][deg+1][idx-1] for i in range(len(ETAS))]) - olds[0][deg+1][idx-1] + offsetx,
                e + offsety,
                np.array([0.0 for i in range(len(ETAS))]),
                label='deg {} ode'.format(str(deg)) if idx==0 else None, color=color
            )
        plot_fill(
            e + offsetx,
            e + offsety,
            np.array([0.0 for i in range(len(ETAS))]),
            label='ideal' if idx==0.0 else None, color='blue'
        )

    plt.plot([0.000, 0.000], [-0.001, +0.001], color='gray')
    plt.plot([-0.0001, +0.0001], [-0.001, -0.001], color='gray')
    plt.plot([-0.0001, +0.0001], [+0.001, +0.001], color='gray')
    plt.plot([-0.001, +0.001], [0.000, 0.000]    , color='gray')
    plt.plot([-0.001, -0.001], [-0.0001, +0.0001], color='gray')
    plt.plot([+0.001, +0.001], [-0.0001, +0.0001], color='gray')

    plt.text(0.0002, 0.0002, '0.002', color='gray')

    plt.axis('equal')
    #plt.axis('square')

    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])

    finish_plot(
        title='TestLoss Prediction \n(after 1 epoch on {} points)'.format(
            T
        ), xlabel='predicted test loss decrease', ylabel='actual test loss decrease', img_filenm=img_filenm
    )


     
def plot_test(img_filenm, T=100, ETAS=[0.005, 0.010, 0.015, 0.020, 0.025]):
    prime_plot()

    olds = []
    for ETA in ETAS:
        expers = [] 
        degr0s = []
        degr1s = []
        degr2s = []
        degr3s = []
        for idx in [1, 2, 3]:
            gradstats_filenm = 'new-data/gs-lenet-{:02d}.data'.format(idx) 
            optimlogs_filenm = 'from-om/ol-lenet-GENO-shorterA-{:02d}.data'.format(idx)

            (X, Y, S), okey = get_optimlogs(optimlogs_filenm, metric='test', optimizer='sgd', beta=0.0, T=T, eta=ETA)
            assert X[0]==ETA
            gradstats = get_grad_stats(gradstats_filenm) 
            expers.append(Y[0])
            degr0s.append(gradstats['()(0)']['mean'])
            degr1s.append(sgd_test_taylor(gradstats, eta=X, T=T, degree=1)[0][0])# - gradstats['()(0)']['mean']) 
            degr2s.append(sgd_test_taylor(gradstats, eta=X, T=T, degree=2)[0][0])# - gradstats['()(0)']['mean'])
            degr3s.append(sgd_test_taylor(gradstats, eta=X, T=T, degree=3)[0][0])# - gradstats['()(0)']['mean'])
         
        expers = np.array(expers)
        degr0s = np.array(degr0s)
        degr1s = np.array(degr1s)
        degr2s = np.array(degr2s)
        degr3s = np.array(degr3s)

        olds.append((expers, degr0s, degr1s, degr2s, degr3s))

        #plt.text(np.mean(degr3s)+0.05, np.mean(expers)+0.05,'\u03B7 = {}'.format(ETA))

        #ang = np.random.random() * 2*np.pi
        #plot_scatter(degr0s, expers, magenta, 'deg 0 poly' if ETA==0.005 else None, ang=ang,  sides=1, op=0.0, bar_width=0.02)

        ang = np.random.random() * 2*np.pi
        plot_scatter(degr1s, expers, red, 'deg 1 poly' if ETA==0.005 else None, ang=ang,  sides=2+int(ETA/0.005), op=0.0, bar_width=0.015)

        ang = np.random.random() * 2*np.pi
        plot_scatter(degr2s, expers, yellow, 'deg 2 poly' if ETA==0.005 else None, ang=ang,  sides=2+int(ETA/0.005), op=0.0, bar_width=0.015)

        ang = np.random.random() * 2*np.pi
        plot_scatter(degr3s, expers, green, 'deg 3 poly' if ETA==0.005 else None, ang=ang,  sides=2+int(ETA/0.005), op=0.0, bar_width=0.015)

        expers = interpolate(expers)
        plot_fill(expers, expers, expers*0.0, label='ideal' if ETA==0.005 else None, color='blue')

    for idx in [1,2,3]:
        for color, deg in [(red, 1),(yellow, 2),(green, 3)]:
            plot_fill(
                np.array([olds[i][deg+1][idx-1] for i in range(len(ETAS))]),
                np.array([olds[i][0][idx-1] for i in range(len(ETAS))]),
                np.array([0.0 for i in range(len(ETAS))]),
                label=None, color=color
            )

    plt.axis('equal')
    #plt.axis('square')

    #plt.gca().axes.xaxis.set_ticklabels([])
    #plt.gca().axes.yaxis.set_ticklabels([])

    finish_plot(
        title='TestLoss Prediction \n(after 1 epoch on {} points)'.format(
            T
        ), xlabel='predicted test loss', ylabel='actual test loss', img_filenm=img_filenm
    )


def plot_gene(img_filenm, T=100, ETAS=[0.005, 0.015, 0.025, 0.035, 0.045]):
    prime_plot()

    olds = []
    for ETA in ETAS:
        expers = [] 
        degr1s = []
        degr2s = []
        degr3s = []
        for idx in [1, 2, 3]:
            gradstats_filenm = 'new-data/gs-lenet-{:02d}.data'.format(idx) 
            optimlogs_filenm = 'from-om/ol-lenet-GENO-shorterA-{:02d}.data'.format(idx)

            (X, Y, S), okey = get_optimlogs(optimlogs_filenm, metric='gen', optimizer='sgd', beta=0.0, T=T, eta=ETA)
            assert X[0]==ETA
            expers.append(Y[0])

            gradstats = get_grad_stats(gradstats_filenm) 
            degr1s.append(sgd_gen(gradstats, eta=X, T=T, degree=1)[0][0]) 
            degr2s.append(sgd_gen(gradstats, eta=X, T=T, degree=2)[0][0])
            if idx != 0:
                degr3s.append(sgd_gen(gradstats, eta=X, T=T, degree=3)[0][0])
         
        expers = np.array(expers)
        degr1s = np.array(degr1s)
        degr2s = np.array(degr2s)
        degr3s = np.array(degr3s)

        olds.append((expers, None, degr1s, degr2s, degr3s))

        plt.text(np.mean(degr3s)+0.005, np.mean(expers)*0.9, '\u03B7 = {}'.format(ETA))

        ang = np.random.random() * 2*np.pi
        plot_scatter(degr1s, expers, red, 'deg 1 poly' if ETA==0.005 else None, ang=ang,  sides=3, op=0.0, bar_width=0.25)

        ang = np.random.random() * 2*np.pi
        plot_scatter(degr2s, expers, yellow, 'deg 1 poly' if ETA==0.005 else None, ang=ang,  sides=5, op=1.3, bar_width=0.25)

        if idx != 0:
            ang = np.random.random() * 2*np.pi
            plot_scatter(degr3s, expers, green, 'deg 3 poly' if ETA==0.005 else None, ang=ang,  sides=4, op=0.0, bar_width=0.45)

        expers = np.array([0.0] + list(expers))
        plot_fill(expers, expers, expers*0.0, label='ideal' if ETA==0.005 else None, color='blue')

    #for idx in [1,2,3]:
    #    for color, deg in [(red, 1),(yellow, 2),(green, 3)]:
    #        plot_fill(
    #            np.array([olds[i][deg+1][idx-1] for i in range(len(ETAS))]),
    #            np.array([olds[i][0][idx-1] for i in range(len(ETAS))]),
    #            np.array([0.0 for i in range(len(ETAS))]),
    #            label=None, color=color
    #        )

    plt.axis('equal')
    #plt.axis('square')

    #plt.gca().axes.xaxis.set_ticklabels([])
    #plt.gca().axes.yaxis.set_ticklabels([])

    finish_plot(
        title='GenGap Prediction \n(after 1 epoch on {} points)'.format(
            T
        ), xlabel='predicted gen gap', ylabel='actual gen gap', img_filenm=img_filenm
    )


def plot_ms(img_filenm):

    models = [
        (32, 64, 0.15, 1000, green),
        (32, 32, 0.15, 1000, green),
        (32, 16, 0.15, 1000, green),
        (16, 32, 0.15, 1000, green),
        (16, 16, 0.15, 1000, green),
        (16, 8 , 0.15, 1000, green),
        (8, 32 , 0.15, 1000, green),
        (8, 16 , 0.15, 1000, green),
        (8,  8 , 0.15, 1000, green),

 
        (32, 64, 0.1, 1000, yellow),
        (32, 32, 0.1, 1000, yellow),
        (32, 16, 0.1, 1000, yellow),
        (16, 32, 0.1, 1000, yellow),
        (16, 16, 0.1, 1000, yellow),
        (16, 8 , 0.1, 1000, yellow),
        (8, 32 , 0.1, 1000, yellow),
        (8, 16 , 0.1, 1000, yellow),
        (8,  8 , 0.1, 1000, yellow),

        #(32, 64, 0.05, 1000, red),
        #(16, 32, 0.05, 1000, red),
        #(16, 8 , 0.05, 1000, red),
        #(8, 32 , 0.05, 1000, red),

        #( 8,  8, 0.025, 1000, red), 
        #( 8, 32, 0.025, 1000, red), 
        #(16,  8, 0.025, 1000, red), 
        #(16, 32, 0.025, 1000, red), 
        #(32, 16, 0.025, 1000, red), 
        #(32, 32, 0.025, 1000, red), 
        #(32, 64, 0.025, 1000, red), 

        ( 8,  8, 0.005, 1000, red), 
        ( 8, 32, 0.005, 1000, red), 
        (16,  8, 0.005, 1000, red), 
        (16, 32, 0.005, 1000, red), 
        (32, 16, 0.005, 1000, red), 
        (32, 32, 0.005, 1000, red), 
        (32, 64, 0.005, 1000, red), 
    ]
    expers = [] 
    degr0s = []
    degr1s = []
    degr2s = []
    degr3s = []
    params= []
    colors = []
    mode = 'LOSS' #'ACC'
    for widthA, widthB, ETA, T, color in models:
        gradstats_filenm2 = 'from-om/gs-new-lenet-ms-0-{}-{:02d}.ord2.data'.format(widthA, widthB) 
        gradstats_filenm3 = 'from-om/gs-new-lenet-ms-0-{}-{:02d}.ord3.data'.format(widthA, widthB) 
        for optimlogs_filenm in [
                'from-om/ol-lenet-ms-{}-{}-00.data'.format(widthA, widthB),
                'from-om/ol-lenet-smallereta-ms-{}-{}-00.data'.format(widthA, widthB),
                'from-om/ol-lenet-eta25-ms-{}-{}-00.data'.format(widthA, widthB),
                'from-om/ol-lenet-smalleta-ms-{}-{}-00.data'.format(widthA, widthB),
                'from-om/ol-lenet-eta15-ms-{}-{}-00.data'.format(widthA, widthB),
                ]:
            try:
                (X, Y, S), okey = get_optimlogs(optimlogs_filenm, metric='testacc'if mode=='ACC' else 'test', optimizer='sgd', beta=None, T=T, eta=ETA)
                assert len(X)
                break
            except:
                continue
        else:
            print('problem with', widthA, widthB)
            continue

        params.append(
            sum(prod(w) for w in MnistLeNet(widthA=widthA, widthB=widthB).subweight_shapes)
        )

        expers.append(Y[0])

        gradstats2 = get_grad_stats(gradstats_filenm2) 
        gradstats3 = get_grad_stats(gradstats_filenm3) 
        gradstats = {k:(gradstats2[k] if gradstats2[k]['mean'] is not None else gradstats3[k]) for k in gradstats2}
        degr0s.append(gradstats['()(0)']['mean'])
        degr1s.append(sgd_test_taylor(gradstats, eta=X, T=T, degree=1)[0][0]) 
        EPOCH_APPROX=10
        degr3s.append(sgd_test_multiepoch_exponential(gradstats, eta=X, T=T//EPOCH_APPROX, degree=3, E=EPOCH_APPROX)[0][0])
        #degr2s.append(sgd_test_taylor(gradstats, eta=X, T=T//2, degree=2)[0][0])
        #degr3s.append(sgd_test_exponential(gradstats, eta=X, T=T, degree=3)[0][0])
        colors.append(color)
     
    print(params)
    prime_plot()
    expersr = get_ranks(+np.array(expers))
    if mode=='ACC':
        degr0sr = get_ranks(-np.array(degr0s))
        paramsr = get_ranks(-np.array(params))
    else:
        degr0sr = get_ranks(+np.array(degr0s))
        paramsr = get_ranks(+np.array(params))
    degr1sr = get_ranks(np.array(degr1s))
    degr2sr = get_ranks(np.array(degr2s))
    degr3sr = get_ranks(np.array(degr3s))

    for color in set(colors): 

        indices = np.array([i for i,c in enumerate(colors) if c==color])
        if not len(indices): continue
        ang = np.random.random() * 2*np.pi
        plot_scatter(degr0sr[indices], expersr[indices], cyan, 'init loss' if 0 in indices else None,  ang=ang, sides=5, op=1.3, bar_width=0.010)
        plot_scatter(paramsr[indices], expersr[indices], magenta, 'nb params' if 0 in indices else None,  ang=ang, sides=5, op=1.3, bar_width=0.010)
        plot_scatter(degr3sr[indices], expersr[indices], color, 'deg 3 ode', ang=ang,  sides=3, op=0.7, bar_width=0.05)

        for m in models:
            if m[4]!=color: continue
            ETA = m[2]
            break
        if mode=='ACC':
            plt.text(np.mean(degr3sr[indices])+2.5, np.mean(expersr[indices])+2.5,'\u03B7 = {}'.format(ETA), color=color)
        else:
            plt.text(np.mean(degr3sr[indices])-4.5, np.mean(expersr[indices])+2.5,'\u03B7 = {}'.format(ETA), color=color)

    if mode=='ACC':
        plot_fill(expersr, len(expersr)-1-expersr, expersr*0.0, label='ideal', color='blue')
    else:
        plot_fill(expersr, expersr, expersr*0.0, label='ideal', color='blue')
    plt.axis('equal')

    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])



    finish_plot(
        title='Model Selection \n(after {} epochs on {} points)'.format(
            10,
            T//10,
            ETA
        ), xlabel='predicted loss rank', ylabel='actual {} rank'.format('accuracy' if mode=='ACC' else 'loss'), img_filenm=img_filenm
    )

#plot_ms('model-selection.png')
#plot_gene('gen-gap.png')
#plot_test('test-loss.png')
#plot_test_conv('test-loss-converged.png')
plot_test_batchmatch('test-loss-bm.png')
