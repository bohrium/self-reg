''' author: samtenka
    change: 2019-08-17
    create: 2019-06-17
    descrp: do gradient descent on landscapes
'''

import numpy as np

from utils import CC
from optimlogs import OptimKey, OptimLog
from landscape import PointedLandscape
from mnist_landscapes import MnistLogistic, MnistLeNet, MnistMLP
from quad_landscapes import Quadratic
import torch
import tqdm

#==============================================================================#
#           0. DEFINE OPTIMIZATION LOOPS (SGD, GD, GDC)                        #
#==============================================================================#

opts = [
    ('SGD', None),
    ('GD', None),
    ('GDC', 1.0),
]
def compute_losses(land, eta, T, N, I=1, idx=None, opts=opts, test_extra=300):
    '''
    '''
    assert N%2==0, 'GDC simulator needs N to be even for covariance estimation'
    ol = OptimLog()

    for i in tqdm.tqdm(range(I)):

    #--------------------------------------------------------------------------#
    #               0.0 sample data (shared for all (opt, beta) pairs)         #
    #--------------------------------------------------------------------------#

        D = land.sample_data(N + (N + test_extra)) 
        D_train, D_test = D[:N], D[N:]

        for opt, beta in opts: 
            nabla = land.nabla
            stalk = land.get_loss_stalk

    #--------------------------------------------------------------------------#
    #               0.1 define optimization updates                            #
    #--------------------------------------------------------------------------#

            compute_gradients = {
                'SGD':  lambda t:   nabla(stalk(D_train[(t%N):(t%N)+1]))  ,
                'GD':   lambda t:   nabla(stalk(D_train               ))  ,
                'GDC':  lambda t: ( nabla(stalk(D_train[:N//2]        ))  ,
                                    nabla(stalk(D_train[N//2:]        )) ),
            }[opt]
            compute_update = {
                'SGD':  lambda g: g,
                'GD':   lambda g: g,
                'GDC':  lambda g: (g[0] + g[1])/2 + nabla(g[0].dot(g[0]-g[1]))*(N//2),
            }[opt]

    #--------------------------------------------------------------------------#
    #               0.2 perform optimization loop                              #
    #--------------------------------------------------------------------------#

            land.switch_to(idx)
            for t in range(T):
                land.update_weights(-eta * compute_update(compute_gradients(t)).detach())

    #--------------------------------------------------------------------------#
    #               0.3 compute losses and accuracies                          #
    #--------------------------------------------------------------------------#

            test_loss = land.get_loss_stalk(D_test)
            test_acc = land.get_accuracy(D_test)
            ol.accum(OptimKey(optimizer=opt.lower(), beta=beta, eta=eta, N=N, T=T, metric='test-loss'), test_loss)
            ol.accum(OptimKey(optimizer=opt.lower(), beta=beta, eta=eta, N=N, T=T, metric='test-acc'), test_acc)

    return ol

#==============================================================================#
#           1. DEFINE SIMULATION HYPERPARAMETER RANGES                         #
#==============================================================================#

    #--------------------------------------------------------------------------#
    #               1.0 sanity check on quadratic landscape                    #
    #--------------------------------------------------------------------------#

def test_on_quad_landscape():
    LC = Quadratic(dim=12)
    DIM = 8
    Q = Quadratic(dim=DIM)
    ol = OptimLog()
    for eta in tqdm.tqdm(np.arange(0.0005, 0.005, 0.001)):
        for T in [100]:
            ol.absorb(compute_losses(Q, eta=eta, T=T, N=T, I=int(100000.0/(T+1))))
    print(ol)
    with open('ol.data', 'w') as f:
        f.write(str(ol))
    print(CC+'measured @G {:+.1f} @W - @G {:+.1f} @W \t expected @Y {:.1f} @W '.format(
        mean - 1.96 * stdv/nb_samples**0.5,
        mean + 1.96 * stdv/nb_samples**0.5,
        DIM/2.0 + DIM/2.0 * (1-ETA)**(2*T)
    ))

    #--------------------------------------------------------------------------#
    #               1.1 lenet                                                  #
    #--------------------------------------------------------------------------#

def simulate_lenet():
    LC = MnistLeNet(digits=list(range(10)))
    LC.load_from('saved-weights/mnist-lenet.npy')
    for idx in tqdm.tqdm(range(0, 4)):
        ol = OptimLog()
        for eta in tqdm.tqdm(np.arange(0.04, 0.241, 0.04)):
            for T in [100]:
                ol.absorb(compute_losses(LC, eta=eta, T=T, N=T, I=int(1000.0/(T+1)), idx=idx))

        with open('ol-lenet-covreg-scan-long-small-2n-{:02d}.data'.format(idx), 'w') as f:
            f.write(str(ol))

if __name__=='__main__':
    simulate_lenet()
