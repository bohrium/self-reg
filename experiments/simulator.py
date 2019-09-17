''' author: samtenka
    change: 2019-08-17
    create: 2019-06-17
    descrp: do gradient descent on landscapes
'''

import numpy as np

from utils import CC
from optimlogs import OptimKey, OptimLog
from landscape import PointedLandscape
from quad_landscapes import Quadratic
import torch
import tqdm

standard_optimizers = [
    ('SGD', None),
    ('GD', None),
    ('GDC', 1.0),
]
def compute_losses(land, eta, T, N, I=1, idx=None, optimizers=standard_optimizers):
    assert N%2==0, 'GDC simulator needs N to be even for covariance estimation'
    ol = OptimLog()

    for i in tqdm.tqdm(range(I)):
        D = land.sample_data(N + (N + 300)) 
        D_train, D_test = D[:N], D[N:]

        for opt, beta in optimizers: 
            nabla = land.nabla
            stalk = land.get_loss_stalk

            compute_gradients = {
                'SGD':  lambda t:  nabla(stalk(D_train[(t%N):(t%N)+1])),
                'GD':   lambda t:  nabla(stalk(D_train               )),
                'GDC':  lambda t: (nabla(stalk(D_train[:N//2]        )), nabla(stalk(D_train[N//2:]        ))),
            }
            compute_update = {
                'SGD':  lambda g: g,
                'GD':   lambda g: g,
                'GDC':  lambda g: (g[0] + g[1])/2 + nabla(g[0].dot(g[0]-g[1]))*(N//2),
            }[opt]

            land.switch_to(idx)
            for t in range(T):
                land.update_weights(-eta * update_computation(compute_gradients(t)).detach())

            sgd_test_loss = land.get_loss_stalk(D_test)
            ol.accum(OptimKey(optimizer=opt.lower(), beta=beta, eta=eta, N=N, T=T, metric='test-loss'), sgd_test_loss)

            sgd_test_acc = land.get_accuracy(D_test)
            ol.accum(OptimKey(optimizer=opt.lower(), beta=beta, eta=eta, N=N, T=T, metric='test-acc'), sgd_test_acc)

    return ol

if __name__=='__main__':
    from mnist_landscapes import MnistLogistic, MnistLeNet, MnistMLP



    #LC = MnistLogistic(digits=list(range(10)))
    #ol = OptimLog()
    #for eta in tqdm.tqdm(np.arange(0.55, 0.76, 0.05)):
    #    for T in [100]:
    #        ol.absorb(compute_losses(LC, eta=eta, T=T, N=T, I=int(10000.0/(T+1))))


    #LC = MnistLogistic(digits=list(range(10)))
    #LC.load_from('saved-weights/mnist-logistic.npy')
    #for i in range(10):
    #    eta = 0.0
    #    T = 10
    #    LC.switch_to(0)
    #    #sgd_test_loss = LC.get_loss_stalk(LC.sample_data(10))
    #    sgd_test_loss = compute_losses(LC, eta=eta, T=T, N=T, I=int(10000.0/(T+1)), idx=0)
    #    print('#'*8, sgd_test_loss.detach().numpy())
    #    input()


    #LC = MnistLogistic(digits=list(range(10)))
    #LC.load_from('saved-weights/mnist-logistic.npy')

    LC = MnistLeNet(digits=list(range(10)))
    LC.load_from('saved-weights/mnist-lenet.npy')
    for idx in tqdm.tqdm(range(0, 4)):
        ol = OptimLog()
        for eta in tqdm.tqdm(np.arange(0.04, 0.241, 0.04)):
            for T in [100]:
                ol.absorb(compute_losses(LC, eta=eta, T=T, N=T, I=int(1000.0/(T+1)), idx=idx))

        with open('ol-lenet-covreg-scan-long-small-2n-{:02d}.data'.format(idx), 'w') as f:
            f.write(str(ol))



    #from quad_landscapes import Quadratic
    #LC = Quadratic(dim=12)
    #DIM = 8
    #Q = Quadratic(dim=DIM)
    #ol = OptimLog()
    #for eta in tqdm.tqdm(np.arange(0.0005, 0.005, 0.001)):
    #    for T in [100]:
    #        ol.absorb(compute_losses(Q, eta=eta, T=T, N=T, I=int(100000.0/(T+1))))
    #print(ol)
    #with open('ol.data', 'w') as f:
    #    f.write(str(ol))
    #print(CC+'measured @G {:+.1f} @W - @G {:+.1f} @W \t expected @Y {:.1f} @W '.format(
    #    mean - 1.96 * stdv/nb_samples**0.5,
    #    mean + 1.96 * stdv/nb_samples**0.5,
    #    DIM/2.0 + DIM/2.0 * (1-ETA)**(2*T)
    #))
