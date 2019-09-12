''' author: samtenka
    change: 2019-06-17
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

def compute_losses(land, eta, T, N, I=1, idx=None):
    assert N%2==0, 'GDC simulator needs N to be even for covariance estimation'
    ol = OptimLog()

    for i in tqdm.tqdm(range(I)):
        D = land.sample_data(2*N) 
        D_train, D_test = D[:N], D[N:]

        # SGD:
        #land.resample_weights()
        land.switch_to(idx)
        for t in range(T):
            loss_stalk = land.get_loss_stalk(D_train[(t%N):(t%N)+1]) 
            grad = land.nabla(loss_stalk, False).detach()
            land.update_weights(-eta*grad)
        sgd_test_loss = land.get_loss_stalk(D_test)
        ol.accum(OptimKey(optimizer='sgd', beta=0.0, eta=eta, N=N, T=T, metric='test'), sgd_test_loss)
        sgd_test_acc = land.get_accuracy(D_test)
        ol.accum(OptimKey(optimizer='sgd', beta=0.0, eta=eta, N=N, T=T, metric='testacc'), sgd_test_acc)

        ## GD:
        #land.resample_weights(w0)
        #for t in range(T):
        #    loss_stalk = land.get_loss_stalk(D_train)
        #    grad = land.nabla(loss_stalk, False).detach()
        #    land.update_weights(-eta*grad)
        #gd_test_loss = land.get_loss_stalk(D_test)
        #ol.accum(OptimKey(optimizer='gd', beta=0.0, eta=eta, N=N, T=T, metric='test'), gd_test_loss)
        #gd_test_acc = land.get_accuracy(D_test)
        #ol.accum(OptimKey(optimizer='gd', beta=0.0, eta=eta, N=N, T=T, metric='testacc'), gd_test_acc)


        ## GDC:
        #for BETA in [10**-3.0, 10**-2.5, 10**-2.0, 10**-1.5, 10**-1.0]:
        #    land.resample_weights(w0)
        #    for t in range(T):
        #        gradA = land.nabla(land.get_loss_stalk(D_train[:int(N//2)]))
        #        gradB = land.nabla(land.get_loss_stalk(D_train[int(N//2):]))
        #        traceC = gradA.dot(gradA-gradB) * (N*N/4)  
        #        grad = ((gradA + gradB)/2).detach()
        #        grad_traceC = land.nabla(traceC, False).detach() 
        #        land.update_weights(-eta*( grad + BETA * grad_traceC ))
        #    gdc_test_loss = land.get_loss_stalk(D_test)
        #    ol.accum(OptimKey(optimizer='gdc', beta=BETA, eta=eta, N=N, T=T, metric='test'), gdc_test_loss)
        #    gdc_test_acc = land.get_accuracy(D_test)
        #    ol.accum(OptimKey(optimizer='gdc', beta=BETA, eta=eta, N=N, T=T, metric='testacc'), gdc_test_acc)

        ## differences: 
        #ol.accum(OptimKey(optimizer='diff', beta=0.0, eta=eta, N=N, T=T, metric='test'), gd_test_loss-sgd_test_loss)
        ##ol.accum(OptimKey(optimizer='diffc', eta=eta, N=N, T=T, metric='test'), gdc_test_loss-sgd_test_loss)

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
    for idx in range(12):
        ol = OptimLog()
        for eta in tqdm.tqdm(np.arange( 0.0, 0.051, 0.005 )):
            for T in [100]:
                ol.absorb(compute_losses(LC, eta=eta, T=T, N=T, I=int(30000.0/(T+1)), idx=idx))

        with open('ol-lenet-{:02d}.data'.format(idx), 'w') as f:
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
