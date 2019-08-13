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
from mnist_landscapes import MnistLeNet
import torch
import tqdm

#def compute_sgd_loss(land, eta, T, N, I=1):
#    ol = OptimLog()
#    for i in tqdm.tqdm(range(I)):
#        land.reset_weights()
#        D = land.sample_data(2*N) 
#        D_train, D_test = D[:N], D[N:]
#        for t in range(T):
#            loss_stalk = land.get_loss_stalk(D_train[(t%N):(t%N)+1]) 
#            grad = land.nabla(loss_stalk, False).detach()
#            land.update_weights(-eta*grad)
#        test_loss = land.get_loss_stalk(D_test)
#        ol.accum(OptimKey(optimizer='sgd', eta=eta, N=N, T=T, metric='test'), test_loss)
#    return ol
#
#def compute_gd_loss(land, eta, T, N, I=1):
#    ol = OptimLog()
#    for i in tqdm.tqdm(range(I)):
#        land.reset_weights()
#        D = land.sample_data(2*N) 
#        D_train, D_test = D[:N], D[N:]
#        for t in range(T):
#            loss_stalk = land.get_loss_stalk(D_train)
#            grad = land.nabla(loss_stalk, False).detach()
#            land.update_weights(-eta*grad)
#        test_loss = land.get_loss_stalk(D_test)
#        ol.accum(OptimKey(optimizer='gd', eta=eta, N=N, T=T, metric='test'), test_loss)
#    return ol

def compute_losses(land, eta, T, N, I=1):
    ol = OptimLog()
    for i in tqdm.tqdm(range(I)):
        D = land.sample_data(2*N) 
        D_train, D_test = D[:N], D[N:]
        w0 = land.get_weights()
        # SGD:
        land.reset_weights(w0)
        for t in range(T):
            loss_stalk = land.get_loss_stalk(D_train[(t%N):(t%N)+1]) 
            grad = land.nabla(loss_stalk, False).detach()
            land.update_weights(-eta*grad)
        sgd_test_loss = land.get_loss_stalk(D_test)
        ol.accum(OptimKey(optimizer='sgd', eta=eta, N=N, T=T, metric='test'), sgd_test_loss)

        # GD:
        land.reset_weights(w0)
        for t in range(T):
            loss_stalk = land.get_loss_stalk(D_train)
            grad = land.nabla(loss_stalk, False).detach()
            land.update_weights(-eta*grad)
        gd_test_loss = land.get_loss_stalk(D_test)
        ol.accum(OptimKey(optimizer='gd', eta=eta, N=N, T=T, metric='test'), gd_test_loss)


        ol.accum(OptimKey(optimizer='diff', eta=eta, N=N, T=T, metric='test'), gd_test_loss-sgd_test_loss)

    return ol


if __name__=='__main__':
    LC = MnistLeNet()
    ol = OptimLog()
    for eta in tqdm.tqdm(np.arange(0.0005, 0.005, 0.001)):
        for T in [100]:
            ol.absorb(compute_losses(LC, eta=eta, T=T, N=T, I=int(1000.0/(T+1))))
    print(ol)
    with open('ol.data', 'w') as f:
        f.write(str(ol))

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
