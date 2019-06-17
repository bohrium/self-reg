''' author: samtenka
    change: 2019-06-17
    create: 2019-06-17
    descrp: do gradient descent on landscapes
'''

import numpy as np

from utils import CC
from landscape import PointedLandscape
from quad_landscapes import Quadratic
from mnist_landscapes import MnistLogistic
import torch
import tqdm

def compute_sgd_loss(land, eta, T, N, I=1):
    losses = []
    for i in range(I):
        land.reset_weights()
        D = land.sample_data(2*N) 
        D_train, D_test = D[:N], D[N:]
        for t in range(T):
            loss_stalk = land.get_loss_stalk(D_train[(t%N):(t%N)+1]) 
            grad = land.nabla(loss_stalk).detach()
            land.update_weights(-eta*grad)
        test_loss = land.get_loss_stalk(D_test).detach().numpy() 
        losses.append(test_loss)
    return np.mean(losses), np.std(losses), len(losses)

def compute_gd_loss(land, eta, T, N, I=1):
    losses = []
    for i in range(I):
        land.reset_weights()
        D = land.sample_data(2*N) 
        D_train, D_test = D[:N], D[N:]
        for t in range(T):
            loss_stalk = land.get_loss_stalk(D_train)
            grad = land.nabla(loss_stalk).detach()
            land.update_weights(-eta*grad)
        test_loss = land.get_loss_stalk(D_test).detach().numpy() 
        losses.append(test_loss)
    return np.mean(losses), np.std(losses), len(losses)



if __name__=='__main__':
    #ETA = 0.001
    #DIM = 8
    #Q = MnistLogistic()
    #for T in range(5, 100, 10):
    #    mean, stdv, nb_samples = compute_gd_loss(Q, eta=ETA, T=T, N=10, I=int(1000.0/(T+1)))
    #    print(CC+'measured @G {:+.1f} @W - @G {:+.1f} @W \t expected @Y {:.1f} @W '.format(
    #        mean - 1.96 * stdv/nb_samples**0.5,
    #        mean + 1.96 * stdv/nb_samples**0.5,
    #        DIM/2.0 + DIM/2.0 * (1-ETA)**(2*T)
    #    ))
    ETA = 0.01
    DIM = 8
    Q = Quadratic(dim=DIM)
    for T in range(5, 100, 10):
        mean, stdv, nb_samples = compute_gd_loss(Q, eta=ETA, T=T, N=100, I=int(1000.0/(T+1)))
        print(CC+'measured @G {:+.1f} @W - @G {:+.1f} @W \t expected @Y {:.1f} @W '.format(
            mean - 1.96 * stdv/nb_samples**0.5,
            mean + 1.96 * stdv/nb_samples**0.5,
            DIM/2.0 + DIM/2.0 * (1-ETA)**(2*T)
        ))
