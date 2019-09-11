''' author: samtenka
    change: 2019-09-03
    create: 2019-06-16
    descrp: instantiate class `Mobius` for toy model with eta-polynomial circular drift  
'''


from utils import device, prod, secs_endured, megs_alloced, CC

from landscape import PointedLandscape

import numpy as np
import torch

class Mobius(PointedLandscape):
    '''
    '''
    def __init__(self, R=1e0, r=1e-2, g=1.5):
        self.resample_weights()
        self.dim = 3
        self.R = R
        self.r = r
        self.g = g

    def resample_weights(self, weights=None):
        self.weights = torch.autograd.Variable(
            1.0 + torch.zeros(3, dtype=torch.float, device=device)
            if weights is None else torch.Tensor(weights)
            ,
            requires_grad=True
        )

    def sample_data(self, N): 
        return torch.randn(N, self.dim)

    def get_loss_stalk(self, data):
        x, y, z = self.weights[0], self.weights[1], self.weight[2]
        sombrero = (x**2 + y**2 - self.R**2)**2 
        torsion = ((x/R) * z + )**2 / self.r**2 

        return

    def update_weights(self, displacement):
        self.weights.data += displacement.detach().data

    def get_weights(self):
        return self.weights.detach().numpy()

    def nabla(self, scalar_stalk, create_graph=True):
        return torch.autograd.grad(
            scalar_stalk,
            self.weights,
            create_graph=create_graph,
        )[0]

if __name__=='__main__':
    BATCH = 1
    ML = MnistLeNet(digits=list(range(10))) 
    LRATE = 1e-3

    for i in range(300):
        D = ML.sample_data(N=BATCH)
        L = ML.get_loss_stalk(D)
        G = ML.nabla(L)
        ML.update_weights(-LRATE * G)

        if (i+1)%10: continue

        L_test = ML.get_loss_stalk(ML.sample_data(N=1000))

        print(CC + 'step {:8d}, loss {:8.2f}'.format(
            i, L_test.numpy() 
        ))
