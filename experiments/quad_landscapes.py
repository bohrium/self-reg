''' author: samtenka
    change: 2019-06-16
    create: 2019-06-16
    descrp: instantiate class `Quadratic` for toy model with constant hessian and covariance
'''


from utils import device, prod, secs_endured, megs_alloced, CC

from landscape import PointedLandscape

import numpy as np
import torch

class Quadratic(PointedLandscape):
    '''
    '''
    def __init__(self, dim, hessian=None, covariance=None):
        self.dim = dim
        self.reset_weights()
        self.hessian    = torch.autograd.Variable(
            hessian    if hessian    is not None else torch.eye(dim)
        )
        self.covariance = covariance if covariance is not None else torch.eye(dim)

        u, s, v = torch.svd(self.covariance)
        self.sqrt_covariance = u.mm(
            s.pow(0.5).diag()
        ).mm(v) 

    def reset_weights(self, weights=None):
        self.weights = torch.autograd.Variable(
            1.0 + torch.zeros(self.dim, dtype=torch.float, device=device)
            if weights is None else torch.Tensor(weights)
            ,
            requires_grad=True
        )

    def sample_data(self, N): 
        return torch.randn(N, self.dim).mm(self.sqrt_covariance)

    def get_loss_stalk(self, data):
        diff = data - (self.weights.unsqueeze(dim=0))
        diff2 = diff.t().mm(diff) 
        loss = 0.5 * diff2.mul(self.hessian).mean()
        return loss + 0.0 * torch.exp(loss)

    def update_weights(self, displacement):
        self.weights.data += displacement.detach().data

    def get_weights(self):
        return self.weights.detach().numpy()

    def nabla(self, scalar_stalk, create_graph=True):
        return torch.autograd.grad(
            scalar_stalk,
            self.weights,
            create_graph=create_graph,
            #allow_unused=True
        )[0]
        #return torch.autograd.Variable(torch.autograd.grad(
        #    scalar_stalk,
        #    self.weights,
        #    create_graph=create_graph,
        #    #allow_unused=True
        #)[0], requires_grad=True)

if __name__=='__main__':
    DIM, N = 36, 196
    Q = Quadratic(dim=DIM)

    l = Q.get_loss_stalk(Q.sample_data(N)) 
    print(CC+'loss @Y {:.2f}@C  expected @R {:.2f} @C '.format(l.item(), DIM))

    gg = Q.nabla(l).pow(2).sum()
    print(CC+'gg @Y {:.2f}@C  expected @R {:.2f} @C '.format(gg.item(), DIM))

    ghhg = (0.5 * Q.nabla(gg)).pow(2).sum()
    print(CC+'ghhg @Y {:.2f}@C  expected @R {:.2f} @C '.format(ghhg.item(), DIM))

