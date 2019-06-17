''' author: samtenka
    change: 2019-06-16
    create: 2019-06-16
    descrp: instantiate abstract class `Landscape` for artificial quadratic model
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
        self.weights = torch.autograd.Variable(
            torch.zeros(dim, dtype=torch.float, device=device),
            requires_grad=True
        )
        self.hessian    = hessian    if hessian    is not None else torch.eye(dim)
        self.covariance = covariance if covariance is not None else torch.eye(dim)

        u, s, v = torch.svd(self.covariance)
        self.sqrt_covariance = u.mm(
            s.pow(0.5).diag()
        ).mm(v) 

    def sample_data(self, N): 
        return torch.randn(N, self.dim).mm(self.sqrt_covariance)

    def get_loss_stalk(self, data):
        diff = data - (self.weights.unsqueeze(dim=0))
        diff2 = diff.t().mm(diff) 
        return 0.5 * torch.mul(self.hessian, diff2).sum() / data.shape[0]

    def update_weights(self, displacement):
        self.weights.data += displacement.detach().data

    def nabla(self, scalar_stalk, create_graph=True):
        return torch.autograd.grad(
            scalar_stalk,
            self.weights,
            create_graph=create_graph,
        )[0] 

if __name__=='__main__':
    DIM, N = 180, 700 
    Q = Quadratic(dim=DIM)

    l = Q.get_loss_stalk(Q.sample_data(N)) 
    print('loss {:.2f}, expected {:.2f}'.format(l.item(), DIM / 2.0))

    gg = Q.nabla(l).pow(2).sum()
    print('gg {:.2f}, expected {:.2f}'.format(gg.item(), DIM/N))

    ghhg = (0.5 * Q.nabla(gg)).pow(2).sum()
    print('ghhg {:.2f}, expected {:.2f}'.format(ghhg.item(), DIM/N))

