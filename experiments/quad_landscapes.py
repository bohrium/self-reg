''' author: samtenka
    change: 2019-08-26
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
        self.hessian    = hessian    if hessian    is not None else torch.eye(dim)
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
        return torch.randn(N, self.dim)

    def get_loss_stalk(self, data):
        #diff = data - (self.weights.unsqueeze(dim=0))
        #diff2 = diff.t().mm(diff) / diff.shape[0] 
        #loss = 0.5 * diff2.mul(self.hessian).sum()
        loss_signal = 0.5 * self.weights.dot(self.hessian.matmul(self.weights))
        loss_noise = data.mean(0).dot(self.sqrt_covariance.matmul(self.weights))
        return loss_signal + loss_noise + 0.0 * torch.tanh(self.weights.sum())

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
    DIM, N = 360, 1000
    assert DIM%4==0
    DIM_4 = DIM//4

    A,B,C,D = 1.0, 2.0, 0.1, 0.3
    hessian    = torch.diag(torch.tensor([A]*DIM_4 + [A]*DIM_4 + [B]*DIM_4 + [B]*DIM_4))
    covariance = torch.diag(torch.tensor([C]*DIM_4 + [D]*DIM_4 + [D]*DIM_4 + [C]*DIM_4))
    Q = Quadratic(dim=DIM, hessian=hessian, covariance=covariance)

    datas  = tuple(Q.sample_data(N) for i in range(4))
    losses = tuple(Q.get_loss_stalk(d) for d in datas)
    grads  = tuple(Q.nabla(l) for l in losses)

    grad2 = grads[0].dot(grads[1])
    ghhg  = Q.nabla(grads[0].detach().dot(grads[1])).dot(Q.nabla(grads[2].dot(grads[3].detach())))
    trcov = (grads[0] - grads[1]).pow(2).sum() * N/2
    tr_hc = (grads[0] - grads[1]).dot(Q.nabla(grads[2].dot((grads[0] - grads[1]).detach()))) * N/2

    print(CC+' @C '.join([
        '\tloss  @Y {:6.1f}'.format(losses[0].item()),   'vs @G {:6.1f}\n'.format(DIM_4*(A+B)),
        '\tgrad2 @Y {:6.1f}'.format(grad2.item()),       'vs @G {:6.1f}\n'.format(DIM_4*(2*A**2+2*B**2)),
        '\tghhg  @Y {:6.1f}'.format(ghhg.item()),        'vs @G {:6.1f}\n'.format(DIM_4*(2*A**4+2*B**4)),
        '\ttrcov @Y {:6.1f}'.format(trcov.item()),       'vs @G {:6.1f}\n'.format(DIM_4*(2*C+2*D)),
        '\ttr_hc @Y {:6.1f}'.format(tr_hc.item()),       'vs @G {:6.1f}\n'.format(DIM_4*((A+B)*(C+D))),
    '']))

