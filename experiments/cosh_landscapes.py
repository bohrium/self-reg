''' author: samtenka
    change: 2019-08-27
    create: 2019-08-27
    descrp: instantiate class `Cosh` for toy model with nontrivial 3rd order tree stats
'''


from utils import device, prod, secs_endured, megs_alloced, CC

from landscape import PointedLandscape

import numpy as np
import torch

class Cosh(PointedLandscape):
    '''
    '''
    def __init__(self, dim):
        self.dim = dim
        self.reset_weights()

    def reset_weights(self, weights=None):
        self.weights = torch.autograd.Variable(
            1.0 + torch.zeros(self.dim, dtype=torch.float, device=device)
            if weights is None else torch.Tensor(weights)
            ,
            requires_grad=True
        )

    def sample_data(self, N): 
        return torch.randint(low=0, high=2, size=(N, self.dim)) * 2 - 1

    def get_loss_stalk(self, data):
        return torch.exp(data.mul(self.weights.unsqueeze(0))).sum(1).mean()

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
    DIM, N = 3600, 100

    Q = Cosh(dim=DIM)

    datas  = tuple(Q.sample_data(N) for i in range(4))
    losses = tuple(Q.get_loss_stalk(d) for d in datas)
    grads  = tuple(Q.nabla(l) for l in losses)

    grad2 = grads[0].dot(grads[1])
    ghhg  = Q.nabla(grads[0].detach().dot(grads[1])).dot(Q.nabla(grads[2].dot(grads[3].detach())))
    trcov = (grads[0] - grads[1]).pow(2).sum() * N/2
    tr_hc = (grads[0] - grads[1]).dot(Q.nabla(grads[2].dot((grads[0] - grads[1]).detach()))) * N/2

    ep = np.exp(1)
    em = np.exp(-1)
    print(CC+' @C '.join([
        '\tloss  @Y {:8.1f}'.format(losses[0].item()),   'vs @G {:8.1f}\n'.format(DIM*(ep+em)/2),
        '\tgrad2 @Y {:8.1f}'.format(grad2.item()),       'vs @G {:8.1f}\n'.format(DIM*((ep-em)/2)**2),
        '\tghhg  @Y {:8.1f}'.format(ghhg.item()),        'vs @G {:8.1f}\n'.format(DIM*((ep-em)/2)**2 * ((ep+em)/2)**2),
        '\ttrcov @Y {:8.1f}'.format(trcov.item()),       'vs @G {:8.1f}\n'.format(DIM*(((ep**2+em**2)/2) - ((ep-em)/2)**2)),
        '\ttr_hc @Y {:8.1f}'.format(tr_hc.item()),       'vs @G {:8.1f}\n'.format(DIM*(((ep**2+em**2)/2) - ((ep-em)/2)**2) * ((ep+em)/2)),
    '']))

