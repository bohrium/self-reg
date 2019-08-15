''' author: samtenka
    change: 2019-08-14
    create: 2019-08-14
    descrp: instantiate concrete class `Tempter` for toy weight-dependent covariance model
'''


from utils import device, prod, secs_endured, megs_alloced, CC

from landscape import PointedLandscape

import numpy as np
import torch

class Tempter(PointedLandscape):
    '''
    '''
    def __init__(self, dim, noise_scale=5.0):
        self.noise_scale = noise_scale
        self.reset_weights()

    def reset_weights(self, weights=None):
        self.weights = torch.autograd.Variable(
            torch.zeros(3, dtype=torch.float, device=device)
            if weights is None else torch.Tensor(weights)
            ,
            requires_grad=True
        )

    def sample_data(self, N): 
        return torch.randn(N) * self.noise_scale

    def get_loss_stalk(self, data):
        mean = (self.weights[0]-1.0).pow(2)
        noise = data.mean() * torch.tanh(self.weights[1]) * self.weights[0]
        return mean + noise

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
    TRAIN = 2
    TESTB = 10000

    ML = Tempter(50.0)
    LRATE = 1e-2

    D = ML.sample_data(N=TRAIN)
    for i in range(500):
        L = ML.get_loss_stalk(D[(i%TRAIN):(i%TRAIN)+1])
        G = ML.nabla(L)
        ML.update_weights(-LRATE * G)

        if (i+1)%50: continue

        La, Lb, Lc = (ML.get_loss_stalk(ML.sample_data(N=TESTB)) for i in range(3))
        Ga, Gb, Gc = (ML.nabla(Lx) for Lx in (La, Lb, Lc))
        GaGa, GaGb = (torch.dot(Ga, Gx) for Gx in (Ga, Gb)) 
        #C = (GaGa-GaGb) * TESTB**2 / (TESTB-1.0) 
        #GaHcGa, GaHcGb = (
        #    torch.dot(Gx, ML.nabla(torch.dot(Gc, Gy.detach())))
        #    for Gx, Gy in ((Ga, Ga), (Ga, Gb)) 
        #)
        #CH = (GaHcGa-GaHcGb) * TESTB**2 / (TESTB-1.0) 
        print(CC+' @C \t'.join([
            'after {:4d} steps'.format(i+1),
            'batch loss @B {:.2f}'.format(L.detach().numpy()),
            'test loss @B {:.2f}'.format(La.detach().numpy()),
            'grad mag2 @G {:+.1e}'.format(GaGb.detach().numpy()),
            #'trace cov @Y {:+.1e}'.format(C.detach().numpy()),
            #'cov hess @R {:+.1e}'.format(CH.detach().numpy()),
        '']))
