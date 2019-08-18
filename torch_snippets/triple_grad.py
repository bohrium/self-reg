''' author: samtenka
    change: 2019-08-19
    create: 2019-08-19
    descrp: show how to differentiate arbitrarily far in pytorch  
'''

import numpy as np
import torch

W = torch.autograd.Variable(
    torch.Tensor(np.random.randn(10)),
    requires_grad=True
)
I = torch.autograd.Variable(
    torch.eye(10),
    #requires_grad=True
)
WW = (W + 0.0*torch.exp(W)).unsqueeze(dim=0) 
D = WW.t().mm(WW)
L = I.mul(D).mean()

##
## NOTE: the exp allows even polynomials to be differentiated arbitrarily far
L = L + 0.0*torch.exp(L)

def nabla(scalar_stalk):
    return torch.autograd.grad(
            scalar_stalk,
            W,
            create_graph=True,
            retain_graph=True,
        )[0]
def out(tensor):
    print(tensor.detach().numpy())

GA = nabla(L)
GB = nabla(L)
GC = nabla(L)
GD = nabla(L)

GA_ = torch.autograd.Variable(torch.Tensor(GA.detach().numpy()), requires_grad=True)
GB_ = torch.autograd.Variable(torch.Tensor(GB.detach().numpy()), requires_grad=True)
GC_ = torch.autograd.Variable(torch.Tensor(GC.detach().numpy()), requires_grad=True)
GD_ = torch.autograd.Variable(torch.Tensor(GD.detach().numpy()), requires_grad=True)

g = GA.dot(GB_)
print('$', g.requires_grad)
print('$$$', nabla(g).requires_grad)
gg = nabla(g).dot(GC_)
ggg = nabla(gg).dot(GD)

out(ggg)


print('%', gg.requires_grad)
