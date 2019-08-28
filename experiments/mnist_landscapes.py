''' author: samtenka
    change: 2019-08-13
    create: 2019-06-11
    descrp: instantiate abstract class `Landscape` for MNIST models (logistic and deep)
'''


from utils import device, prod, secs_endured, megs_alloced, CC

from landscape import PointedLandscape

import numpy as np
import torch
from torch import conv2d, matmul, tanh
from torch.nn.functional import log_softmax, nll_loss 
from torchvision import datasets, transforms


################################################################################
#           0. MNIST                                                           #
################################################################################

    #--------------------------------------------------------------------------#
    #               0.0 begin defining landscape by providing data population  #
    #--------------------------------------------------------------------------#

class MNIST(PointedLandscape):
    ''' load specified digits of MNIST, e.g. just 0s and 1s for binary classification subtask.
        implements PointedLandscape's `get_data_sample` but not its `update_weight`, `nabla`,
        `get_random_loss_field`, or `evaluate_as_tensor`.
    '''
    def __init__(self, digits=list(range(10))):
        train_set, test_set = (
            datasets.MNIST(
                '../data',
                train=train_flag,
                download=True,
                transform=transforms.ToTensor()
            )      
            for train_flag in (True, False) 
        )
        self.imgs = torch.cat([train_set.train_data  , test_set.test_data  ], dim=0).numpy()
        self.lbls = torch.cat([train_set.train_labels, test_set.test_labels], dim=0).numpy()
        indices_to_keep = np.array([i for i, lbl in enumerate(self.lbls) if lbl in digits])        
        self.imgs = torch.Tensor(self.imgs[indices_to_keep]).view(-1, 1, 28, 28)
        self.lbls = torch.Tensor([digits.index(l) for l in self.lbls[indices_to_keep]]).view(-1).long()
        self.nb_classes = len(digits)
        self.nb_datapts = len(indices_to_keep)
        self.idxs = np.arange(self.nb_datapts)

    def sample_data(self, N):
        return np.random.choice(self.idxs, N, replace=False)

    #--------------------------------------------------------------------------#
    #               0.1 finish landscape definition by providing architecture  #
    #--------------------------------------------------------------------------#

class MnistAbstractArchitecture(MNIST):
    def __init__(self, digits=list(range(10)), weight_scale=1.0):
        super().__init__(digits)
        self.weight_scale = weight_scale

    def reset_weights(self, weights=None):
        self.subweight_offsets = [
            sum(prod(shape) for shape in self.subweight_shapes[:depth])
            for depth in range(len(self.subweight_shapes)+1) 
        ]
        self.subweight_scales = [
            self.weight_scale * (shape[0] + prod(shape[1:]))**(-0.5)
            for shape in self.subweight_shapes
        ]

        self.weights = torch.autograd.Variable(
            torch.randn(self.subweight_offsets[-1], device=device)
            if weights is None else torch.Tensor(weights)
            ,
            requires_grad=True
        )
        self.get_subweight = lambda depth: ( 
            (self.subweight_scales[depth] * 
            self.weights[self.subweight_offsets[depth]:
                         self.subweight_offsets[depth+1]])
            .view(self.subweight_shapes[depth])
        )

    def get_weights(self):
        return self.weights.detach().numpy()

    def update_weights(self, displacement):
        self.weights.data += displacement.detach().data

    def nabla(self, scalar_stalk, create_graph=True):
        return torch.autograd.grad(
            scalar_stalk,
            self.weights,
            create_graph=create_graph,
        )[0] 

    def get_loss_stalk(self, data_indices):
        logits, labels = self.logits_and_labels(data_indices)
        return nll_loss(logits, labels)

    def get_accuracy(self, data_indices):
        logits, labels = self.logits_and_labels(data_indices)
        _, argmax = logits.max(1) 
        return argmax.eq(labels).sum() / labels.shape[0]

class MnistLogistic(MnistAbstractArchitecture):
    def __init__(self, digits=list(range(10)), weight_scale=0.01):
        super().__init__(digits, weight_scale)
        self.subweight_shapes = [
            (self.nb_classes , 28*28        ),
            (self.nb_classes , 1            )
        ]
        self.reset_weights()

    def logits_and_labels(self, data_indices):
        x, y = self.imgs[data_indices], self.lbls[data_indices]
        x = x.view(-1, 28*28, 1)
        x = matmul(self.get_subweight(0), x) + self.get_subweight(1).unsqueeze(0)
        x = x.view(-1, self.nb_classes)
        return log_softmax(x, dim=1), y

    def get_loss_stalk(self, data_indices):
        logits, labels = self.logits_and_labels(data_indices)
        return nll_loss(logits, labels)

    def get_accuracy(self, data_indices):
        logits, labels = self.logits_and_labels(data_indices)
        _, argmax = logits.max(1) 
        return argmax.eq(labels).double().mean()

class MnistLeNet(MnistAbstractArchitecture):
    def __init__(self, digits=list(range(10)), weight_scale=1.0):
        super().__init__(digits, weight_scale)
        self.subweight_shapes = [
            (16              ,  1     , 5, 5),      #(16,), 
            (16              , 16     , 5, 5),      #(16,),
            (16              , 4*4*16       ),      #(16,),
            (self.nb_classes , 16           ),      #(self.nb_classes,)
        ]
        self.reset_weights()

    def logits_and_labels(self, data_indices):
        x, y = self.imgs[data_indices], self.lbls[data_indices]
        x = tanh(conv2d(x, self.get_subweight(0), bias=None, stride=2))
        x = tanh(conv2d(x, self.get_subweight(1), bias=None, stride=2))
        x = x.view(-1, 4*4*16, 1)
        x = tanh(matmul(self.get_subweight(2), x))
        x = matmul(self.get_subweight(3), x)
        x = x.view(-1, self.nb_classes)
        logits = log_softmax(x, dim=1)
        return logits, y

    def get_loss_stalk(self, data_indices):
        logits, labels = self.logits_and_labels(data_indices)
        return nll_loss(logits, labels)

    def get_accuracy(self, data_indices):
        logits, labels = self.logits_and_labels(data_indices)
        _, argmax = logits.max(1) 
        return argmax.eq(labels).double().mean()


    #--------------------------------------------------------------------------#
    #               0.2 demonstrate interface by descending with grad stats    #
    #--------------------------------------------------------------------------#
 
if __name__=='__main__':
    BATCH = 200

    ML = MnistLeNet(digits=list(range(10))) 
    LRATE = 1e+1

    #ML = MnistLogistic(digits=list(range(10))) 
    #LRATE = 1e-0

    for i in range(1000):
        D = ML.sample_data(N=BATCH)
        L = ML.get_loss_stalk(D)
        G = ML.nabla(L)
        ML.update_weights(-LRATE * G)

        if (i+1)%100: continue

        La, Lb, Lc = (ML.get_loss_stalk(ML.sample_data(N=BATCH)) for i in range(3))
        acc = ML.get_accuracy(ML.sample_data(N=BATCH))
        Ga, Gb, Gc = (ML.nabla(Lx) for Lx in (La, Lb, Lc))
        GaGa, GaGb = (torch.dot(Ga, Gx) for Gx in (Ga, Gb)) 
        C = (GaGa-GaGb) * BATCH**2 / (BATCH-1.0) 
        GaHcGa, GaHcGb = (
            torch.dot(Gx, ML.nabla(torch.dot(Gc, Gy.detach())))
            for Gx, Gy in ((Ga, Ga), (Ga, Gb)) 
        )
        CH = (GaHcGa-GaHcGb) * 25**2 / (25-1.0) 
        print(CC+' @C \t'.join([
            'after {:4d} steps'.format(i+1),
            'batch loss @B {:.2f}'.format(L.detach().numpy()),
            'test loss @B {:.2f}'.format(La.detach().numpy()),
            'test acc @B {:.2f}'.format(acc.detach().numpy()),
            'grad mag2 @G {:+.1e}'.format(GaGb.detach().numpy()),
            'trace cov @Y {:+.1e}'.format(C.detach().numpy()),
            'cov hess @R {:+.1e}'.format(CH.detach().numpy()),
        '']))
