''' author: samtenka
    change: 2019-08-13
    create: 2019-06-11
    descrp: instantiate abstract class `Landscape` for MNIST models (logistic and deep)
'''


from utils import device, prod, secs_endured, megs_alloced, CC

from landscape import PointedLandscape

import tqdm
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
            (shape[0] + prod(shape[1:]))**(-0.5)
            for shape in self.subweight_shapes
        ]

        self.weights = torch.autograd.Variable(
            self.weight_scale * torch.randn(self.subweight_offsets[-1], device=device)
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
    def __init__(self, digits=list(range(10)), weight_scale=0.01**(1.0/1)):
        super().__init__(digits, weight_scale)
        self.subweight_shapes = [
            (self.nb_classes , 28*28        ), (self.nb_classes , 1            )
        ]
        #print('Logistic has {} parameters'.format(sum(prod(w) for w in self.subweight_shapes)))
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


class MnistMLP(MnistAbstractArchitecture):
    def __init__(self, digits=list(range(10)), weight_scale=10**(-0.5), widthA=16, widthB=16):
        super().__init__(digits, weight_scale)
        self.subweight_shapes = [
            (widthA         ,  1     , 5, 5), 
            (widthB         ,  widthA, 5, 5),
            (self.nb_classes, 4*4*widthB)
        ]
        #print('MLP has {} parameters'.format(sum(prod(w) for w in self.subweight_shapes)))
        self.widthA = widthA
        self.widthB = widthB
        self.reset_weights()

    def logits_and_labels(self, data_indices):
        x, y = self.imgs[data_indices], self.lbls[data_indices]
        x = tanh(conv2d(x, self.get_subweight(0), bias=None, stride=2)) 
        x = tanh(conv2d(x, self.get_subweight(1), bias=None, stride=2))
        x = x.view(-1, 4 * 4* self.widthB, 1)
        x =     matmul(self.get_subweight(2), x)
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
    def __init__(self, digits=list(range(10)), weight_scale=10**(-0.5), widthA=20, widthB=20):
        super().__init__(digits, weight_scale)
        self.subweight_shapes = [
            (widthA          ,  1     , 5, 5),      #(widthA,), 
            (widthB          , widthA , 5, 5),      #(widthB,),
            (self.nb_classes , 4*4*widthB ),        #(self.nb_classes, 1)
        ]
        #print('LeNet has {} parameters'.format(sum(prod(w) for w in self.subweight_shapes)))
        self.widthA = widthA
        self.widthB = widthB
        self.reset_weights()

    def logits_and_labels(self, data_indices):
        x, y = self.imgs[data_indices], self.lbls[data_indices]
        x = tanh(conv2d(x, self.get_subweight(0), bias=None, stride=2))
        x = tanh(conv2d(x, self.get_subweight(1), bias=None, stride=2))
        x = x.view(-1, 4*4*self.widthB, 1)
        x = matmul(self.get_subweight(2), x) 
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

def grid_search(BATCH=1, TIME=200, NB_ITERS=10): 
    for widthA in [16]:
        for widthB in [16]:
            print()
            for LRATE in [10**i for i in [-3.0, -2.5, -2.0]]:
                for weight_scale in [(10**i)**(1.0/2) for i in [1.5, 1.0, 0.5]]:
                    accs = []
                    for i in tqdm.tqdm(range(NB_ITERS)):
                        ML = MnistMLP(
                            digits=list(range(10)),
                            weight_scale=weight_scale,
                            widthA=widthA,
                            widthB=widthB
                        ) 

                        for i in range(TIME):
                            D = ML.sample_data(N=BATCH)
                            L = ML.get_loss_stalk(D)
                            G = ML.nabla(L)
                            ML.update_weights(-LRATE * G)

                        acc = ML.get_accuracy(ML.sample_data(N=1000))
                        accs.append(acc)
                    print('\033[1A' + ' '*200 + '\033[1A')
                    acc_mean = np.mean(accs)
                    acc_stdv = np.std(accs)
                    low_bd = int(100 * (acc_mean - 3 * acc_stdv/NB_ITERS**0.5))
                    up_bd = int(100 * (acc_mean  + 3 * acc_stdv/NB_ITERS**0.5))
                    print(CC + '@C lrate @G {:8.2e} @C \t widthA @B {} \t widthB @B {} \t ws {:8.2e} \t accs @M {:.3f} @C \t accm @Y {:.3f} \t @G {} @C '.format(
                        LRATE, widthA, widthB, weight_scale, acc_stdv, acc_mean,
                        '@G ' + '~'*(10) + '@R '
                        '@G ' + '='*(low_bd-10-1) + '@R ' + str(low_bd) +
                        '@Y ' + '-'*(up_bd-low_bd-2) + '@R ' + str(up_bd) +
                        '@C ' + ' '*(100-up_bd-1)
                    ))

 
if __name__=='__main__':
    #grid_search()
    #ML = MnistLogistic(digits=list(range(10))) 
    #LRATE = 10e-0



    BATCH = 1

    ML = MnistLeNet(digits=list(range(10))) 
    LRATE = 10**0.5
    #ML = MnistMLP(digits=list(range(10))) 
    #LRATE = 10**0.0

    for i in range(300):
        D = ML.sample_data(N=BATCH)
        L = ML.get_loss_stalk(D)
        G = ML.nabla(L)
        ML.update_weights(-LRATE * G)

        if (i+1)%10: continue

        La, Lb, Lc = (ML.get_loss_stalk(ML.sample_data(N=1000)) for i in range(3))
        acc = ML.get_accuracy(ML.sample_data(N=1000))
        Ga, Gb, Gc = (ML.nabla(Lx) for Lx in (La, Lb, Lc))
        GaGa, GaGb = (torch.dot(Ga, Gx) for Gx in (Ga, Gb)) 
        #C = (GaGa-GaGb) * BATCH**2 / (BATCH-1.0) 
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
            #'trace cov @Y {:+.1e}'.format(C.detach().numpy()),
            'cov hess @R {:+.1e}'.format(CH.detach().numpy()),
        '']))
