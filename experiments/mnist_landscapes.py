''' author: samtenka
    change: 2019-09-17
    create: 2019-06-11
    descrp: instantiate abstract class `Landscape` for MNIST models (logistic and deep)
'''


from utils import device, prod, secs_endured, megs_alloced, CC

from landscape import PointedLandscape, FixedInitsLandscape

import tqdm
import numpy as np
import torch
from torch import conv2d, matmul, tanh
from torch.nn.functional import log_softmax, nll_loss 
from torchvision import datasets, transforms


#==============================================================================#
#           0. MNIST                                                           #
#==============================================================================#

    #--------------------------------------------------------------------------#
    #               0.0 begin defining landscape by providing data population  #
    #--------------------------------------------------------------------------#

class MNIST(PointedLandscape):
    ''' load specified digits of MNIST, e.g. just 0s and 1s for binary classification subtask.
        implements PointedLandscape's `sample_data` but not its `update_weights`, `get_weights`,
        `set_weights`, `get_loss_stalk`, or `nabla`.
    '''
    def __init__(self, digits=list(range(10))):
        '''
        '''
        # load all of mnist:
        train_set, test_set = (
            datasets.MNIST(
                '../data',
                train=train_flag,
                download=True,
                transform=transforms.ToTensor()
            )      
            for train_flag in (True, False) 
        )
        self.imgs = torch.cat([train_set.train_data  , test_set.test_data  ], dim=0).numpy() / 255.0
        self.lbls = torch.cat([train_set.train_labels, test_set.test_labels], dim=0).numpy()

        # filter for requested digits:
        indices_to_keep = np.array([i for i, lbl in enumerate(self.lbls) if lbl in digits]) 
        self.imgs = torch.Tensor(self.imgs[indices_to_keep]).view(-1, 1, 28, 28)
        self.lbls = torch.Tensor([digits.index(l) for l in self.lbls[indices_to_keep]]).view(-1).long()

        # record index bounds and range:
        self.nb_classes = len(digits)
        self.nb_datapts = len(indices_to_keep)
        self.idxs = np.arange(self.nb_datapts)

    def sample_data(self, N):
        '''
        '''
        return np.random.choice(self.idxs, N, replace=False)

    #--------------------------------------------------------------------------#
    #               0.1 finish landscape definition by providing architecture  #
    #--------------------------------------------------------------------------#

class MnistAbstractArchitecture(MNIST, FixedInitsLandscape):
    ''' 
    '''
    def __init__(self, digits=list(range(10)), weight_scale=1.0):
        ''' '''
        super().__init__(digits)
        self.weight_scale = weight_scale

    def resample_weights(self, weights=None):
        ''' '''
        self.subweight_offsets = [
            sum(prod(shape) for shape in self.subweight_shapes[:depth])
            for depth in range(len(self.subweight_shapes)+1) 
        ]
        self.subweight_scales = [
            self.weight_scale * (2.0 / (shape[0] + prod(shape[1:])))**0.5
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
        ''' '''
        return self.weights.detach().numpy()

    def set_weights(self, weights):
        ''' '''
        #self.weights.data = torch.Tensor(weights)
        self.weights = torch.autograd.Variable(torch.Tensor(weights), requires_grad=True)

    def update_weights(self, displacement):
        ''' '''
        self.weights.data += displacement.detach().data

    def nabla(self, scalar_stalk, create_graph=True):
        ''' '''
        return torch.autograd.grad(
            scalar_stalk,
            self.weights,
            create_graph=create_graph,
        )[0] 

    def get_loss_stalk(self, data_indices):
        ''' '''
        logits, labels = self.logits_and_labels(data_indices)
        return nll_loss(logits, labels)

    def get_accuracy(self, data_indices):
        ''' '''
        logits, labels = self.logits_and_labels(data_indices)
        _, argmax = logits.max(1) 
        return argmax.eq(labels).sum() / labels.shape[0]

class MnistLogistic(MnistAbstractArchitecture):
    '''
    '''
    def __init__(self, digits=list(range(10)), weight_scale=1.0, verbose=False):
        ''' '''
        super().__init__(digits, weight_scale)
        self.subweight_shapes = [
            (self.nb_classes , 28*28        ), (self.nb_classes , 1            )
        ]
        self.resample_weights()
        if verbose:
            print('Logistic has {} parameters'.format(
                sum(prod(w) for w in self.subweight_shapes)
            ))

    def logits_and_labels(self, data_indices):
        ''' '''
        x, y = self.imgs[data_indices], self.lbls[data_indices]
        x = x.view(-1, 28*28, 1)
        x = matmul(self.get_subweight(0), x) + self.get_subweight(1).unsqueeze(0)
        x = x.view(-1, self.nb_classes)
        return log_softmax(x, dim=1), y

    def get_loss_stalk(self, data_indices):
        ''' '''
        logits, labels = self.logits_and_labels(data_indices)
        return nll_loss(logits, labels)

    def get_accuracy(self, data_indices):
        ''' '''
        logits, labels = self.logits_and_labels(data_indices)
        _, argmax = logits.max(1) 
        return argmax.eq(labels).double().mean()


class MnistMLP(MnistAbstractArchitecture):
    '''
    '''
    def __init__(self, digits=list(range(10)), weight_scale=1.0, width=32, verbose=False):
        ''' '''
        super().__init__(digits, weight_scale)
        self.subweight_shapes = [
            (width           , 28*28        ), (width           , 1            ),
            (self.nb_classes , width        ), (self.nb_classes , 1            ),
        ]

        self.width = width
        self.resample_weights()

        if verbose:
            print('MLP has {} parameters'.format(
                sum(prod(w) for w in self.subweight_shapes)
            ))

    def logits_and_labels(self, data_indices):
        ''' '''
        x, y = self.imgs[data_indices], self.lbls[data_indices]

        x = x.view(-1, 28*28, 1)
        x = tanh(matmul(self.get_subweight(0), x) + self.get_subweight(1).unsqueeze(0))
        x =      matmul(self.get_subweight(2), x) + self.get_subweight(3).unsqueeze(0)

        x = x.view(-1, self.nb_classes)
        return log_softmax(x, dim=1), y

    def get_loss_stalk(self, data_indices):
        ''' '''
        logits, labels = self.logits_and_labels(data_indices)
        return nll_loss(logits, labels)

    def get_accuracy(self, data_indices):
        ''' '''
        logits, labels = self.logits_and_labels(data_indices)
        _, argmax = logits.max(1) 
        return argmax.eq(labels).double().mean()



class MnistLeNet(MnistAbstractArchitecture):
    def __init__(self, digits=list(range(10)), weight_scale=1.0, widthA= 8, widthB=16, verbose=False):
        super().__init__(digits, weight_scale)
        self.subweight_shapes = [
            (widthA          ,  1     , 5, 5),      (widthA,), 
            (widthB          , widthA , 5, 5),      (widthB,),
            (self.nb_classes , widthB * 4*4 ),      (self.nb_classes, 1),
        ]

        self.widthA = widthA
        self.widthB = widthB
        self.resample_weights()

        if verbose:
            print('LeNet has {} parameters'.format(
                sum(prod(w) for w in self.subweight_shapes)
            ))

    def logits_and_labels(self, data_indices):
        x, y = self.imgs[data_indices], self.lbls[data_indices]
        x = tanh(conv2d(x, self.get_subweight(0), bias=self.get_subweight(1), stride=2)) # 12 x 12
        x = tanh(conv2d(x, self.get_subweight(2), bias=self.get_subweight(3), stride=2)) #  4 x  4
        x = x.view(-1, self.widthB*4*4, 1)
        x = matmul(self.get_subweight(4), x) + self.get_subweight(5).unsqueeze(0)
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

def grid_search(TIME=1000, NB_ITERS=4): 
    ''' '''
    for LRATE in [10**i for i in [0.0, 0.5, 1.0]]:
        for weight_scale in [10**i for i in [-1.0, -0.5, 0.0]]:
            accs_train = []
            accs_test = []
            for i in tqdm.tqdm(range(NB_ITERS)):
                ML = MnistMLP(
                    digits=list(range(10)),
                    weight_scale=weight_scale,
                ) 

                D = ML.sample_data(N=TIME) 
                for i in range(TIME):
                    L = ML.get_loss_stalk(D[i:i+1])
                    G = ML.nabla(L)
                    ML.update_weights(-LRATE * G)

                accs_train.append(ML.get_accuracy(D))
                accs_test.append(ML.get_accuracy(ML.sample_data(N=1000)))
            print('\033[1A' + ' '*200 + '\033[1A')
            acc_mean = np.mean(accs_test)
            acc_stdv = np.std(accs_test)
            low_bd = int(100 * (acc_mean - 3 * acc_stdv/NB_ITERS**0.5))
            up_bd = int(100 * (acc_mean  + 3 * acc_stdv/NB_ITERS**0.5))
            accuracy_bar = ''.join((
                '@G ' + '~'*(10) + '@R ',
                '@G ' + '='*(low_bd-10-1) + '@R ' + str(low_bd),
                '@Y ' + '-'*(up_bd-low_bd-2) + '@R ' + str(up_bd),
                '@C ' + ' '*(100-up_bd-1),
            ))

            print(CC + ' @C '.join((
                'lrate @G {:8.2e}'.format(LRATE),
                'wscale @B {:8.2e}'.format(weight_scale),
                'train @O {:.2f}'.format(np.mean(accs_train)),
                'accstdv @M {:.2f}'.format(acc_stdv),
                'accmean @Y {:.2f}'.format(acc_mean),
                accuracy_bar
            )))

def sample_models():
    for widthA in [4, 8, 16, 32]:
        for widthB in [int(widthA * offset) for offset in [0.5, 1, 2, 4]]:
            ML = MnistLeNet(widthA=widthA, widthB=widthB)
            ML.resample_to('saved-weights/mnist-lenet-ms-{}-{}.npy'.format(widthA, widthB), 2)

if __name__=='__main__':
    #grid_search()
    sample_models()

    #BATCH = 1
    #TIME = 100

    #model_nm = 'LENET'
    #model_data = {
    #    'LOGISTIC': {'class': MnistLogistic, 'weight_scale': 1.0**(1.0/1), 'lrate':1e-1, 'file_nm': 'mnist-logistic.npy'},
    #    'MLP'     : {'class': MnistMLP,      'weight_scale': 1.0**(1.0/2), 'lrate':1e-1, 'file_nm': 'mnist-mlp.npy'},
    #    'LENET'   : {'class': MnistLeNet,    'weight_scale': 1.0**(1.0/3), 'lrate':1e-1, 'file_nm': 'mnist-lenet.npy'},
    #}[model_nm]
    #ML = model_data['class'](weight_scale = model_data['weight_scale'], verbose=True)
    #ML.load_from('saved-weights/{}'.format(model_data['file_nm']), 12)
    #ML.switch_to(0)
    #print(ML.get_weights())
    #input()
    #LRATE = model_data['lrate']

    #D = ML.sample_data(N=TIME) 
    #for i in range(TIME):
    #    L = ML.get_loss_stalk(D[i:i+1])
    #    G = ML.nabla(L)
    #    ML.update_weights(-LRATE * G)

    #    if (i+1)%10: continue

    #    L_train= ML.get_loss_stalk(D)
    #    data = ML.sample_data(N=3000)
    #    L_test = ML.get_loss_stalk(data[:1500])
    #    L_test_= ML.get_loss_stalk(data[1500:])
    #    acc = ML.get_accuracy(ML.sample_data(N=3000))

    #    print(CC+' @C \t'.join([
    #        'after @M {:4d} @C steps'.format(i+1),
    #        'grad2 @G {:.2e}'.format(ML.nabla(L_test).dot(ML.nabla(L_test_)).detach().numpy()),
    #        'train loss @Y {:.2f}'.format(L_train.detach().numpy()),
    #        'test loss @L {:.2f}'.format(L_test.detach().numpy()),
    #        'test acc @O {:.2f}'.format(acc.detach().numpy()),
    #    '']))
