''' author: samtenka
    change: 2019-09-17
    create: 2019-06-11
    descrp: instantiate abstract class `Landscape` for CIFAR models (logistic and deep)
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
#           0. CIFAR                                                           #
#==============================================================================#

    #--------------------------------------------------------------------------#
    #               0.0 begin defining landscape by providing data population  #
    #--------------------------------------------------------------------------#

class CIFAR(PointedLandscape):
    ''' load specified class_nms of CIFAR, e.g. just 0s and 1s for binary classification subtask.
        implements PointedLandscape's `sample_data` but not its `update_weights`, `get_weights`,
        `set_weights`, `get_loss_stalk`, or `nabla`.
    '''

    CLASS_NMS = (
        'airplane', 
        'automobile', 
        'bird', 
        'cat', 
        'deer', 
        'dog', 
        'frog', 
        'horse', 
        'ship', 
        'truck'
    )

    def __init__(self, class_nms=CLASS_NMS): 
        '''
        '''
        # load all of cifar:
        train_set, test_set = (
            datasets.CIFAR10(
                '../data',
                train=train_flag,
                download=True,
                transform=transforms.ToTensor()
            )
            for train_flag in (True, False) 
        )
        assert train_set.classes == test_set.classes
        self.imgs = np.concatenate([train_set.data   , test_set.data   ], axis=0) / 255.0
        self.lbls = np.concatenate([train_set.targets, test_set.targets], axis=0)

        # filter for requested classes:
        indices_to_keep = np.array([i for i, lbl in enumerate(self.lbls) if train_set.classes[lbl] in class_nms]) 
        self.imgs = torch.Tensor(self.imgs[indices_to_keep]).view(-1, 3, 32, 32)
        self.lbls = torch.Tensor([class_nms.index(train_set.classes[l]) for l in self.lbls[indices_to_keep]]).view(-1).long()

        # record index bounds and range:
        self.nb_classes = len(class_nms)
        self.nb_datapts = len(indices_to_keep)
        self.idxs = np.arange(self.nb_datapts)

    def sample_data(self, N):
        '''
        '''
        return np.random.choice(self.idxs, N, replace=False)

    #--------------------------------------------------------------------------#
    #               0.1 finish landscape definition by providing architecture  #
    #--------------------------------------------------------------------------#

class CifarAbstractArchitecture(CIFAR, FixedInitsLandscape):
    ''' 
    '''
    def __init__(self, class_nms=CIFAR.CLASS_NMS, weight_scale=1.0):
        ''' '''
        super().__init__(class_nms)
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



class CifarLeNet(CifarAbstractArchitecture):
    def __init__(self, class_nms=CIFAR.CLASS_NMS, weight_scale=1.0, widthA=16, widthB=32, widthC=64, verbose=False):
        super().__init__(class_nms, weight_scale)
        self.subweight_shapes = [
            (widthA          ,  3     , 5, 5),      (widthA,), 
            (widthB          , widthA , 5, 5),      (widthB,),
            (widthC          , widthB , 5, 5),      (widthC,),
            (self.nb_classes , widthC * 4*4 ),      (self.nb_classes, 1),
        ]

        self.widthA = widthA
        self.widthB = widthB
        self.widthC = widthC
        self.resample_weights()

        if verbose:
            print('LeNet has {} parameters'.format(
                sum(prod(w) for w in self.subweight_shapes)
            ))

    def logits_and_labels(self, data_indices):
        x, y = self.imgs[data_indices], self.lbls[data_indices]
        x = tanh(conv2d(x, self.get_subweight(0), bias=self.get_subweight(1), stride=1)) # 28 x 28
        x = tanh(conv2d(x, self.get_subweight(2), bias=self.get_subweight(3), stride=2)) # 12 x 12
        x = tanh(conv2d(x, self.get_subweight(4), bias=self.get_subweight(5), stride=2)) #  4 x  4
        x = x.view(-1, self.widthC*4*4, 1)
        x = matmul(self.get_subweight(6), x) + self.get_subweight(7).unsqueeze(0)
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
                ML = CifarMLP(
                    class_nms=CIFAR.CLASS_NMS,
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

 
if __name__=='__main__':
    #grid_search()

    BATCH = 1
    TIME = 10000
    model_nm = 'LENET'
    ML = CifarLeNet(verbose=True)
    ML.load_from('saved-weights/cifar-lenet.npy', nb_inits=6)
    ML.switch_to(0)
    LRATE = 0.1

    D = ML.sample_data(N=TIME) 
    for i in range(TIME):
        L = ML.get_loss_stalk(D[i:i+1])
        G = ML.nabla(L)
        ML.update_weights(-LRATE * G)

        if (i+1)%500: continue

        L_train= ML.get_loss_stalk(D)
        data = ML.sample_data(N=3000)
        L_test = ML.get_loss_stalk(data[:1500])
        L_test_= ML.get_loss_stalk(data[1500:])
        acc = ML.get_accuracy(ML.sample_data(N=3000))

        print(CC+' @C \t'.join([
            'after @M {:4d} @C steps'.format(i+1),
            'grad2 @G {:.2e}'.format(ML.nabla(L_test).dot(ML.nabla(L_test_)).detach().numpy()),
            'train loss @Y {:.2f}'.format(L_train.detach().numpy()),
            'test loss @L {:.2f}'.format(L_test.detach().numpy()),
            'test acc @O {:.2f}'.format(acc.detach().numpy()),
        '']))

    #for idx in range(6):
    #    ML = CifarLeNet(verbose=True)
    #    np.save('saved-weights/cifar-lenet-{:02d}.npy'.format(idx), [ML.get_weights()])