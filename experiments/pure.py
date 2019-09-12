import numpy as np

from utils import CC
from landscape import PointedLandscape
from gradstats import grad_stat_names, GradStats
import torch
import tqdm


#from mnist_landscapes import MnistLogistic, MnistLeNet, MnistMLP
#
#LC = MnistLeNet(digits=list(range(10)))
#LC.load_from('saved-weights/mnist-logistic.npy')
#
#for i in range(10):
#    LC.switch_to(0)
#    sgd_test_loss = LC.get_loss_stalk(LC.sample_data(10))
#    print(sgd_test_loss)

 

#if __name__ == '__main__':
#    from mnist_landscapes import MnistLogistic, MnistLeNet, MnistMLP
#    LC = MnistLogistic(digits=list(range(10)))
#    LC.load_from('saved-weights/mnist-logistic.npy')
#
#    for idx in range(1):
#        for i in range(10):
#            LC.switch_to(0)
#            sgd_test_loss = LC.get_loss_stalk(LC.sample_data(10))
#            print('#'*8, sgd_test_loss.detach().numpy())
#
