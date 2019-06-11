''' author: samtenka
    change: 2019-06-10
    create: 2019-06-10
    descrp: define loss landscape type 
'''

from abc import ABC, abstractmethod
from itertools import chain 
import numpy as np
import tensorflow as tf

class PointedLandscape(ABC):
    '''
    '''

    @abstractmethod
    def get_data_sample(self, N): 
        ''' sample N datapoints (i.e. memory-light objects indexing deterministic loss landscapes)
            independently and identically distributed from the population.
        '''
        pass

    @abstractmethod
    def get_weight(self):
        ''' give weight as tensorflow variable node (so treatable as reference, e.g. writable) '''
        pass

    @abstractmethod
    def get_random_loss_field(self):
        ''' give loss as a random tensor field (i.e. differentiable tensor still awaiting actual
            data in the form of feed_dict arguments)
        '''
        pass

    @abstractmethod
    def evaluate_as_tensor(self, session, random_tensor_fields, datapts):
        ''' give tensors (i.e. numpy arrays) corresponding to datapoints '''
        pass

class MNIST(PointedLandscape):
    ''' load specified digits of MNIST, e.g. just 0s and 1s for binary classification subtask.
        implements PointedLandscape's `get_data_sample` but not its `get_weight`,
        `get_random_loss_field`, or `evaluate_as_tensor`.
    '''
    def __init__(self, digits=list(range(10))):
        (ins_imgs, ins_lbls), (out_imgs, out_lbls) = tf.keras.datasets.mnist.load_data()

        self.nb_classes = len(digits)
        self.nb_datapoints = len(ins_imgs) + len(out_imgs) 
        self.idxs = np.arange(self.nb_datapoints)

        self.imgs = []
        self.lbls = []
        for img, lbl in zip(chain(ins_imgs, out_imgs),
                            chain(ins_lbls, out_lbls)):
            if lbl not in digits: continue 
            self.imgs.append(np.reshape(img, (28, 28)))
            self.lbls.append(np.eye(self.nb_classes)[lbl])
        self.imgs = np.array(self.imgs)
        self.lbls = np.array(self.lbls)

    def get_data_sample(self, N):
        return np.random.choice(self.idxs, N, replace=False)

class MNIST_Logistic(MNIST):
    def __init__(self, digits=list(range(10))):
        super().__init__(digits)

        self.true_imgs = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
        self.true_lbls = tf.placeholder(dtype=tf.float32, shape=[None, self.nb_classes])
        self.weights = tf.get_variable(
            'mnist-logistic-weights',
            shape=[28*28, self.nb_classes],
            dtype=tf.float32
        )
        flattened = tf.reshape(self.true_imgs, shape=[-1, 28*28])
        self.logits = tf.matmul(flattened, self.weights) 
        self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.true_lbls,
            logits=self.logits
        )

    def get_weight(self):
        return self.weights

    def get_random_loss_field(self):
        return self.losses

    def evaluate_as_tensor(self, session, random_tensor_fields, datapts):
        return session.run(random_tensor_fields, feed_dict={
            self.true_imgs : self.imgs[datapts],
            self.true_lbls : self.lbls[datapts]
        })

m = MNIST_Logistic()
