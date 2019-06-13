''' author: samtenka
    change: 2019-06-12
    create: 2019-06-10
    descrp: define loss landscape type 
'''

from abc import ABC, abstractmethod
from itertools import chain 
import numpy as np

import tensorflow as tf
from utils import CC, suppress_tf_warnings
suppress_tf_warnings()



################################################################################
#           0. DEFINE LANDSCAPE INTERFACE                                      #
################################################################################

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
    def update_weight(self, displacement):
        pass

    @abstractmethod
    def get_random_loss_field(self):
        ''' give loss as a random tensor field (i.e. differentiable tensor still awaiting actual
            data in the form of feed_dict arguments)
        '''
        pass

    @abstractmethod
    def nabla(self, random_scalar_field):
        ''' differentiate scalar field to return random tensor field of same shape as weights '''
        pass

    @abstractmethod
    def evaluate_as_tensor(self, random_tensor_fields, datapts):
        ''' give tensors (i.e. numpy arrays) corresponding to datapoints '''
        pass



################################################################################
#           1. MNIST LOGISTIC EXAMPLE                                          #
################################################################################

    #--------------------------------------------------------------------------#
    #               1.0 begin defining landscape by providing data population  #
    #--------------------------------------------------------------------------#

class MNIST(PointedLandscape):
    ''' load specified digits of MNIST, e.g. just 0s and 1s for binary classification subtask.
        implements PointedLandscape's `get_data_sample` but not its `update_weight`, `nabla`,
        `get_random_loss_field`, or `evaluate_as_tensor`.
    '''
    def __init__(self, digits=list(range(10))):
        (ins_imgs, ins_lbls), (out_imgs, out_lbls) = tf.keras.datasets.mnist.load_data()

        self.nb_classes = len(digits)

        self.imgs = []
        self.lbls = []
        for img, lbl in zip(chain(ins_imgs, out_imgs),
                            chain(ins_lbls, out_lbls)):
            if lbl not in digits: continue 
            self.imgs.append(np.reshape(img, (28, 28)))
            self.lbls.append(np.eye(self.nb_classes)[lbl])
        self.imgs = np.array(self.imgs)
        self.lbls = np.array(self.lbls)

        self.nb_datapts = len(self.imgs)
        self.idxs = np.arange(self.nb_datapts)

    def get_data_sample(self, N):
        return np.random.choice(self.idxs, N, replace=False)

    #--------------------------------------------------------------------------#
    #               1.1 finish defining landscape by providing loss field      #
    #--------------------------------------------------------------------------#

class MNIST_Logistic(MNIST):
    def __init__(self, digits=list(range(10))):
        super().__init__(digits)

        self.true_imgs = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
        self.true_lbls = tf.placeholder(dtype=tf.float32, shape=[None, self.nb_classes])
        self.weights = tf.get_variable(
            'mnist-logistic-weights',
            shape=[28*28, self.nb_classes],
            dtype=tf.float32,
            initializer=tf.zeros_initializer
        )
        flattened = tf.reshape(self.true_imgs, shape=[-1, 28*28])
        self.logits = tf.matmul(flattened, self.weights) 
        self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.true_lbls,
            logits=self.logits
        )

        self.displacement = tf.placeholder(dtype=tf.float32, shape=self.weights.shape)  
        self.updater = tf.assign(self.weights, self.weights+self.displacement)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def update_weight(self, certain_tensor):
        self.session.run(self.updater, feed_dict={
            self.displacement : certain_tensor
        })

    def nabla(self, random_scalar_field):
        assert random_scalar_field.shape == (), 'can only differentiate scalar field'
        return tf.convert_to_tensor(
            tf.gradients(random_scalar_field, self.weights)[0]
        )

    def get_random_loss_field(self):
        return self.losses

    def evaluate_as_tensor(self, random_tensor_fields, datapts):
        return self.session.run(random_tensor_fields, feed_dict={
            self.true_imgs : self.imgs[datapts],
            self.true_lbls : self.lbls[datapts]
        })

    #--------------------------------------------------------------------------#
    #               1.2 demonstrate gradstats and descent on example           #
    #--------------------------------------------------------------------------#

if __name__ == '__main__':
    m = MNIST_Logistic(digits=[0, 1])
    print(CC+'@G initialized @W MNIST landscape!')
    print(CC+'    @R {} @W samples and @R {} @W classes'.format(m.nb_datapts, m.nb_classes))

    batch_size = 4
    ins = m.get_data_sample(N=batch_size)
    out = m.get_data_sample(N=batch_size)
    print(CC+'@G sampled @W datapoints!')
    print(CC+'    @R {} @W samples'.format(batch_size))

    l = m.get_random_loss_field()
    print(CC+'@G compiled @W losses!')
    for name, data in {'train':ins, 'test':out}.items():
        print(CC+'    {} losses are @R {} @W '.format(
            name, '@W , @R '.join(str(el) for el in m.evaluate_as_tensor(l, data))
        ))
    
    mean_l = tf.reduce_mean(l, axis=0)
    g = m.nabla(mean_l)
    gg = tf.reduce_sum(tf.square(g))
    print(CC+'@G compiled @W gradients!')
    print(CC+'    on datapoint 0 squarenorm is @R {} @W '.format(
        m.evaluate_as_tensor(gg, ins)
    ))

    hg = 0.5 * m.nabla(gg)
    ghhg = tf.reduce_sum(tf.square(hg))
    print(CC+'@G compiled @W hessian-gradient!')
    print(CC+'    on datapoint 0 squarenorm is @R {} @W '.format(
        m.evaluate_as_tensor(ghhg, ins)
    ))


    for i in range(10):
        m.update_weight(m.evaluate_as_tensor(-1e-7 * g, ins))
        print(CC+'@G descended @W on gradients!')
        print(CC+'    '+';\t'.join(
            'average {} loss is now @R {:.3} @W '.format(
                name, float(m.evaluate_as_tensor(mean_l, data))
            )
            for name, data in {'train':ins, 'test':out}.items()
        ))

