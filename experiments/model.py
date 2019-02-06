''' author: samtenka
    change: 2019-01-11
    create: 2017-10-07
    descrp: Compare SGD and GD out-of-sample performances (for a shallow dense network on small MNIST training sets)
            over a log-spaced range of learning rates, then APPEND summary statistics to `results.txt`.  To run, type:
                python model.py 100 256 64 ests.txt
            The     100 represents the number of trials to perform per experimental condition;
            the     256 represents the training set size;
            the      64 represents the desired floating point precision (32 or 64).

            
'''

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys 

assert(len(sys.argv)==5)
NB_TRIALS = int(sys.argv[1]) 
INS_SIZE  = int(sys.argv[2]) 
PRECISION = {32:tf.float32, 64:tf.float64}[int(sys.argv[3])]
FILE_NM   = sys.argv[4]


################################################################################
#            0. DEFINE STATISTICS                                              #
################################################################################

    #--------------------------------------------------------------------------#
    #                0.0 define unbiased estimator helper function             #
    #--------------------------------------------------------------------------#

def second_order_stats(x, y, BATCH_SIZE): 
    ''' Return unbiased estimates of
            PRODUCT_MEAN =  trace({x}{y}),
            COVARIANCE   =  trace({xy}-{x}{y}), and
            AVG_PRODUCT  =  trace({xy})
            PRODUCT_AVG  =  trace({x}{y})
        for given tensors x and y of shape (nb samples)-by-(dim)
    '''
    BESSEL_FACTOR = BATCH_SIZE / (BATCH_SIZE-1)

    avg_x = tf.reduce_mean(x, axis=0)
    avg_y = tf.reduce_mean(y, axis=0) if x is not y else avg_x

    avg_product = tf.reduce_sum(tf.reduce_mean(tf.multiply(x, y), axis=0))
    product_avg = tf.reduce_sum(tf.multiply(avg_x, avg_y))
    
    covariance = BESSEL_FACTOR * (avg_product - product_avg)
    product_mean = avg_product - covariance

    return product_mean, covariance, avg_product, product_avg

def gradient_stats(Losses, Weights, BATCH_SIZE): 
    ''' Given loss batch and weight tensors, return unbiased estimates of
        sentiment, intensity, uncertainty, passion, audacity, and peril.
        Assumes that Weights has shape (nb_weights, batch_size), i.e. consists of weight replicas.
        Assumes batch size is even.
    '''

    assert(BATCH_SIZE % 2 == 0)

    #--------------------------------------------------------------------------#
    #                0.1 compute SENTIMENT                                     #
    #--------------------------------------------------------------------------#
    
    # use of `Linker` forces tf's gradient computations to return an actual 0 instead of `None` when there is no
    #       dependency
    Linker = 0.0*tf.reduce_sum(tf.square(Weights))
    LinkedLosses = Losses + Linker
    AvgLoss = tf.reduce_mean(LinkedLosses)
    
    #--------------------------------------------------------------------------#
    #                0.2 compute INTENSITY and UNCERTAINTY                     #
    #--------------------------------------------------------------------------#
    
    Gradients = tf.transpose(tf.convert_to_tensor(tf.gradients(LinkedLosses, Weights)))
    MeanSqrGrad, TraceCovar, AvgSqrNorm, SqrAvgNorm = second_order_stats(Gradients, Gradients, BATCH_SIZE)
    
    #--------------------------------------------------------------------------#
    #                0.3 compute PASSION and AUDACITY                          #
    #--------------------------------------------------------------------------#
    
    # below, we multiply by BATCH_SIZE to counter tf.gradients' averaging behavior
    GradMeanSqrGrad = tf.transpose(tf.convert_to_tensor(tf.gradients(MeanSqrGrad, Weights))) * BATCH_SIZE
    GradTraceCovar =  tf.transpose(tf.convert_to_tensor(tf.gradients(TraceCovar, Weights))) * BATCH_SIZE
    Passion, _, __, ___ = second_order_stats(Gradients, GradMeanSqrGrad, BATCH_SIZE)
    Audacity, _, __, ___ = second_order_stats(Gradients, GradTraceCovar, BATCH_SIZE)
    
    #--------------------------------------------------------------------------#
    #                0.4 compute PERIL                                         #
    #--------------------------------------------------------------------------#
    
    # to estimate Peril, we split the batch into two subbatches and invoke multiplicativity of expectation for
    #       independents
    Gradients_0 = Gradients[:BATCH_SIZE//2, :] 
    Gradients_1 = Gradients[BATCH_SIZE//2:, :] 
    InterSubbatchGradientDots = tf.reduce_sum(tf.multiply(Gradients_0, Gradients_1), axis=1)
    # HessesTimesGrads is the derivative with respect to subbatch_0 of (gradients_0 * gradients_1):
    HessesTimesGrads = tf.transpose(tf.convert_to_tensor(tf.gradients(InterSubbatchGradientDots, Weights)))[:BATCH_SIZE//2, :]
    UncenteredPeril = tf.reduce_mean(tf.reduce_sum(tf.multiply(Gradients_1, HessesTimesGrads), axis=1))
    Peril = UncenteredPeril - Passion/2 

    return AvgLoss, MeanSqrGrad, TraceCovar, Passion, Audacity, Peril





###############################################################################
#                            0. READ DATASET                                  #
###############################################################################

from tensorflow.examples.tutorials.mnist import input_data

class Dataset(object): 
    ''' MNIST is a classic image-classification dataset containing 28x28 grayscale photographs of handwritten digits (0
        through 9).  This class provides access to MNIST via in-sample and out-of-sample batches.  It allows us to
        sample from a training set potentially smaller than the the overall MNIST training set, and to sample with or
        without replacement.  Note that we load labels as one-hot vectors, making it easier to define losses.  Thus,
        `get_batch` and `get_all` each return two arrays of shape (???, 28*28) and (???, 10), respectively.
    '''
    def __init__(self):
        ''' Read MNIST, with labels one-hot. '''
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.resample_ins(len(self.mnist.train.images))

    def resample_ins(self, ins_size):
        ''' MNIST is a classic image-classification dataset.  Its images are 28x28 grayscale photographs of handwritten
            digits (0 through 9).  Note that we load labels as one-hot vectors, making it easier to define the loss.
        '''
        self.ins_inputs, self.ins_outputs = self.mnist.train.next_batch(ins_size) 
        self.ins_size = ins_size
        self.index = 0

    def get_batch(self, batch_size, split='ins', opt='sgd', with_replacement=False):
        ''' Returns batch, by default from training set.  In case the specified optimizer is 'sgd', then `batch_size`
            and `with_replacment` become salient parameters; if the optimizer is 'gd', then the returned batch is the
            same (and of size equal to the number of training points) each time. 
        '''
        if split == 'out': 
            return self.mnist.test.next_batch(batch_size) 
        if opt == 'gd': 
            return self.ins_inputs, self.ins_outputs
        if with_replacement:
            indices = np.random.choice(self.ins_size, size)
            return self.ins_inputs[indices], self.ins_outputs[indices]
        if self.index + batch_size > self.ins_size: 
            assert(self.index == self.ins_size)
            indices = np.random.shuffle(np.arange(self.ins_size))
            self.ins_inputs = self.ins_inputs[indices] 
            self.ins_outputs = self.ins_outputs[indices] 
            self.index = 0
        rtrn = self.ins_inputs[self.index:self.index+batch_size], self.ins_outputs[self.index:self.index+batch_size]
        self.index += batch_size
        return rtrn

    def get_all(self, split='ins'):
        ''' Returns whole in-sample or out-of-sample points.  Good for evaluating train and test scores. ''' 
        if split == 'out':
            return self.mnist.test.images, self.mnist.test.labels
        return self.ins_inputs, self.ins_outputs



###############################################################################
#                            1. DEFINE MODEL                                  #
###############################################################################

def slrelu(x, leak=0.1):
    ''' Smooth leaky ReLU activation function: slope is `leak` near -infinity and slope is 1 near +infinity. '''
    return tf.log(tf.exp(leak*x) + tf.exp(x))

class Classifier(object):
    ''' Creates, (re)initializes, trains, and evaluates a shallow neural network. '''
    def __init__(self, batch_size, precision=PRECISION):
        self.create_model(batch_size, precision)
        self.create_trainer(batch_size, precision)
        self.session = tf.Session()
        self.initialize_weights(*self.sample_init_weights())

    def create_model(self, batch_size, precision=tf.float32):
        ''' Construct a densely connected 28*28 --> 64 --> 10 neural network (with slrelu activations). '''
        self.TrueInputs = tf.placeholder(precision, shape=[batch_size, 28*28])
        self.TrueOutputs= tf.placeholder(precision, shape=[batch_size, 10])

        self.Weights = tf.get_variable('flattened', shape=[28*28*64 + 64 + 64*10 + 10, batch_size], dtype=precision)

        WeightsA = tf.transpose(tf.reshape(self.Weights[:28*28*64, :],                               [28*28, 64, batch_size]), perm=[2, 0, 1])
        BiasesA =  tf.transpose(tf.reshape(self.Weights[ 28*28*64:28*28*64+64, :],                       [1, 64, batch_size]), perm=[2, 0, 1])
        WeightsB = tf.transpose(tf.reshape(self.Weights[          28*28*64+64:28*28*64+64+64*10, :],    [64, 10, batch_size]), perm=[2, 0, 1])
        BiasesB =  tf.transpose(tf.reshape(self.Weights[                      28*28*64+64+64*10:, :],    [1, 10, batch_size]), perm=[2, 0, 1])

        self.InitWeightsA= tf.placeholder(precision, shape=[  28*28,  64])
        self.InitBiasesA = tf.placeholder(precision, shape=[          64])
        self.InitWeightsB= tf.placeholder(precision, shape=[     64,  10])
        self.InitBiasesB = tf.placeholder(precision, shape=[          10])
        self.Inits = tf.concat([tf.reshape(init, [-1]) for init in [self.InitWeightsA, self.InitBiasesA, self.InitWeightsB, self.InitBiasesB]], axis=0)
        self.Replicated = tf.stack([self.Inits]*batch_size, axis=1) 
        self.Initializer = tf.assign(self.Weights, self.Replicated)

        HiddenLayerA = slrelu(tf.matmul(tf.stack([self.TrueInputs], axis=1), WeightsA) + BiasesA)
        HiddenLayerB = tf.matmul(HiddenLayerA, WeightsB) + BiasesB
        self.PredictedLogits = tf.reshape(HiddenLayerB, shape=[batch_size, 10])

    def create_trainer(self, batch_size, precision=tf.float32):
        ''' Define the loss and corresponding gradient-based update.  The difference between gd and sgd is not codified
            here; instead, the difference lies in the size and correlations between batches we use to train the
            classifier, i.e. in the values assigned to `TrueInputs` and `TrueOutputs` at each gradient update step.
        '''
        CrossEntropies = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.TrueOutputs,
            logits=self.PredictedLogits
        ))
        self.Loss = tf.reduce_mean(CrossEntropies)
        
        PredictionIsCorrect = tf.equal(tf.argmax(self.PredictedLogits, 1), tf.argmax(self.TrueOutputs, 1))
        self.Accuracy = tf.reduce_mean(tf.cast(PredictionIsCorrect, precision))

        self.Sentiment, self.Intensity, self.Uncertainty, self.Passion, self.Audacity, self.Peril = gradient_stats(CrossEntropies, self.Weights, batch_size)
 

    def sample_init_weights(self): 
        ''' Sample weights (as numpy arrays) distributed according to Glorot-Bengio recommended length scales.  These
            weights are intended to be initializers. 
        '''
        wa = np.random.randn(28*28, 64) * np.sqrt(2.0 / (28*28 + 64))
        ba = np.random.randn(64) * np.sqrt(2.0 / 64)
        wb = np.random.randn(64, 10) * np.sqrt(2.0 / (64 + 10))
        bb = np.random.randn(10) * np.sqrt(2.0 / 10)
        return (wa, ba, wb, bb)

    def initialize_weights(self, wa, ba, wb, bb): 
        ''' Initialize weights as a RUNTIME OPERATION, not by creating new graph nodes. '''
        self.session.run(self.Initializer, feed_dict={
            self.InitWeightsA:wa,
            self.InitBiasesA:ba,
            self.InitWeightsB:wb,
            self.InitBiasesB:bb
        })

    def run(self, dataset): 
        ''' Compute post-training metrics for given dataset and hyperparameters.  Return in-sample loss, out-of-sample
            loss, in-sample accuracy, and out-of-sample accuracy --- in that order.
        '''
        ins_inputs, ins_outputs = dataset.get_all('ins') 
        sentiment, intensity, uncertainty, passion, audacity, peril = self.session.run([self.Sentiment, self.Intensity, self.Uncertainty, self.Passion, self.Audacity, self.Peril], feed_dict={
            self.TrueInputs:ins_inputs,
            self.TrueOutputs:ins_outputs
        })
        return sentiment, intensity, uncertainty, passion, audacity, peril



###############################################################################
#                            3. EXPERIMENT LOOP                               #
###############################################################################

def run_experiment(nb_trials, ins_size):
    ''' Compute the 'eta curve', that is, sgd vs gd test performance as a function of learning rate.  Record summary 
        statistics by appending to the given filename.
    '''
    mnist = Dataset()
    mlp = Classifier(ins_size)

    sentiment_acc, intensity_acc, uncertainty_acc, passion_acc, audacity_acc, peril_acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for h in tqdm(range(nb_trials)):
        mnist.resample_ins(ins_size)
        init_weights = mlp.sample_init_weights() 
        mlp.initialize_weights(*init_weights)
        sentiment, intensity, uncertainty, passion, audacity, peril = mlp.run(mnist)
        sentiment_acc   += sentiment
        intensity_acc   += intensity
        uncertainty_acc += uncertainty
        passion_acc     += passion
        audacity_acc    += audacity
        peril_acc       += peril
    print('sent %.8f, intense %.8f, uncert %.8f, pass %.8f, aud %.8f, peril %.8f' % (
        sentiment_acc  /nb_trials, 
        intensity_acc  /nb_trials, 
        uncertainty_acc/nb_trials, 
        passion_acc    /nb_trials, 
        audacity_acc   /nb_trials, 
        peril_acc      /nb_trials))

    with open(FILE_NM, 'w') as f:
        f.write('sent %.8f, intense %.8f, uncert %.8f, pass %.8f, aud %.8f, peril %.8f' % (
        sentiment_acc  /nb_trials, 
        intensity_acc  /nb_trials, 
        uncertainty_acc/nb_trials, 
        passion_acc    /nb_trials, 
        audacity_acc   /nb_trials, 
        peril_acc      /nb_trials))


if __name__=='__main__':
    run_experiment(nb_trials=NB_TRIALS, ins_size=INS_SIZE)
