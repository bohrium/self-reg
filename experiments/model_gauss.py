''' author: samtenka
    change: 2019-01-11
    create: 2017-10-07
    descrp: Compare SGD and GD out-of-sample performances (for a shallow dense network on small MNIST training sets)
            over a log-spaced range of learning rates, then APPEND summary statistics to `results.txt`.  To run, type:
                python model_gauss.py 1000 24 32 ests_gauss.txt
            The    1000 represents the number of trials to perform
            the      24 represents the training set size;
            the      32 represents the desired floating point precision (32 or 64).

            
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
    BESSEL_FACTOR = float(BATCH_SIZE) / (BATCH_SIZE-1)

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
        Assumes batch size is divisible by 4.
    '''

    assert(BATCH_SIZE % 4 == 0)

    #--------------------------------------------------------------------------#
    #                0.1 compute SENTIMENT                                     #
    #--------------------------------------------------------------------------#
    
    # use of `Linker` forces tf's gradient computations to return an actual 0 instead of `None` when there is no
    #       dependency
    Linker = 0.0*tf.reduce_sum(Weights)
    LinkedLosses = Losses + Linker
    AvgLoss = tf.reduce_mean(LinkedLosses)
    
    #--------------------------------------------------------------------------#
    #                0.2 compute INTENSITY and UNCERTAINTY                     #
    #--------------------------------------------------------------------------#
    
    Gradients = tf.transpose(tf.convert_to_tensor(tf.gradients(LinkedLosses, Weights)))
    Gradients_0 = Gradients[:BATCH_SIZE//2, :] 
    Gradients_1 = Gradients[BATCH_SIZE//2:, :] 
    MeanSqrGrad, TraceCovar, _, __ = second_order_stats(Gradients, Gradients, BATCH_SIZE)
    
    #--------------------------------------------------------------------------#
    #                0.3 compute PASSION and AUDACITY                          #
    #--------------------------------------------------------------------------#
    
    MeanSqrGrad_1, TraceCovar_1, _, __ = second_order_stats(Gradients_1, Gradients_1, BATCH_SIZE//2)
    # below, we multiply by BATCH_SIZE//2 to counter tf.gradients' averaging behavior
    GradMeanSqrGrad = tf.transpose(tf.convert_to_tensor(tf.gradients(MeanSqrGrad_1, Weights))) * BATCH_SIZE
    GradTraceCovar =  tf.transpose(tf.convert_to_tensor(tf.gradients(TraceCovar_1, Weights))) * BATCH_SIZE
    Passion = tf.reduce_sum(tf.multiply(tf.reduce_mean(Gradients_0, axis=0), tf.reduce_mean(GradMeanSqrGrad, axis=0)))
    Audacity= tf.reduce_sum(tf.multiply(tf.reduce_mean(Gradients_0, axis=0), tf.reduce_mean(GradTraceCovar, axis=0)))
    
    #--------------------------------------------------------------------------#
    #                0.4 compute PERIL                                         #
    #--------------------------------------------------------------------------#
    
    # to estimate Peril, we split the batch into two subbatches and invoke multiplicativity of expectation for
    #       independents
    InterSubbatchGradientDots = tf.reduce_sum(tf.multiply(Gradients_0, Gradients_1), axis=1)
    # HessesTimesGrads is the derivative with respect to subbatch_0 of (gradients_0 * gradients_1):
    HessesTimesGrads = tf.transpose(tf.convert_to_tensor(tf.gradients(InterSubbatchGradientDots, Weights)))[:BATCH_SIZE//2, :]
    UncenteredPeril = tf.reduce_mean(tf.reduce_sum(tf.multiply(Gradients_1, HessesTimesGrads), axis=1))
    Peril = UncenteredPeril - Passion/2 

    return AvgLoss, MeanSqrGrad, TraceCovar, Passion, Audacity, Peril




###############################################################################
#                            0. READ DATASET                                  #
###############################################################################

class Dataset(object): 
    '''
    '''
    def __init__(self, max_size=1000):
        self.resample_ins(max_size)

    def resample_ins(self, ins_size):
        self.ins_inputs = np.random.randn(ins_size, 1)
        self.ins_size = ins_size
        self.index = 0

    def get_batch(self, batch_size, split='ins', opt='sgd', with_replacement=False):
        ''' Returns batch, by default from training set.  In case the specified optimizer is 'sgd', then `batch_size`
            and `with_replacment` become salient parameters; if the optimizer is 'gd', then the returned batch is the
            same (and of size equal to the number of training points) each time. 
        '''
        if split == 'out': 
            self.ins_inputs = np.random.randn(batch_size, 1)
            return out_inputs 
        if opt == 'gd': 
            return self.ins_inputs
        if with_replacement:
            indices = np.random.choice(self.ins_size, size)
            return self.ins_inputs[indices]
        if self.index + batch_size > self.ins_size: 
            assert(self.index == self.ins_size)
            indices = np.random.shuffle(np.arange(self.ins_size))
            self.ins_inputs = self.ins_inputs[indices] 
            self.index = 0
        rtrn = self.ins_inputs[self.index:self.index+batch_size]
        self.index += batch_size
        return rtrn

    def get_all(self, split='ins', max_size=1000):
        ''' Returns whole in-sample or out-of-sample points.  Good for evaluating train and test scores. ''' 
        if split == 'out':
            self.ins_inputs = np.random.randn(max_size, 1)
            return out_inputs
        return self.ins_inputs



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
        self.Data = tf.placeholder(precision, shape=[batch_size, 1])

        self.Weights = tf.get_variable('flattened', shape=[1+1, batch_size], dtype=precision)

        self.WeightsA = tf.transpose(tf.reshape(self.Weights[0, :],  [1, batch_size]), perm=[1, 0])
        self.WeightsB =  tf.transpose(tf.reshape(self.Weights[1, :],  [1, batch_size]), perm=[1, 0])

        self.InitWeightsA = tf.placeholder(precision, shape=[1])
        self.InitWeightsB = tf.placeholder(precision, shape=[1])
        self.Inits = tf.concat([tf.reshape(init, [-1]) for init in [self.InitWeightsA, self.InitWeightsB]], axis=0)
        self.Replicated = tf.stack([self.Inits]*batch_size, axis=1) 
        self.Initializer = tf.assign(self.Weights, self.Replicated)


    def create_trainer(self, batch_size, precision=tf.float32):
        ''' Define the loss and corresponding gradient-based update.  The difference between gd and sgd is not codified
            here; instead, the difference lies in the size and correlations between batches we use to train the
            classifier, i.e. in the values assigned to `Data` and `TrueOutputs` at each gradient update step.
        '''
        self.Losses = self.WeightsA - tf.square(self.WeightsA) + tf.square(self.WeightsB - tf.multiply(self.WeightsA, self.Data))
        self.Loss = tf.reduce_mean(self.Losses)
        
        self.Accuracy = 0.0#tf.reduce_mean(tf.cast(PredictionIsCorrect, precision))

        self.Sentiment, self.Intensity, self.Uncertainty, self.Passion, self.Audacity, self.Peril = gradient_stats(self.Losses, self.Weights, batch_size)
 

    def sample_init_weights(self): 
        ''' Sample weights (as numpy arrays) distributed according to Glorot-Bengio recommended length scales.  These
            weights are intended to be initializers. 
        '''
        wa = 1.0 + 0.0 * np.random.randn(1)
        ba = 1.0 + 0.0 * np.random.randn(1)
        return (wa, ba)

    def initialize_weights(self, wa, ba):
        ''' Initialize weights as a RUNTIME OPERATION, not by creating new graph nodes. '''
        self.session.run(self.Initializer, feed_dict={
            self.InitWeightsA:wa,
            self.InitWeightsB:ba,
        })

    def run(self, dataset): 
        ''' Compute post-training metrics for given dataset and hyperparameters.  Return in-sample loss, out-of-sample
            loss, in-sample accuracy, and out-of-sample accuracy --- in that order.
        '''
        ins_inputs = dataset.get_all('ins') 
        sentiment, intensity, uncertainty, passion, audacity, peril = self.session.run([self.Sentiment, self.Intensity, self.Uncertainty, self.Passion, self.Audacity, self.Peril], feed_dict={
            self.Data:ins_inputs,
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

    sentiment, intensity, uncertainty, passion, audacity, peril = [], [], [], [], [], []
    for h in tqdm(range(nb_trials)):
        mnist.resample_ins(ins_size)
        init_weights = mlp.sample_init_weights() 
        mlp.initialize_weights(*init_weights)
        se_, in_, un_, pa_, au_, pe_ = mlp.run(mnist)
        sentiment  .append(se_)
        intensity  .append(in_)
        uncertainty.append(un_)
        passion    .append(pa_)
        audacity   .append(au_)
        peril      .append(pe_)
    print('MEAN: sent %.8f, intense %.8f, uncert %.8f, pass %.8f, aud %.8f, peril %.8f' % (
        np.mean(sentiment), 
        np.mean(intensity), 
        np.mean(uncertainty), 
        np.mean(passion), 
        np.mean(audacity), 
        np.mean(peril)))
    print('SDEV: sent %.8f, intense %.8f, uncert %.8f, pass %.8f, aud %.8f, peril %.8f' % (
        np.std(sentiment), 
        np.std(intensity), 
        np.std(uncertainty), 
        np.std(passion), 
        np.std(audacity), 
        np.std(peril)))

    with open(FILE_NM, 'w') as f:
        f.write('%d TRIALS, %d SAMPLES\n' % (nb_trials, ins_size))
        f.write('MEAN: sent %.8f, intense %.8f, uncert %.8f, pass %.8f, aud %.8f, peril %.8f\n' % (
            np.mean(sentiment), 
            np.mean(intensity), 
            np.mean(uncertainty), 
            np.mean(passion), 
            np.mean(audacity), 
            np.mean(peril)))
        f.write('SDEV: sent %.8f, intense %.8f, uncert %.8f, pass %.8f, aud %.8f, peril %.8f\n' % (
            np.std(sentiment), 
            np.std(intensity), 
            np.std(uncertainty), 
            np.std(passion), 
            np.std(audacity), 
            np.std(peril)))

if __name__=='__main__':
    run_experiment(nb_trials=NB_TRIALS, ins_size=INS_SIZE)
