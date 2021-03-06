''' author: samtenka
    change: 20125.04-14
    create: 2017-10-07
    descrp: Estimate and save gradient statistics (on binary MNIST)
            To run, type:
                python differentiate_deep17.py 10000 24 32 gradstats_deep17.txt
            The                 10000   gives   a number of trials to perform;
            the                    24   gives   a training set size;
            the                    32   gives   a desired floating point precision (32 or 64);
            the    gradstats_deep17.txt   gives   a filename to overwrite with results.
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
    
    # Use of `Linker` forces tf.gradient to return an actual 0 instead of `None` when there is no dependency:
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
    # Below, we multiply by BATCH_SIZE//2 to counter tf.gradients' averaging behavior:
    GradMeanSqrGrad = tf.transpose(tf.convert_to_tensor(tf.gradients(MeanSqrGrad_1, Weights))) * BATCH_SIZE
    GradTraceCovar =  tf.transpose(tf.convert_to_tensor(tf.gradients(TraceCovar_1, Weights))) * BATCH_SIZE
    Passion = tf.reduce_sum(tf.multiply(tf.reduce_mean(Gradients_0, axis=0), tf.reduce_mean(GradMeanSqrGrad, axis=0)))
    Audacity= tf.reduce_sum(tf.multiply(tf.reduce_mean(Gradients_0, axis=0), tf.reduce_mean(GradTraceCovar, axis=0)))
    
    #--------------------------------------------------------------------------#
    #                0.4 compute PERIL                                         #
    #--------------------------------------------------------------------------#
    
    # To estimate Peril, we split the batch in two and invoke multiplicativity of expectation for independents:
    InterSubbatchGradientDots = tf.reduce_sum(tf.multiply(Gradients_0, Gradients_1), axis=1)
    # HessesTimesGrads is the derivative with respect to subbatch_0 of (gradients_0 * gradients_1):
    HessesTimesGrads = tf.transpose(tf.convert_to_tensor(tf.gradients(InterSubbatchGradientDots, Weights)))[:BATCH_SIZE//2, :]
    UncenteredPeril = tf.reduce_mean(tf.reduce_sum(tf.multiply(Gradients_1, HessesTimesGrads), axis=1))
    Peril = UncenteredPeril - Passion/2 

    #--------------------------------------------------------------------------#
    #                0.5 compute SERENDIPITY                                   #
    #--------------------------------------------------------------------------#
    IGD = tf.reduce_sum(tf.multiply(Gradients, Gradients), axis=1)
    HG = tf.transpose(tf.convert_to_tensor(tf.gradients(IGD, Weights))) / 2.0
    GHG = tf.reduce_mean(tf.reduce_sum(tf.multiply(Gradients, HG), axis=1), axis=0)

    IGD_ = tf.reduce_sum(tf.multiply(Gradients_0, Gradients_1), axis=1)
    HG_  = tf.transpose(tf.convert_to_tensor(tf.gradients(IGD_, Weights)))[BATCH_SIZE//2:, :]
    GHG_ = tf.reduce_mean(tf.reduce_sum(tf.multiply(Gradients_0, HG_), axis=1), axis=0)

    Serendipity = GHG - GHG_ 

    return AvgLoss, MeanSqrGrad, TraceCovar, Passion, Audacity, Peril, Serendipity



################################################################################
#            1. DEFINE DATASET BATCHES                                         #
################################################################################

from tensorflow.examples.tutorials.mnist import input_data

class Dataset(object): 
    ''' MNIST is a classic image-classification dataset containing 28x28 grayscale photographs of handwritten digits (0
        through 9).  This class provides access to MNIST via in-sample and out-of-sample batches.  It allows us to
        sample from a training set potentially smaller than the the overall MNIST training set, and to sample with or
        without replacement.  Note that we DO NOT load labels as one-hot vectors.
        Thus, `get_batch` and `get_all` each return two arrays of shape (???, 28) and (???), respectively.
    '''
    def __init__(self):
        ''' Read MNIST, with labels one-hot. '''
        mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

        self.ins_images = []
        self.ins_labels = []
        for img, lbl in zip(mnist.train.images, mnist.train.labels):  
            if 2<=lbl: continue
            self.ins_images.append(np.reshape(img, (28, 28))[:, 14])
            self.ins_labels.append(lbl)

        self.out_images = []
        self.out_labels = []
        for img, lbl in zip(mnist.test.images, mnist.test.labels):  
            if 2<=lbl: continue
            self.out_images.append(np.reshape(img, (28, 28))[:, 14])
            self.out_labels.append(lbl)                      


        total_length = len(self.ins_images)+len(self.out_images)
        self.ins_images, self.out_images = ((self.ins_images + self.out_images)[:total_length//2],
                                            (self.ins_images + self.out_images)[total_length//2:])
        self.ins_labels, self.out_labels = ((self.ins_labels + self.out_labels)[:total_length//2],
                                            (self.ins_labels + self.out_labels)[total_length//2:])   
        print(total_length//2, total_length-total_length//2)

        self.ins_images = np.array(self.ins_images)
        self.ins_labels = np.array(self.ins_labels)
        self.out_images = np.array(self.out_images)
        self.out_labels = np.array(self.out_labels)

        print(len(self.ins_images))
        print(len(self.out_images))

    def resample(self, sample_size):
        ''' MNIST is a classic image-classification dataset.  Its images are 28x28 grayscale photographs of handwritten
            digits (0 through 9).  Note that we load labels as one-hot vectors, making it easier to define the loss.
        '''
        indices = np.random.choice(np.arange(len(self.ins_images)), size=sample_size, replace=False)
        self.sample_images = self.ins_images[indices]
        self.sample_labels = self.ins_labels[indices]
        self.sample_size = sample_size
        self.index = 0

    def get_batch(self, batch_size, split='ins', opt='sgd', with_replacement=False):
        ''' Returns batch, by default from training set.  In case the specified optimizer is 'sgd', then `batch_size`
            and `with_replacment` become salient parameters; if the optimizer is 'gd', then the returned batch is the
            same (and of size equal to the number of training points) each time. 
        '''
        if split == 'out': 
            indices = np.random.choice(np.arange(len(self.out_images)), size=batch_size, replace=False)
            return self.out_images[indices], self.out_labels[indices]
        if opt == 'gd': 
            return self.sample_images, self.sample_labels
        if with_replacement:
            indices = np.random.choice(self.sample_images, size)
            return self.sample_images[indices], self.sample_labels[indices]
        if self.index + batch_size > self.sample_size: # then shuffle 
            assert(self.index == self.sample_size)
            indices = np.random.shuffle(np.arange(self.sample_size))
            self.sample_images = self.sample_images[indices]
            self.sample_labels = self.sample_labels[indices] 
            self.index = 0
        rtrn = self.sample_images[self.index:self.index+batch_size], self.sample_labels[self.index:self.index+batch_size]
        self.index += batch_size
        return rtrn

    def get_all(self, split='ins'):
        ''' Returns whole in-sample or out-of-sample points.  Good for evaluating train and test scores. ''' 
        if split == 'out':
            return self.out_images, self.out_labels
        return self.sample_images, self.sample_labels




################################################################################
#            2. DEFINE LEARNING MODEL                                          #
################################################################################

class Learner(object):
    ''' Creates, (re)initializes, trains, and evaluates a deep17 neural network. '''
    def __init__(self, batch_size, precision=PRECISION):
        self.create_model(batch_size, precision)
        self.create_trainer(batch_size, precision)
        self.session = tf.Session()
        self.initialize_weights(*self.sample_init_weights())

    def create_model(self, batch_size, precision=tf.float32):
        ''' Define the loss landscape as a function of weights and data. '''
        self.Images = tf.placeholder(precision, shape=[batch_size, 28])
        self.Labels= tf.placeholder(precision, shape=[batch_size])

        sizeA = 28*25
        sizeB = 25*25
        sizeC = 25*1
        self.Weights = tf.get_variable('flattened', shape=[sizeA + sizeB + sizeC, batch_size], dtype=precision)
        self.WeightsA = tf.transpose(tf.reshape(self.Weights[:sizeA,:], [28, 25, batch_size]), perm=(2,0,1))
        self.WeightsB = tf.transpose(tf.reshape(self.Weights[sizeA:sizeA+sizeB,:], [25, 25, batch_size]), perm=(2,0,1))
        self.WeightsC = tf.transpose(tf.reshape(self.Weights[sizeA+sizeB:,:], [25, 1, batch_size]), perm=(2,0,1))

        self.InitWeightsA = tf.placeholder(precision, shape=[28*25])
        self.InitWeightsB = tf.placeholder(precision, shape=[25*25])
        self.InitWeightsC = tf.placeholder(precision, shape=[25*1])
        self.Inits = tf.concat([tf.reshape(init, [-1]) for init in [self.InitWeightsA, self.InitWeightsB, self.InitWeightsC]], axis=0)
        self.Replicated = tf.stack([self.Inits]*batch_size, axis=1) 
        self.Initializer = tf.assign(self.Weights, self.Replicated)

        self.Hidden = tf.math.tanh(tf.matmul(tf.expand_dims(self.Images, 1), self.WeightsA))
        self.Hidden = tf.math.tanh(tf.matmul(self.Hidden, self.WeightsB))
        self.Logits = tf.reshape(tf.matmul(self.Hidden, self.WeightsC), [-1])
        self.Losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Labels, logits=self.Logits) 

    def create_trainer(self, batch_size, precision=tf.float32):
        ''' Define the loss and corresponding gradient-based update.  The difference between gd and sgd is not codified
            here; instead, the difference lies in the size and correlations between batches we use to train the
            classifier, i.e. in the values assigned to `Data` and `TrueOutputs` at each gradient update step.
        '''
        self.Loss = tf.reduce_mean(self.Losses)
        self.Sentiment, self.Intensity, self.Uncertainty, self.Passion, self.Audacity, self.Peril, self.Serendipity  = gradient_stats(self.Losses, self.Weights, batch_size)

    def sample_init_weights(self): 
        ''' Sample weights (as numpy arrays) distributed according to Glorot-Bengio recommended length scales.  These
            weights are intended to be initializers. 
        '''
        wa = 1.0*(np.arange(0.0,28.0*25.0)/(28.0*25.0)-0.5) #/(28.0+25.0)**0.5
        wb = 1.0*(np.arange(0.0,25.0*25.0)/(25.0*25.0)-0.5) #/(25.0+25.0)**0.5
        wc = 1.0*(np.arange(0.0,25.0* 1.0)/(25.0* 1.0)-0.5) #/(25.0+ 1.0)**0.5
        return (wa,wb,wc)

    def initialize_weights(self, wa, wb, wc):
        ''' Initialize weights as a RUNTIME OPERATION, not by creating new graph nodes. '''
        self.session.run(self.Initializer, feed_dict={
            self.InitWeightsA:wa,
            self.InitWeightsB:wb,
            self.InitWeightsC:wc,
        })
       
    def run(self, dataset): 
        ''' Compute gradient statistics on the given training set. '''
        ins_inputs, ins_outputs = dataset.get_all('ins') 
        sentiment, intensity, uncertainty, passion, audacity, peril, serendipity = self.session.run(
            [self.Sentiment, self.Intensity, self.Uncertainty, self.Passion, self.Audacity, self.Peril, self.Serendipity],
            feed_dict = {
                self.Images:ins_inputs,
                self.Labels:ins_outputs
            }
        )
        return sentiment, intensity, uncertainty, passion, audacity, peril, serendipity



################################################################################
#            3. EXPERIMENT LOOP                                                #
################################################################################

def run_experiment(nb_trials, ins_size):
    ''' Compute the 'eta curve', that is, sgd vs gd test performance as a function of learning rate.  Record summary 
        statistics by appending to the given filename.
    '''
    dataset = Dataset()
    learner = Learner(ins_size)

    sentiment, intensity, uncertainty, passion, audacity, peril, serendipity = [], [], [], [], [], [], []
    statlists = [sentiment, intensity, uncertainty, passion, audacity, peril, serendipity]
    for h in tqdm(range(nb_trials)):
        dataset.resample(ins_size)
        init_weights = learner.sample_init_weights() 
        learner.initialize_weights(*init_weights)
        for s, slist in zip(learner.run(dataset), statlists): 
            slist.append(s)
    print('MEAN: sent %.8f, intense %.8f, uncert %.8f, pass %.8f, aud %.8f, peril %.8f, serendipity %.8f' % tuple(np.mean(sl) for sl in statlists))
    print('SDEV: sent %.8f, intense %.8f, uncert %.8f, pass %.8f, aud %.8f, peril %.8f, serendipity %.8f' % tuple(np.std(sl) for sl in statlists))

    with open(FILE_NM, 'w') as f:
        f.write('%d TRIALS, %d SAMPLES\n' % (nb_trials, ins_size))
        f.write('MEAN: sent %.8f, intense %.8f, uncert %.8f, pass %.8f, aud %.8f, peril %.8f, serendipity %.8f\n' % tuple(np.mean(sl) for sl in statlists))
        f.write('SDEV: sent %.8f, intense %.8f, uncert %.8f, pass %.8f, aud %.8f, peril %.8f, serendipity %.8f\n' % tuple(np.std(sl) for sl in statlists))



if __name__=='__main__':
    run_experiment(nb_trials=NB_TRIALS, ins_size=INS_SIZE)
