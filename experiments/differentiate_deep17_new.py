''' author: samtenka
    change: 2019-02-20
    create: 2017-10-07
    descrp: Estimate and save gradient statistics (on a toy learning task described in `eta_curve_deep17.py`).
            To run, type:
                python differentiate_deep17.py 10000 24 32 gradstats_deep17.txt
            The                 10000   gives   a number of trials to perform;
            the                    24   gives   a training set size;
            the                    32   gives   a desired floating point precision (32 or 64);
            the   gradstats_deep17.txt   gives   a filename to overwrite with results.
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
NB_STATS = 19


################################################################################
#            0. DEFINE STATISTICS                                              #
################################################################################

    #--------------------------------------------------------------------------#
    #                0.0 define tf helper functions                            #
    #--------------------------------------------------------------------------#

def tfdot(s, t):
    return tf.reduce_sum(tf.multiply(s, t), axis=1)

def tfgrad(y, x, batch_size, block, stop_gradients=[]):
    start =  block    * (batch_size//4)
    end   = (block+1) * (batch_size//4)
    return tf.transpose(tf.convert_to_tensor(tf.gradients(y, x, stop_gradients=stop_gradients)))[start:end, :, 0]

def gradient_stats(Losses, Weights, BATCH_SIZE): 
    ''' Given loss batch and weight tensors, return unbiased estimates of
        sentiment, intensity, uncertainty, passion, audacity, and peril.
        Assumes that Weights has shape (nb_weights, batch_size), i.e. consists of weight replicas.
        Assumes batch size is divisible by 4.
    '''

    #--------------------------------------------------------------------------#
    #                0.1 compute starting tensors                              #
    #--------------------------------------------------------------------------#
    
    # Use of `Linker` forces tf.gradient to return an actual 0 instead of `None` when there is no dependency:
    Linker = 0.0*tf.reduce_sum(Weights)
    LinkedLosses = Losses + Linker
    
    assert(BATCH_SIZE % 4 == 0)
    
    Gradients = tf.transpose(tf.convert_to_tensor(tf.gradients(LinkedLosses, Weights)))[:, :, 0]
    Gradients0  = Gradients[:BATCH_SIZE//2, :] 
    Gradients1  = Gradients[BATCH_SIZE//2:, :] 
    Gradients00 = Gradients0[:BATCH_SIZE//4, :] 
    Gradients01 = Gradients0[BATCH_SIZE//4:, :] 
    Gradients10 = Gradients1[:BATCH_SIZE//4, :] 
    Gradients11 = Gradients1[BATCH_SIZE//4:, :] 

    #--------------------------------------------------------------------------#
    #                0.2 combine tensors up to 3rd order                       #
    #--------------------------------------------------------------------------#

    A_As    = tfdot(Gradients00, Gradients01)
    AAs     = tfdot(Gradients00, Gradients00)

    A_Abs   =       tfgrad(A_As, Weights, BATCH_SIZE, 1)
    AAbs    = 0.5 * tfgrad(AAs , Weights, BATCH_SIZE, 0)

    A_Ab_Bs = tfdot(A_Abs, Gradients10)
    AAb_Bs  = tfdot(AAbs,  Gradients10)
    AB_Abs  = tfdot(A_Abs, Gradients00)
    AAbBs   = tfdot(AAbs,  Gradients00)

    A_Ab_Bcs=       tfgrad(A_Ab_Bs, Weights, BATCH_SIZE, 2)
    A_Abc_Bs=       tfgrad(A_Ab_Bs, Weights, BATCH_SIZE, 1)
    AAb_Bcs =       tfgrad(AAb_Bs , Weights, BATCH_SIZE, 1)
    ABc_Abs = 0.5 * tfgrad(AB_Abs , Weights, BATCH_SIZE, 0)
    AB_Abcs =       tfgrad(AB_Abs , Weights, BATCH_SIZE, 1)

    A_Ab_Bc_Cs  = tfdot(A_Ab_Bcs, Gradients11)
    AAb_Bc_Cs   = tfdot(AAb_Bcs , Gradients10)
    ABc_Ab_Cs   = tfdot(ABc_Abs , Gradients10)
    AC_Ab_Bcs   = tfdot(A_Ab_Bcs, Gradients00)

    A_AbBcs= tfgrad(tfdot(A_Abs, Gradients01), Weights, BATCH_SIZE, 1, stop_gradients=[A_Abs])
    AAbBcs = tfgrad(tfdot(AAbs , Gradients00), Weights, BATCH_SIZE, 0, stop_gradients=[A_Abs])

    A_AbBc_Cs   = tfdot(A_AbBcs , Gradients10)
    AAbBc_Cs    = tfdot(AAbBcs  , Gradients01)  
    AAbC_Bcs    = tfdot(AAb_Bcs , Gradients00)

    A_Abc_B_Cs  = tfdot(A_Abc_Bs, Gradients11)
    AB_Abc_Cs   = tfdot(AB_Abcs , Gradients10)
    A_AbcC_Bs   = tfdot(A_Abc_Bs, Gradients01)
    ABC_Abcs    = tfdot(AB_Abcs , Gradients00)
    A_AbcBCs    = tfdot(A_AbBcs , Gradients01)

    #--------------------------------------------------------------------------#
    #                0.3 return unbiased estimates                             #
    #--------------------------------------------------------------------------#

    return [tf.reduce_mean(x) for x in (
        Losses, 
        A_As, AAs,
        A_Ab_Bs, AAb_Bs, AB_Abs, AAbBs,
        A_Ab_Bc_Cs, AAb_Bc_Cs, ABc_Ab_Cs, AC_Ab_Bcs, A_AbBc_Cs, AAbBc_Cs, AAbC_Bcs,
        A_Abc_B_Cs, AB_Abc_Cs, A_AbcC_Bs, ABC_Abcs, A_AbcBCs
    )]

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

    def resample_ins(self, sample_size):
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
        self.stat_tensors = gradient_stats(self.Losses, self.Weights, batch_size) 

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
        stats = self.session.run(self.stat_tensors, feed_dict={ self.Images:ins_inputs, self.Labels:ins_outputs })
        return stats






################################################################################
#            3. EXPERIMENT LOOP                                                #
################################################################################

def run_experiment(nb_trials, ins_size):
    ''' Compute the 'eta curve', that is, sgd vs gd test performance as a function of learning rate.  Record summary 
        statistics by appending to the given filename.
    '''
    dataset = Dataset()
    learner = Learner(ins_size)

    stats_array = np.zeros((NB_STATS, nb_trials), dtype=np.float32) 
    for h in tqdm(range(nb_trials)):
        dataset.resample_ins(ins_size)
        init_weights = learner.sample_init_weights() 
        learner.initialize_weights(*init_weights)

        stats_array[:,h] = learner.run(dataset)

    string_format = '\n'.join('%s %%16.8f\t %%16.8f' % name.ljust(10) for name in (
        'Loss, A_A, AA, A_Ab_B, AAb_B, AB_Ab, AAbB, A_Ab_Bc_C, AAb_Bc_C, ABc_Ab_C, AC_Ab_Bc, A_AbBc_C, AAbBc_C, AAbC_Bc, A_Abc_B_C, AB_Abc_C, A_AbcC_B, ABC_Abc, A_AbcBC'.split(', ')
    ))

    report = (string_format % tuple(proc(sl) for sl in stats_array for proc in (np.mean, np.std)))

    print(report)
    with open(FILE_NM, 'w') as f:
        f.write('%d TRIALS, %d SAMPLES PER TRIAL\n' % (nb_trials, ins_size))
        f.write(report)

if __name__=='__main__':
    run_experiment(nb_trials=NB_TRIALS, ins_size=INS_SIZE)
