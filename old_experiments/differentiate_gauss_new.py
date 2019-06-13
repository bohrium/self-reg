''' author: samtenka
    change: 2019-02-20
    create: 2017-10-07
    descrp: Estimate and save gradient statistics (on a toy learning task described in `eta_curve_gauss.py`).
            To run, type:
                python differentiate_gauss.py 10000 24 32 gradstats_gauss.txt
            The                 10000   gives   a number of trials to perform;
            the                    24   gives   a training set size;
            the                    32   gives   a desired floating point precision (32 or 64);
            the   gradstats_gauss.txt   gives   a filename to overwrite with results.
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

class Dataset(object): 
    ''' This class provides access to the aforementioned toy dataset via in-sample and out-of-sample batches.  It
        allows us to sample a finite training set and from there and to sample with or without replacement.  Since our
        learning task involves no labels, `get_batch` and `get_all` each return one array of shape (???, 1).
    '''
    def __init__(self, max_size=1000):
        self.resample_ins(max_size)

    def resample_ins(self, ins_size):
        ''' Resample the finite training set. '''
        self.ins_inputs = np.random.randn(ins_size, 2)
        self.ins_size = ins_size
        self.index = 0

    def get_batch(self, batch_size, split='ins', opt='sgd', with_replacement=False):
        ''' Return batch, by default from training set.  In case the specified optimizer is 'sgd', then `batch_size`
            and `with_replacment` become salient parameters; if the optimizer is 'gd', then the returned batch is the
            same (and of size equal to the number of training points) each time. 
        '''
        if split == 'out': 
            out_inputs = np.random.randn(max_size, 2)
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
        ''' Return whole in-sample or out-of-sample points.  Good for evaluating train and test scores. ''' 
        if split == 'out':
            out_inputs = np.random.randn(max_size, 2)
            return out_inputs
        return self.ins_inputs




################################################################################
#            2. DEFINE LEARNING MODEL                                          #
################################################################################

class Learner(object):
    ''' Creates, (re)initializes, trains, and evaluates a shallow neural network. '''
    def __init__(self, batch_size, precision=PRECISION):
        self.create_model(batch_size, precision)
        self.create_trainer(batch_size, precision)
        self.session = tf.Session()
        self.initialize_weights(*self.sample_init_weights())

    def create_model(self, batch_size, precision=tf.float32):
        ''' Define the loss landscape as a function of weights and data. '''
        self.Data = tf.placeholder(precision, shape=[batch_size, 2])

        self.Weights = tf.get_variable('flattened', shape=[1+1, batch_size], dtype=precision)

        # Crucially, we repeat weights along batch axis to jibe with tf.gradient:
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
        self.Losses = (
                tf.square(self.WeightsA + self.Data[:, 0:1] * (self.WeightsB-1.0) - self.Data[:, 1:2])
            )
        
        self.stat_tensors = gradient_stats(self.Losses, self.Weights, batch_size)

    def sample_init_weights(self): 
        ''' Sample weights (as numpy arrays) distributed according to Glorot-Bengio recommended length scales.  These
            weights are intended to be initializers. 
        '''
        wa = 0.0 + 0.0 * np.random.randn(1)
        ba = 0.0 + 0.0 * np.random.randn(1)
        return (wa, ba)

    def initialize_weights(self, wa, ba):
        ''' Initialize weights as a RUNTIME OPERATION, not by creating new graph nodes. '''
        self.session.run(self.Initializer, feed_dict={
            self.InitWeightsA:wa,
            self.InitWeightsB:ba,
        })

    def run(self, dataset): 
        ''' Compute gradient statistics on the given training set. '''
        ins_inputs = dataset.get_all('ins') 
        stats = self.session.run(self.stat_tensors, feed_dict={ self.Data:ins_inputs })
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
