''' author: samtenka
    change: 2019-02-20
    create: 2017-10-07
    descrp: Append summary of SGD and GD losses (on a toy learning task over a range of learning rates) to a file.
            Here, `SGD` means `batch size 1 without replacement`.  Our task in particular is to decrease this loss:

                loss(data, (weightA, weightB)) = A + (B - A*data)**4 - 3*A**4 

            where data is distributed according to a univariate standard normal.  Observe that for any fixed A, setting  
            B=0 minimizes expected loss. In fact, this minimal expected loss decreases linearly as a function of A. 
            However, as A travels away from 0, the dependence of loss on the specific data sample grows.  Intuitively,
            for 'good' A, the corresponding optimal B is hard to estimate.  Herein lies danger!

            To run, type:
                python simulate_death.py 1000 10 0.00 0.005 12 32 experdata_death.txt 
            The                  1000   gives   a number of trials to perform per experimental condition;
            the                    10   gives   a training set size and number of gradient updates;
            the                  0.00   gives   a starting learning rate to sweep from;
            the                  0.05   gives   a ending learning rate to sweep to;
            the                    12   gives   (one less than) the number of learning rates to sweep through;
            the                    32   gives   a desired floating point precision (32 or 64);
            the   experdata_death.txt   gives   a filename of a log to which to append.
'''

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys 

assert(len(sys.argv)==8)
NB_TRIALS = int(sys.argv[1]) 
INS_SIZE  = int(sys.argv[2]) 
MIN_LR    = float(sys.argv[3]) 
MAX_LR    = float(sys.argv[4]) 
LR_SWEEP  = int(sys.argv[5]) 
PRECISION = {32:tf.float32, 64:tf.float64}[int(sys.argv[6])]
FILE_NM   = sys.argv[7]



################################################################################
#           0. DATASET BATCHES                                                 #
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
        self.ins_inputs = np.random.randn(ins_size, 1)
        self.ins_size = ins_size
        self.index = 0

    def get_batch(self, batch_size, split='ins', opt='sgd', with_replacement=False):
        ''' Return batch, by default from training set.  In case the specified optimizer is 'sgd', then `batch_size`
            and `with_replacment` become salient parameters; if the optimizer is 'gd', then the returned batch is the
            same (and of size equal to the number of training points) each time. 
        '''
        if split == 'out': 
            out_inputs = np.random.randn(max_size, 1)
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
            out_inputs = np.random.randn(max_size, 1)
            return out_inputs
        return self.ins_inputs



################################################################################
#           1. LEARNING MODEL                                                  #
################################################################################

class Learner(object):
    ''' Creates, (re)initializes, trains, and evaluates a differentiable learner. '''
    def __init__(self, precision=PRECISION):
        self.create_model(precision)
        self.create_trainer(precision)
        self.session = tf.Session()
        self.initialize_weights(*self.sample_init_weights())

    def create_model(self, precision=tf.float32):
        ''' Define the loss landscape as a function of weights and data. '''
        self.Data = tf.placeholder(precision, shape=[None, 1])

        self.Weights = tf.get_variable('flattened', shape=[1+1], dtype=precision)

        self.WeightsA = self.Weights[0]
        self.WeightsB = self.Weights[1]

        self.InitWeightsA = tf.placeholder(precision, shape=[1])
        self.InitWeightsB = tf.placeholder(precision, shape=[1])
        self.Inits = tf.concat([tf.reshape(init, [-1]) for init in [self.InitWeightsA, self.InitWeightsB]], axis=0)
        self.Initializer = tf.assign(self.Weights, self.Inits)

        #self.Losses = self.WeightsA - 3*tf.square(tf.square(self.WeightsA)) + tf.square(tf.square(self.WeightsB - tf.multiply(self.WeightsA, self.Data)))
        self.Losses = (
                  tf.square(self.WeightsA - 1.0)
                + tf.square(self.WeightsB - 1.0)
                + self.Data * self.WeightsB * tf.exp(self.WeightsA)
            )


    def create_trainer(self, precision=tf.float32):
        ''' Define the loss and corresponding gradient-based update.  The difference between gd and sgd is not codified
            here; instead, the difference lies in the size and correlations between batches we use to train the
            classifier, i.e. in the values assigned to `Data` and `TrueOutputs` at each gradient update step.
        '''
        self.LearningRate = tf.placeholder(dtype=precision)

        self.Loss = tf.reduce_mean(self.Losses)
        self.GradientWeights = tf.convert_to_tensor(tf.gradients(self.Loss, self.Weights))[0]
        self.Update = tf.tuple([
            tf.assign(self.Weights, self.Weights - self.LearningRate * self.GradientWeights), 
        ])

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

    def run(self, dataset, ins_time, batch_size, learning_rate, opt='sgd'): 
        ''' Compute post-training metrics for given dataset and hyperparameters.  Return in-sample loss and
            out-of-sample loss --- in that order.  This learning task has no auxilliary metrics such as accuracy. 
        '''
        for t in range(ins_time):
            batch_inputs = dataset.get_batch(batch_size=batch_size, split='ins', opt=opt) 
            self.session.run([self.Update], feed_dict={
                self.Data:batch_inputs,
                self.LearningRate:learning_rate
            }) 
        
        ins_inputs = dataset.get_all('ins') 
        out_inputs = dataset.get_all('out') 
        ins_los = self.session.run(self.Loss, feed_dict={self.Data:ins_inputs})
        out_los = self.session.run(self.Loss, feed_dict={self.Data:out_inputs})
        return ins_los, out_los



################################################################################
#           2. STATISTICS LOGGER                                               #
################################################################################

class Logger(object):
    ''' Collects and aggregates the metric scores of each trial.  Maintains a dictionary (of score-lists) indexed by
        keys that are intended to represent the experimental conditions yielding those scores.
    '''
    def __init__(self): 
        self.scores = {}

    def append(self, key, value):
        ''' Log (key, value) pair.  Intended typical use: key is tuple describing experimental condition, for instance
            recording learning rate, optimizer type, and so forth; value is individual score TO BE APPENDED TO A LIST.  
        '''
        if key not in self.scores:
            self.scores[key] = [] 
        self.scores[key].append(value)

    def gen_diffs(self): 
        ''' From GD and SGD logs, create DIFF log. '''
        for key in list(self.scores.keys()): 
            if key[5] != 'gd': continue
            key_ = key[:5] + ('sgd',) + key[6:]
            key__= key[:5] + ('diff',) + key[6:]
            if key_ not in self.scores: continue
            self.scores[key__] = [gd_val-sgd_val for gd_val, sgd_val in zip(self.scores[key], self.scores[key_])] 

    def get_stats(self, key):
        ''' Compute mean, sample deviation, min, and max '''
        s = np.array(self.scores[key])
        mean = np.mean(s)
        var = np.mean(np.square(s - mean)) * len(s)/(len(s)-1)
        return mean, np.sqrt(var), np.amin(s), np.amax(s)

    def write_summary(self, file_nm, key_renderer):
        ''' Compute statistics and append to file. '''
        self.gen_diffs()
        with open(file_nm, 'a') as f: 
            for key in self.scores:  
                stats = self.get_stats(key)
                f.write('%s:\t%.9f\t%.9f\t%.9f\t%.9f' % ((key_renderer(*key),)+stats)) 
                f.write('\n')



################################################################################
#           3. EXPERIMENT LOOP                                                 #
################################################################################

def run_experiment(nb_trials, ins_size, ins_time, batch_size, learning_rates):
    ''' Compute the 'eta curve', that is, sgd vs gd test performance as a function of learning rate.  Record summary 
        statistics by appending to the given filename.
    '''
    dataset = Dataset()
    learner = Learner()
    logger = Logger() 

    try:
        for learning_rate in learning_rates:
            print('LR =', learning_rate)
            for h in tqdm(range(nb_trials)):
                dataset.resample_ins(ins_size)
                init_weights = learner.sample_init_weights() 
                for opt in ('gd', 'sgd'): 
                    learner.initialize_weights(*init_weights)
                    il, ol = learner.run(dataset, ins_time, batch_size, learning_rate, opt) 
                    logger.append((nb_trials, ins_size, ins_time, batch_size, learning_rate, opt, 'IL'), il)
                    logger.append((nb_trials, ins_size, ins_time, batch_size, learning_rate, opt, 'OL'), ol)
    except KeyboardInterrupt:
        pass

    logger.write_summary(FILE_NM,
        key_renderer=lambda nb_trials, ins_size, ins_time, batch_size, learning_rate, opt, metric:
            'NB_TRIALS=%d\tINS_SIZE=%d\tINS_TIME=%d\tBATCH_SIZE=%d\tLEARNING_RATE=%f\tOPT=%s\tMETRIC=%s' %
            (nb_trials, ins_size, ins_time, batch_size, learning_rate, opt, metric) 
    )

if __name__=='__main__':
    arith_prog = [MIN_LR + (MAX_LR-MIN_LR)*(float(i)/LR_SWEEP) for i in range(0, LR_SWEEP+1)]
    run_experiment(nb_trials=NB_TRIALS, ins_size=INS_SIZE, ins_time=INS_SIZE, batch_size=1, learning_rates=arith_prog)
