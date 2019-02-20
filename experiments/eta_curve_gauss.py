''' author: samtenka
    change: 2019-01-11
    create: 2017-10-07
    descrp: Compare SGD and GD out-of-sample performances (for a shallow dense network on small MNIST training sets)
            over a log-spaced range of learning rates, then APPEND summary statistics to `results.txt`.  To run, type:
                python eta_curve_gauss.py 1000 10 0.000 0.001 10 32 results_gauss.txt 
            The    1000 represents the number of trials to perform per experimental condition;
            the      10 represents the training set size;
            the   0.000 represents the starting learning rate to sweep from;
            the   0.001 represents the ending learning rate to sweep to;
            the      10 represents (one less than) the number of learning rates to sweep through; and
            the      32 represents the desired floating point precision (32 or 64).

            
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
        ''' Returns whole in-sample or out-of-sample points.  Good for evaluating train and test scores. ''' 
        if split == 'out':
            out_inputs = np.random.randn(max_size, 1)
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
    def __init__(self, precision=PRECISION):
        self.create_model(precision)
        self.create_trainer(precision)
        self.session = tf.Session()
        self.initialize_weights(*self.sample_init_weights())

    def create_model(self, precision=tf.float32):
        ''' Construct a densely connected 28*28 --> 64 --> 10 neural network (with slrelu activations). '''
        self.Data = tf.placeholder(precision, shape=[None, 1])

        self.Weights = tf.get_variable('flattened', shape=[1+1], dtype=precision)

        self.WeightsA = self.Weights[0]
        self.WeightsB = self.Weights[1]

        self.InitWeightsA = tf.placeholder(precision, shape=[1])
        self.InitWeightsB = tf.placeholder(precision, shape=[1])
        self.Inits = tf.concat([tf.reshape(init, [-1]) for init in [self.InitWeightsA, self.InitWeightsB]], axis=0)
        self.Initializer = tf.assign(self.Weights, self.Inits)


    def create_trainer(self, precision=tf.float32):
        ''' Define the loss and corresponding gradient-based update.  The difference between gd and sgd is not codified
            here; instead, the difference lies in the size and correlations between batches we use to train the
            classifier, i.e. in the values assigned to `Data` and `TrueOutputs` at each gradient update step.
        '''
        self.Losses = self.WeightsA - tf.square(self.WeightsA) + tf.square(self.WeightsB - tf.multiply(self.WeightsA, self.Data))
        self.Loss = tf.reduce_mean(self.Losses)

        self.LearningRate = tf.placeholder(dtype=precision)

        self.GradientWeights = tf.convert_to_tensor(tf.gradients(self.Loss, self.Weights))[0]

        self.Update = tf.tuple([
            tf.assign(self.Weights, self.Weights - self.LearningRate * self.GradientWeights), 
        ])
        
        self.Accuracy = 0.0*tf.reduce_mean(self.Weights)


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

    def run(self, dataset, ins_time, batch_size, learning_rate, opt='sgd'): 
        ''' Compute post-training metrics for given dataset and hyperparameters.  Return in-sample loss, out-of-sample
            loss, in-sample accuracy, and out-of-sample accuracy --- in that order.
        '''
        for t in range(ins_time):
            batch_inputs = dataset.get_batch(batch_size=batch_size, split='ins', opt=opt) 
            self.session.run([self.Update], feed_dict={
                self.Data:batch_inputs,
                self.LearningRate:learning_rate
            }) 
        
        ins_inputs = dataset.get_all('ins') 
        out_inputs = dataset.get_all('out') 
        ins_acc, ins_los = self.session.run([self.Accuracy, self.Loss], feed_dict={
            self.Data:ins_inputs,
        })
        out_acc, out_los = self.session.run([self.Accuracy, self.Loss], feed_dict={
            self.Data:out_inputs,
        })
        return ins_los, out_los, ins_acc, out_acc





###############################################################################
#                            2. STATISTICS LOGGER                             #
###############################################################################

class Logger(object):
    ''' Collects and aggregates the metric scores of each trial.  The aggregation occurs both  Maintains a dictionary (of score-lists) indexed by
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
        ''' From GD and SGD logs, create DIFF log. 
        '''
        for key in list(self.scores.keys()): 
            if key[5] != 'gd': continue
            key_ = key[:5] + ('sgd',) + key[6:]
            key__= key[:5] + ('diff',) + key[6:]
            if key_ not in self.scores: continue
            self.scores[key__] = [gd_val-sgd_val for gd_val, sgd_val in zip(self.scores[key], self.scores[key_])] 

    def get_stats(self, key):
        ''' Compute mean, sample variance, min, and max '''
        s = np.array(self.scores[key])
        mean = np.mean(s)
        var = np.mean(np.square(s - mean)) * (1.0 + 1.0/(len(s)-1))
        return mean, np.sqrt(var), np.amin(s), np.amax(s)

    def write_summary(self, file_nm, key_renderer):
        ''' Compute statistics and append to file. '''
        self.gen_diffs()
        with open(file_nm, 'a') as f: 
            for key in self.scores:  
                stats = self.get_stats(key)
                f.write('%s:\t%.9f\t%.9f\t%.9f\t%.9f' % ((key_renderer(*key),)+stats)) 
                f.write('\n')



###############################################################################
#                            3. EXPERIMENT LOOP                               #
###############################################################################

def run_experiment(nb_trials, ins_size, ins_time, batch_size, learning_rates):
    ''' Compute the 'eta curve', that is, sgd vs gd test performance as a function of learning rate.  Record summary 
        statistics by appending to the given filename.
    '''
    mnist = Dataset()
    mlp = Classifier()
    log = Logger() 

    try:
        for learning_rate in learning_rates:
            print('LR =', learning_rate)
            for h in tqdm(range(nb_trials)):
                mnist.resample_ins(ins_size)
                init_weights = mlp.sample_init_weights() 
                for opt in ('gd', 'sgd'): 
                    mlp.initialize_weights(*init_weights)
                    il, ol, ia, oa = mlp.run(mnist, ins_time, batch_size, learning_rate, opt) 
                    log.append((nb_trials, ins_size, ins_time, batch_size, learning_rate, opt, 'IL'), il)
                    log.append((nb_trials, ins_size, ins_time, batch_size, learning_rate, opt, 'OL'), ol)
                    log.append((nb_trials, ins_size, ins_time, batch_size, learning_rate, opt, 'IA'), ia)
                    log.append((nb_trials, ins_size, ins_time, batch_size, learning_rate, opt, 'OA'), oa)
    except KeyboardInterrupt:
        pass

    log.write_summary(FILE_NM,
        key_renderer=lambda nb_trials, ins_size, ins_time, batch_size, learning_rate, opt, metric:
            'NB_TRIALS=%d\tINS_SIZE=%d\tINS_TIME=%d\tBATCH_SIZE=%d\tLEARNING_RATE=%f\tOPT=%s\tMETRIC=%s' %
            (nb_trials, ins_size, ins_time, batch_size, learning_rate, opt, metric) 
    )

if __name__=='__main__':
    run_experiment(nb_trials=NB_TRIALS, ins_size=INS_SIZE, ins_time=INS_SIZE, batch_size=1,
        #learning_rates=[MIN_LR * (MAX_LR/MIN_LR)**(float(i)/LR_SWEEP) for i in range(0, LR_SWEEP+1)] #geometric
        learning_rates=[MIN_LR + (MAX_LR-MIN_LR)*(float(i)/LR_SWEEP) for i in range(0, LR_SWEEP+1)] # arithmetic
    )
