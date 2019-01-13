''' author: samtenka
    change: 2019-01-12
    create: 2019-01-11
    descrp: Illustrate how to use Tensorflow's automatic differentiation to compute bespoke gradient statistics.
            Given a loss function l(data, weights), where `data` is drawn from a fixed distribution and we imagine
            perturbing `weights` around a fixed initialization, we give unbiased estimates for these scalars: 
                A. mean loss                                    {()}
                B. square-norm of mean-gradient                 {(a)}{(a)}
                C. trace of covariance of gradients             {(a)(a)} - {(a)}{(a)}
                D. 1st order increase of (B) along gradient     2{(a)}{(ab)(b)}
                E. 1st order increase of (C) along gradient     2{(a)}{(ab)}{(b)} - 2{(a)}{(ab)(b)}
            Above, a sequence of k letters in parentheses indicates the rank-k tensor obtained by differentiating the
            loss k times with respect to weights.  The letters are to be read as tensor indices and contracted as usual
            (so, though not used here, a parenthesized expression could actually be a rank-(k minus 2) tensor etc).
            Curly braces indicate an expectation over the data distribution.  As another example, {(aa)} is the trace
            of the expected hessian.  We call (D) the `ADVENTUROUSNESS` and (E) the `PROGRESSIVENESS`.

            For this demonstration, we write data=(noise_a, noise_b) and weights=(weights_a, weights_b), and we set
                l(data, weights) = (alpha + noise_a) * weights_a + (beta + noise_b) * weights_b**2 
            We set the coefficients (alpha, beta) and the weight initialization (A, B) in the hyperparameter section
            below.  The data is distributed as normal spherical Gaussian.  In this case, an easily calculation shows: 
                A. mean loss                                    {()}                                = alpha A + beta B^2
                B. square-norm of mean-gradient                 {(a)}{(a)}                          = alpha^2 + 4 beta^2 B^2
                C. trace of covariance of gradients             {(a)(a)} - {(a)}{(a)}               = 1 + 4 B^2
                D. 1st order increase of (B) along gradient     2{(a)}{(ab)(b)}                     = 16 beta^3 b^2 
                E. 1st order increase of (C) along gradient     2{(a)}{(ab)}{(b)} - 2{(a)}{(ab)(b)} = 16 beta b^2
'''

import tensorflow as tf
import numpy as np



###############################################################################
#                            0. HYPERPARAMETERS                               #
###############################################################################

BATCH_SIZE = 65536
ALPHA = 3.0
BETA  = 2.0
A = 1.0
B = 1.0



###############################################################################
#                            0. READ DATASET                                  #
###############################################################################

NoiseA = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
NoiseB = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
WeightA = tf.placeholder(tf.float32, shape=[BATCH_SIZE]) # TODO: explain COPYING
WeightB = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
Weights = [WeightA, WeightB]

Outputs = (ALPHA + NoiseA) * WeightA + (BETA  + NoiseB) * tf.square(WeightB) + 0.0*tf.reduce_sum(tf.square(Weights)) # TODO: explain Trick

AvgOut = tf.reduce_mean(Outputs)
Gradients = tf.convert_to_tensor(tf.gradients(Outputs, Weights))
AvgGrad = tf.reduce_mean(Gradients, axis=1)
SqrNorms = tf.reduce_sum(tf.square(Gradients), axis=0)

AvgSqrNorm = tf.reduce_mean(SqrNorms, axis=0)
SqrNormAvg = tf.reduce_sum(tf.square(AvgGrad), axis=0)
TraceCovar = (AvgSqrNorm - SqrNormAvg) * BATCH_SIZE / (BATCH_SIZE-1)
MeanSqrGrad = AvgSqrNorm - TraceCovar

GradSqrNormAvg = tf.reduce_sum(tf.convert_to_tensor(tf.gradients(SqrNormAvg, Weights)), axis=1)
GradAvgSqrNorm = tf.reduce_sum(tf.convert_to_tensor(tf.gradients(AvgSqrNorm, Weights)), axis=1)
DeltaSqrNormAvg = tf.reduce_sum(tf.math.multiply(AvgGrad, GradSqrNormAvg))
DeltaAvgSqrNorm = tf.reduce_sum(tf.math.multiply(AvgGrad, GradAvgSqrNorm)) # TODO: oops, this is biased! 

Adventurousness = (DeltaAvgSqrNorm - DeltaSqrNormAvg) * BATCH_SIZE / (BATCH_SIZE-1)
Progressiveness = DeltaAvgSqrNorm - Adventurousness



###############################################################################
#                            0. READ DATASET                                  #
###############################################################################

def get_batch(batch_size=BATCH_SIZE):
    noise_a = np.random.randn(batch_size) 
    noise_b = np.random.randn(batch_size) 
    return {NoiseA:noise_a, NoiseB:noise_b, WeightA:A*np.ones(batch_size), WeightB:B*np.ones(batch_size)}

with tf.Session() as session:
    batch = get_batch()
    out =   session.run(AvgOut,         feed_dict=batch)
    gg  =   session.run(MeanSqrGrad,    feed_dict=batch)
    cc  =   session.run(TraceCovar,     feed_dict=batch)

    prog=   session.run(Progressiveness,feed_dict=batch)
    advn=   session.run(Adventurousness,feed_dict=batch)
    print('%.2f --- expected %.2f' % ( out, ALPHA*A + BETA *B**2))
    print('%.2f --- expected %.2f' % (  gg, ALPHA**2 + 4.0*BETA **2*B**2))
    print('%.2f --- expected %.2f' % (  cc, 1.0 + 4.0*B**2))
    print('%.2f --- expected %.2f' % (prog, 16.0*BETA **3*B**2))
    print('%.2f --- expected %.2f' % (advn, 16.0*BETA *B**2))
