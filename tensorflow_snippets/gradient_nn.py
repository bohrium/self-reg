''' author: samtenka
    change: 2019-02-05
    create: 2019-01-11
    descrp: Illustrate how to use Tensorflow's automatic differentiation to compute bespoke gradient statistics.
            Given a loss function l(data, weights), where `data` has a fixed distribution and we imagine perturbing
            `weights` around a fixed initialization, we give UNBIASED estimates for these scalars in O(BATCHSIZE) time: 
                A. mean loss                                    {()}                                    SENTIMENT
                B. trace of square gradient                     {(a)}{(a)}                              INTENSITY
                C. trace of covariance of gradients             {(a)(a)} - {(a)}{(a)}                   UNCERTAINTY
                D. 1st order increase of (B) along gradient     2{(a)}{(ab)}{(b)}                       PASSION
                E. 1st order increase of (C) along gradient     2{(a)}{(ab)(b)} - 2{(a)}{(ab)}{(b)}     AUDACITY
                F. trace of hessian times square gradient       {(ab)}{(a)}{(b)}                        PASSION/2
                G. trace of hessian times covariance            {(ab)}{(a)(b)} - {(ab)}{(a)}{(b)}       PERIL 
            Above, a sequence of k letters in parentheses indicates the rank-k tensor obtained by differentiating the
            loss k times with respect to weights.  The letters are to be read as tensor indices and contracted as usual
            (so, though not used here, a parenthesized expression could actually be a rank-(k minus 2) tensor etc).
            Curly braces indicate an expectation over the data distribution.  As another example, {(aa)} is the trace
            of the hessian.  The rightmost column lists intuition-pumping names; for instance, we call (D)
            `PASSION` and (E) `AUDACITY`.  Note finally that (D) and (F) are proportional. 
'''

import tensorflow as tf
import numpy as np



################################################################################
#            0. HYPERPARAMETERS                                                #
################################################################################

BATCH_SIZE = 256 
ALPHA = 3.0
BETA  = 2.0
A = 1.0
B = 1.0



################################################################################
#            1. DEFINE STATISTICS                                              #
################################################################################

    #--------------------------------------------------------------------------#
    #                1.0 define unbiased estimator helper function             #
    #--------------------------------------------------------------------------#

BESSEL_FACTOR = BATCH_SIZE / (BATCH_SIZE-1)
def second_order_stats(x, y): 
    ''' Return unbiased estimates of
            PRODUCT_MEAN =  trace({x}{y}),
            COVARIANCE   =  trace({xy}-{x}{y}), and
            AVG_PRODUCT  =  trace({xy})
            PRODUCT_AVG  =  trace({x}{y})
        for given tensors x and y of shape (nb samples)-by-(dim)
    '''
    avg_x = tf.reduce_mean(x, axis=0)
    avg_y = tf.reduce_mean(y, axis=0) if x is not y else avg_x

    avg_product = tf.reduce_sum(tf.reduce_mean(tf.multiply(x, y), axis=0))
    product_avg = tf.reduce_sum(tf.multiply(avg_x, avg_y))
    
    covariance = BESSEL_FACTOR * (avg_product - product_avg)
    product_mean = avg_product - covariance

    return product_mean, covariance, avg_product, product_avg

def gradient_stats(Losses, Weights): 
    ''' Given loss batch and weight tensors, return unbiased estimates of
        sentiment, intensity, uncertainty, passion, audacity, and peril.
        Assumes that Weights has shape (nb_weights, batch_size), i.e. consists of weight replicas.
    '''
    #--------------------------------------------------------------------------#
    #                1.1 compute SENTIMENT                                     #
    #--------------------------------------------------------------------------#
    
    # use of `Linker` forces tf's gradient computations to return an actual 0 instead of `None` when there is no dependency
    Linker = 0.0*tf.reduce_sum(tf.square(Weights))
    LinkedLosses = Losses + Linker
    AvgLoss = tf.reduce_mean(LinkedLosses)
    
    #--------------------------------------------------------------------------#
    #                1.2 compute INTENSITY and UNCERTAINTY                     #
    #--------------------------------------------------------------------------#
    
    Gradients = tf.transpose(tf.convert_to_tensor(tf.gradients(LinkedLosses, Weights)))
    MeanSqrGrad, TraceCovar, AvgSqrNorm, SqrAvgNorm = second_order_stats(Gradients, Gradients)
    
    #--------------------------------------------------------------------------#
    #                1.3 compute PASSION and AUDACITY                          #
    #--------------------------------------------------------------------------#
    
    # below, we multiply by BATCH_SIZE to counter tf.gradients' averaging behavior
    GradMeanSqrGrad = tf.transpose(tf.convert_to_tensor(tf.gradients(MeanSqrGrad, Weights))) * BATCH_SIZE
    GradTraceCovar =  tf.transpose(tf.convert_to_tensor(tf.gradients(TraceCovar, Weights))) * BATCH_SIZE
    Passion, _, __, ___ = second_order_stats(Gradients, GradMeanSqrGrad)
    Audacity, _, __, ___ = second_order_stats(Gradients, GradTraceCovar)
    
    #--------------------------------------------------------------------------#
    #                1.4 compute PERIL                                         #
    #--------------------------------------------------------------------------#
    
    # to estimate Peril, we split the batch into two subbatches and invoke multiplicativity of expectation for independents
    Gradients_0 = Gradients[:BATCH_SIZE//2, :] 
    Gradients_1 = Gradients[BATCH_SIZE//2:, :] 
    InterSubbatchGradientDots = tf.reduce_sum(tf.multiply(Gradients_0, Gradients_1), axis=1)
    # HessesTimesGrads is the derivative with respect to subbatch_0 of (gradients_0 * gradients_1):
    HessesTimesGrads = tf.transpose(tf.convert_to_tensor(tf.gradients(InterSubbatchGradientDots, Weights)))[:BATCH_SIZE//2, :]
    UncenteredPeril = tf.reduce_mean(tf.reduce_sum(tf.multiply(Gradients_1, HessesTimesGrads), axis=1))
    Peril = UncenteredPeril - Passion/2 

    return AvgLoss, MeanSqrGrad, TraceCovar, Passion, Audacity, Peril



if __name__ == '__main__':
    ################################################################################
    #            2. DEFINE TOY LOSS LANDSCAPE                                      #
    ################################################################################
    
    Noise = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1, 8])
    # below, we use BATCH_SIZE many copies of Weight to address the egregious summing operation implicit in `tf.gradients`
    Weights = tf.placeholder(tf.float32, shape=[64+8, BATCH_SIZE])
    WeightA = tf.reshape(tf.transpose(Weights[:64, :], perm=[1, 0]), [BATCH_SIZE, 8, 8]) 
    WeightB = tf.reshape(tf.transpose(Weights[64:64+8, :], perm=[1, 0]), [BATCH_SIZE, 8, 1]) 
    Hidden = tf.math.tanh(tf.matmul(Noise, WeightA))  
    Losses = tf.reshape(tf.square(tf.matmul(Hidden, WeightB)), [BATCH_SIZE])
    
    Sentiment, Intensity, Uncertainty, Passion, Audacity, Peril = gradient_stats(Losses, Weights)
    
    
    
    ################################################################################
    #            3. RUN SESSION                                                    #
    ################################################################################
    
    def get_batch(batch_size=BATCH_SIZE):
        ''' return (independent) noise samples in the format of a tensorflow feed_dict '''
        noise = np.random.randn(batch_size, 1, 8)  
        weights = np.ones((64+8, batch_size), np.float32) / 8.0**0.5
        return {Noise:noise, Weights:weights}
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
    
        batch = get_batch()
    
        # Though, for the purpose of testing, we compute the following 7 scalars in independent session-runs, one would in
        #       practice use a single call: `session.run([AvgOut, MeanSqrGrad, ...], ...)`
        sentiment   =   session.run(Sentiment,      feed_dict=batch)
        intensity   =   session.run(Intensity,      feed_dict=batch)
        uncertainty =   session.run(Uncertainty,    feed_dict=batch)
        passion     =   session.run(Passion,    feed_dict=batch)
        audacity    =   session.run(Audacity,   feed_dict=batch)
        peril       =   session.run(Peril,      feed_dict=batch)
    
        print('loss         %.2f --- ' % sentiment      )
        print('intensity    %.2f --- ' % intensity      )
        print('uncertainty  %.2f --- ' % uncertainty    )
        print('passion      %.2f --- ' % passion        ) 
        print('audacity     %.2f --- ' % audacity       ) 
        print('peril        %.2f --- ' % peril          )
