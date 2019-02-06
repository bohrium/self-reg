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
            of the expected hessian.  The rightmost column lists intuition-pumping names; for instance, we call (D)
            `PASSION` and (E) `AUDACITY`.  Note finally that (D) and (F) are proportional. 

            For this demonstration, we write data=(noise_a, noise_b) and weights=(weights_a, weights_b), and we set
                l(data, weights) = (alpha + noise_a) * weights_a + (beta + noise_b) * weights_b**2 
            We set the coefficients (alpha, beta) and the weight initialization (A, B) in the hyperparameter section
            below.  The data is distributed as a normal spherical Gaussian.  In this case, a routine calculation shows: 
                A. SENTIMENT    {()}                                     = alpha A + beta B^2
                B. INTENSITY    {(a)}{(a)}                               = alpha^2 + 4 beta^2 B^2
                C. UNCERTAINTY  {(a)(a)} - {(a)}{(a)}                    = 1 + 4 B^2
                D. PASSION      2{(a)}{(ab)}{(b)}                        = 16 beta^3 B^2  <---+
                E. AUDACITY     2{(a)}{(ab)(b)} - 2{(a)}{(ab)}{(b)}      = 16 beta B^2        |
                F. PASSION/2    {(ab)}{(a)}{(b)}                         = -------------------+ divided by 2 = 8 beta^3 B^2
                G. PERIL        {(ab)}{(a)(b)} - {(ab)}{(a)}{(b)}        = 8 beta B^2 
'''

import tensorflow as tf
import numpy as np



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



################################################################################
#            1. RUN ON TOY LOSS LANDSCAPE                                      #
################################################################################
 
if __name__ == '__main__':

    BATCH_SIZE = 4096
    ALPHA = 3.0
    BETA  = 2.0
    A = 1.0
    B = 1.0

    #--------------------------------------------------------------------------#
    #                1.0 define toy loss landscape                             #
    #--------------------------------------------------------------------------#
   
    NoiseA = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
    NoiseB = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
    # below, we use BATCH_SIZE many copies of Weight to address the egregious summing operation implicit in
    #       `tf.gradients`
    Weights = tf.placeholder(tf.float32, shape=[2, BATCH_SIZE]) # i.e. [WeightA, WeightB]
    Losses = (ALPHA + NoiseA) * Weights[0] + (BETA  + NoiseB) * tf.square(Weights[1])
    
    Sentiment, Intensity, Uncertainty, Passion, Audacity, Peril = gradient_stats(Losses, Weights, BATCH_SIZE)
    
    #--------------------------------------------------------------------------#
    #                1.1 display stats on batch                                #
    #--------------------------------------------------------------------------#
   
    def get_batch(batch_size=BATCH_SIZE):
        ''' return (independent) noise samples in the format of a tensorflow feed_dict '''
        noise_a = np.random.randn(batch_size) 
        noise_b = np.random.randn(batch_size) 
        return {NoiseA:noise_a, NoiseB:noise_b, Weights:np.stack([A*np.ones(batch_size), B*np.ones(batch_size)], axis=0)}

    with tf.Session() as session:
        batch = get_batch()
    
        # Though, for the purpose of testing, we compute the following 7 scalars in independent session-runs, one would
        #       in practice use a single call: `session.run([AvgOut, MeanSqrGrad, ...], ...)`
        sentiment   =   session.run(Sentiment,      feed_dict=batch)
        intensity   =   session.run(Intensity,      feed_dict=batch)
        uncertainty =   session.run(Uncertainty,    feed_dict=batch)
        passion     =   session.run(Passion,        feed_dict=batch)
        audacity    =   session.run(Audacity,       feed_dict=batch)
        peril       =   session.run(Peril,          feed_dict=batch)

        print('sentiment    %6.2f --- expected %6.2f' % (sentiment,   ALPHA*A + BETA *B**2))
        print('intensity    %6.2f --- expected %6.2f' % (intensity,   ALPHA**2 + 4.0*BETA **2*B**2))
        print('uncertainty  %6.2f --- expected %6.2f' % (uncertainty, 1.0 + 4.0*B**2))
        print('passion      %6.2f --- expected %6.2f' % (passion,     16.0*BETA **3*B**2))
        print('audacity     %6.2f --- expected %6.2f' % (audacity,    16.0*BETA *B**2))
        print('peril        %6.2f --- expected %6.2f' % (peril,       8.0*BETA *B**2))
