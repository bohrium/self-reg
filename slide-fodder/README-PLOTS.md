The directories logistic/ and shallow/ each contain 5 plots.  These plots all depict losses on binary MNIST
classification ('0' vs '1') for GD or SGD for very small train sets and relatively few updates.  Throughout, SGD has
batch-size 1, sampled without repetition. 

For LOGISTIC, number of updates = trainset size = 100.
    Here, we perform logistic regression with no explicit bias term.  We initialize weights to 0. 

For SHALLOW , number of updates = trainset size = 10.
    Here, we train a shallow densely connected neural net with tanh activation for its 28 hidden units
    and no explicit bias terms.  We initialize to a fixed weight vector consisting of 0s in the first layer, and in 
    the second layer, to break symmetry, uniformly spaced weights between -1/2 and +1/2. 

In all 5+5 plots, the vertical axis has units of nits and the horizontal axis has units of inverse nits (assuming
weightspace is dimensionless).  Experimental results are shown in blue, while predictions are shown in red and green.
The error bars for experimental data indicate the error of estimating expected losses by finitary averages.
The error bars for predictions indicate the error of the estimating parameters by finitary averages over the training
set PLUS the error of modeling the ambient distribution by the training set and test set. 
That latter term in the prediction error thus accounts for the fact that we estimate experimental out-of-sample losses
by finitary averages across a test set.

OUT-SGD
    This is the out-of-sample loss for SGD.  It starts at log(2) ~ 0.69, as expected. 

OUT-GD
    This is the out-of-sample loss for GD.  It starts at log(2) ~ 0.69, as expected.

OUT-DIFF
    This is the out-of-sample loss for GD MINUS the out-of-sample loss for SGD.  Positive means SGD is better than GD.
    One finds that SGD indeed optimizes better.  In agreement with predictions, this stochasticity benefit appears to
    be a 2nd order effect i.e. appears to scale with (learning rate)^2.  This second order term is controlled by the
        (A) 1st order increase in trace-of-covariance-of-gradient --- when this is high, SGD is favored --- and
        (B) trace-of-(covariance-times-hessian) --- when this is high, GD is favored.
    This makes sense: B measures how much noise correlates with curvature and thus how much Jensen would advise us to
    average away this noise. 

GEN-SGD
    This is the in-sample loss MINUS the out-of-sample loss for SGD.  Positive means worse generalization.
    The first order term is controlled by the trace-of-covariance-of-gradient.

GEN-GD
    This is the in-sample loss MINUS the out-of-sample loss for SGD.  Positive means worse generalization.
    The first order term is controlled by the trace-of-covariance-of-gradient.
    There is no green (quadratic) prediction for GD's generalization gap because I haven't yet computed the prediction. 
