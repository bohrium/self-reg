''' author: samtenka
    change: 2019-06-18
    create: 2019-06-18
    descrp: Predict based on diagrams
'''

from matplotlib import pyplot as plt
import numpy as np
from optimlogs import OptimKey
import sys 

red  ='#cc4444'
green='#44cc44'
blue ='#4444cc'

def from_string(gradstats, formula, eta, T):
    Y = formula[:]
    S = formula.replace('- ', '+ ').replace(' -', ' +')
    for k in gradstats:
        Y = Y.replace(k, 'gradstats["{}"]["mean"]'.format(k))
        S = S.replace(k, 'gradstats["{}"]["stdv"]/gradstats["{}"]["nb_samples"]**0.5'.format(k, k))
    return (eval(form, {'eta':eta, 'T':T, 'gradstats':gradstats, 'np':np}) for form in (Y, S))

sgd_test_coeffs = (
    '(+ (()(0)))',
    '(- (T*(01)(0-1)))',
    '(+ ((T*(T-1)/2.0)*2*(01-02)(0-1-2) + T*0.5*(01-02)(0-12)))',
)
def sgd_test_taylor(gradstats, eta, T, degree):
    assert 1 <= degree, 'need strictly positive degree of approximation!'
    formula = ' + '.join('eta*'*d + sgd_test_coeffs[d] for d in range(degree+1))
    Y, S = from_string(gradstats, formula, eta, T)
    return Y, S

def sgd_test_exponential(gradstats, eta, T, degree):
    # TODO: correct error bars 
    cs = [sgd_test_coeffs[d] for d in range(3)]
    if degree==2:
        rate = '(-2 * {} / {})'.format(cs[2], cs[1])
        scale = '({} * {} / (2 * {}))'.format(cs[1], cs[1], cs[2])
        offset = '({} - {} * {} / (2 * {}))'.format(cs[0], cs[1], cs[1], cs[2])
        formula = '{} * np.exp(- {} * eta) + {}'.format(scale, rate, offset)
        Y, S = from_string(gradstats, formula, eta, T)
    return Y, S


