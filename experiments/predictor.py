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

with open('gs.data') as f:
    gs = eval(f.read())

def from_string(formula, eta, T):
    Y = formula[:]
    S = formula.replace('- ', '+ ').replace(' -', ' +')
    for k in gs:
        Y = Y.replace(k, 'gs["{}"]["mean"]'.format(k))
        S = S.replace(k, 'gs["{}"]["stdv"]/gs["{}"]["nb_samples"]**0.5'.format(k, k))
    return (eval(form, {'eta':eta, 'T':T, 'gs':gs, 'np':np}) for form in (Y, S))

sgd_test_coeffs = (
    '(+ ((0)()))',
    '(- (T*(0-1)(01)))',
    '(+ ((T*(T-1)/2.0)*2*(0-1-2)(01-02) + T*0.5*(0-12)(01-02)))',
)
def sgd_test_taylor(eta, T, degree):
    assert 1 <= degree, 'need strictly positive degree of approximation!'
    formula = ' + '.join('eta*'*d + sgd_test_coeffs[d] for d in range(degree+1))
    print(formula)
    Y, S = from_string(formula, eta, T)
    return Y, S
def sgd_test_exponential(eta, T, degree):
    # TODO: correct error bars 
    cs = [sgd_test_coeffs[d] for d in range(3)]
    if degree==2:
        rate = '(-2 * {} / {})'.format(cs[2], cs[1])
        scale = '({} * {} / (2 * {}))'.format(cs[1], cs[1], cs[2])
        offset = '({} - {} * {} / (2 * {}))'.format(cs[0], cs[1], cs[1], cs[2])
        formula = '{} * np.exp(- {} * eta) + {}'.format(scale, rate, offset)
        Y, S = from_string(formula, eta, T)
    return Y, S


