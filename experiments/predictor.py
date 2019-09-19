''' author: samtenka
    change: 2019-06-26
    create: 2019-06-18
    descrp: Predict based on diagrams
'''

from matplotlib import pyplot as plt
import numpy as np
from optimlogs import OptimKey
import solver
import sys 
from utils import prod

red  ='#cc4444'
green='#44cc44'
blue ='#4444cc'

def choose(T, t):
    return prod(range(T-t+1, T+1)) / float(prod(range(1, t+1)))

def from_string(gradstats, formula, eta, T, E=1):
    Y = formula[:]
    S = formula.replace('- ', '+ ').replace(' -', ' +')
    for k in gradstats:
        Y = Y.replace(k, 'gradstats["{}"]["mean"]'.format(k))
        S = S.replace(k, 'gradstats["{}"]["stdv"]/gradstats["{}"]["nb_samples"]**0.5'.format(k, k))
    return tuple(eval(form, {'eta':eta, 'T':T, 'E':E, 'gradstats':gradstats, 'np':np, 'choose':choose}) for form in (Y, S))

sgd_test_coeffs = (
    '(+ ( ()(0) ) )',
    '(- (choose(T, 1) * (01)(0-1)) )',
    '(+ (choose(T, 2) * 2*(01-02)(0-1-2) + choose(T, 1) * 0.5*(01-02)(0-12)) )',
    '(- (choose(T, 3) * (4*(01-02-13)(0-1-2-3) + 2*(01-02-03)(0-1-2-3)) + choose(T, 2) * (1.5 * (01-02-03)(01-23) + (01-02-13)(0-12-3)) + choose(T, 1) * ((1.0/6) * (01-02-03)(0-123))) )',
)

sgd_test_multiepoch_coeffs = (
    '(+ ( ()(0) ) )',
    '(- (choose(E, 1) * choose(T, 1) * (01)(0-1)) )',
    '(+ ((2 * choose(E, 2) + choose(E, 1)) * choose(T, 2) * 2*(01-02)(0-1-2) + choose(E, 2)*choose(T, 1)*((01-02)(0-12) + (01-02)(01-2)) + choose(E, 1)*choose(T, 1)*0.5*(01-02)(0-12)) )',
)

def sgd_test_multiepoch_diff_e2h2(gradstats, eta, T, degree, E=None):
    assert 2 == degree, 'i only know the answer for degree 2!'
    formula = 'eta*eta * choose(E, 2) * choose(T, 1) * (01-02)(01-2)'
    print(formula)
    Y, S = from_string(gradstats, formula, eta, T, E)
    return Y, S


def sgd_test_multiepoch(gradstats, eta, T, degree, E=None):
    assert 1 <= degree, 'need strictly positive degree of approximation!'
    formula = ' + '.join('eta*'*d + sgd_test_multiepoch_coeffs[d] for d in range(degree+1))
    print(formula)
    Y, S = from_string(gradstats, formula, eta, T, E)
    return Y, S

def sgd_test_taylor(gradstats, eta, T, degree):
    assert 1 <= degree, 'need strictly positive degree of approximation!'
    formula = ' + '.join('eta*'*d + sgd_test_coeffs[d] for d in range(degree+1))
    print(formula)
    Y, S = from_string(gradstats, formula, eta, T)
    return Y, S

def sgd_test_exponential(gradstats, eta, T, degree):
    # TODO: correct error bars 
    if degree==2:
        cs = [from_string(gradstats, sgd_test_coeffs[d], None, T)[0] for d in range(3)]
        print(cs)
        rate = -2*cs[2]/cs[1]
        scale = cs[1]**2 / (2*cs[2])
        offset = cs[0] - cs[1]**2/(2*cs[2])

        formula = '{} * np.exp(- {} * eta) + {}'.format(scale, rate, offset)
        print(formula)

        Y, S = from_string(gradstats, formula, eta, T)
        S *= 0
    elif degree==3:
        cs = [from_string(gradstats, sgd_test_coeffs[d], None, T)[0] for d in range(4)]
        rate = abs(3 * (cs[2]/cs[1])**2 - 2 * (cs[3]/cs[1]))**0.5
        scale = (1.0/2 + (cs[2]/cs[1])/(2*rate))**2 * rate / cs[1]  
        shift = abs((scale*rate)/cs[1])**0.5   
        offset = cs[0] - 1.0/shift

        formula = '1.0 / ( {} * (np.exp(- {} * eta) - 1) + {} ) + {}'.format(scale, rate, shift, offset)

        Y, S = from_string(gradstats, formula, eta, T)
        S *= 0

    return Y, S

