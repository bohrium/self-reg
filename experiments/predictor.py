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
    return (eval(form, {'eta':eta, 'T':T, 'gs':gs}) for form in (Y, S))

def sgd_test_linear(eta, T):
    Y, S = from_string('(0)() - eta*T*(0-1)(01)', eta, T)
    return Y, S
def sgd_test_quadratic(eta, T):
    Y, S = from_string('(0)() - eta*T*(0-1)(01) + eta*eta*(T*(T-1)/2.0)*2*(0-1-2)(01-02) + eta*eta*T*0.5*(0-12)(01-02)', eta, T)
    return Y, S


