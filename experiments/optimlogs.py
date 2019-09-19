''' author: samtenka
    change: 2019-08-17
    create: 2019-06-17
    descrp: interface for optimization results
'''

from collections import namedtuple
import numpy as np
from utils import CC
import torch

#OptimKey = namedtuple('OptimKey', ('sampler', 'beta', 'eta', 'T', 'N', 'evalset', 'metric')) 
OptimKey = namedtuple('OptimKey', ('optimizer', 'beta', 'eta', 'T', 'N', 'metric')) 

class OptimLog(object):
    def __init__(self):
        self.logs = {}
    def accum(self, okey, value):
        if okey not in self.logs:
            self.logs[okey] = []
        self.logs[okey].append(value.detach().numpy())

    def recent(self, okey):
        return self.logs[okey][-1]

    def compute_diffs(self):
        diffs = {}
        for okey_base, value_base in self.logs.items():
            for okey_comp, value_comp in self.logs.items():
                if okey_base == okey_comp: continue
                if okey_base.metric != okey_comp.metric: continue
                if len(value_base) != len(value_comp): continue

                value_diff = np.array(value_comp) - np.array(value_base)
                okey_diff = OptimKey(
                    sampler     =(okey_comp.sampler, okey_base.sampler),
                    beta        =(okey_comp.beta,   okey_base.beta),
                    eta         =(okey_comp.eta,    okey_base.eta),
                    T           =(okey_comp.T,      okey_base.T),
                    N           =(okey_comp.N,      okey_base.N),
                    evalset     =(okey_comp.evalset,okey_base.evalset),
                    metric      =(okey_comp.metric, okey_base.metric),
                )  
                diffs[okey_diff] = value_diff
        for k in diffs:
            self.logs[k] = diffs[k]

    def __str__(self):
        self.compute_diffs()
        return '{\n'+',\n'.join(
            '    {}: {{ "mean":{}, "stdv":{}, "nb_samples":{} }}'.format(
                okey, np.mean(values), np.std(values), len(values)
            )
            for okey, values in sorted(self.logs.items())
        )+'\n}'

    def absorb(self, rhs):
        for okey, values in rhs.logs.items():
            if okey not in self.logs:
                self.logs[okey] = []
            self.logs[okey] += values
