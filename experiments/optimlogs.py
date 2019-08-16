''' author: samtenka
    change: 2019-06-17
    create: 2019-06-17
    descrp: interface for optimization results
'''

from collections import namedtuple
import numpy as np
from utils import CC
import torch

OptimKey = namedtuple('OptimKey', ('optimizer', 'eta', 'T', 'N', 'metric', 'beta')) 

class OptimLog(object):
    def __init__(self):
        self.logs = {}
    def accum(self, okey, value):
        if okey not in self.logs:
            self.logs[okey] = []
        self.logs[okey].append(value.detach().numpy())
    def recent(self, okey):
        return self.logs[okey][-1]
    def __str__(self):
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
