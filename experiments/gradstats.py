''' author: samtenka
    change: 2019-06-17
    create: 2019-06-17
    descrp: interface for gradient statistics data structure
'''

import numpy as np
from utils import CC
import torch

grad_stat_names = set([
    '(0)()',
    '(0-1)(01)',
    '(01)(01)',
    '(0-1-2)(01-02)',
    '(0-12)(01-02)',
    '(0-12)(01-12)',
    '(012)(01-02)',
])

class GradStats(object):
    def __init__(self):
        self.logs = {
            nm:[] for nm in grad_stat_names
        }
    def accum(self, name, value):
        self.logs[name].append(value.detach().numpy())
    def recent(self, name):
        return self.logs[name][-1]
    def __str__(self):
        return '{\n'+',\n'.join(
            '    "{}": {{ "mean":{}, "stdv":{}, "nb_samples":{} }}'.format(
                name, np.mean(values), np.std(values), len(values)
            )
            for name, values in sorted(self.logs.items())
        )+'\n}'
