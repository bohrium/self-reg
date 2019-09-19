''' author: samtenka
    change: 2019-08-17
    create: 2019-06-12
    descrp: helpers for torch, math, profiling, and ansi commands
'''

import torch
import functools
import time
import sys
try:
    import memory_profiler
except ImportError:
    print('failed attempt to import `memory_profiler`')



################################################################################
#           0. TORCH                                                           #
################################################################################

device, _ = (
    (torch.device("cuda:0"), torch.device("cuda:1"))
    if torch.cuda.is_available() else
    (torch.device("cpu"), torch.device("cpu"))
) 



################################################################################
#           1. MATH                                                            #
################################################################################

prod = lambda seq: functools.reduce(lambda a,b:a*b, seq, 1) 



################################################################################
#           2. PROFILING                                                       #
################################################################################

start_time = time.time()
secs_endured = lambda: (time.time()-start_time) 
megs_alloced = None if 'memory_profile' not in sys.modules else lambda: (
    memory_profiler.memory_usage(
        -1, interval=0.001, timeout=0.0011
    )[0]
)



################################################################################
#           3. ANSI COMMANDS                                                   #
################################################################################

class Colorizer(object):
    def __init__(self):
        self.ANSI_by_name = {
            '@K ': '\033[38;2;000;000;000m',  # color: black
            '@R ': '\033[38;2;255;064;064m',  # color: red
            '@O ': '\033[38;2;255;128;000m',  # color: orange
            '@Y ': '\033[38;2;192;192;000m',  # color: yellow
            '@L ': '\033[38;2;128;255;000m',  # color: lime 
            '@G ': '\033[38;2;064;255;064m',  # color: green
            '@J ': '\033[38;2;000;255;192m',  # color: jade
            '@C ': '\033[38;2;000;192;192m',  # color: cyan
            '@T ': '\033[38;2;000;192;255m',  # color: teal
            '@B ': '\033[38;2;064;064;255m',  # color: blue
            '@P ': '\033[38;2;128;000;255m',  # color: purple  
            '@M ': '\033[38;2;192;000;192m',  # color: magenta
            '@S ': '\033[38;2;255;000;128m',  # color: salmon  
            '@W ': '\033[38;2;255;255;255m',  # color: white
            '@^ ': '\033[1A',   # motion: up
        }
        self.text = ''

    def __add__(self, rhs):
        assert type(rhs) == type(''), 'expected types (Colorizer + string)'
        for name, ansi in self.ANSI_by_name.items():
            rhs = rhs.replace(name, ansi)
        self.text += rhs
        return self

    def __str__(self):
        rtrn = self.text 
        self.text = ''
        return rtrn

CC = Colorizer()

if __name__=='__main__':
    print(CC + '@K moo')

    print(CC + '@R moo')
    print(CC + '@O moo')
    print(CC + '@Y moo')
    print(CC + '@L moo')
    print(CC + '@G moo')
    print(CC + '@J moo')
    print(CC + '@C moo')
    print(CC + '@T moo')
    print(CC + '@B moo')
    print(CC + '@P moo')
    print(CC + '@M moo')
    print(CC + '@S moo')
    print(CC + '@R moo')

    print(CC + '@W moo')
    print(CC + '@R moo')
    print(CC + 'hi @M moo' + 'cow @C ')
