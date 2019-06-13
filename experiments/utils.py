''' author: samtenka
    change: 2019-06-11
    create: 2019-06-11
    descrp: provide command line helpers including argument parsing and ANSI coloration  
'''

import os
import tensorflow as tf

def suppress_tf_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.ERROR)

class Colorizer(object):
    def __init__(self):
        self.ANSI_by_name = {
            '@R ': '\033[31m',
            '@G ': '\033[32m',
            '@Y ': '\033[33m',
            '@B ': '\033[34m',
            '@W ': '\033[37m',
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
    print(CC + 'moo')
    print(CC + '@R moo')
    print(CC + 'hi @W moo' + 'cow')
