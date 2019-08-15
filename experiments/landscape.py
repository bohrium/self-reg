''' author: samtenka
    change: 2019-06-12
    create: 2019-06-10
    descrp: define loss landscape type 
'''

from abc import ABC, abstractmethod
from itertools import chain 
import numpy as np

from utils import CC



################################################################################
#           0. DEFINE LANDSCAPE INTERFACE                                      #
################################################################################

class PointedLandscape(ABC):
    ''' 
    '''

    @abstractmethod
    def sample_data(self, N): 
        ''' sample N datapoints (i.e. memory-light objects indexing deterministic loss landscapes)
            independently and identically distributed from the population.
        '''
        pass

    @abstractmethod
    def reset_weights(self):
        ''' reset Point to a (potentially dirac) distribution that is a property of this Pointed
            Landscape class (but not conceptually a property of the mathematical object).
            should be automatically called also at initialization
        '''
        pass

    @abstractmethod
    def update_weights(self, displacement):
        ''' add displacement to weights '''
        pass

    @abstractmethod
    def get_loss_stalk(self, data_indices):
        ''' present loss, averaged over given data, as a deterministic scalar stalk '''
        pass

    @abstractmethod
    def nabla(self, scalar_stalk):
        ''' differentiate deterministic scalar stalk (assumed to be on current weight),
            returning as deterministic vector stalk (with same shape as weights)
        '''
        pass

