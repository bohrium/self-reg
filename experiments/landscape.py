''' author: samtenka
    change: 2019-09-03
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
    ''' Interface for a stochastic loss landscape.  In particular, presents a distribution over
        smooth loss functions on weight space by providing: 

            get_loss_stalk : Datapoints --> Weights* --> Reals

            sample_data : 1 --> Datapoints 

            nabla : ( Weights* --> Reals ) --> ( Weights* --> Reals )

        This landscape class is *pointed*, meaning that it maintains a weight via the lens

            get_weights: Landscape --> Weights

            set_weights: Weights --> Landscape

        --- a weight sampled according to a distribution 

            resample_weights: 1 --> Landscape
        
        It is at this weight that `get_loss_stalk` and `nabla` act.  Likewise, via
            
            update_weights: Landscape --> TangentSpace(Weights) --> Landscape

        gradient descent is cleanly implementable.  In the above, the type `Weights*` actually
        indicates the weights in an infinitesimal neighborhood around the weight maintained by the
        Landscape at the time of execution of that function; we read the type `Weights* --> Reals`
        as type of germs of smooth functions at the landscape's weight.
    '''

    @abstractmethod
    def sample_data(self, N): 
        ''' sample N datapoints (i.e. memory-light objects indexing deterministic loss landscapes)
            independently and identically distributed from the population.
        '''
        pass

    @abstractmethod
    def resample_weights(self):
        ''' reset Point to a (potentially dirac) distribution that is a property of this Pointed
            Landscape class (but not conceptually a property of the mathematical object).
            should be automatically called also at initialization
        '''
        pass

    @abstractmethod
    def get_weights(self):
        ''' '''
        pass

    @abstractmethod
    def set_weights(self, weights):
        ''' '''
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



################################################################################
#           1. DEFINE WEIGHT READ/WRITING                                      #
################################################################################

class FixedInitsLandscape(PointedLandscape):
    '''
    '''
    def resample(self, file_nm, nb_inits):
        '''  '''
        self.inits = [(self.reset_weights(), self.get_weights())[1] for _ in range(nb_inits)] 
        np.save(file_nm, self.inits)

    def load_from(self, file_nm):
        ''' '''
        self.inits = np.load(file_nm)

    def switch_to(self, init_idx):  
        ''' '''
        self.set_weights(self.inits[init_idx]) 
