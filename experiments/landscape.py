''' author: samtenka
    change: 2019-09-10
    create: 2019-06-10
    descrp: define loss landscape type 
'''

from abc import ABC, abstractmethod
from itertools import chain 
import numpy as np
import os.path

from utils import CC



#==============================================================================#
#           0. DECLARE LANDSCAPE INTERFACE                                     #
#==============================================================================#

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

    #--------------------------------------------------------------------------#
    #               0.0 declare samplers for data and weights                  #
    #--------------------------------------------------------------------------#

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

    #--------------------------------------------------------------------------#
    #               0.1 declare lens for current weights                       #
    #--------------------------------------------------------------------------#

    @abstractmethod
    def get_weights(self):
        ''' return numpy value of current weights '''
        pass

    @abstractmethod
    def set_weights(self, weights):
        ''' overwrite current weights from numpy value '''
        pass

    @abstractmethod
    def update_weights(self, displacement):
        ''' add displacement to weights '''
        pass

    #--------------------------------------------------------------------------#
    #               0.2 declare loss stalk and its derivative operator         #
    #--------------------------------------------------------------------------#

    @abstractmethod
    def get_loss_stalk(self, data_indices):
        ''' present loss, averaged over given data, as a deterministic scalar stalk '''
        pass

    @abstractmethod
    def nabla(self, scalar_stalk):
        ''' differentiate the given deterministic scalar stalk (assumed to be on current weight),
            returning a deterministic vector stalk (with the same shape as weights have)
        '''
        pass



#==============================================================================#
#           1. IMPLEMENT READING/WRITING OF WEIGHT INITIALIZATIONS             #
#==============================================================================#

class FixedInitsLandscape(PointedLandscape):
    ''' For reduced estimation-error, we may choose to initialize only at a few points.  This
        wrapper on the PointedLandscape class implements some handy methods for this purpose.  
    '''

    def resample_to(self, file_nm, nb_inits):
        ''' save a random list of weight initializations to the file named '''
        assert file_nm.endswith('.npy'), 'file name must end with .npy'
        assert not os.path.isfile(file_nm), 'avoided overwriting {}'.format(file_nm)
        self.inits = [(self.resample_weights(), self.get_weights())[1] for _ in range(nb_inits)] 
        np.save(file_nm, self.inits)
        print(CC + 'saved @R {} @C initial weights to @M {} @C '.format(
            len(self.inits), file_nm
        ))

    def load_from(self, file_nm, nb_inits=None):
        ''' load set of weight initializations from the file named '''
        assert file_nm.endswith('.npy'), 'file name must end with .npy'
        if not os.path.isfile(file_nm):
            assert nb_inits is not None, 'attempted resample before load: nb_inits is unspecified!'
            self.resample_to(file_nm, nb_inits)
        else:
            self.inits = np.load(file_nm)
        print(CC + 'loaded @R {} @C initial weights from @M {} @C '.format(
            len(self.inits), file_nm
        ))
        self.switch_to(0)

    def switch_to(self, init_idx):  
        ''' switch current weight to that of the given index '''
        self.set_weights(self.inits[init_idx]) 
        #print(CC + 'switched to initial weight @R {} @C of @R {} @C '.format(
        #    init_idx, len(self.inits)
        #))
