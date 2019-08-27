''' author: samtenka
    change: 2019-08-26
    create: 2019-06-16
    descrp: define gradient statistics up to 3rd order
'''

import numpy as np

from utils import CC
from landscape import PointedLandscape
from gradstats import grad_stat_names, GradStats
import torch
import tqdm

def compute_grad_stats(land, N, I=1):
    nab = land.nabla

    gs = GradStats()
    for i in tqdm.tqdm(range(I)):
        land.reset_weights()
        A, B, C, D = (
            land.get_loss_stalk(land.sample_data(N))
            #land.weights.pow(2).sum()
            for i in range(4)
        )
        
        GA, GB, GC, GD = (
            nab(X) 
            for X in (A, B, C, D)
        )
        GA_, GB_, GC_, GD_ = (
            Gi.detach()
            for Gi in (GA, GB, GC, GD)
        )

        gs.accum('()(0)', (
            (A+B+C+D)/4
        ))

        gs.accum('(01)(0-1)', (
            (GA.dot(GB) + GC.dot(GD))/2
        ))
        gs.accum('(01)(01)', (
            (GA.dot(GA)+GB.dot(GB)+GC.dot(GC)+GD.dot(GD))/4 * N  
            - gs.recent('(01)(0-1)') * (N-1)
        ))

        gs.accum('(01-02)(0-1-2)', (
            (nab(GA.dot(GB_))).dot(GC)
        )) 
        gs.accum('(01-02)(0-12)', (
            ((nab(GA.dot(GB_))).dot(GB) +
             (nab(GC.dot(GD_))).dot(GD))/2 * N 
            - gs.recent('(01-02)(0-1-2)') * (N-1)
        ))
        gs.accum('(01-02)(01-2)', (
            (GA.dot(nab(GB.dot(GB_))) +
             GC.dot(nab(GD.dot(GD_))))/2 * N
            - gs.recent('(01-02)(0-1-2)') * (N-1)
        ))

        gs.accum('(01-02)(012)', (
            ((nab(GA.dot(GA_))).dot(GA) +
             (nab(GB.dot(GB_))).dot(GB) +
             (nab(GC.dot(GC_))).dot(GC) +
             (nab(GD.dot(GD_))).dot(GD))/4 * N*N
            -     gs.recent('(01-02)(0-12)') * (N-1)
            - 2 * gs.recent('(01-02)(01-2)') * (N-1)
            -     gs.recent('(01-02)(0-1-2)') * (N-1)*(N-2)
        ))

        #tree
        gs.accum('(01-02-03)(0-1-2-3)', (
            nab(nab(GA.dot(GB_)).dot(GC_)).dot(GD)
        ))
        #tree leaves
        gs.accum('(01-02-03)(0-1-23)', (
            nab(nab(GA.dot(GB_)).dot(GC_)).dot(GC) * N
            -     gs.recent('(01-02-03)(0-1-2-3)') * (N-1)
        ))
        #tree branch
        gs.accum('(01-02-03)(01-2-3)', (
            nab(nab(GA.dot(GB_)).dot(GC_)).dot(GA) * N
            -     gs.recent('(01-02-03)(0-1-2-3)') * (N-1)
        ))
        #tree but root 
        gs.accum('(01-02-03)(0-123)', (
            nab(nab(GA.dot(GC_)).dot(GC_)).dot(GC) * N*N
            - 3 * gs.recent('(01-02-03)(0-1-23)')  * (N-1)
            -     gs.recent('(01-02-03)(0-1-2-3)') * (N-1)*(N-2)
        ))
        #tree but leaf
        gs.accum('(01-02-03)(012-3)', (
            nab(nab(GA.dot(GA_)).dot(GA_)).dot(GC) * N*N
            - 2 * gs.recent('(01-02-03)(01-2-3)')  * (N-1)
            -     gs.recent('(01-02-03)(0-1-23)')  * (N-1)
            -     gs.recent('(01-02-03)(0-1-2-3)') * (N-1)*(N-2)
        ))
        #tree split
        gs.accum('(01-02-03)(01-23)', (
            nab(nab(GA.dot(GA_)).dot(GC_)).dot(GC) * N*N
            -     gs.recent('(01-02-03)(01-2-3)')  * (N-1)
            -     gs.recent('(01-02-03)(0-1-23)')  * (N-1)
            -     gs.recent('(01-02-03)(0-1-2-3)') * (N-1)*(N-1)
        ))
        #tree all
        gs.accum('(01-02-03)(0123)', (
            nab(nab(GA.dot(GA_)).dot(GA_)).dot(GA) * N*N*N
            - 3 * gs.recent('(01-02-03)(01-23)')   * (N-1)
            - 3 * gs.recent('(01-02-03)(012-3)')   * (N-1)
            -     gs.recent('(01-02-03)(0-123)')   * (N-1)
            - 3 * gs.recent('(01-02-03)(01-2-3)')  * (N-1)*(N-2)
            - 3 * gs.recent('(01-02-03)(0-1-23)')  * (N-1)*(N-2)
            -     gs.recent('(01-02-03)(0-1-2-3)') * (N-1)*(N-2)*(N-3)
        ))

        #vine
        gs.accum('(01-02-13)(0-1-2-3)', (
            nab(GC_.dot(GA)).dot(nab(GB.dot(GD_)))
        ))
        #vine leaves 
        gs.accum('(01-02-13)(0-1-23)', (
            nab(GC_.dot(GA)).dot(nab(GB.dot(GC_))) * N
            - gs.recent('(01-02-13)(0-1-2-3)') * (N-1)
        ))
        #vine alternating
        gs.accum('(01-02-13)(0-12-3)', (
            nab(GC_.dot(GA)).dot(nab(GC.dot(GD_))) * N
            - gs.recent('(01-02-13)(0-1-2-3)') * (N-1)
        ))
        #vine branch 
        gs.accum('(01-02-13)(0-13-2)', (
            nab(GA_.dot(GA)).dot(nab(GB.dot(GD_))) * N
            - gs.recent('(01-02-13)(0-1-2-3)') * (N-1)
        ))
        #vine middle
        gs.accum('(01-02-13)(01-2-3)', (
            nab(GC_.dot(GA)).dot(nab(GA.dot(GD_))) * N
            - gs.recent('(01-02-13)(0-1-2-3)') * (N-1)
        ))


        #vine but middle
        gs.accum('(01-02-13)(0-123)', (
            nab(GC_.dot(GA)).dot(nab(GC.dot(GC_))) * N*N
            -     gs.recent('(01-02-13)(0-13-2)')  * (N-1)
            -     gs.recent('(01-02-13)(0-12-3)')  * (N-1)
            -     gs.recent('(01-02-13)(0-1-23)')  * (N-1)
            -     gs.recent('(01-02-13)(0-1-2-3)') * (N-1)*(N-2)
        ))
        #vine but leaf 
        gs.accum('(01-02-13)(012-3)', (
            nab(GC_.dot(GA)).dot(nab(GA.dot(GA_))) * N*N
            -     gs.recent('(01-02-13)(0-13-2)')  * (N-1)
            -     gs.recent('(01-02-13)(0-12-3)')  * (N-1)
            -     gs.recent('(01-02-13)(01-2-3)')  * (N-1)
            -     gs.recent('(01-02-13)(0-1-2-3)') * (N-1)*(N-2)
        ))
        #vine split middle leaves 
        gs.accum('(01-02-13)(01-23)', (
            nab(GC_.dot(GA)).dot(nab(GA.dot(GC_))) * N*N
            -     gs.recent('(01-02-13)(01-2-3)')  * (N-1)
            -     gs.recent('(01-02-13)(0-1-23)')  * (N-1)
            -     gs.recent('(01-02-13)(0-1-2-3)') * (N-1)*(N-1)
        ))
        #vine split branches
        gs.accum('(01-02-13)(02-13)', (
            nab(GC_.dot(GC)).dot(nab(GA.dot(GA_))) * N*N
            - 2 * gs.recent('(01-02-13)(0-13-2)')  * (N-1)
            -     gs.recent('(01-02-13)(0-1-2-3)') * (N-1)*(N-1)
        ))
        #vine split alternating
        gs.accum('(01-02-13)(03-12)', (
            nab(GC_.dot(GA)).dot(nab(GC.dot(GA_))) * N*N
            - 2 * gs.recent('(01-02-13)(0-12-3)')  * (N-1)
            -     gs.recent('(01-02-13)(0-1-2-3)') * (N-1)*(N-1)
        ))
        #vine all
        gs.accum('(01-02-13)(0123)', (
            nab(GA_.dot(GA)).dot(nab(GA.dot(GA_))) * N*N*N
            -     gs.recent('(01-02-13)(0-1-2-3)') * (N-1)*(N-2)*(N-3)
            - 2 * gs.recent('(01-02-13)(0-12-3)') * (N-1)*(N-2)
            - 2 * gs.recent('(01-02-13)(0-13-2)') * (N-1)*(N-2)
            -     gs.recent('(01-02-13)(01-2-3)') * (N-1)*(N-2)
            -     gs.recent('(01-02-13)(0-1-23)') * (N-1)*(N-2)
            - 2 * gs.recent('(01-02-13)(0-123)') * (N-1)
            - 2 * gs.recent('(01-02-13)(012-3)') * (N-1)
            -     gs.recent('(01-02-13)(01-23)') * (N-1)
            -     gs.recent('(01-02-13)(02-13)') * (N-1)
            -     gs.recent('(01-02-13)(03-12)') * (N-1)
        ))

    return gs

def test_derivator_on_cosh():
    from cosh_landscapes import Cosh 

    DIM = 256
    Q = Cosh(dim=DIM)
    ep = np.exp(1)
    em = np.exp(-1)

    L = (ep + em)/2
    G = (ep - em)/2
    H = (ep + em)/2
    J = (ep - em)/2

    GG = (ep**2 + em**2)/2
    HG = (ep**2 - em**2)/2
    HH = (ep**2 + em**2)/2
    JG = (ep**2 + em**2)/2

    GGG = (ep**3 - em**3)/2
    HGG = (ep**3 + em**3)/2
    HHG = (ep**3 - em**3)/2
    JGG = (ep**3 - em**3)/2

    HHGG = (ep**4 + em**4)/2
    JGGG = (ep**4 + em**4)/2

    predictions = {
        '()(0)': DIM * ( L ),
        '(01)(0-1)': DIM * ( G**2 ),
        '(01)(01)':  DIM * ( GG ), 
        '(01-02)(0-1-2)': DIM * ( H * G**2 ), 
        '(01-02)(0-12)':  DIM * ( H * GG ),
        '(01-02)(01-2)':  DIM * ( HG * G  ),
        '(01-02)(012)':   DIM * ( HGG ),
        '(01-02-03)(0-1-2-3)': DIM * ( J * G**3 ),
        '(01-02-03)(0-1-23)':  DIM * ( J * G * GG ),
        '(01-02-03)(0-123)':   DIM * ( J * GGG ),
        '(01-02-03)(01-2-3)':  DIM * ( JG * G * G ),
        '(01-02-03)(01-23)':   DIM * ( JG * GG ),
        '(01-02-03)(012-3)':   DIM * ( JGG * G ),
        '(01-02-03)(0123)':    DIM * ( JGGG ),
        '(01-02-13)(0-1-2-3)': DIM * ( H**2 * G**2 ), 
        '(01-02-13)(0-1-23)':  DIM * ( H**2 * GG ), 
        '(01-02-13)(0-12-3)':  DIM * ( H * HG * G ), 
        '(01-02-13)(0-123)':   DIM * ( H * HGG ), 
        '(01-02-13)(0-13-2)':  DIM * ( H * HG * G ), 
        '(01-02-13)(01-2-3)':  DIM * ( HH * G**2 ), 
        '(01-02-13)(01-23)':   DIM * ( HH * GG ), 
        '(01-02-13)(012-3)':   DIM * ( HHG * G ), 
        '(01-02-13)(0123)':    DIM * ( HHGG ), 
        '(01-02-13)(02-13)':   DIM * ( HG * HG ), 
        '(01-02-13)(03-12)':   DIM * ( HG * HG ), 
    }

    grad_stats = str(compute_grad_stats(Q, N=4, I=3000))
    for name, stats in sorted(eval(grad_stats).items()):
        mean = stats["mean"]
        halfrange = 1.96 * stats["stdv"]/stats["nb_samples"]**0.5 
        print(CC + ' @C \t'.join([
            'stat @R {:24s}'.format(name),
            'measured @B {:+8.2f}'.format(mean - halfrange),
            'to @B {:+8.2f}'.format(mean + halfrange),
            'expected @Y {:8.2f}'.format(predictions[name]),
            ' @R violation! ' if halfrange < abs(predictions[name]-mean) else ' @G okay ',
        '']))


def test_derivator_on_quad():
    from quad_landscapes import Quadratic

    DIM = 256
    assert DIM%4==0
    DIM_4 = DIM//4

    A,B,C,D = 0.5, 1.0, 0.1, 0.3
    hessian    = torch.diag(torch.tensor([A]*DIM_4 + [A]*DIM_4 + [B]*DIM_4 + [B]*DIM_4))
    covariance = torch.diag(torch.tensor([C]*DIM_4 + [D]*DIM_4 + [D]*DIM_4 + [C]*DIM_4))
    Q = Quadratic(dim=DIM, hessian=hessian, covariance=covariance)

    predictions = {
        '()(0)': DIM_4 * (A + B),
        '(01)(0-1)': DIM_4 * (2*A**2 + 2*B**2),
        '(01)(01)':  DIM_4 * (2*A**2 + 2*B**2 + 2*C + 2*D),
        '(01-02)(0-1-2)': DIM_4 * (2*A**3 + 2*B**3),
        '(01-02)(0-12)':  DIM_4 * (2*A**3 + 2*B**3 + (A+B)*(C+D)),
        '(01-02)(01-2)':  DIM_4 * (2*A**3 + 2*B**3),
        '(01-02)(012)':   DIM_4 * (2*A**3 + 2*B**3 + (A+B)*(C+D)),
        '(01-02-03)(0-1-2-3)': 0.0,
        '(01-02-03)(0-1-23)':  0.0,
        '(01-02-03)(0-123)':   0.0,
        '(01-02-03)(01-2-3)':  0.0,
        '(01-02-03)(01-23)':   0.0,
        '(01-02-03)(012-3)':   0.0,
        '(01-02-03)(0123)':    0.0,
        '(01-02-13)(0-1-2-3)': DIM_4 * (2*A**4 + 2*B**4),
        '(01-02-13)(0-1-23)':  DIM_4 * (2*A**4 + 2*B**4 + (A**2+B**2)*(C+D)),
        '(01-02-13)(0-12-3)':  DIM_4 * (2*A**4 + 2*B**4),
        '(01-02-13)(0-123)':   DIM_4 * (2*A**4 + 2*B**4 + (A**2+B**2)*(C+D)),
        '(01-02-13)(0-13-2)':  DIM_4 * (2*A**4 + 2*B**4),
        '(01-02-13)(01-2-3)':  DIM_4 * (2*A**4 + 2*B**4),
        '(01-02-13)(01-23)':   DIM_4 * (2*A**4 + 2*B**4 + (A**2+B**2)*(C+D)),
        '(01-02-13)(012-3)':   DIM_4 * (2*A**4 + 2*B**4),
        '(01-02-13)(0123)':    DIM_4 * (2*A**4 + 2*B**4 + (A**2+B**2)*(C+D)),
        '(01-02-13)(02-13)':   DIM_4 * (2*A**4 + 2*B**4),
        '(01-02-13)(03-12)':   DIM_4 * (2*A**4 + 2*B**4),
    }

    grad_stats = str(compute_grad_stats(Q, N=4, I=3000))
    for name, stats in sorted(eval(grad_stats).items()):
        mean = stats["mean"]
        halfrange = 1.96 * stats["stdv"]/stats["nb_samples"]**0.5 
        print(CC + ' @C \t'.join([
            'stat @R {:24s}'.format(name),
            'measured @B {:+8.2f}'.format(mean - halfrange),
            'to @B {:+8.2f}'.format(mean + halfrange),
            'expected @Y {:8.2f}'.format(predictions[name]),
            ' @R violation! ' if halfrange < abs(predictions[name]-mean) else ' @G okay ',
        '']))

if __name__ == '__main__':

    from mnist_landscapes import MnistLogistic
    LC = MnistLogistic(digits=list(range(10)))
    #from quad_landscapes import Quadratic
    #LC = Quadratic(dim=12)

    grad_stats = str(compute_grad_stats(LC, N=10, I=10000))
    with open('gs.data', 'w') as f:
        f.write(grad_stats)


    #test_derivator_on_cosh()
    #test_derivator_on_quad()

