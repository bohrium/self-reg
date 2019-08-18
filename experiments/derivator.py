''' author: samtenka
    change: 2019-06-16
    create: 2019-06-16
    descrp: define gradient statistics
'''

import numpy as np

from utils import CC
from landscape import PointedLandscape
from gradstats import grad_stat_names, GradStats
import torch
import tqdm

def compute_grad_stats(land, N, I=1):
    gs = GradStats()
    for i in tqdm.tqdm(range(I)):
        land.reset_weights()
        A, B, C, D = (
            land.get_loss_stalk(land.sample_data(N))
            #land.weights.pow(2).sum()
            for i in range(4)
        )
        
        GA, GB, GC, GD = (
            land.nabla(X) 
            for X in (A, B, C, D)
        )
        GA_, GB_, GC_, GD_ = (
            torch.Tensor(Gi.detach().numpy())
            for Gi in (GA, GB, GC, GD)
        )

        gs.accum('(0)()', (
            (A+B+C+D)/4
        ))

        gs.accum('(0-1)(01)', (
            (GA.dot(GB) + GC.dot(GD))/2
        ))
        gs.accum('(01)(01)', (
            (GA.dot(GA)+GB.dot(GB)+GC.dot(GC)+GD.dot(GD))/4 * N  
            + gs.recent('(0-1)(01)') * (1.0-N)
        ))

        gs.accum('(0-1-2)(01-02)', (
            (land.nabla(GA.dot(GB_))).dot(GC)
        )) 
        gs.accum('(0-12)(01-02)', (
            ((land.nabla(GA.dot(GB_))).dot(GB) +
             (land.nabla(GC.dot(GD_))).dot(GD))/2 * N 
            + gs.recent('(0-1-2)(01-02)') * (1.0-N)
        ))
        gs.accum('(0-12)(01-12)', (
            (GA.dot(land.nabla(GB.dot(GB_))) +
             GC.dot(land.nabla(GD.dot(GD_))))/2 * N
            + gs.recent('(0-1-2)(01-02)') * (1.0-N)
        ))
        gs.accum('(012)(01-02)', (
            ((land.nabla(GA.dot(GA_))).dot(GA) +
             (land.nabla(GB.dot(GB_))).dot(GB) +
             (land.nabla(GC.dot(GC_))).dot(GC) +
             (land.nabla(GD.dot(GD_))).dot(GD))/4 * N
            + gs.recent('(0-1-2)(01-02)') * (1.0-N)
        ))

        #
        # -T-O-D-O-: figure out why 3rd order computations complain:
        #   RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # --- something to do with `detach`?
        # FIXED!

        gs.accum('(0-1-2-3)(01-02-03)', (
            land.nabla(land.nabla(GA.dot(GB_)).dot(GC_)).dot(GD)
        ))

        #gs.accum('(0-1-2-3)(01-02-13)', (
        #    land.nabla(GA.dot(land.nabla(GB.dot(GD.detach())).detach())).dot(GC)
        #))
        #gs.accum('(0-1-23)(01-02-03)', (
        #    land.nabla(land.nabla(GA.dot(GB.detach())).dot(GC.detach())).dot(GC.detach()) * N
        #    + gs.recent('(0-1-2-3)(01-02-03)') * (1.0-N)
        #))

        #gs.accum('(0-1-23)(01-02-13)', (
        #))
        #gs.accum('(0-1-23)(01-02-23)', (
        #))
        #gs.accum('(0-1-23)(02-03-12)', (
        #))
        #gs.accum('(0-1-23)(02-12-23)', (
        #))
        #gs.accum('(0-1-23)(02-13-23)', (
        #))
        #gs.accum('(0-123)(01-02-03)', (
        #))
        #gs.accum('(0-123)(01-02-13)', (
        #))
        #gs.accum('(0-123)(01-12-13)', (
        #))
        #gs.accum('(0-123)(01-12-23)', (
        #))
        #gs.accum('(01-23)(01-02-03)', (
        #))
        #gs.accum('(01-23)(01-02-13)', (
        #))
        #gs.accum('(01-23)(01-02-23)', (
        #))
        #gs.accum('(01-23)(02-03-12)', (
        #))
        #gs.accum('(0123)(01-02-03)', (
        #))
        #gs.accum('(0123)(01-02-13)', (
        #))

    return gs

if __name__ == '__main__':

    #from mnist_landscapes import MnistLogistic
    #LC = MnistLogistic(digits=list(range(10)))
    ##from quad_landscapes import Quadratic
    ##LC = Quadratic(dim=12)

    #grad_stats = str(compute_grad_stats(LC, N=10, I=10000))
    #with open('gs.data', 'w') as f:
    #    f.write(grad_stats)

    from quad_landscapes import Quadratic

    DIM = 8
    #hessian = torch.eye(DIM) 
    #hessian[:int(DIM/2)] *= 2
    hessian=None
    Q = Quadratic(dim=DIM, hessian=hessian)
    grad_stats = str(compute_grad_stats(Q, N=30, I=1000))
    #with open('gs.data', 'w') as f:
    #    f.write(grad_stats)
    for name, stats in sorted(eval(grad_stats).items()):
        print(CC + ' @C \t'.join([
            'stat @R {:16s}'.format(name),
            'measured @G {:+.2f}'.format(stats["mean"] - 1.96 * stats["stdv"]/stats["nb_samples"]**0.5),
            'to @G {:+.2f}'.format(stats["mean"] + 1.96 * stats["stdv"]/stats["nb_samples"]**0.5),
            'expected @Y {:.2f}'.format({
                '(0)()': DIM/2 * (3.0/2 + 3.0/2)  ,
                '(0-1)(01)': DIM/2 * (5.0),
                '(01)(01)': DIM/2 * (5.0 + 5.0),
                '(0-1-2)(01-02)': DIM/2 * (9.0),
                '(0-12)(01-02)': DIM/2 * (9.0 + 9.0),
                '(0-12)(01-12)': DIM/2 * (9.0),
                '(012)(01-02)': DIM/2 * (9.0 + 9.0),
                '(0-1-2-3)(01-02-03)': 0.0,
                #'(0-1-2-3)(01-02-13)': 0.0,
                #'(0-1-23)(01-02-03)': 0.0,
            }[name]),
        '']))
