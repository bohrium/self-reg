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
        A, B, C, D = (
            land.get_loss_stalk(land.sample_data(N))
            for i in range(4)
        )
        
        GA, GB, GC, GD = (
            land.nabla(X) 
            for X in (A, B, C, D)
        )

        gs.accum('(0)()', (
            (A+B+C+D)/4
        ))

        gs.accum('(0-1)(01)', (
            (GA.dot(GB) + GC.dot(GD))/2
        ))
        gs.accum('(01)(01)', (
            gs.recent('(0-1)(01)') * (1.0-N) +
            (GA.dot(GA)+GB.dot(GB)+GC.dot(GC)+GD.dot(GD))/4 * N  
        ))

        gs.accum('(0-1-2)(01-02)', (
            (land.nabla(GA.dot(GB.detach()))).dot(GC)
        )) 
        gs.accum('(0-12)(01-02)', (
            gs.recent('(0-1-2)(01-02)') * (1.0-N) +
            ((land.nabla(GA.dot(GB.detach()))).dot(GB) +
             (land.nabla(GC.dot(GD.detach()))).dot(GD))/2 * N 
        ))
        gs.accum('(0-12)(01-12)', (
            gs.recent('(0-1-2)(01-02)') * (1.0-N) +
            (GA.dot(land.nabla(GB.dot(GB.detach())))+
             GC.dot(land.nabla(GD.dot(GD.detach()))))/2 * N
        ))
        gs.accum('(012)(01-02)', (
            gs.recent('(0-1-2)(01-02)') * (1.0-N) +
            ((land.nabla(GA.dot(GA.detach()))).dot(GA) +
             (land.nabla(GB.dot(GB.detach()))).dot(GB) +
             (land.nabla(GC.dot(GC.detach()))).dot(GC) +
             (land.nabla(GD.dot(GD.detach()))).dot(GD))/4 * N
        ))

    return gs

if __name__ == '__main__':
    from quad_landscapes import Quadratic
    DIM = 2
    hessian = torch.eye(DIM) 
    hessian[:int(DIM/2)] *= 2
    Q = Quadratic(dim=DIM, hessian=hessian)
    grad_stats = eval(str(compute_grad_stats(Q, N=30, I=10000)))
    for name, stats in sorted(grad_stats.items()):
        print(CC+'stat @R {:16s} @W \t measured @G {:+.2f} @W - @G {:+.2f} @W \t expected @Y {:.2f} @W '.format(
            name,
            stats["mean"] - 1.96 * stats["stdv"]/stats["nb_samples"]**0.5,
            stats["mean"] + 1.96 * stats["stdv"]/stats["nb_samples"]**0.5,
            {
                '(0)()': DIM/2 * (3.0/2 + 3.0/2)  ,
                '(0-1)(01)': DIM/2 * (5.0),
                '(01)(01)': DIM/2 * (5.0 + 5.0),
                '(0-1-2)(01-02)': DIM/2 * (9.0),
                '(0-12)(01-02)': DIM/2 * (9.0 + 9.0),
                '(0-12)(01-12)': DIM/2 * (9.0),
                '(012)(01-02)': DIM/2 * (9.0 + 9.0)
            }[name]
        ))
