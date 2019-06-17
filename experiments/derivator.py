''' author: samtenka
    change: 2019-06-16
    create: 2019-06-16
    descrp: define gradient statistics
'''

from abc import ABC, abstractmethod
from itertools import chain 
import numpy as np

from utils import CC
from landscape import PointedLandscape
import torch
import tqdm

grad_stat_names = set([
    '(0)()',
    '(0-1)(01)',
    '(01)(01)',
    '(0-1-2)(01-02)',
    '(0-12)(01-02)',
    '(0-12)(01-12)',
    '(012)(01-02)',
])

def compute_grad_stats(land, N, I=1):
    gs = {name:torch.Tensor([0.0]) for name in grad_stat_names}
    gs_= {name:None for name in grad_stat_names}
    for i in tqdm.tqdm(range(I)):
        A, B, C, D = (
            land.get_loss_stalk(land.sample_data(N))
            for i in range(4)
        )
        
        GA, GB, GC, GD = (
            land.nabla(X) 
            for X in (A, B, C, D)
        )

        gs_['(0)()']  = (
            A
        )

        gs_['(0-1)(01)']  = (
            GA.dot(GB)
        )
        gs_['(01)(01)']  = (
            gs_['(0-1)(01)'] * (1.0-N) +
            GA.dot(GA) * N  
        )

        gs_['(0-1-2)(01-02)']  = (
            (land.nabla(GA.dot(GB.detach()))).dot(GC)
        ) 
        gs_['(0-12)(01-02)']  = (
            gs_['(0-1-2)(01-02)'] * (1.0-N) +
            (land.nabla(GA.dot(GB.detach()))).dot(GB) * N 
        ) 
        gs_['(0-12)(01-12)']  = (
            gs_['(0-1-2)(01-02)'] * (1.0-N) +
            GA.dot(land.nabla(GB.dot(GB.detach()))) * N
        ) 
        gs_['(012)(01-02)']  = (
            gs_['(0-1-2)(01-02)'] * (1.0-N) +
            (land.nabla(GA.dot(GA.detach()))).dot(GA) * N
        )

        for name in gs.keys():
            gs[name] += gs_[name].detach()

    return {name: float(tensor.detach().numpy())/I for name, tensor in gs.items()}

if __name__ == '__main__':
    from quad_landscapes import Quadratic
    DIM = 8
    hessian = torch.eye(DIM) 
    hessian[:int(DIM/2)] *= 2
    Q = Quadratic(dim=DIM, hessian=hessian)
    gs = compute_grad_stats(Q, N=100, I=1000)
    for name, value in sorted(gs.items()):
        print(CC+'stat @R {:16s} @W \t measured @G {:+.2f} @W \t expected @Y {:.2f} @W '.format(
            name,
            value,
            {
                '(0)()': DIM * 3.0/2,
                '(0-1)(01)': 0.0,
                '(01)(01)': DIM * 5.0/2,
                '(0-1-2)(01-02)': 0.0,
                '(0-12)(01-02)': DIM * 9.0/2,
                '(0-12)(01-12)': 0.0,
                '(012)(01-02)': DIM * 9.0/2 
            }[name]
        ))
