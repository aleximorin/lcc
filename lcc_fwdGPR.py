import numpy as np
import matplotlib.pyplot as plt
import torch

from optim import ShiftOptim
from LCC2D_torch import LCC2D

if __name__ == '__main__':
    path0 = '/home/amorin/GeoHawk_GPR/IceFlowGPR/depot_SIMU/fwd_t0/out.npy'
    path1 = '/home/amorin/GeoHawk_GPR/IceFlowGPR/depot_SIMU/fwd_t1/out.npy'

    im0 = np.load(path0, allow_pickle=True).T
    im1 = np.load(path1, allow_pickle=True).T

    maxlag = 20
    wlags = np.arange(-maxlag, maxlag + 1)
    hlags = np.arange(-maxlag, maxlag + 1)
    search_sigma = 15

    lcc = LCC2D(im0, im1,
                hlags, wlags,
                search_sigma,
                threshold=2)

    shift = torch.stack((lcc.subdw, lcc.subdh)).to(torch.float32)

    lmbda = 1e2
    beta = 0
    niter = 1e4

    model = ShiftOptim(lcc.f, lcc.g, shift, lmbda=lmbda, beta=beta, correlation=lcc.convolutions)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out, losses = model.fit(optimizer, n=int(niter))

    out_dict = {'lcc': lcc, 'out': out, 'losses': losses}

    import pickle

    with open('out.p', 'wb') as file:

        pickle.dump(out_dict, file)