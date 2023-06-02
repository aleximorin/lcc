import numpy as np
import matplotlib.pyplot as plt
import torch
from optim import ShiftOptim
from LCC2D_torch import LCC2D

if __name__ == '__main__':
    path0 = '/home/amorin/GeoHawk_GPR/IceFlowGPR/depot_SIMU/fwd_t0/out.npy'
    path1 = '/home/amorin/GeoHawk_GPR/IceFlowGPR/depot_SIMU/fwd_t1/out.npy'

    im0 = np.load(path0, allow_pickle=True).T[:2100:5, ::3]
    im1 = np.load(path1, allow_pickle=True).T[:2100:5, ::3]

    print(im0.shape)

    maxlag = 40
    wlags = np.arange(-maxlag, maxlag + 1).astype(int)
    hlags = np.arange(-maxlag, maxlag + 1).astype(int)
    search_sigma = 18

    lcc = LCC2D(im0, im1,
                hlags, wlags,
                search_sigma,
                threshold=2)

    shift = torch.stack((lcc.subdw, lcc.subdh)).to(torch.float32)

    lmbda = 1e3
    beta = 0
    niter = 2e4

    model = ShiftOptim(lcc.f, lcc.g, shift, lmbda=lmbda, beta=beta, correlation=lcc.convolutions)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out, losses = model.fit(optimizer, n=int(niter))

    out_dict = {'lcc': lcc, 'out': out, 'losses': losses}

    import pickle

    with open('out.p', 'wb') as file:
        pickle.dump(out_dict, file)
    print()
    print('saving done')


"""
import matplotlib
matplotlib.use('Qt5Agg')
vmax = np.quantile(np.abs(im0), 0.90)

kws = dict(aspect='auto', vmin=-vmax, vmax=vmax, cmap='coolwarm')

DX = 0.10
DZ = 0.10
DT = 3e-10
maxT = 1.28e-6
NT = int(1.15 * maxT/DT)
CF = 80e6

eps = np.load('data/eps.npy', allow_pickle=True)

eps0 = eps[0]
eps1 = eps[1]

vmax = np.quantile(np.abs(im0), 0.9)

fig, axs = plt.subplots(1, 2, sharex='all', sharey='all')
axs[0].imshow(im0, extent=[100, 400, NT * DT, 0], vmin=-vmax, vmax=vmax, cmap='coolwarm')
axs[0].imshow(eps0, extent=[0, 500, maxT, 0], alpha=0.3, origin='lower')    

axs[1].imshow(im1, extent=[100, 400, NT * DT, 0], vmin=-vmax, vmax=vmax, cmap='coolwarm')
axs[1].imshow(eps1, extent=[0, 500, maxT, 0], alpha=0.3, origin='lower')

for ax in axs:
    ax.set_aspect('auto')

"""