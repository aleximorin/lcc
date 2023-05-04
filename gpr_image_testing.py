import numpy as np
import pickle

from LCC2D_torch import LCC2D
from optim import ShiftOptim

import torch


if __name__ == '__main__':
    
    
    niter = 12500
    
    import os
    savepath = 'gpr_gridsearch//'
    try:
        os.makedirs(savepath)
    except OSError:
        pass
        
    https://wp.unil.ch/eureka/concours-fns-images-scientifiques-2023/
    # loading the data
    im0 = np.load('data/gpr_slice1.npy', allow_pickle=True)
    im1 = np.load('data/gpr_slice2.npy', allow_pickle=True)
    
    # we take a section of the data and normalize it between 1 and -1
    im0 = im0[150:, 10:-5]
    im0 = 2 * (im0 - im0.min())/(im0.max() - im0.min()) - 1

    im1 = im1[150:, 10:-5]
    im1 = 2 * (im1 - im1.min())/(im1.max() - im1.min()) - 1
    
    maxlag = 30
    wlags = np.arange(-maxlag, maxlag + 1)
    hlags = np.arange(-maxlag, maxlag + 1)
    
    sigmas = [20]
    
    lambdas = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    betas = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    
    for search_sigma in sigmas:
       
        lcc = LCC2D(im0, im1,
                    hlags, wlags,
                    search_sigma,
                    threshold=1)
                
        shift = torch.stack((lcc.subdw, lcc.subdh)).to(torch.float64)

        for lmbda in lambdas:
            for beta in betas:

                model = ShiftOptim(lcc.f, lcc.g, shift, lmbda=lmbda, beta=beta, correlation=lcc.convolutions)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                out, losses = model.fit(optimizer, n=niter)
                
                savedict = {'out': out, 'loss': losses}
                fp = savepath + f'sigma{search_sigma}_lmbda{lmbda:.0e}_beta{beta:.0e}.p'
                with open(fp, 'wb') as f:
                    pickle.dump(savedict, f)
                    
                
                
                
                
