import numpy as np
import matplotlib.pyplot as plt
import torch

from optim import ShiftOptim
from LCC2D_torch import LCC2D

if __name__ == '__main__':
    im0 = np.load('data/im1.npy', allow_pickle=True)
    im1 = np.load('data/im3.npy', allow_pickle=True)

    maxlag = 15
    wlags = np.arange(-maxlag, maxlag + 1)
    hlags = np.arange(-maxlag, maxlag + 1)
    search_sigma = 15

    lcc = LCC2D(im0, im1,
                hlags, wlags,
                search_sigma,
                threshold=1)

    shift = torch.stack((lcc.subdw, lcc.subdh)).to(torch.float64)

    lmbda = 1e3
    beta = 1e2
    model = ShiftOptim(lcc.f, lcc.g, shift, lmbda=lmbda, beta=beta, correlation=lcc.convolutions)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out, losses = model.fit(optimizer, n=2500)
    print()

    plt.figure()
    plt.plot(losses[:, 0], ls='dashed', label='$\epsilon$')
    plt.plot(losses[:, 1], ls='dashed', label=f'$\lambda$={lmbda:.1e}')
    plt.plot(losses[:, 2], ls='dashed', label=f'$\\beta$={beta:.1e}')
    plt.plot(losses.sum(axis=1), c='k', label='$\Sigma$')
    plt.fill_between(np.arange(len(losses)), 0, losses.sum(axis=1), alpha=0.3, fc='tab:grey', ec='k')
    plt.xlim(0, len(losses) - 1)
    plt.ylim(0, None)
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('loss')

    from image_warping import warp_cv2
    fig, axs = plt.subplots(1, 5, sharex='all', sharey='all', figsize=(14, 3))
    axs[0].imshow(im0, origin='lower')
    axs[0].text(0.99, 0.99, f'Original', ha='right', va='top', transform=axs[0].transAxes)

    axs[1].imshow(im1, origin='lower')
    axs[1].text(0.99, 0.99, f'Warped', ha='right', va='top', transform=axs[1].transAxes)

    axs[2].imshow(warp_cv2(im0, torch.stack((lcc.dw, lcc.dh)).numpy()), origin='lower')
    axs[2].text(0.99, 0.99, f'Integer disp.', ha='right', va='top', transform=axs[2].transAxes)

    axs[3].imshow(warp_cv2(im0, shift.numpy()), origin='lower')
    axs[3].text(0.99, 0.99, f'Subpixel disp.', ha='right', va='top', transform=axs[3].transAxes)

    axs[4].imshow(model.forward().detach(), origin='lower')
    axs[4].text(0.99, 0.99, f'Optimized disp.', ha='right', va='top', transform=axs[4].transAxes)

    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    fig, axs = plt.subplots(1, 5, sharex='all', sharey='all', figsize=(14, 3))
    axs[0].imshow(im0, origin='lower')
    axs[0].text(0.99, 0.99, f'Original', ha='right', va='top', transform=axs[0].transAxes)

    axs[1].imshow(im1, origin='lower')
    axs[1].text(0.99, 0.99, f'Warped', ha='right', va='top', transform=axs[1].transAxes)

    axs[2].imshow(warp_cv2(im0, torch.stack((lcc.dw, lcc.dh)).numpy()), origin='lower')
    axs[2].text(0.99, 0.99, f'Integer disp.', ha='right', va='top', transform=axs[2].transAxes)

    axs[3].imshow(warp_cv2(im0, shift.numpy()), origin='lower')
    axs[3].text(0.99, 0.99, f'Subpixel disp.', ha='right', va='top', transform=axs[3].transAxes)

    axs[4].imshow(model.forward().detach(), origin='lower')
    axs[4].text(0.99, 0.99, f'Optimized disp.', ha='right', va='top', transform=axs[4].transAxes)

    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
    axs[0, 0].imshow(shift[0], origin='lower', vmin=wlags.min(), vmax=wlags.max())
    axs[0, 0].text(0.99, 0.99, 'lcc $\hat{v}_x$', ha='right', va='top', transform=axs[0, 0].transAxes)
    axs[0, 1].imshow(shift[1], origin='lower', vmin=hlags.min(), vmax=hlags.max())
    axs[0, 1].text(0.99, 0.99, 'lcc $\hat{v}_y$', ha='right', va='top', transform=axs[0, 1].transAxes)
    axs[1, 0].imshow(out[0], origin='lower', vmin=wlags.min(), vmax=wlags.max())
    axs[1, 0].text(0.99, 0.99, 'optim. $\hat{v}_x$', ha='right', va='top', transform=axs[1, 0].transAxes)
    axs[1, 1].imshow(out[1], origin='lower', vmin=hlags.min(), vmax=hlags.max())
    axs[1, 1].text(0.99, 0.99, 'optim $\hat{v}_y$', ha='right', va='top', transform=axs[1, 1].transAxes)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.show()
