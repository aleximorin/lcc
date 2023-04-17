from numba_phase_correlation import translation
from velocity_field import RandomVelocityField
from numba_phase_correlation import windowed_phase_correlation

import imreg_dft as ird

import numpy as np
import matplotlib.pyplot as plt

import time


if __name__ == '__main__':
    # coordinate generation

    n = 100
    x = np.linspace(0, 100, n)
    X = np.array(np.meshgrid(x, x))

    # gstools can generate a random vector field
    vf = RandomVelocityField(X, var=5, len_scale=5, seed=42069)

    h1 = np.load('h1.npy')
    h2 = np.load('h2.npy')

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(h1, extent=[x.min(), x.max(), x.min(), x.max()], origin='lower')
    axs[1].imshow(h2, extent=[x.min(), x.max(), x.min(), x.max()], origin='lower')

    window_size = 25
    vx, vy = windowed_phase_correlation(h1, h2, window_size)
    subx, suby = vf.X[:, window_size // 2:-window_size // 2, window_size // 2:-window_size // 2]

    errx = vx - vf.V[1, window_size // 2:-window_size // 2, window_size // 2:-window_size // 2]
    erry = vy - vf.V[1, window_size // 2:-window_size // 2, window_size // 2:-window_size // 2]

    fig, axs = plt.subplots(2, 4, sharex='all', sharey='all', figsize=(15, 8))

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    axs[0, 0].imshow(h1, extent=[subx.min(), subx.max(), suby.min(), suby.max()])
    axs[1, 0].imshow(h2, extent=[subx.min(), subx.max(), suby.min(), suby.max()])
    axs[0, 1].imshow(vf.V[0], extent=[vf.X[0].min(), vf.X[0].max(), vf.X[1].min(), vf.X[1].max()])
    axs[0, 2].imshow(vx, extent=[subx.min(), subx.max(), suby.min(), suby.max()])
    axs[1, 1].imshow(vf.V[1], extent=[vf.X[0].min(), vf.X[0].max(), vf.X[1].min(), vf.X[1].max()])
    axs[1, 2].imshow(vy, extent=[subx.min(), subx.max(), suby.min(), suby.max()])
    axs[0, 3].imshow(errx, extent=[subx.min(), subx.max(), suby.min(), suby.max()])
    axs[1, 3].imshow(erry, extent=[subx.min(), subx.max(), suby.min(), suby.max()])

    plt.figure()
    plt.hist(errx.flatten(), 50, histtype='stepfilled', alpha=0.3)
    plt.hist(erry.flatten(), 50, histtype='stepfilled', alpha=0.3)

    plt.show()