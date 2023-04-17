import numpy as np
import matplotlib.pyplot as plt
import imreg_dft as ird
from scipy.signal import correlate, convolve, fftconvolve


# gaussian filter implementation
# could be implemented recursively to 2d?
# could also use np.outer?
def gaussian_filter(kernel_size, sigma=1):
    x = np.arange(kernel_size) - kernel_size / 2
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x ** 2 + y ** 2)
    gauss = np.exp(-(r / sigma) ** 2)
    gauss = gauss / gauss.sum()
    return gauss


# CROSS-CORRELATIONS #######################################################################
def cross_corr_translation(a1, a2):
    corr = fftconvolve(a1, a2[::-1, ::-1])
    I, J = _translation(corr)
    Ihat, Jhat = _poly_interp(corr, I, J)
    return Ihat, Jhat


def _local_cross_correlation(h1, h2, center_i, center_j, g):
    w, _ = g.shape
    dw = w // 2

    ilow = center_i - dw
    iup = center_i + dw + 1

    jleft = center_j - dw
    jright = center_j + dw + 1

    slice_ = np.s_[jleft:jright, ilow:iup]
    dx, dy = cross_corr_translation(h1[slice_], h2[slice_])
    return dy, dx


def mp_windowed_cross_correlation(h1, h2, sigma=12):
    window_size = 3 * sigma
    window_size = window_size + 1 if window_size % 2 == 0 else window_size
    g = gaussian_filter(window_size, sigma=sigma)
    ni, nj = np.array(h1.shape) - window_size

    h1 = h1/np.sqrt(convolve(h1*h1, g, mode='same'))
    h2 = h2/np.sqrt(convolve(h2*h2, g, mode='same'))

    from multiprocessing import Pool

    K = np.arange(ni * nj)
    I = K // ni + window_size // 2  # ypos or row
    J = K % ni + window_size // 2  # xpos or col

    pool = Pool()
    generator = ((h1, h2, I[k], J[k], g) for k in range(ni * nj))
    results = pool.starmap(_local_cross_correlation, generator)
    vx, vy = np.array(results).T

    return vx.reshape((ni, nj), order='F'), vy.reshape((ni, nj), order='F')


def serial_windowed_cross_correlation(h1, h2, sigma=12, normalized=True):
    window_size = 3 * sigma
    window_size = window_size + 1 if window_size % 2 == 0 else window_size
    g = gaussian_filter(window_size, sigma=sigma)
    ni, nj = np.array(h1.shape) - window_size

    ch1 = 1/np.sqrt(convolve(h1*h1, g, mode='same'))
    ch2 = 1/np.sqrt(convolve(h2*h2, g, mode='same'))

    if normalized:
        nh1 = h1 * ch1
        nh2 = h2 * ch2
    else:
        nh1 = h1 * 1
        nh2 = h2 * 1

    vx = np.zeros(ni * nj)
    vy = np.zeros(ni * nj)

    K = np.arange(ni * nj)
    I = K // ni + window_size // 2  # ypos or row
    J = K % ni + window_size // 2  # xpos or col
    nk = ni * nj
    for k in range(nk):
        print(f'\rk {k}/{nk} = {k / nk * 100:.2f}%', end='')
        dx, dy = _local_cross_correlation(nh1, nh2, I[k], J[k], g)
        vx[k] = dx
        vy[k] = dy
    return vx.reshape(ni, nj), vy.reshape(ni, nj)


def visualise_windowed_cross_correlation(h1, h2, sigma=12, normalized=True):
    window_size = 3 * sigma
    window_size = window_size + 1 if window_size % 2 == 0 else window_size
    g = gaussian_filter(window_size, sigma=sigma)
    ni, nj = np.array(h1.shape) - window_size

    ch1 = 1/np.sqrt(convolve(h1*h1, g, mode='same'))
    ch2 = 1/np.sqrt(convolve(h2*h2, g, mode='same'))

    if normalized:
        nh1 = h1 * ch1
        nh2 = h2 * ch2
    else:
        nh1 = h1 * 1
        nh2 = h2 * 1

    vx = np.zeros(ni * nj)
    vy = np.zeros(ni * nj)

    K = np.arange(ni * nj)
    I = K // ni + window_size // 2  # ypos or row
    J = K % ni + window_size // 2  # xpos or col
    nk = ni * nj

    correlations = np.zeros((nk, window_size*2 - 1, window_size*2 - 1))

    for k in range(nk):
        print(f'\rk {k}/{nk} = {k / nk * 100:.2f}%', end='')
        w, _ = g.shape
        dw = w // 2

        center_i, center_j = I[k], J[k]

        ilow = center_i - dw
        iup = center_i + dw + 1

        jleft = center_j - dw
        jright = center_j + dw + 1

        subh1 = nh1[jleft:jright, ilow:iup] * g
        subh2 = nh2[jleft:jright, ilow:iup] * g

        corr = fftconvolve(subh2, subh1[::-1, ::-1])  # , method='fft')
        i, j = _translation(corr)
        dy, dx = _poly_interp(corr, i, j)

        correlations[k] = corr

        vx[k] = dx
        vy[k] = dy

    return vx.reshape(ni, nj), vy.reshape(ni, nj)


def argmax2d(a):
    w, h = np.array(a.shape)
    index = a.argmax()
    return index // h - w // 2, index % h - h // 2

import itertools
def polyfit2d(x, y, z, order=3, product=False):
    # https://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent

    if product:
        ij = itertools.product(range(order+1), range(order+1))
    else:
        ij = np.vstack((np.arange(order + 1), np.zeros(order + 1)))
        ij = np.hstack((ij, np.roll(ij[:, 1:], 1, axis=0))).T

    ncols = (order + 1)**2 if product else len(ij)
    G = np.zeros((x.size, ncols))

    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z, rcond=1e-10)
    return m


def polyval2d(x, y, m, order, product=False):
    # https://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent
    if product:
        ij = itertools.product(range(order+1), range(order+1))
    else:
        ij = np.vstack((np.arange(order + 1), np.zeros(order + 1)))
        ij = np.hstack((ij, np.roll(ij[:, 1:], 1, axis=0))).T

    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z = z + a * x**i * y**j
    return z


def _poly_interp(corr, I, J, threshold=2):
    H, W = corr.shape
    h, w = np.arange(H) - H // 2, np.arange(W) - W // 2
    xx, yy = np.meshgrid(h, w)

    threshold = 3
    ii = (np.abs(xx - J) <= threshold) & (np.abs(yy - I) <= threshold)

    m = polyfit2d(xx[ii].flatten(), yy[ii].flatten(), corr[ii].flatten(), order=2, product=False)
    Ihat = -0.5 * m[1] / m[2]
    Jhat = -0.5 * m[3] / m[4]

    """m = polyfit2d(xx[ii].flatten(), yy[ii].flatten(), corr[ii].flatten(), order=2, product=False)
    corr_hat = polyval2d(xx.flatten(), yy.flatten(), m, order=2, product=False).reshape(corr.shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.plot_surface(xx, yy, corr, alpha=0.3)
    zmax = polyval2d(Ihat, Jhat, m, order=2, product=False)
    ax.scatter(xx[ii], yy[ii], corr[ii])
    ax.scatter(xx[ii], yy[ii], corr_hat[ii])
    ax.scatter(J, I, corr.flatten()[corr.argmax()], c='red', marker='*', s=100)
    ax.scatter(Ihat, Jhat, zmax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')"""
    return Jhat, Ihat


def _translation(corr):
    I, J = argmax2d(corr)
    return I, J


# PHASE-CORRELATIONS #######################################################################
def phase_corr_translation(a1, a2):
    corr = phase_correlation(a1, a2)
    I, J = _translation(corr)
    return I, J

def _local_phase_correlation(h1, h2, center_i, center_j, g):
    w, _ = g.shape
    dw = w // 2

    ilow = center_i - dw
    iup = center_i + dw + 1

    jleft = center_j - dw
    jright = center_j + dw + 1

    subh1 = h1[jleft:jright, ilow:iup] * g
    subh2 = h2[jleft:jright, ilow:iup] * g

    dx, dy = phase_corr_translation(subh2, subh1)
    return dy, dx


def phase_correlation(f, g):
    #  From Hale (2007)
    F = np.fft.fft2(f)
    G = np.fft.fft2(g)

    eps = max(1e-15 * np.abs(f).max(), 1e-15)

    P = F * G.conj()
    P = P / (np.abs(P) + eps)

    p = np.fft.ifft2(P)
    p = np.fft.fftshift(p)

    return p.real


def serial_windowed_phase_correlation(h1, h2, sigma=12):
    window_size = 3 * sigma
    window_size = window_size + 1 if window_size % 2 == 0 else window_size
    g = gaussian_filter(window_size, sigma=sigma)
    ni, nj = np.array(h1.shape) - window_size

    vx = np.zeros(ni * nj)
    vy = np.zeros(ni * nj)

    K = np.arange(ni * nj)
    I = K // ni + window_size // 2  # ypos or row
    J = K % ni + window_size // 2  # xpos or col
    nk = ni * nj
    for k in range(nk):
        print(f'\rk {k}/{nk} = {k / nk * 100:.2f}%', end='')
        dx, dy = _local_phase_correlation(h1, h2, I[k], J[k], g)
        vx[k] = dx
        vy[k] = dy
    print()
    return vx.reshape(ni, nj), vy.reshape(ni, nj)


def mp_windowed_phase_correlation(h1, h2, sigma=12):
    window_size = 3 * sigma
    window_size = window_size + 1 if window_size % 2 == 0 else window_size
    g = gaussian_filter(window_size, sigma=sigma)
    ni, nj = np.array(h1.shape) - window_size

    from multiprocessing import Pool

    K = np.arange(ni * nj)
    I = K // ni + window_size // 2  # ypos or row
    J = K % ni + window_size // 2  # xpos or col

    pool = Pool()
    generator = ((h1, h2, I[k], J[k], g) for k in range(ni * nj))
    results = pool.starmap(_local_phase_correlation, generator)
    vx, vy = np.array(results).T

    return vx.reshape((ni, nj), order='F'), vy.reshape((ni, nj), order='F')


def vec_windowed_phase_correlation(h1, h2, sigma=12):
    window_size = 3 * sigma
    window_size = window_size + 1 if window_size % 2 == 0 else window_size
    g = gaussian_filter(window_size, sigma=sigma)

    from numpy.lib.stride_tricks import sliding_window_view

    subh1 = sliding_window_view(h1, (window_size, window_size)) * g
    subh2 = sliding_window_view(h2, (window_size, window_size)) * g

    # vectorized phase correlation
    f1 = np.fft.fft2(subh1)
    f2 = np.fft.fft2(subh2)
    eps = (np.abs(f1.max(axis=(-1, -2))) * 1e-15 + 1e-15)[:, :, None, None]
    P = f1 * f2.conj()
    P = P / (np.abs(P) + eps)
    p = np.fft.ifft2(P)
    p = np.fft.fftshift(p).real

    dw = window_size // 2
    p = p.reshape(-1, window_size, window_size)
    vx, vy = np.zeros(shape=(2, len(p)))

    for i in range(len(p)):
        index = p[i].argmax()
        vx[i] = index % window_size - dw
        vy[i] = index // window_size - dw

    w, h = P.shape[:2]
    return vx.reshape(w, h), vy.reshape(w, h)

def vec_windowed_cross_correlation(h1, h2, sigma=12):
    window_size = 3 * sigma
    window_size = window_size + 1 if window_size % 2 == 0 else window_size
    g = gaussian_filter(window_size, sigma=sigma)

    from numpy.lib.stride_tricks import sliding_window_view

    subh1 = sliding_window_view(h1, (window_size, window_size)) * g
    subh2 = sliding_window_view(h2, (window_size, window_size)) * g

    # vectorized phase correlation
    f1 = np.fft.fft2(subh1)
    f2 = np.fft.fft2(subh2)
    P = f1 * f2.conj()
    p = np.fft.ifft2(P)
    p = np.fft.fftshift(p).real

    dw = window_size // 2
    p = p.reshape(-1, window_size, window_size)
    vx, vy = np.zeros(shape=(2, len(p)))

    for i in range(len(p)):
        index = p[i].argmax()
        vx[i] = index % window_size - dw
        vy[i] = index // window_size - dw

    w, h = P.shape[:2]
    return vx.reshape(w, h), vy.reshape(w, h)



if __name__ == '__main__':
    """# simple example with the horse
    im = plt.imread('data/horse.png')[..., -1]
    im2 = plt.imread('data/horse_translated.png')[..., -1]

    # we get approximately the same value as the python library
    dx, dy = ird.translation(im, im2)['tvec']
    dx2, dy2 = cross_corr_translation(im, im2)"""

    # can we do localized phase corr????
    from time import time

    im = plt.imread('data/spongebob.png')[::-1, :, 0]
    im2 = np.load('data/warp_sponge.np', allow_pickle=True)
    print('Starting mp func')
    t0 = time()
    vec_windowed_cross_correlation(im, im2, sigma=12)
    serial_windowed_cross_correlation(im, im2, sigma=12)
    out2 = mp_windowed_cross_correlation(im, im2, sigma=12)

    t1 = time()
    print(f'Took {t1 - t0:.2f} seconds')


"""
x = np.linspace(0, 100, 1000)
x, y = np.meshgrid(x, x)
dt = x[1] - x[0]

z = np.sin(2*np.pi*x/100) + np.sin(2*np.pi*y/50)  + 0.1 * np.random.randn(*x.shape)

sigma = 3
xg = np.arange(3*sigma)
xg = xg - xg.mean()
g = np.exp(-(xg/sigma)**2)/sigma/np.sqrt(2*np.pi)
g = np.outer(g, g)

from scipy.signal import correlate, convolve
smooth = convolve(z, g, mode='same')
conv = convolve(z, z[::-1, ::-1], mode='same')
corr = correlate(z, z, mode='same')


fig, axs = plt.subplots(2, 2, sharex='col', sharey='col')
axs[0, 0].set_aspect(1)
axs[0, 0].imshow(z)
axs[0, 1].imshow(smooth)
axs[1, 0].imshow(conv)
axs[1, 1].imshow(corr)
"""

"""
n = 10
fig, axs = plt.subplots(n, n, sharex='all', sharey='all', figsize=(9.8, 9.8))
corr2 = correlations.reshape(ni, nj, window_size*2-1, window_size*2-1, order='F')
for i in range(n):
    for j in range(n):
        axs[i, j].imshow(corr2[i, j])
        axs[i, j].axis('off')
fig.subplots_adjust(hspace=0, wspace=0)
axs[0, 0].set_xticks([])
axs[0, 0].set_yticks([])
"""