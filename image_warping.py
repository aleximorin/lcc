from velocity_field import RandomVelocityField

import numpy as np
import matplotlib.pyplot as plt
import cv2

from time import time

from local_crosscorrelations import mp_windowed_phase_correlation,\
    mp_windowed_cross_correlation,\
    serial_windowed_cross_correlation

from twod_hale_localcc import windowed_2d_cc, cyclic_search2d
from scipy.interpolate import RectBivariateSpline


def warp_cv2(im, u, fact=1):
    # f[j1, j2] = g(j1 + u1[j1, j2], j2 + u2[j1, j2])
    h, w = im.shape
    h, w = np.arange(h), np.arange(w)
    w, h = np.meshgrid(w, h)
    mapx, mapy = w - u[0] * fact, h - u[1] * fact
    warp = cv2.remap(im, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR)
    return warp


def warp(im, u, t=1):

    h, w = im.shape
    h = np.arange(h)
    w = np.arange(w)
    ww, hh = np.meshgrid(w, h)

    f = RectBivariateSpline(w, h, im.T)
    ww2, hh2 = np.array((ww, hh)) - t*u

    im2 = f(ww2.flatten(), hh2.flatten(), grid=False).reshape(ww2.shape)
    return im2


def phase_corr_image(im, var=5, lengthscale=25, interpolate_step=None,
                     fact=1, sigma=12, seed=None):
    h, w = im.shape
    h, w = np.arange(h), np.arange(w)
    ww, hh = np.meshgrid(w, h)

    vf = RandomVelocityField(np.array((ww, hh)),
                             var=var,
                             len_scale=lengthscale,
                             seed=seed,
                             interpolate_step=interpolate_step)

    warp = warp_cv2(im, vf.V, fact=fact)

    lags1 = np.arange(int(-2*sigma/2), int(2*sigma/2) + 1)

    t0 = time()
    corr, (vx, vy) = windowed_2d_cc(warp, im, lags1, lags1, sigma=sigma)
    #_, (vx, vy) = cyclic_search2d(im, warp, lags1, lags1, sigma, 10)
    t1 = time()

    return vf, warp, vx, vy, t1-t0


def phase_corr_plot(vf, im, warp, vx, vy, sigma):
    h, w = im.shape
    w, h = np.arange(w), np.arange(h)
    window_size = 3 * sigma
    window_size = window_size + 1 if window_size % 2 == 0 else window_size

    fig, axs = plt.subplots(2, 3, sharex='all', sharey='all', figsize=(16, 6))

    # we plot the vector field over the warped images
    vf.plot(ax=axs[0, 0])
    vf.plot(ax=axs[1, 0])

    # we plot the original and the warped image
    axs[0, 0].imshow(im, origin='lower', extent=[w[0], w[-1], h[0], h[-1]], cmap='gray')
    axs[1, 0].imshow(warp, origin='lower', extent=[w[0], w[-1], h[0], h[-1]], cmap='gray')
    axs[0, 0].text(0.99, 1.01, 'Original image', ha='right', va='bottom', transform=axs[0, 0].transAxes)
    axs[1, 0].text(0.99, 1.01, 'Warped image', ha='right', va='bottom', transform=axs[1, 0].transAxes)

    # we plot the vx and vy components of the vector field
    axs[0, 1].imshow(vf.V[0], origin='lower', extent=[w[0], w[-1], h[0], h[-1]])
    axs[1, 1].imshow(vf.V[1], origin='lower', extent=[w[0], w[-1], h[0], h[-1]])
    axs[0, 1].text(0.99, 1.01, '$v_x$', ha='right', va='bottom', transform=axs[0, 1].transAxes)
    axs[1, 1].text(0.99, 1.01, '$v_y$', ha='right', va='bottom', transform=axs[1, 1].transAxes)

    subw = window_size//2
    # we plot the recovered vx and vy displacement from the phase correlation
    axs[0, 2].imshow(vx, origin='lower', extent=[w[subw], w[-subw], h[subw], h[-subw]],
                     vmin=np.quantile(vx, 0.01), vmax=np.quantile(vx, 0.99))
    axs[1, 2].imshow(vy, origin='lower', extent=[w[subw], w[-subw], h[subw], h[-subw]],
                     vmin=np.quantile(vy, 0.01), vmax=np.quantile(vy, 0.99))
    axs[0, 2].text(0.99, 1.01, '$\hat{v}_x$', ha='right', va='bottom', transform=axs[0, 2].transAxes)
    axs[1, 2].text(0.99, 1.01, '$\hat{v}_y$', ha='right', va='bottom', transform=axs[1, 2].transAxes)

    fig.tight_layout()
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    return fig, axs


def maxcorr(im, im_crop, sigma):
    window_size = 3 * sigma
    window_size = window_size + 1 if window_size % 2 == 0 else window_size
    subw = window_size // 2

    h, w = im.shape
    a = im[subw + 1:h - subw, subw + 1:w - subw]
    corr = np.correlate(a.flatten()/np.linalg.norm(im),
                        im_crop.flatten()/np.linalg.norm(im_crop))
    return corr


if __name__ == '__main__':
    """# simple horse example
    im = plt.imread('horse.png')[::-1, :, -1]
    #phase_corr_image(im, fact=10, seed=42069, window_size=16)"""

    from time import time

    # spongebob example
    im = plt.imread('data/spongebob.png')[::-1, :, 0]

    from scipy import ndimage
    ndimage.gaussian_filter(im, 3)
    lpf = ndimage.gaussian_filter(im, 3)
    hpf = im - lpf

    im = hpf

    sigma = 30

    vf, warp, vx, vy, t = phase_corr_image(im, var=1, lengthscale=25,  # vector field generation parameters
                                        interpolate_step=5,  # we're interpolating the vector field for large images
                                        fact=10,  # warping parameter, how much the image is deformed
                                        sigma=sigma,  # phase correlation window parameter
                                        seed=42069)  # random seed

    print(f'Took {t:.2f} seconds')
    fig, axs = phase_corr_plot(vf, im, warp, vx, vy, sigma=sigma)
    plt.show()

"""
for fact in [0, 3, 5, 10]:
    g = warp_image(im, vf.V, fact=fact)
    path = f'data/spongebob{fact}.npy'
    plt.figure()
    plt.imshow(g)
    g.dump(path)

vf.V.dump('data/vf.npy')
"""

"""
n = 1001
im = np.zeros((n, n))

im[::50] = 1
im[1::50] = 1
im[2::50] = 1
im[3::50] = 1

im[:, ::50] = 1
im[:, 1::50] = 1
im[:, 2::50] = 1
im[:, 3::50] = 1

t = np.linspace(-n/2, n/2, n)
xx, yy = np.meshgrid(t, t)

ux = np.cos(np.arctan2(yy, xx))
uy = np.sin(np.arctan2(yy, xx))

gauss = np.exp(-(t**2)/1e4)
gauss = np.outer(gauss, gauss)
u = gauss * np.stack((ux, uy))


warped = warp_cv2(im, u, fact=50)
fig, axs = plt.subplots(1, 2)#, sharex='all', sharey='all')
axs[0].imshow(im, cmap='gray_r', extent=[-n/2, n/2, -n/2, n/2], origin='lower')
#saxs[0].quiver(xx[::50, ::50], yy[::50, ::50], u[0, ::50, ::50], u[1, ::50, ::50], scale=5e0)
axs[0].set_title('original image', loc='left')

axs[1].imshow(warped, cmap='gray_r', extent=[-n/2, n/2, -n/2, n/2], origin='lower')
#axs[1].quiver(xx[::50, ::50], yy[::50, ::50], u[0, ::50, ::50], u[1, ::50, ::50], scale=5e0)
axs[1].set_title('warped image', loc='left')

for ax in axs:
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    
fig.subplots_adjust(hspace=0, wspace=0.05)
fig.savefig('warping.png', dpi=800, bbox_inches='tight')
"""