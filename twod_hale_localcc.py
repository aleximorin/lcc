import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import itertools
from sinc_interp import sinc_interp
import poly_interp as poly
from scipy.ndimage import convolve1d


def prediction_error_filter(im, window):
    R0 = cfg(im, im, 0, 0, window)
    Rlag = cfg(im, im, 1, -1, window)

    r1 = cfg(im, im, -1, 0, window)
    r2 = cfg(im, im, 0, -1, window)

    height, width = im.shape

    e = np.zeros((height, width))
    for k1 in range(1, height - 1):
        for k2 in range(1, width - 1):

            R = np.array([[R0[k1 - 1, k2], Rlag[k1 - 1, k2]],
                          [Rlag[k1 - 1, k2], R0[k1, k2 - 1]]])

            r = np.array([r1[k1, k2],
                          r2[k1, k2]])

            a1, a2 = np.linalg.solve(R, r)
            e[k1, k2] = im[k1, k2] - a1 * im[k1 - 1, k2] - a2 * im[k1, k2 - 1]

    return e





"""
height, width = corr.shape
h, w = np.arange(height), np.arange(width)
xx, yy = np.meshgrid(h, w)

threshold = 1
ii = (np.abs(xx - J) <= threshold) & (np.abs(yy - I) <= threshold)
N = 50
x2 = np.linspace(xx[ii].min() - 0.5, xx[ii].max() + 0.5, N)
y2 = np.linspace(yy[ii].min() - 0.5, yy[ii].max() + 0.5, N)
x2, y2 = np.meshgrid(x2, y2)
ii = (np.abs(xx - J) <= threshold) & (np.abs(yy - I) <= threshold)
m = polyfit2d(xx[ii].flatten(), yy[ii].flatten(), corr[ii].flatten(), order=degree, product=True)

dfdx = lambda X: m[3] + m[4] * X[1] + m[5] * X[1] * X[1] + m[6] * 2 * X[0] + \
                 m[7] * 2 * X[0] * X[1] + m[8] * 2 * X[0] * X[1] * X[1]
dfdy = lambda X: m[1] + m[2] * 2 * X[1] + m[4] * X[0] + m[5] * X[0] * 2 * X[1] + \
                 m[7] * X[0] * X[0] + m[8] * X[0] * X[0] * 2 * X[1]

grad = lambda X: np.array([dfdx(X), dfdy(X)])

j0, i0 = root(grad, x0=np.array([J, I])).x

z = polyval2d(x2.flatten(), y2.flatten(), m, order=degree, product=True).reshape(x2.shape)

fig, ax = plt.subplots()
ax.imshow(ii, alpha=0.3, zorder=10)
ax.imshow(corr)
ax.set_xlim(x2.min() - 1, x2.max() + 1)
ax.set_ylim(y2.min() - 1, y2.max() + 1)
ax.contour(x2, y2, z, colors='k', linewidths=0.5)
ax.scatter(j0, i0)
"""


def argmax2d(im):
    h, w = im.shape  # number of rows, number of columns
    index = im.argmax()  # flattened index

    j = index % w  # row
    i = index // w  # column

    return i, j


def _get_h(f, g, l1, l2):

    height, width = f.shape
    h = np.zeros((height, width))

    if l1 > 0:
        if l2 > 0:
            h[l1:, l2:] = f[l1:, l2:] * g[:height - l1, :width - l2]
        else:
            l2 = -l2
            h[l1:, :width - l2] = f[l1:, :width - l2] * g[:height - l1, l2:]
    else:
        l1 = -l1
        if l2 > 0:
            h[:height - l1, l2:] = f[:height - l1, l2:] * g[l1:, :width - l2]
        else:
            l2 = -l2
            h[:height - l1, :width - l2] = f[:height - l1, :width - l2] * g[l1:, l2:]

    return h


def gaussian_window1d(sigma):
    nw = 3 * sigma + 1 if sigma % 2 == 0 else 3 * sigma
    x = np.arange(nw) - (nw - 1) / 2
    window = np.exp(-(x/sigma)**2)
    return window


def gaussian_window2d(sigma):

    window = gaussian_window1d(sigma)
    window = np.outer(window, window)
    return window/window.sum()


def cfg(f, g, l1, l2, window):
    hfg = _get_h(f, g, l1, l2)
    out = convolve(hfg, window, mode='same')

    """
    # debugging plot
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
    axs[0, 0].imshow(f, origin='lower')
    axs[0, 1].imshow(g, origin='lower')
    axs[1, 0].imshow(hfg, origin='lower')
    axs[1, 1].imshow(out, origin='lower')
    """

    return out


def windowed_2d_cc(f, g, lags1, lags2, sigma,
                   poly_degree=2, threshold=1, plot=False):

    # image dimensions
    height, width = f.shape

    # we define a gaussian window which will be reused constantly
    window = gaussian_window2d(sigma)

    # apply PEFs to the images according to what Dave says
    # # it is probably wrong however, recheck i,j,k1,k2 indices from Hale's paper?
    #f = prediction_error_filter(f, window)
    #g = prediction_error_filter(g, window)

    # smoothing the images with a sigma = 1 gaussian window
    smooth_window = gaussian_window2d(1)
    smooth_window = smooth_window / smooth_window.sum()
    f = convolve(f, smooth_window, mode='same')
    g = convolve(g, smooth_window, mode='same')

    # normalization factors
    cff = 1/np.sqrt(cfg(f, f, 0, 0, window))
    cgg = 1/np.sqrt(cfg(g, g, 0, 0, window))

    # convolution matrix which will store the cross-correlations
    convolutions = np.zeros((len(lags1), len(lags2), height, width))

    # we loop over every lag in both dimensions
    # # could be parallelized
    for i, l1 in enumerate(lags1):
        for j, l2 in enumerate(lags2):
            conv = cfg(f, g, l1, l2, window)
            nconv = conv * _get_h(cff, cgg, l1, l2)
            convolutions[i, j] = nconv

    # WARNING: ATTEMPT
    # here we smooth the correlations in the xy space for every lag
    window1d = gaussian_window1d(10)
    window1d = window1d / window1d.sum()
    convolutions = convolve1d(convolutions, window1d, axis=-1)
    convolutions = convolve1d(convolutions, window1d, axis=-2)

    # we loop in 2d to compute subpixel displacements
    # # also could be parallelized
    shift = np.zeros((2, height, width))
    coeffs = np.zeros((height, width), dtype='object')
    for h in range(height):
        for w in range(width):
            dw, dh = argmax2d(convolutions[:, :, h, w])

            dh, dw, m = poly.poly_interp_newton(convolutions[:, :, h, w], dw, dh,
                                                degree=poly_degree, threshold=threshold)

            shift[0, h, w] = dh - lags1.max()
            shift[1, h, w] = dw - lags2.max()
            coeffs[h, w] = m

    return convolutions, shift


def _directional_cfg(f, g, cff, cgg, lags, axis, search_window, smooth_window):

    """
    :param f: 2d image that we want to morph g into
    :param g: see f
    :param lags: number of lags to try and compute cross correlation
    :param axis: eiher 'horizontal' or 'vertical'
    :param search_window: needs to be a kernel (2d matrix)
    :return:
    """

    height, width = f.shape

    # hacky way to set the direction in which we will do the directional cfg
    if axis == 'horizontal':
        a1 = 0
        a2 = 1
        t = np.arange(width)
    elif axis == 'vertical':
        a1 = 1
        a2 = 0
        t = np.arange(height)
    else:
        raise ValueError('parameter "axis" needs to be set to either "horizontal" or "vertical"')

    # convolution matrix which will store the cross-correlations
    convolutions = np.zeros((len(lags), height, width))

    # we loop over every lag in the horizontal dimension
    # # could be parallelized
    for i, l in enumerate(lags):
        conv = cfg(f, g, a1 * l, a2 * l, search_window)
        nconv = conv * _get_h(cff, cgg, a1*l, a2*l)
        convolutions[i] = nconv

    window1d = gaussian_window1d(10)
    window1d = window1d / window1d.sum()
    convolutions = convolve1d(convolutions, window1d, axis=-1)
    convolutions = convolve1d(convolutions, window1d, axis=-2)

    # we now check every cell for the maximum lag
    # # could be parallelized
    dlag = np.zeros((height, width))
    max_corr = np.zeros((height, width))

    # is multiplying by a 0 centered window a good idea to avoid extreme lags?
    small_window = np.exp(-((lags / (lags.max())) ** 2))

    # here we find the index (or lag) with the maximum correlation value for every pixel
    for h in range(height):
        for w in range(width):

            # multiplication by a 0 centered window, is it a good idea? it further complicates the process
            cc = convolutions[:, h, w]
            index = cc.argmax()

            # we compute an offset according to Hale, letting us have non integer shifts
            # the offset can only be computed for non-edge cases
            offset = 0
            if index not in (0, len(lags) - 1):
                up = cc[index - 1] - cc[index + 1]
                down = 2 * cc[index - 1] + 2 * cc[index + 1] - 4 * cc[index]
                offset = up/down

            # we store both the maximum correlation value and the corresponding lag
            max_corr[h, w] = cc[index]
            dlag[h, w] = index + offset - lags.max()

    # here we warp the image according to the lag computed
    from image_warping import warp
    zeros = np.zeros_like(dlag)
    if axis == 'horizontal':
        dlag = np.array((zeros, dlag))
    else:
        dlag = np.array((dlag, zeros))

    tmp = warp(f, dlag)

    return tmp, dlag


def cyclic_search2d(f, g, vertical_lags, horizontal_lags, search_sigma, smooth_sigma, nsearch):

    # image dimensions
    height, width = f.shape

    # we define a gaussian window which will be reused constantly
    search_window = gaussian_window2d(search_sigma)
    smooth_window = gaussian_window2d(smooth_sigma)

    # apply PEFs to the images according to what Dave says
    # # it is probably wrong however, recheck i,j,k1,k2 indices from Hale's paper?
    """f = prediction_error_filter(f, search_window)
    g = prediction_error_filter(g, search_window)"""

    # normalization factors
    cff = 1/np.sqrt(cfg(f, f, 0, 0, search_window))
    cgg = 1/np.sqrt(cfg(g, g, 0, 0, search_window))

    tmp = g.copy()
    dw = np.zeros((nsearch + 1, height, width))
    dh = np.zeros((nsearch + 1, height, width))

    W = np.arange(width)
    H = np.arange(height)

    for n in range(nsearch):
        tmp, dw_n = _directional_cfg(f, tmp, cff, cgg, horizontal_lags, 'horizontal', search_window, smooth_window)
        tmp, dh_n = _directional_cfg(f, tmp, cff, cgg, vertical_lags, 'vertical', search_window, smooth_window)

        if n % 2 == 0:
            for w in range(width):
                dw[n + 1, :, w] = dw[n, :, w] + sinc_interp(H, dw_n[:, w], H + dw_n[:, w])
                dh[n + 1, :, w] = sinc_interp(H, dh_n[:, w], H + dh_n[:, w])
        else:
            for h in range(height):
                dw[n + 1, h] = sinc_interp(W, dw_n[h], W + dw_n[h])
                dh[n + 1, h] = dh[n, h] + sinc_interp(W, dh_n[h], W + dh_n[h])

    return tmp, (dw, dh)


if __name__ == '__main__':

    im0 = np.load('data/spongebob_warp2_0.npy', allow_pickle=True)
    im3 = np.load('data/spongebob_warp2_3.npy', allow_pickle=True)
    im5 = np.load('data/spongebob_warp2_5.npy', allow_pickle=True)
    im10 = np.load('data/spongebob_warp2_10.npy', allow_pickle=True)
    vf = np.load('data/vf_warp2.npy', allow_pickle=True)*5

    stepy = 25
    stepx = 12

    # just a regular homogeneous offset
    """im2 = im1[stepy:, stepx:]
    im1 = im1[:-stepy, :-stepx]"""

    maxlag = 25
    wlags = np.arange(-maxlag, maxlag + 1)
    hlags = np.arange(-maxlag, maxlag + 1)
    sigma = 10

    """
    hconv = cyclic_search2d(im1, im3, hlags, wlags,
                            search_sigma=12,
                            smooth_sigma=5,
                            nsearch=5)
    """

    cyclic_search2d(im10, im0, hlags, wlags, search_sigma=sigma, smooth_sigma=10, nsearch=5)

    conv, shift = windowed_2d_cc(im10, im0, hlags, wlags, sigma=sigma,
                                 poly_degree=2, threshold=1,
                                 plot=True)

    plt.show()

    # figure showing the deformation field as a streamplot
    fig, axs = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(10, 5))
    height, width = im0.shape
    H, W = np.arange(height), np.arange(width)
    ww, hh = np.meshgrid(W, H)
    norm = np.linalg.norm(vf, axis=0)
    axs[0].streamplot(ww, hh, vf[0], vf[1], color=norm, linewidth=0.5)
    norm2 = np.linalg.norm(shift, axis=0)
    norm2 = norm2 / norm.max()
    norm2[norm2 >= 1] = 1
    im = axs[1].streamplot(ww, hh, shift[1], shift[0], color=norm2, linewidth=0.5)

    axs[0].set_title('Original field', loc='left')
    axs[1].set_title('LCC approx.', loc='left')

    for ax in axs:
        ax.set_aspect(1)
        ax.set_ylim(0, height)
        ax.set_xlim(0, width)
    fig.subplots_adjust(wspace=0.05)
    fig.savefig('progress/infered_deformation_field.png', dpi=800, bbox_inches='tight')

    # figure showing the x and y components respectively
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(8, 6))
    axs[0, 0].imshow(vf[0], origin='lower')
    axs[0, 0].text(0.01, 0.99, '$u_x$', ha='left', va='top', transform=axs[0, 0].transAxes)
    axs[0, 1].imshow(vf[1], origin='lower')
    axs[0, 1].text(0.01, 0.99, '$u_y$', ha='left', va='top', transform=axs[0, 1].transAxes)
    axs[1, 0].imshow(shift[1], vmin=vf[0].min(), vmax=vf[0].max(), origin='lower')
    axs[1, 0].text(0.01, 0.99, '$\hat{u}_x$', ha='left', va='top', transform=axs[1, 0].transAxes)
    axs[1, 1].imshow(shift[0], vmin=vf[1].min(), vmax=vf[1].max(), origin='lower')
    axs[1, 1].text(0.01, 0.99, '$\hat{u}_y$', ha='left', va='top', transform=axs[1, 1].transAxes)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.savefig('progress/vf_2_comparison.png', dpi=800, bbox_inches='tight')

    plt.show()

"""
fig, axs = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(10, 5))
img = plt.imread('data/spongebob.png')[::-1]
axs[0].imshow(img, origin='lower')
axs[0].set_title('Original signal', loc='left')
axs[1].streamplot(hh, ww, vf[0], vf[1], color=norm, linewidth=1)
axs[1].set_aspect(1)
axs[1].set_title('Deformation field', loc='left')
fig.savefig('progress/spongebob_example.png', dpi=800, bbox_inches='tight')

fig, axs = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(10, 5))
axs[0].imshow(im0, origin='lower', cmap='gray')
axs[0].set_title('Original signal', loc='left')

axs[1].imshow(im5, origin='lower', cmap='gray')
axs[1].set_title('Deformed signal', loc='left')

height, width = im0.shape
axs[0].set_xlim(0, width)
axs[0].set_ylim(0, height)
H, W = np.arange(height), np.arange(width)
hh, ww = np.meshgrid(W, H)
norm = np.linalg.norm(vf, axis=0)
for ax in axs:
    ax.streamplot(hh, ww, vf[0], vf[1], color=norm, linewidth=1)

fig.savefig('progress/deformed_spongebob.png', dpi=800, bbox_inches='tight')
"""


"""
fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
axs[0, 0].imshow(im1, origin='lower', cmap='gray')
axs[1, 0].imshow(im3, origin='lower', cmap='gray')

height, width = im1.shape

axs[0, 0].set_xlim(0, width)
axs[0, 0].set_ylim(0, height)

H, W = np.arange(height), np.arange(width)
hh, ww = np.meshgrid(W, H)
norm = np.linalg.norm(vf, axis=0)
for ax in axs[:, 0]:
    ax.streamplot(hh, ww, vf[0], vf[1], color=norm, linewidth=0.5)

norm = np.linalg.norm(shift, axis=0)
for ax in axs[:, 1]:
    ax.streamplot(hh, ww, shift[0], shift[1], color=norm, linewidth=0.5)
"""

"""
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# moving lags
fig = plt.figure()
ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))

l10 = 0
l20 = 0

im = ax.imshow(convolutions[int(lags1.max()), int(lags2.max())], vmin=-1, vmax=1)

x0, y0, HEIGHT, WIDTH = make_axes_locatable(ax).get_position()
x1 = x0 + WIDTH
y1 = y0 + HEIGHT

l1_ax = fig.add_axes((x0, y1 + 0.01 * HEIGHT, WIDTH, 0.05 * HEIGHT))
l1_slider = Slider(l1_ax, '', valmin=lags2.min(), valmax=lags2.max(), valinit=0, valstep=lags2,
                   valfmt='$l_2$: %.f')

l2_ax = fig.add_axes((x1 + 0.01 * WIDTH, y0, 0.05 * WIDTH, HEIGHT))
l2_slider = Slider(l2_ax, '', valmin=lags1.min(), valmax=lags1.max(), valinit=0, valstep=lags1,
                   valfmt='$l_1$: %.f',
                   orientation='vertical')

def update_l2(l2):
    l1 = l1_slider.val
    im.set_data(convolutions[int(l1 - lags1.max()), int(l2 - lags2.max())])
    fig.canvas.draw_idle()

def update_l1(l1):
    l2 = l2_slider.val
    im.set_data(convolutions[int(l1 - lags1.max()), int(l2 - lags2.max())])
    fig.canvas.draw_idle()

l2_slider.on_changed(update_l2)
l1_slider.on_changed(update_l1)"""

"""# moving i, j
fig2 = plt.figure(figsize=(8, 8))
ax2 = fig2.add_axes((0.1, 0.1, 0.8, 0.8))

nlags1, nlags2, height, width = convolutions.shape
i0, j0 = height // 2, width // 2

im2 = ax2.imshow(convolutions[:, :, i0, j0], vmin=-1, vmax=1, aspect='auto',
               extent=[lags2.min(), lags2.max(), lags1.min(), lags2.max()])

x0, y0, HEIGHT, WIDTH = make_axes_locatable(ax2).get_position()
x1 = x0 + WIDTH
y1 = y0 + HEIGHT

i_ax = fig2.add_axes((x0, y1 + 0.01 * HEIGHT, WIDTH, 0.05 * HEIGHT))
i_slider = Slider(i_ax, '', valmin=0, valmax=height - 1, valinit=i0, valstep=1,
                  valfmt='$i$: %.f')

j_ax = fig2.add_axes((x1 + 0.01 * WIDTH, y0, 0.05 * WIDTH, HEIGHT))
j_slider = Slider(j_ax, '', valmin=0, valmax=width - 1, valinit=j0, valstep=1,
                  valfmt='$j$: %.f',
                  orientation='vertical')

def update_j(j):
    i = i_slider.val
    im2.set_data(convolutions[:, :, int(i), int(j)])
    fig2.canvas.draw_idle()

def update_i(i):
    j = j_slider.val
    im2.set_data(convolutions[:, :, int(i), int(j)])
    fig2.canvas.draw_idle()

i_slider.on_changed(update_i)
j_slider.on_changed(update_j)"""