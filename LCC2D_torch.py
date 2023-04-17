import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import GaussianBlur
import gaussian_windows as gw

from optim import ShiftOptim
from poly_interp import vec_polyfit2d, PytorchPolyFit2D

from itertools import product

import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _get_h(f, l1, l2):

    height, width = f.shape
    h = torch.zeros((height, width))

    if l1 > 0:
        if l2 > 0:
            h[l1:, l2:] = f[:height - l1, :width - l2]
        else:
            l2 = -l2
            h[l1:, :width - l2] = f[:height - l1, l2:]
    else:
        l1 = -l1
        if l2 > 0:
            h[:height - l1, l2:] = f[l1:, :width - l2]
        else:
            l2 = -l2
            h[:height - l1, :width - l2] = f[l1:, l2:]

    return h


class LCC2D:

    def __init__(self, f, g,
                 hlags,
                 wlags,
                 search_sigma,
                 threshold=1,
                 verbose=True):

        assert f.shape == g.shape, 'f and g need to have the same shape'
        self.height, self.width = f.shape

        self.f = torch.tensor(f, device=device, dtype=torch.float32)
        self.g = torch.tensor(g, device=device, dtype=torch.float32)

        self.threshold = threshold
        self.verbose = verbose

        self.hlags = hlags
        self.wlags = wlags

        self.search_sigma = search_sigma

        self.dw, self.dh, self.corr, self.convolutions = self.lcc()

        self.subdw, self.subdh = self.subpixel()

    def lcc(self):


        t0 = time.time()
        if self.verbose:
            print(f'computing localized cross correlations')

        search_window = gw.torch_gaussian_window2d(self.search_sigma).to(self.f.dtype)

        # convolutions is a tensor of size (hlags * wlags, 1 channel, height, width)
        convolutions = torch.zeros((len(self.hlags)*len(self.wlags), 1, self.height, self.width), device=device)

        # we also need to apply normalization factors to the images
        cff = self.f/(F.conv2d((self.f * self.f).reshape(1, 1, self.height, self.width),
                               weight=search_window, padding='same').sqrt()[0, 0])
        cgg = self.g/(F.conv2d((self.g * self.g).reshape(1, 1, self.height, self.width),
                               weight=search_window, padding='same').sqrt()[0, 0])

        # this is the only part involving for loops,
        for i, (l1, l2) in enumerate(product(self.hlags, self.wlags)):
            convolutions[i, 0] = _get_h(cff, l1, l2)

        # multiplication of the moved image with the unmoved image
        convolutions = convolutions * cgg.reshape(1, 1, self.height, self.width)

        # we finally localize the cross correlation with a gaussian blur
        convolutions = F.conv2d(convolutions, weight=search_window, padding='same')

        # we look for the maximum indices
        values, indices = torch.max(convolutions.reshape(len(self.hlags) * len(self.wlags), -1), dim=0)

        # reshaping them like the initial image was
        values = torch.reshape(values, (self.height, self.width))
        indices = torch.reshape(indices, (self.height, self.width))
        convolutions = convolutions.reshape(len(self.hlags), len(self.wlags), self.height, self.width)

        # changing the indices to i, j coordinates
        dw = indices % len(self.wlags) - self.hlags.max()
        dh = torch.div(indices, len(self.wlags), rounding_mode='floor') - self.wlags.max()

        if self.verbose:
            t1 = time.time()
            print(f'\rtook {(t1-t0):.2f} seconds')

        return dw, dh, values, convolutions

    def subpixel(self):

        t0 = time.time()

        if self.verbose:
            print(f'computing subpixel displacements')

        # taking into account where the index is at the border
        dh = self.dh.clone()
        dh[dh == self.hlags.min()] += 1
        dh[dh == self.hlags.max()] -= 1

        dw = self.dw.clone()
        dw[dw == self.wlags.min()] += 1
        dw[dw == self.wlags.max()] -= 1

        # we need to extract the surrounding images to the integer maximum to be able to fit the quadratic surface
        i = dh.flatten() + self.hlags.max()
        j = dw.flatten() + self.wlags.max()

        # there has to be a better way to index this lol
        arange = torch.arange(self.width * self.height)
        conv = self.convolutions.reshape(len(self.hlags), len(self.wlags), self.height * self.width)

        offset = torch.arange(-self.threshold, self.threshold + 1)
        n = 2 * self.threshold + 1

        # we store the 3x3 cross-correlation around the maximum in the images tensor
        images = torch.zeros(n, n, self.width * self.height)

        # we fill the images tensor by iterating every combination of the offset [-1, 0, 1]
        for k, (di, dj) in enumerate(product(offset, offset)):
            i2, j2 = i + di, j + dj
            images[di + self.threshold, dj + self.threshold] = conv[i2, j2, arange]

        # the coordinates are the same for every quadratic surface
        x = torch.arange(n, dtype=torch.float32) - self.threshold
        X = torch.stack(torch.meshgrid(x, x)).reshape(2, -1)

        # this vectorized implementation for fitting polynomials goes hard
        poly = PytorchPolyFit2D(X, images.reshape(n * n, -1))
        dh, dw = poly.fit()

        if self.verbose:
            t1 = time.time()
            print(f'\rtook {(t1-t0):.2f} seconds')

        return self.dw + dw.reshape(self.height, self.width), self.dh + dh.reshape(self.height, self.width)


if __name__ == '__main__':

    im0 = np.load('data/spongebob_warp2_0.npy', allow_pickle=True).astype(np.float32)
    im5 = np.load('data/spongebob_warp2_5.npy', allow_pickle=True).astype(np.float32)
    vf = np.load('data/vf_warp2.npy', allow_pickle=True) * 5

    maxlag = 20
    wlags = np.arange(-maxlag, maxlag + 1)
    hlags = np.arange(-maxlag, maxlag + 1)
    search_sigma = 15

    lcc = LCC2D(im0, im5,
                hlags, wlags,
                search_sigma,
                threshold=1)

    shift = torch.stack((lcc.subdw, lcc.subdh)).to(torch.float32)

    lmbda = 1e2
    beta = 1e4

    model = ShiftOptim(lcc.f, lcc.g, shift, lmbda=lmbda, beta=beta, correlation=lcc.convolutions)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out, losses = model.fit(optimizer, n=3000)

    from image_warping import warp_cv2

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

    fig, axs = plt.subplots(1, 5, sharex='all', sharey='all', figsize=(14, 3))
    axs[0].imshow(im0, origin='lower')
    axs[0].text(0.99, 0.99, f'Original', ha='right', va='top', transform=axs[0].transAxes)

    axs[1].imshow(im5, origin='lower')
    axs[1].text(0.99, 0.99, f'Warped', ha='right', va='top', transform=axs[1].transAxes)

    axs[2].imshow(warp_cv2(im0, torch.stack((lcc.dw, lcc.dh)).numpy()), origin='lower')
    axs[2].text(0.99, 0.99, f'Integer disp.', ha='right', va='top', transform=axs[2].transAxes)

    axs[3].imshow(warp_cv2(im0, shift.numpy()), origin='lower')
    axs[3].text(0.99, 0.99, f'Subpixel disp.', ha='right', va='top', transform=axs[3].transAxes)

    axs[4].imshow(model.forward().detach(), origin='lower')
    axs[4].text(0.99, 0.99, f'Optimized disp.', ha='right', va='top', transform=axs[4].transAxes)

    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    fig, axs = plt.subplots(3, 2, sharex='all', sharey='all')
    axs[0, 0].imshow(shift[0], origin='lower', vmin=vf[0].min(), vmax=vf[0].max())
    axs[0, 0].text(0.99, 0.99, 'lcc $\hat{v}_x$', ha='right', va='top', transform=axs[0, 0].transAxes)

    axs[0, 1].imshow(shift[1], origin='lower', vmin=vf[1].min(), vmax=vf[1].max())
    axs[0, 1].text(0.99, 0.99, 'lcc $\hat{v}_y$', ha='right', va='top', transform=axs[0, 1].transAxes)

    axs[1, 0].imshow(out[0], origin='lower', vmin=vf[0].min(), vmax=vf[0].max())
    axs[1, 0].text(0.99, 0.99, 'optim. $\hat{v}_x$', ha='right', va='top', transform=axs[1, 0].transAxes)

    axs[1, 1].imshow(out[1], origin='lower', vmin=vf[1].min(), vmax=vf[1].max())
    axs[1, 1].text(0.99, 0.99, 'optim $\hat{v}_y$', ha='right', va='top', transform=axs[1, 1].transAxes)

    axs[2, 0].imshow(vf[0], origin='lower', vmin=vf[0].min(), vmax=vf[0].max())
    axs[2, 0].text(0.99, 0.99, '$v_x$', ha='right', va='top', transform=axs[2, 0].transAxes)

    axs[2, 1].imshow(vf[1], origin='lower', vmin=vf[1].min(), vmax=vf[1].max())
    axs[2, 1].text(0.99, 0.99, '$v_y$', ha='right', va='top', transform=axs[2, 1].transAxes)

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.show()


"""
im0 = (im0 - im0.min())/(im0.max() - im0.min())
im5 = (im5 - im5.min())/(im5.max() - im5.min())
fig = plt.figure(figsize=(12.5, 3))
gs = fig.add_gridspec(2, 8)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])
h_ax = fig.add_subplot(gs[:, 2:4])
g_ax = fig.add_subplot(gs[:, 4:6])
corr_ax = fig.add_subplot(gs[:, 6:])
i, j = 150, 120
ax0.imshow(im0, origin='lower')
ax1.imshow(im5, origin='lower')
ax2.imshow(im0, origin='lower')
ax3.imshow(im5, origin='lower')
offset = 30
for ax in (ax2, ax3):
    ax.set_xlim(j-offset, j+offset)
    ax.set_ylim(i-offset, i+offset)
ax0.set_title('original', loc='left', size='small', pad=0.5)
ax1.set_title('deformed', loc='left', size='small', pad=0.5)

h, w = im0.shape
h = np.arange(h)
w = np.arange(w)
ww, hh = np.meshgrid(w, h)
gaussw = np.exp((-(w - j)**2)/10**2)
gaussh = np.exp((-(h - i)**2)/10**2)
gauss = np.outer(gaussh, gaussw)
h_ax.imshow(im0 * im5, origin='lower')
h_ax.set_xlim(j - offset, j + offset)
h_ax.set_ylim(i - offset, i + offset)
h_ax.set_xticks([])
h_ax.set_yticks([])
h_ax.set_title('original $\\times$ deformed', loc='left', size='small', pad=0.5)

g_ax.imshow(im0 * im5 * gauss, origin='lower')
g_ax.set_xlim(j - offset, j + offset)
g_ax.set_ylim(i - offset, i + offset)
g_ax.set_xticks([])
g_ax.set_yticks([])
g_ax.set_title('gaussian window', loc='left', size='small', pad=0.5)


corr_ax.imshow(lcc.convolutions[:, :, i, j], extent=[-20, 20, -20, 20], origin='lower')
corr_ax.set_xlabel('$l_w$')
corr_ax.set_ylabel('$l_h$', rotation=0)
corr_ax.yaxis.tick_right()
corr_ax.yaxis.set_label_position('right')
corr_ax.set_title(f'Localized cross correlation', loc='left', size='small', pad=0.5)
for ax in (ax0, ax1, ax2, ax3):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(j, i, c='red', ec='k', s=15)
    

fig.subplots_adjust(wspace=0.00)
fig.savefig('powerpoints/figures/lcc2d_exemple.png', dpi=800, bbox_inches='tight')"""

"""
# figure showing the deformation field as a streamplot
fig, axs = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(15, 4))
height, width = im0.shape
H, W = np.arange(height), np.arange(width)
ww, hh = np.meshgrid(W, H)
norm = np.linalg.norm(vf, axis=0)
axs[0].streamplot(ww, hh, vf[0], vf[1], color=norm, linewidth=0.5)

norm2 = np.linalg.norm(shift, axis=0)
norm2[norm2 >= norm.max()] = norm.max()
axs[1].streamplot(ww, hh, shift[0], shift[1], color=norm2, linewidth=0.5)

norm3 = np.linalg.norm(out, axis=0)
norm3[norm3 >= norm.max()] = norm.max()
axs[2].streamplot(ww, hh, out[0], out[1], color=norm3, linewidth=0.5)

axs[0].set_title('original field', loc='left')
axs[1].set_title('LCC approx.', loc='left')
axs[2].set_title('optimized LCC', loc='left')
for ax in axs:
    ax.set_aspect(1)
    ax.set_ylim(0, height)
    ax.set_xlim(0, width)
fig.subplots_adjust(wspace=0.05)
"""